from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import phase1_2_rad5_64_pipeline as p12
from dc_balanced_fixed_total import make_measurement_from_rows, tv_baseline
from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src import phase69B_controlled_gauge_cgan_pilot as p69b
from src import phase1_4v4b0_scoring as b0
from src.dc_balanced import (
    build_dc_balanced_rows,
    centered_rmse,
    dct_band_rmse,
    full_rmse,
    mean_abs_error,
    row_audit,
)
from src.operator_conditioned_nullspace import (
    MatrixFreeNullProjector,
    SmallNullspaceUNet,
    reconstruct_with_projected_residual,
)
from src.phase2_fresh_operator import resolve_device
from src.phase2_witness import paired_percentile_bootstrap, repo_state, sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "operator_conditioned_dct_canary_smoke.yaml"


class OperatorConditionedCanaryError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val):
            return None
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise OperatorConditionedCanaryError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def sha256_numpy(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def split_indices(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    split_cfg = dict(config["splits"])
    out: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "source": "STL10 train+unlabeled",
        "train_indices_full_sha256": p69a.sha256_np(train_full, sort_int64=True),
        "final_v4_or_locked_used": False,
    }
    used: list[int] = []
    for name in ["train", "val", "test"]:
        offset = int(split_cfg[name]["offset"])
        count = int(split_cfg[name]["count"])
        if offset < 0 or offset + count > train_full.shape[0]:
            raise OperatorConditionedCanaryError(f"SPLIT_OUT_OF_RANGE:{name}:{offset}:{count}:{train_full.shape[0]}")
        arr = train_full[offset : offset + count].copy()
        out[name] = arr
        manifest[name] = {
            "offset": offset,
            "count": count,
            "indices_sha256": p69a.sha256_np(arr),
        }
        used.extend(int(v) for v in arr.tolist())
    if len(set(used)) != len(used):
        raise OperatorConditionedCanaryError("TRAIN_VAL_TEST_SPLITS_OVERLAP")
    manifest["combined_indices_sha256"] = p69a.sha256_np(np.asarray(used, dtype=np.int64))
    return out, manifest


def make_loader(indices: np.ndarray, *, batch_size: int, shuffle: bool, seed: int):
    base = p69a.stl10_dataset("train+unlabeled")
    subset = p69a.IndexedDataset(base, np.asarray(indices, dtype=np.int64))
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(subset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, drop_last=False, generator=gen)


def batch_to_flat(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, labels, indices = batch
    return x.to(device=device, dtype=torch.float32), torch.as_tensor(labels).long(), torch.as_tensor(indices).long()


def make_rows(config: Mapping[str, Any]) -> np.ndarray:
    op = dict(config["operator"])
    img_size = int(op.get("img_size", 64))
    total_m = int(op.get("total_m", 41))
    family = str(op.get("family", "dct"))
    rows = build_dc_balanced_rows(family, total_m - 1, dim=img_size * img_size, img_size=img_size, seed=int(op.get("seed", 0)))
    return rows.astype(np.float32)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    projector: MatrixFreeNullProjector,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    img_size: int,
    cond_scalar: float,
    clip_grad: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    rels: list[float] = []
    for batch in loader:
        x_img, _labels, _indices = batch_to_flat(batch, device)
        x = x_img.reshape(x_img.shape[0], -1)
        y = projector.measurement(x)
        r = projector.data_anchor(y)
        target_null = x - r
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = reconstruct_with_projected_residual(model, projector, r, y, img_size=img_size, cond_scalar=cond_scalar)
                loss = F.mse_loss(out.null_hat, target_null)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
        else:
            out = reconstruct_with_projected_residual(model, projector, r, y, img_size=img_size, cond_scalar=cond_scalar)
            loss = F.mse_loss(out.null_hat, target_null)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        rels.append(float(out.relmeaserr.detach().max().cpu()))
    return {"loss_mean": float(np.mean(losses)), "relmeaserr_max": float(np.max(rels))}


@torch.no_grad()
def predict_model(
    *,
    model: torch.nn.Module,
    projector: MatrixFreeNullProjector,
    loader,
    device: torch.device,
    img_size: int,
    cond_scalar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    xs, xhats, rels, labels, indices = [], [], [], [], []
    for batch in loader:
        x_img, lab, idx = batch_to_flat(batch, device)
        x = x_img.reshape(x_img.shape[0], -1)
        y = projector.measurement(x)
        r = projector.data_anchor(y)
        out = reconstruct_with_projected_residual(model, projector, r, y, img_size=img_size, cond_scalar=cond_scalar)
        xs.append(x.detach().cpu().numpy().astype(np.float32))
        xhats.append(out.x_hat.detach().cpu().numpy().astype(np.float32))
        rels.append(out.relmeaserr.detach().cpu().numpy().astype(np.float64))
        labels.append(lab.numpy().astype(np.int64))
        indices.append(idx.numpy().astype(np.int64))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(xhats, axis=0),
        np.concatenate(rels, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(indices, axis=0),
    )


@torch.no_grad()
def predict_joint(projector: MatrixFreeNullProjector, x: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y = projector.measurement(xb)
    r = projector.data_anchor(y)
    rel = projector.relmeaserr(r, y)
    return r.detach().cpu().numpy().astype(np.float32), rel.detach().cpu().numpy().astype(np.float64)


def tikhonov_anchor(rows: np.ndarray, x: np.ndarray, *, lambda_: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    A = torch.from_numpy(rows).to(device=device, dtype=torch.float64)
    xb = torch.from_numpy(x).to(device=device, dtype=torch.float64)
    y = xb @ A.T
    eye = torch.eye(A.shape[0], device=device, dtype=torch.float64)
    gram = A @ A.T + float(lambda_) * eye
    coeff = torch.linalg.solve(gram, y.T).T
    out = coeff @ A
    rel = torch.linalg.norm(out @ A.T - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    return out.detach().cpu().numpy().astype(np.float32), rel.detach().cpu().numpy().astype(np.float64)


def ssim_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int) -> np.ndarray:
    from skimage.metrics import structural_similarity

    pred = np.clip(xhat.reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(x.reshape(-1, img_size, img_size), 0.0, 1.0)
    return np.asarray([structural_similarity(truth[i], pred[i], data_range=1.0, win_size=7, channel_axis=None) for i in range(pred.shape[0])])


def rapsd_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int) -> np.ndarray:
    pred = np.clip(xhat.reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(x.reshape(-1, img_size, img_size), 0.0, 1.0)
    return np.asarray([b0.rapsd_distance(pred[i], truth[i], bins=32) for i in range(pred.shape[0])])


def lpips_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int, device_name: str) -> np.ndarray | None:
    if xhat.shape[0] == 0:
        return None
    pred = np.clip(xhat.reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(x.reshape(-1, img_size, img_size), 0.0, 1.0)
    try:
        return b0.compute_lpips_matrix(pred[:, None, :, :], truth, device_name=device_name).reshape(pred.shape[0])
    except Exception:
        return None


def metric_bundle(
    *,
    method: str,
    xhat: np.ndarray,
    x: np.ndarray,
    rel: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    img_size: int,
    compute_lpips: bool,
    lpips_device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    f = full_rmse(xhat, x)
    c = centered_rmse(xhat, x)
    mean_err = mean_abs_error(xhat, x)
    bands = dct_band_rmse(xhat, x, img_size=img_size, low_count=40)
    psnr = -20.0 * np.log10(np.maximum(f, 1e-12))
    ssim = ssim_values(xhat, x, img_size=img_size)
    rapsd = rapsd_values(xhat, x, img_size=img_size)
    lp = lpips_values(xhat, x, img_size=img_size, device_name=lpips_device) if compute_lpips else None
    summary = {
        "method": method,
        "n": int(x.shape[0]),
        "centered_rmse_mean": float(c.mean()),
        "full_rmse_mean": float(f.mean()),
        "dc_error_mean": float(mean_err.mean()),
        "dct_non_dc_low_rmse_mean": float(bands["dct_non_dc_low_rmse"].mean()),
        "dct_mid_rmse_mean": float(bands["dct_mid_rmse"].mean()),
        "dct_high_rmse_mean": float(bands["dct_high_rmse"].mean()),
        "psnr_mean": float(psnr.mean()),
        "ssim_mean": float(ssim.mean()),
        "rapsd_mean": float(rapsd.mean()),
        "lpips_mean": "[DATA MISSING]" if lp is None else float(lp.mean()),
        "relmeaserr_max": float(np.max(rel)),
        "relmeaserr_mean": float(np.mean(rel)),
    }
    per = []
    for i in range(x.shape[0]):
        per.append(
            {
                "method": method,
                "row": int(i),
                "source_index": int(indices[i]),
                "label": int(labels[i]),
                "centered_rmse": float(c[i]),
                "full_rmse": float(f[i]),
                "dc_error": float(mean_err[i]),
                "dct_non_dc_low_rmse": float(bands["dct_non_dc_low_rmse"][i]),
                "dct_mid_rmse": float(bands["dct_mid_rmse"][i]),
                "dct_high_rmse": float(bands["dct_high_rmse"][i]),
                "psnr": float(psnr[i]),
                "ssim": float(ssim[i]),
                "rapsd": float(rapsd[i]),
                "lpips": "[DATA MISSING]" if lp is None else float(lp[i]),
                "relmeaserr": float(rel[i]),
            }
        )
    return summary, per


def paired_delta(per_rows: Sequence[Mapping[str, Any]], method: str, reference: str, metric: str, *, reps: int, seed: int) -> dict[str, Any]:
    a = {int(r["source_index"]): float(r[metric]) for r in per_rows if r["method"] == method}
    b = {int(r["source_index"]): float(r[metric]) for r in per_rows if r["method"] == reference}
    keys = sorted(set(a) & set(b))
    if not keys:
        raise OperatorConditionedCanaryError(f"NO_PAIRING_KEYS:{method}:{reference}")
    delta = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    boot = paired_percentile_bootstrap(delta, reps=int(reps), seed=int(seed))
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "n": int(delta.shape[0]),
        "mean_delta": float(delta.mean()),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "wins": int(np.sum(delta < 0)),
        "losses": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
    }


def write_math(path: Path) -> None:
    text = r"""# Operator-Conditioned Null-Space Canary Math

For a fixed final operator \(A\), write \(r=A^\dagger y\) and \(P_0^A=I-A^\dagger A\).
The joint minimum-norm estimate is \(r\), which corresponds to using zero conditional
null-space mean.  A squared-error Bayes estimator has the form
\[
\hat x=r+g(y,A),\qquad g(y,A)\in\ker A,
\]
and minimizes
\[
\mathbb E\|P_0^A X-g(Y,A)\|^2.
\]
The unique minimizer is
\[
g^\star(Y,A)=\mathbb E[P_0^A X\mid Y,A].
\]
Thus any improvement over joint minimum-norm must come from predicting the
remaining null-space component, not from changing measured row-space content.

The canary implements
\[
\hat x=r+P_0^A f_\theta(r,\mathcal E(A)).
\]
Since \(A P_0^A=0\),
\[
A\hat x=Ar+A P_0^Af_\theta=y.
\]
The implementation stores only \(A\) and \((AA^\top)^\dagger\), and computes
\(P_0^Av=v-A^\top(AA^\top)^\dagger Av\).  It never forms dense \(P_0^A\).
"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def evaluate_frozen_prior(
    *,
    config: Mapping[str, Any],
    rows: np.ndarray,
    test_indices: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    fp_cfg = dict(config.get("frozen_prior", {}))
    if not bool(fp_cfg.get("enabled", False)):
        return None, {"status": "SKIPPED_BY_CONFIG"}
    measurement, base_config = make_measurement_from_rows(
        rows,
        config={"operator": {"img_size": config["operator"]["img_size"], "matrix_normalization": "orthonormal_rows"}, "batch_size": config.get("batch_size", 8)},
        run_config={"operator_seed": int(config["operator"].get("seed", 0))},
        role="operator_conditioned_canary_frozen_prior_eval",
        arm_id="dc_plus_40_non_dc_dct",
        device=device,
    )
    generator, gen_config, _ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
        Path(fp_cfg.get("checkpoint", config.get("checkpoint", p12.PHASE79_CKPT))),
        base_config,
        measurement,
        device,
    )
    if missing or unexpected:
        raise OperatorConditionedCanaryError(f"FROZEN_PRIOR_LOAD_NOT_STRICT:{missing}:{unexpected}")
    base = p69a.stl10_dataset("train+unlabeled")
    subset = p69a.IndexedDataset(base, np.asarray(test_indices, dtype=np.int64))
    loader = DataLoader(subset, batch_size=int(config.get("batch_size", 8)), shuffle=False, num_workers=0)
    xs, labels, indices = [], [], []
    for batch in loader:
        xb, lab, idx = batch_to_flat(batch, device)
        xs.append(xb.detach().cpu())
        labels.append(lab)
        indices.append(idx)
    x = torch.cat(xs, 0)
    split = {
        "name": "frozen_prior_test",
        "x": x,
        "y": torch.from_numpy(np.asarray(x.reshape(x.shape[0], -1).numpy(), dtype=np.float32) @ rows.T).float(),
        "labels": torch.cat(labels, 0),
        "indices": torch.cat(indices, 0),
    }
    cache = p12.build_candidate_cache(
        generator,
        measurement,
        gen_config,
        split,
        out=output_dir,
        k=int(fp_cfg.get("candidate_k", 16)),
        seed=int(fp_cfg.get("candidate_seed", 424242)),
        device=device,
    )
    xhat = (cache["r"] + cache["cand_n"].mean(axis=1)).numpy().astype(np.float32)
    return xhat, {
        "status": "PASS",
        "checkpoint_state_key": state_key,
        "candidate_k": int(fp_cfg.get("candidate_k", 16)),
        "cache_path": str(output_dir / "candidate_cache" / "frozen_prior_test_k16.pt"),
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/operator_conditioned_nullspace/dct_canary_smoke"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_math(reports / "math_derivation.md")
    device = resolve_device(str(config.get("device", "cuda")))
    seed = int(config.get("seed", 20260625))
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = make_rows(config)
    rows_t = torch.from_numpy(rows).to(device=device, dtype=torch.float32)
    projector = MatrixFreeNullProjector(rows_t)
    img_size = int(config["operator"].get("img_size", 64))
    cond_scalar = float(rows.shape[0] / rows.shape[1])
    splits, split_manifest = split_indices(config)
    loaders = {
        "train": make_loader(splits["train"], batch_size=int(config.get("batch_size", 16)), shuffle=True, seed=seed + 1),
        "val": make_loader(splits["val"], batch_size=int(config.get("batch_size", 16)), shuffle=False, seed=seed + 2),
        "test": make_loader(splits["test"], batch_size=int(config.get("batch_size", 16)), shuffle=False, seed=seed + 3),
    }

    model_cfg = dict(config.get("model", {}))
    model = SmallNullspaceUNet(
        in_channels=2,
        base_channels=int(model_cfg.get("base_channels", 32)),
        blocks=int(model_cfg.get("blocks", 2)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("lr", 2e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    epochs = int(config.get("training", {}).get("epochs", 5))
    clip_grad = float(config.get("training", {}).get("clip_grad", 1.0))
    train_log: list[dict[str, Any]] = []
    best = {"epoch": -1, "val_centered_rmse": float("inf"), "state": None}
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            projector=projector,
            loader=loaders["train"],
            optimizer=opt,
            device=device,
            img_size=img_size,
            cond_scalar=cond_scalar,
            clip_grad=clip_grad,
            scaler=scaler,
        )
        x_val, pred_val, rel_val, lab_val, idx_val = predict_model(
            model=model,
            projector=projector,
            loader=loaders["val"],
            device=device,
            img_size=img_size,
            cond_scalar=cond_scalar,
        )
        val_centered = float(centered_rmse(pred_val, x_val).mean())
        val_joint, _rel_joint = predict_joint(projector, x_val, device=device)
        val_joint_centered = float(centered_rmse(val_joint, x_val).mean())
        row = {
            "epoch": epoch,
            **train_stats,
            "val_centered_rmse": val_centered,
            "val_joint_centered_rmse": val_joint_centered,
            "val_delta_vs_joint": val_centered - val_joint_centered,
            "val_relmeaserr_max": float(rel_val.max()),
        }
        train_log.append(row)
        if val_centered < best["val_centered_rmse"]:
            best = {
                "epoch": epoch,
                "val_centered_rmse": val_centered,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
    if best["state"] is None:
        raise OperatorConditionedCanaryError("NO_BEST_STATE_RECORDED")
    model.load_state_dict(best["state"])
    ckpt_path = output_dir / "checkpoints" / "best.pt"
    ensure_dir(ckpt_path.parent)
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": json_safe(config),
            "best_epoch": int(best["epoch"]),
            "rows_sha256": sha256_numpy(rows),
            "projector": projector.diagnostics(),
        },
        tmp,
    )
    os.replace(tmp, ckpt_path)

    x_test, pred_test, rel_model, labels, indices = predict_model(
        model=model,
        projector=projector,
        loader=loaders["test"],
        device=device,
        img_size=img_size,
        cond_scalar=cond_scalar,
    )
    joint, rel_joint = predict_joint(projector, x_test, device=device)
    tikh, rel_tikh = tikhonov_anchor(rows, x_test, lambda_=float(config.get("classical", {}).get("tikhonov_lambda", 1e-5)), device=device)
    tv, tv_diag = tv_baseline(rows, rows @ x_test.T if False else x_test @ rows.T, img_size=img_size, config=config, device=device)
    frozen, frozen_diag = evaluate_frozen_prior(config=config, rows=rows, test_indices=splits["test"], output_dir=output_dir / "frozen_prior", device=device)

    estimates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("joint_minimum_norm", joint, rel_joint),
        ("joint_tikhonov", tikh, rel_tikh),
        ("operator_specific_null_predictor", pred_test, rel_model),
    ]
    if tv.shape == x_test.shape:
        rel_tv = np.linalg.norm(tv @ rows.T - x_test @ rows.T, axis=1) / np.maximum(np.linalg.norm(x_test @ rows.T, axis=1), 1e-12)
        estimates.append(("joint_tv_pgd", tv, rel_tv))
    if frozen is not None:
        rel_frozen = np.linalg.norm(frozen @ rows.T - x_test @ rows.T, axis=1) / np.maximum(np.linalg.norm(x_test @ rows.T, axis=1), 1e-12)
        estimates.append(("frozen_prior_posterior_mean", frozen, rel_frozen))

    metric_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    qcfg = dict(config.get("quality", {}))
    for method, xhat, rel in estimates:
        summary, per = metric_bundle(
            method=method,
            xhat=xhat,
            x=x_test,
            rel=rel,
            labels=labels,
            indices=indices,
            img_size=img_size,
            compute_lpips=bool(qcfg.get("compute_lpips", False)),
            lpips_device=str(qcfg.get("lpips_device", "cuda")),
        )
        metric_rows.append(summary)
        per_rows.extend(per)
    write_csv(reports / "method_metrics.csv", metric_rows)
    write_csv(reports / "per_image_metrics.csv", per_rows)
    write_csv(reports / "train_log.csv", train_log)

    stats_cfg = dict(config.get("statistics", {}))
    reps = int(stats_cfg.get("bootstrap_replicates", 1000))
    comparisons = [
        paired_delta(per_rows, "operator_specific_null_predictor", "joint_minimum_norm", "centered_rmse", reps=reps, seed=seed + 100),
        paired_delta(per_rows, "operator_specific_null_predictor", "joint_minimum_norm", "full_rmse", reps=reps, seed=seed + 101),
    ]
    if frozen is not None:
        comparisons.append(paired_delta(per_rows, "operator_specific_null_predictor", "frozen_prior_posterior_mean", "centered_rmse", reps=reps, seed=seed + 102))
    write_json(reports / "paired_comparisons.json", comparisons)

    model_row = next(r for r in metric_rows if r["method"] == "operator_specific_null_predictor")
    joint_row = next(r for r in metric_rows if r["method"] == "joint_minimum_norm")
    frozen_row = next((r for r in metric_rows if r["method"] == "frozen_prior_posterior_mean"), None)
    cmp_joint = comparisons[0]
    gate_conditions = {
        "beats_joint_centered_ci": bool(cmp_joint["mean_delta"] < -float(config.get("gate", {}).get("min_centered_rmse_gain", 1e-4)) and cmp_joint["ci_upper"] < 0),
        "relmeaserr_ok": bool(float(model_row["relmeaserr_max"]) < float(config.get("gate", {}).get("relmeaserr_max", 1e-4))),
        "beats_frozen_prior_if_available": True,
    }
    if frozen_row is not None:
        cmp_frozen = next(c for c in comparisons if c["reference"] == "frozen_prior_posterior_mean")
        gate_conditions["beats_frozen_prior_if_available"] = bool(cmp_frozen["mean_delta"] < 0 and cmp_frozen["ci_upper"] < 0)
    decision = "CANARY_PASS_EXPAND_BUDGET_CONDITIONED" if all(gate_conditions.values()) else "CANARY_FAIL_DIAGNOSE_BEFORE_EXPANDING"
    gate = {
        "status": "PASS",
        "decision": decision,
        "scope": "operator-specific DCT development canary, not locked",
        "conditions": gate_conditions,
        "model": model_row,
        "joint": joint_row,
        "frozen_prior": frozen_row,
        "paired_comparisons": comparisons,
        "locked_test_authorized": False,
    }
    write_json(reports / "gate_report.json", gate)
    write_json(
        reports / "lineage_and_leakage_audit.json",
        {
            "status": "PASS",
            "split_manifest": split_manifest,
            "repo_state": repo_state(),
            "final_v4_or_locked_used_for_training_or_selection": False,
            "rows_sha256": sha256_numpy(rows),
            "row_audit": row_audit(rows, name="operator_conditioned_dct_canary"),
            "projector": projector.diagnostics(),
            "tv_baseline": tv_diag,
            "frozen_prior": frozen_diag,
        },
    )
    conclusion = [
        "# Operator-Specific DCT Null-Space Canary",
        "",
        f"- Decision: `{decision}`",
        f"- Best epoch: `{best['epoch']}`",
        f"- Model centered RMSE: `{model_row['centered_rmse_mean']}`",
        f"- Joint centered RMSE: `{joint_row['centered_rmse_mean']}`",
        f"- Model minus joint centered delta: `{cmp_joint['mean_delta']}` CI `[{cmp_joint['ci_lower']}, {cmp_joint['ci_upper']}]`",
        f"- RelMeasErr max: `{model_row['relmeaserr_max']}`",
        "",
        "This is a development canary only. It authorizes expansion only if the gate passes; it never authorizes a locked test directly.",
    ]
    (reports / "research_decision.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "best.pt": sha256_file(ckpt_path),
        "method_metrics.csv": sha256_file(reports / "method_metrics.csv"),
        "per_image_metrics.csv": sha256_file(reports / "per_image_metrics.csv"),
        "paired_comparisons.json": sha256_file(reports / "paired_comparisons.json"),
        "gate_report.json": sha256_file(reports / "gate_report.json"),
    }
    runtime = {
        "status": "PASS",
        "started_utc": now_utc(),
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "artifact_hashes": hashes,
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "OPERATOR_CONDITIONED_DCT_CANARY_COMPLETE",
        "output_dir": str(output_dir),
        "gate": gate,
        "runtime": runtime,
        "key_artifacts": {
            "research_decision": str(reports / "research_decision.md"),
            "gate_report": str(reports / "gate_report.json"),
            "method_metrics": str(reports / "method_metrics.csv"),
            "per_image_metrics": str(reports / "per_image_metrics.csv"),
            "checkpoint": str(ckpt_path),
        },
    }
    write_json(reports / "summary.json", summary)
    atomic_json(output_dir / "OPERATOR_CONDITIONED_DCT_CANARY_COMPLETE.json", {"status": summary["status"], "decision": decision, "summary_sha256": sha256_file(reports / "summary.json")})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate an operator-specific DCT null-space Bayes canary.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.config))
    print(json.dumps(json_safe({"status": summary["status"], "output_dir": summary["output_dir"], "decision": summary["gate"]["decision"], "key_artifacts": summary["key_artifacts"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
