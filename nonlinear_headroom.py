from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import nonlinear_operator_transfer as nlt
from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src.dc_balanced import centered_rmse, dct_band_rmse, full_rmse, row_audit
from src.operator_conditioned_nullspace import MatrixFreeNullProjector
from src.phase2_fresh_operator import resolve_device
from src.phase2_witness import sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "nonlinear_headroom_dct_canary.yaml"


class HeadroomError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arm:
    arm_id: str
    budget: int
    operator_seed: int
    rows: np.ndarray
    projector: MatrixFreeNullProjector


@dataclass
class LMMSEFit:
    mu: np.ndarray
    z: np.ndarray
    lambda_by_budget: dict[int, float]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise HeadroomError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def sha256_np(arr: np.ndarray, *, sort_int64: bool = False) -> str:
    x = np.asarray(arr)
    if sort_int64:
        x = np.sort(x.astype(np.int64))
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def select_clean_splits(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split_cfg = dict(config["splits"])
    counts = {k: int(split_cfg[k]["count"]) for k in ["train", "val", "dev"]}
    offsets_cfg = {k: split_cfg[k].get("offset") for k in ["train", "val", "dev"]}
    train_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    if all(v is not None for v in offsets_cfg.values()):
        offsets = {k: int(offsets_cfg[k]) for k in offsets_cfg}
        out = {k: train_full[offsets[k] : offsets[k] + counts[k]].copy() for k in counts}
    else:
        start = int(split_cfg.get("search_start", 43000))
        stride = int(split_cfg.get("search_stride", 700))
        max_start = int(split_cfg.get("search_max_start", 49000))
        candidates = list(range(start, max_start, stride))
        out = {}
        for tr in candidates:
            for va in candidates:
                for de in candidates:
                    intervals = [(tr, tr + counts["train"]), (va, va + counts["val"]), (de, de + counts["dev"])]
                    if len({tr, va, de}) < 3:
                        continue
                    if any(max(a[0], b[0]) < min(a[1], b[1]) for i, a in enumerate(intervals) for b in intervals[i + 1 :]):
                        continue
                    trial = {
                        "train": train_full[tr : tr + counts["train"]].copy(),
                        "val": train_full[va : va + counts["val"]].copy(),
                        "dev": train_full[de : de + counts["dev"]].copy(),
                    }
                    dup, _rows = nlt.duplicate_audit(trial)
                    if dup["status"] == "PASS":
                        out = trial
                        offsets = {"train": tr, "val": va, "dev": de}
                        break
                if out:
                    break
            if out:
                break
        if not out:
            raise HeadroomError("NO_DUPLICATE_CLEAN_SPLIT_FOUND")
    dup, rows = nlt.duplicate_audit(out)
    if dup["status"] != "PASS":
        raise HeadroomError(f"DUPLICATE_AUDIT_FAILED:{dup}")
    used = np.concatenate([out["train"], out["val"], out["dev"]]).astype(np.int64)
    manifest = {
        "source": "STL10 train+unlabeled fresh development",
        "train_full_sorted_sha256": sha256_np(train_full, sort_int64=True),
        "offsets": offsets,
        "counts": counts,
        "combined_indices_sha256": sha256_np(used),
        "duplicate_audit": dup,
        "sample_hash_rows": rows,
        "final_v4_or_existing_locked_used": False,
    }
    for k, arr in out.items():
        manifest[k] = {"indices_sha256": sha256_np(arr), "indices_sorted_sha256": sha256_np(arr, sort_int64=True)}
    return out, manifest


def build_dct_arms(config: Mapping[str, Any], pool: str, *, device: torch.device) -> list[Arm]:
    op = dict(config["operator_protocol"])
    budgets = [int(v) for v in op["budgets"]]
    seeds = [int(v) for v in op[f"{pool}_operator_seeds"]]
    img_size = int(op.get("img_size", 64))
    arms: list[Arm] = []
    for budget in budgets:
        for seed in seeds:
            rows = nlt.build_rows("dct", budget, img_size=img_size, seed=seed)
            proj = MatrixFreeNullProjector(torch.from_numpy(rows).to(device=device, dtype=torch.float32))
            arms.append(Arm(f"{pool}_dct_m{budget}_op{seed}", budget, seed, rows, proj))
    return arms


def load_x(indices: np.ndarray, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_t, labels, seen = nlt.load_split_arrays(indices, batch_size=batch_size, device=device)
    return x_t.numpy().astype(np.float32), labels, seen


def split_fit_calib(x_train: np.ndarray, *, fit_count: int) -> tuple[np.ndarray, np.ndarray]:
    if fit_count <= 0 or fit_count >= x_train.shape[0]:
        raise HeadroomError(f"BAD_FIT_COUNT:{fit_count}:{x_train.shape[0]}")
    return x_train[:fit_count].copy(), x_train[fit_count:].copy()


def fit_lmmse(x_fit: np.ndarray) -> LMMSEFit:
    mu = x_fit.astype(np.float64).mean(axis=0)
    z = x_fit.astype(np.float64) - mu[None, :]
    return LMMSEFit(mu=mu, z=z, lambda_by_budget={})


def lmmse_raw(x: np.ndarray, arm: Arm, fit: LMMSEFit, lambda_: float) -> np.ndarray:
    a = np.asarray(arm.rows, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    y = x64 @ a.T
    y_mu = fit.mu @ a.T
    u = fit.z @ a.T
    scale = max(fit.z.shape[0] - 1, 1)
    s = (u.T @ u) / scale
    ca_t = (fit.z.T @ u) / scale
    coeff = np.linalg.solve(s + float(lambda_) * np.eye(s.shape[0]), (y - y_mu[None, :]).T).T
    return (fit.mu[None, :] + coeff @ ca_t.T).astype(np.float32)


def audit(xhat: np.ndarray, x_truth: np.ndarray, arm: Arm, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xb = torch.from_numpy(np.asarray(x_truth, dtype=np.float32)).to(device)
    y = arm.projector.measurement(xb)
    pred = torch.from_numpy(np.asarray(xhat, dtype=np.float32)).to(device)
    out = arm.projector.data_anchor(y) + arm.projector.null_project(pred)
    rel = arm.projector.relmeaserr(out, y)
    return out.detach().cpu().numpy().astype(np.float32), rel.detach().cpu().numpy().astype(np.float64)


def lmmse_predict(x: np.ndarray, arm: Arm, fit: LMMSEFit, lambda_: float, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    return audit(lmmse_raw(x, arm, fit, lambda_), x, arm, device=device)


def select_lambda_by_budget(val_x: np.ndarray, val_arms: Sequence[Arm], fit: LMMSEFit, lambdas: Sequence[float], *, device: torch.device) -> dict[int, dict[str, Any]]:
    by_budget: dict[int, dict[str, Any]] = {}
    for budget in sorted({a.budget for a in val_arms}):
        rows = []
        for lam in lambdas:
            vals = []
            for arm in [a for a in val_arms if a.budget == budget]:
                pred, _rel = lmmse_predict(val_x, arm, fit, float(lam), device=device)
                vals.append(centered_rmse(pred, val_x).mean())
            rows.append({"lambda": float(lam), "val_centered_rmse": float(np.mean(vals))})
        best = min(rows, key=lambda r: r["val_centered_rmse"])
        by_budget[int(budget)] = {"best_lambda": float(best["lambda"]), "grid": rows}
    return by_budget


def residual_bank(calib_x: np.ndarray, arm: Arm, fit: LMMSEFit, lambda_: float, *, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_l, _rel = lmmse_predict(calib_x, arm, fit, lambda_, device=device)
    residual = calib_x.astype(np.float32) - x_l.astype(np.float32)
    residual, _rel2 = audit(residual, np.zeros_like(calib_x, dtype=np.float32), arm, device=device)
    y_calib = calib_x.astype(np.float64) @ arm.rows.astype(np.float64).T
    u = fit.z @ arm.rows.astype(np.float64).T
    scale = max(fit.z.shape[0] - 1, 1)
    s = (u.T @ u) / scale + float(lambda_) * np.eye(arm.rows.shape[0])
    sinv = np.linalg.pinv(s)
    return y_calib.astype(np.float64), residual.astype(np.float32), s.astype(np.float64), sinv.astype(np.float64)


def local_residual_predict(
    x: np.ndarray,
    arm: Arm,
    fit: LMMSEFit,
    lambda_: float,
    calib_y: np.ndarray,
    calib_residual: np.ndarray,
    sinv: np.ndarray,
    *,
    k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_l, _rel = lmmse_predict(x, arm, fit, lambda_, device=device)
    y = x.astype(np.float64) @ arm.rows.astype(np.float64).T
    residuals = []
    for yy in y:
        diff = calib_y - yy[None, :]
        d2 = np.einsum("nm,mp,np->n", diff, sinv, diff)
        kk = min(int(k), calib_y.shape[0])
        idx = np.argpartition(d2, kk - 1)[:kk]
        scale = max(float(np.median(d2[idx])), 1e-8)
        w = np.exp(-0.5 * d2[idx] / scale)
        w = w / np.maximum(w.sum(), 1e-12)
        residuals.append((w[:, None] * calib_residual[idx].astype(np.float64)).sum(axis=0))
    raw = x_l.astype(np.float64) + np.stack(residuals, axis=0)
    return audit(raw.astype(np.float32), x, arm, device=device)


def kmeans(data: np.ndarray, k: int, *, seed: int, iters: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = data.shape[0]
    centers = data[rng.choice(np.arange(n), size=int(k), replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(iters)):
        d2 = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        for j in range(int(k)):
            mask = labels == j
            if np.any(mask):
                centers[j] = data[mask].mean(axis=0)
    return labels


@dataclass
class ComponentFit:
    weight: float
    fit: LMMSEFit
    lambda_: float
    meas_mean: np.ndarray
    meas_cov: np.ndarray
    meas_cov_inv: np.ndarray
    logdet: float


def fit_mfa_components(x_fit: np.ndarray, arm: Arm, *, k: int, lambda_: float, seed: int) -> list[ComponentFit]:
    y_fit = x_fit.astype(np.float64) @ arm.rows.astype(np.float64).T
    # Use whitened low-dimensional measurements for unsupervised components.
    labels = np.zeros(x_fit.shape[0], dtype=np.int64) if int(k) == 1 else kmeans(y_fit, int(k), seed=seed, iters=10)
    comps: list[ComponentFit] = []
    for j in range(int(k)):
        mask = labels == j
        if int(mask.sum()) < max(8, arm.rows.shape[0] // 2):
            continue
        sub = x_fit[mask]
        f = fit_lmmse(sub)
        u = f.z @ arm.rows.astype(np.float64).T
        scale = max(f.z.shape[0] - 1, 1)
        cov = (u.T @ u) / scale + float(lambda_) * np.eye(arm.rows.shape[0])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-4 * np.eye(cov.shape[0])
            sign, logdet = np.linalg.slogdet(cov)
        comps.append(
            ComponentFit(
                weight=float(mask.mean()),
                fit=f,
                lambda_=float(lambda_),
                meas_mean=f.mu @ arm.rows.astype(np.float64).T,
                meas_cov=cov,
                meas_cov_inv=np.linalg.pinv(cov),
                logdet=float(logdet),
            )
        )
    if not comps:
        comps.append(fit_mfa_components(x_fit, arm, k=1, lambda_=lambda_, seed=seed)[0])
    total = sum(c.weight for c in comps)
    for c in comps:
        c.weight = c.weight / total
    return comps


def mfa_predict(x: np.ndarray, arm: Arm, comps: Sequence[ComponentFit], *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    y = x.astype(np.float64) @ arm.rows.astype(np.float64).T
    preds = []
    logps = []
    for c in comps:
        raw = lmmse_raw(x, arm, c.fit, c.lambda_).astype(np.float64)
        preds.append(raw)
        diff = y - c.meas_mean[None, :]
        quad = np.einsum("nm,mp,np->n", diff, c.meas_cov_inv, diff)
        logps.append(np.log(max(c.weight, 1e-12)) - 0.5 * (quad + c.logdet))
    logits = np.stack(logps, axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    gamma = np.exp(logits)
    gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
    stack = np.stack(preds, axis=1)
    raw = np.sum(gamma[:, :, None] * stack, axis=1)
    return audit(raw.astype(np.float32), x, arm, device=device)


def metric_arrays(xhat: np.ndarray, x: np.ndarray, rel: np.ndarray, *, img_size: int) -> dict[str, np.ndarray]:
    f = full_rmse(xhat, x)
    c = centered_rmse(xhat, x)
    bands = dct_band_rmse(xhat, x, img_size=img_size, low_count=40)
    return {
        "centered_rmse": c,
        "full_rmse": f,
        "psnr": -20.0 * np.log10(np.maximum(f, 1e-12)),
        "relmeaserr": rel,
        **bands,
    }


def append_rows(
    per: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    *,
    method: str,
    arm: Arm,
    xhat: np.ndarray,
    x: np.ndarray,
    rel: np.ndarray,
    source_indices: np.ndarray,
    labels: np.ndarray,
    img_size: int,
) -> None:
    arr = metric_arrays(xhat, x, rel, img_size=img_size)
    summary.append(
        {
            "method": method,
            "arm_id": arm.arm_id,
            "budget": arm.budget,
            "operator_seed": arm.operator_seed,
            "n": int(x.shape[0]),
            "centered_rmse_mean": float(arr["centered_rmse"].mean()),
            "full_rmse_mean": float(arr["full_rmse"].mean()),
            "psnr_mean": float(arr["psnr"].mean()),
            "dct_non_dc_low_rmse_mean": float(arr["dct_non_dc_low_rmse"].mean()),
            "dct_mid_rmse_mean": float(arr["dct_mid_rmse"].mean()),
            "dct_high_rmse_mean": float(arr["dct_high_rmse"].mean()),
            "relmeaserr_max": float(arr["relmeaserr"].max()),
        }
    )
    for i in range(x.shape[0]):
        per.append(
            {
                "method": method,
                "arm_id": arm.arm_id,
                "budget": arm.budget,
                "operator_seed": arm.operator_seed,
                "source_index": int(source_indices[i]),
                "label": int(labels[i]),
                **{k: float(v[i]) for k, v in arr.items() if k != "relmeaserr"},
                "relmeaserr": float(arr["relmeaserr"][i]),
            }
        )


def image_clustered_delta(per: Sequence[Mapping[str, Any]], method: str, reference: str, *, budget: int | None = None, reps: int = 1000, seed: int = 0) -> dict[str, Any]:
    vals: dict[tuple[int, str], dict[str, float]] = {}
    for r in per:
        if budget is not None and int(r["budget"]) != int(budget):
            continue
        if r["method"] not in {method, reference}:
            continue
        key = (int(r["source_index"]), str(r["arm_id"]))
        vals.setdefault(key, {})[str(r["method"])] = float(r["centered_rmse"])
    by_img: dict[int, list[float]] = {}
    for (src, _arm), d in vals.items():
        if method in d and reference in d:
            by_img.setdefault(src, []).append(d[method] - d[reference])
    means = np.asarray([np.mean(by_img[k]) for k in sorted(by_img)], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(reps)):
        idx = rng.choice(np.arange(means.shape[0]), size=means.shape[0], replace=True)
        boots.append(float(means[idx].mean()))
    return {
        "method": method,
        "reference": reference,
        "budget": "ALL" if budget is None else int(budget),
        "n_images": int(means.shape[0]),
        "n_pairs": int(sum(len(v) for v in by_img.values())),
        "mean_delta": float(means.mean()),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5)),
        "image_wins": int(np.sum(means < 0)),
        "image_losses": int(np.sum(means > 0)),
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    out = ROOT / str(config.get("output_dir", "outputs/compatibility/nonlinear_headroom/dct_canary"))
    reports = out / "reports"
    ensure_dir(reports)
    (out / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    device = resolve_device(str(config.get("device", "cuda")))
    splits, split_manifest = select_clean_splits(config)
    write_csv(reports / "sample_hash_audit.csv", split_manifest["sample_hash_rows"])
    slim_manifest = {k: v for k, v in split_manifest.items() if k != "sample_hash_rows"}
    write_json(reports / "split_and_duplicate_audit.json", slim_manifest)
    x_train, train_labels, train_seen = load_x(splits["train"], batch_size=int(config.get("batch_size", 32)), device=device)
    x_val, val_labels, val_seen = load_x(splits["val"], batch_size=int(config.get("batch_size", 32)), device=device)
    x_dev, dev_labels, dev_seen = load_x(splits["dev"], batch_size=int(config.get("batch_size", 32)), device=device)
    fit_count = int(config["splits"].get("fit_count", int(0.75 * x_train.shape[0])))
    x_fit, x_calib = split_fit_calib(x_train, fit_count=fit_count)
    fit = fit_lmmse(x_fit)
    val_arms = build_dct_arms(config, "val", device=device)
    dev_arms = build_dct_arms(config, "dev", device=device)
    lgrid = [float(v) for v in config["lmmse"].get("lambda_grid", [1e-5, 1e-4, 1e-3])]
    lambda_selection = select_lambda_by_budget(x_val, val_arms, fit, lgrid, device=device)
    write_json(reports / "lambda_selection.json", lambda_selection)

    k_grid = [int(v) for v in config["local"].get("k_grid", [8, 16, 32, 64])]
    mfa_k_grid = [int(v) for v in config["mfa"].get("component_grid", [1, 2, 4])]
    local_selection: dict[int, Any] = {}
    mfa_selection: dict[int, Any] = {}
    for budget in sorted({a.budget for a in val_arms}):
        lam = float(lambda_selection[budget]["best_lambda"])
        local_rows = []
        for k in k_grid:
            vals = []
            for arm in [a for a in val_arms if a.budget == budget]:
                cy, cr, _s, sinv = residual_bank(x_calib, arm, fit, lam, device=device)
                pred, _rel = local_residual_predict(x_val, arm, fit, lam, cy, cr, sinv, k=k, device=device)
                vals.append(centered_rmse(pred, x_val).mean())
            local_rows.append({"k": int(k), "val_centered_rmse": float(np.mean(vals))})
        local_selection[budget] = {"best_k": int(min(local_rows, key=lambda r: r["val_centered_rmse"])["k"]), "grid": local_rows}
        mfa_rows = []
        for kk in mfa_k_grid:
            vals = []
            for arm in [a for a in val_arms if a.budget == budget]:
                comps = fit_mfa_components(x_fit, arm, k=kk, lambda_=lam, seed=int(config.get("seed", 20260625)) + budget * 10 + kk)
                pred, _rel = mfa_predict(x_val, arm, comps, device=device)
                vals.append(centered_rmse(pred, x_val).mean())
            mfa_rows.append({"components": int(kk), "val_centered_rmse": float(np.mean(vals))})
        mfa_selection[budget] = {"best_components": int(min(mfa_rows, key=lambda r: r["val_centered_rmse"])["components"]), "grid": mfa_rows}
    write_json(reports / "local_selection.json", local_selection)
    write_json(reports / "mfa_selection.json", mfa_selection)

    per: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for arm in dev_arms:
        lam = float(lambda_selection[arm.budget]["best_lambda"])
        joint, rel_joint = nlt.joint_predict(x_dev, nlt.TransferArm(arm.arm_id, "dct", arm.budget, "dev", arm.operator_seed, arm.rows, arm.projector, torch.empty(0)), device=device)
        append_rows(per, summaries, method="minimum_norm", arm=arm, xhat=joint, x=x_dev, rel=rel_joint, source_indices=dev_seen, labels=dev_labels, img_size=int(config["operator_protocol"].get("img_size", 64)))
        lmmse, rel_l = lmmse_predict(x_dev, arm, fit, lam, device=device)
        append_rows(per, summaries, method="global_lmmse", arm=arm, xhat=lmmse, x=x_dev, rel=rel_l, source_indices=dev_seen, labels=dev_labels, img_size=int(config["operator_protocol"].get("img_size", 64)))
        cy, cr, _s, sinv = residual_bank(x_calib, arm, fit, lam, device=device)
        local, rel_loc = local_residual_predict(x_dev, arm, fit, lam, cy, cr, sinv, k=int(local_selection[arm.budget]["best_k"]), device=device)
        append_rows(per, summaries, method="local_knn_residual", arm=arm, xhat=local, x=x_dev, rel=rel_loc, source_indices=dev_seen, labels=dev_labels, img_size=int(config["operator_protocol"].get("img_size", 64)))
        comps = fit_mfa_components(x_fit, arm, k=int(mfa_selection[arm.budget]["best_components"]), lambda_=lam, seed=int(config.get("seed", 20260625)) + arm.operator_seed + arm.budget)
        mfa, rel_mfa = mfa_predict(x_dev, arm, comps, device=device)
        append_rows(per, summaries, method="mfa_posterior", arm=arm, xhat=mfa, x=x_dev, rel=rel_mfa, source_indices=dev_seen, labels=dev_labels, img_size=int(config["operator_protocol"].get("img_size", 64)))
    write_csv(reports / "method_metrics.csv", summaries)
    write_csv(reports / "per_image_operator_metrics.csv", per)

    reps = int(config["statistics"].get("bootstrap_replicates", 500))
    comparisons = []
    for method in ["local_knn_residual", "mfa_posterior"]:
        comparisons.append(image_clustered_delta(per, method, "global_lmmse", reps=reps, seed=int(config.get("seed", 0)) + len(comparisons)))
        for budget in sorted({a.budget for a in dev_arms}):
            comparisons.append(image_clustered_delta(per, method, "global_lmmse", budget=budget, reps=reps, seed=int(config.get("seed", 0)) + budget + len(comparisons)))
    write_json(reports / "clustered_comparisons.json", comparisons)

    heldout = int(config["operator_protocol"].get("heldout_budget", 41))
    local_all = next(c for c in comparisons if c["method"] == "local_knn_residual" and c["budget"] == "ALL")
    mfa_all = next(c for c in comparisons if c["method"] == "mfa_posterior" and c["budget"] == "ALL")
    local_held = next(c for c in comparisons if c["method"] == "local_knn_residual" and c["budget"] == heldout)
    mfa_held = next(c for c in comparisons if c["method"] == "mfa_posterior" and c["budget"] == heldout)
    relmax = max(float(r["relmeaserr_max"]) for r in summaries)
    conditions = {
        "local_or_mfa_beats_lmmse_all_ci": bool((local_all["ci_upper"] < 0) or (mfa_all["ci_upper"] < 0)),
        "heldout_budget_beats_lmmse_ci": bool((local_held["ci_upper"] < 0) or (mfa_held["ci_upper"] < 0)),
        "measurement_consistency_ok": bool(relmax < float(config["gate"].get("relmeaserr_max", 1e-4))),
        "duplicate_audit_ok": bool(slim_manifest["duplicate_audit"]["status"] == "PASS"),
    }
    if conditions["local_or_mfa_beats_lmmse_all_ci"] and conditions["heldout_budget_beats_lmmse_ci"]:
        classification = "NONLINEAR_MIXTURE_GAIN_CONFIRMED" if mfa_all["ci_upper"] < 0 else "STRUCTURED_ONLY_NONLINEAR_GAIN"
    else:
        classification = "NO_NONLINEAR_HEADROOM_DETECTED"
    gate = {
        "status": "PASS",
        "classification": classification,
        "locked_test_authorized": classification in {"NONLINEAR_MIXTURE_GAIN_CONFIRMED", "STRUCTURED_ONLY_NONLINEAR_GAIN"},
        "conditions": conditions,
        "relmeaserr_max": relmax,
        "scope": "DCT structured canary, fresh development only",
        "comparisons": comparisons,
    }
    write_json(reports / "gate_report.json", gate)
    write_reports(reports, gate, summaries, comparisons, lambda_selection, local_selection, mfa_selection, config)
    package = make_package(out, config_path)
    runtime = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package": str(package),
        "hashes": {
            "config_used": sha256_file(out / "config_used.yaml"),
            "method_metrics": sha256_file(reports / "method_metrics.csv"),
            "comparisons": sha256_file(reports / "clustered_comparisons.json"),
            "gate": sha256_file(reports / "gate_report.json"),
            "package": sha256_file(package),
        },
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {"status": "NONLINEAR_HEADROOM_COMPLETE", "output_dir": str(out), "classification": classification, "gate": gate, "package": str(package)}
    write_json(reports / "summary.json", summary)
    return summary


def write_reports(reports: Path, gate: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]], lambdas: Mapping[int, Any], local_sel: Mapping[int, Any], mfa_sel: Mapping[int, Any], config: Mapping[str, Any]) -> None:
    math_text = r"""# Nonlinear Headroom Math

Let \(x_L(Y,A)\) be the exact-audited empirical LMMSE estimator and
\(P_0^A=I-A^\dagger A\). Any further measurement-consistent improvement can be
written as
\[
\hat x=x_L+P_0^A h(Y,A).
\]
The target residual is
\[
Z=P_0^A(X-x_L).
\]
The nonlinear headroom under squared error is
\[
R_L-R_*=\mathbb E\|\mathbb E[Z\mid Y,A]\|^2.
\]
This canary estimates that residual with two non-neural, cross-fit methods:
measurement-whitened kNN residual averaging and a mixture posterior. The
calibration residual bank is disjoint from the LMMSE fit images and from dev.
"""
    (reports / "math_derivation.md").write_text(math_text, encoding="utf-8")
    by_method: dict[str, list[float]] = {}
    for r in summaries:
        by_method.setdefault(str(r["method"]), []).append(float(r["centered_rmse_mean"]))
    risk = [{"method": k, "centered_rmse_mean": float(np.mean(v))} for k, v in sorted(by_method.items())]
    write_csv(reports / "risk_ladder.csv", risk)
    lines = [
        "# Nonlinear Headroom DCT Canary",
        "",
        f"- Classification: `{gate['classification']}`",
        f"- Locked test authorized: `{str(gate['locked_test_authorized']).lower()}`",
        "- Scope: DCT/low-frequency structured canary, fresh development only.",
        "",
        "## Risk Ladder",
        "",
        "| Method | Centered RMSE mean |",
        "|---|---:|",
    ]
    for row in risk:
        lines.append(f"| `{row['method']}` | {row['centered_rmse_mean']:.6f} |")
    lines.extend(["", "## Comparisons vs Global LMMSE", ""])
    for c in comparisons:
        lines.append(f"- `{c['method']}`, budget `{c['budget']}`: delta `{c['mean_delta']:.6f}`, CI `[{c['ci_lower']:.6f}, {c['ci_upper']:.6f}]`, wins `{c['image_wins']}/{c['n_images']}`")
    lines.extend(["", "## Selection", ""])
    lines.append(f"- LMMSE lambda selection: `{lambdas}`")
    lines.append(f"- local k selection: `{local_sel}`")
    lines.append(f"- MFA component selection: `{mfa_sel}`")
    lines.extend(["", "## Gate", ""])
    for k, v in gate["conditions"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(
        [
            "",
            "## Claim-Evidence Ledger",
            "",
            "| Claim | Evidence | Status |",
            "|---|---|---|",
            "| Global LMMSE is the zero-residual baseline | All methods are compared as corrections over exact-audited LMMSE. | supported |",
            "| Local/kernel residual detects nonlinear headroom | Requires local CI upper < 0 vs LMMSE. | see gate |",
            "| MFA detects nonlinear headroom | Requires MFA CI upper < 0 vs LMMSE. | see gate |",
            "| Locked test is warranted | Requires held-out budget and overall DCT canary success. | see gate |",
            "",
            "## Unique Recommended Next Step",
            "",
        ]
    )
    if gate["classification"] == "NO_NONLINEAR_HEADROOM_DETECTED":
        lines.append("Do not train a residual NN or open a locked test. Treat global empirical LMMSE as the current ceiling for this DCT canary, then improve nonlinear estimators only if a stronger local/MFA residual diagnostic first beats LMMSE.")
    else:
        lines.append("Freeze a larger DCT-only development replicate or locked protocol before claiming neural residual value; residual NN must beat the best local/MFA posterior, not just LMMSE.")
    (reports / "research_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "claim_evidence_ledger.md").write_text("\n".join(lines[lines.index("## Claim-Evidence Ledger") :]) + "\n", encoding="utf-8")


def make_package(out: Path, config_path: Path) -> Path:
    stage = out.parent / "nonlinear_headroom_nextstep_package"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "reports").mkdir(parents=True)
    (stage / "code").mkdir()
    (stage / "configs").mkdir()
    reports = out / "reports"
    for name in [
        "research_decision.md",
        "math_derivation.md",
        "risk_ladder.csv",
        "method_metrics.csv",
        "clustered_comparisons.json",
        "gate_report.json",
        "lambda_selection.json",
        "local_selection.json",
        "mfa_selection.json",
        "split_and_duplicate_audit.json",
        "claim_evidence_ledger.md",
    ]:
        p = reports / name
        if p.exists():
            shutil.copy2(p, stage / "reports" / name)
    shutil.copy2(config_path, stage / "configs" / config_path.name)
    shutil.copy2(ROOT / "nonlinear_headroom.py", stage / "code" / "nonlinear_headroom.py")
    shutil.copy2(ROOT / "tests" / "test_nonlinear_headroom.py", stage / "code" / "test_nonlinear_headroom.py")
    manifest = []
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            manifest.append({"relative_path": str(p.relative_to(stage)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = out.parent / "nonlinear_headroom_nextstep_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in stage.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    return zip_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DCT nonlinear headroom canary beyond empirical LMMSE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.config))
    print(json.dumps({"status": summary["status"], "classification": summary["classification"], "output_dir": summary["output_dir"], "package": summary["package"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
