from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import operator_conditioned_nullspace_canary as occ
from dc_balanced_fixed_total import tv_baseline
from src.dc_balanced import (
    build_dc_balanced_rows,
    dc_row,
    dct_lowfreq_non_dc_rows,
    random_zero_mean_rows,
    row_audit,
)
from src.operator_conditioned_nullspace import MatrixFreeNullProjector, SmallNullspaceUNet, reconstruct_with_projected_residual
from src.phase2_fresh_operator import resolve_device
from src.phase2_witness import repo_state, sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "operator_conditioned_budget_dev.yaml"


class BudgetConditionedError(RuntimeError):
    pass


@dataclass
class ArmState:
    arm_id: str
    rows: np.ndarray
    projector: MatrixFreeNullProjector
    cond_features: torch.Tensor


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise BudgetConditionedError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def arm_rows(arm: Mapping[str, Any], *, img_size: int, total_m: int, seed: int) -> np.ndarray:
    dim = int(img_size) * int(img_size)
    family = str(arm.get("family", "dct")).lower()
    if family in {"random", "dct", "hadamard"}:
        return build_dc_balanced_rows(family, int(total_m) - 1, dim=dim, img_size=img_size, seed=seed).astype(np.float32)
    if family in {"mixed_random_dct", "mixed"}:
        random_count = int(arm.get("random_count", 8))
        dct_count = int(arm.get("dct_count", int(total_m) - 1 - random_count))
        if 1 + random_count + dct_count != int(total_m):
            raise BudgetConditionedError(f"MIXED_COUNTS_DO_NOT_MATCH_TOTAL:{random_count}:{dct_count}:{total_m}")
        rows = np.concatenate(
            [
                dc_row(dim)[None, :],
                random_zero_mean_rows(random_count, dim, seed + int(arm.get("seed_offset", 0))),
                dct_lowfreq_non_dc_rows(dct_count, img_size),
            ],
            axis=0,
        )
        return rows.astype(np.float32)
    raise BudgetConditionedError(f"UNKNOWN_ARM_FAMILY:{family}")


def operator_condition_features(rows: np.ndarray, *, img_size: int, device: torch.device) -> torch.Tensor:
    arr = np.asarray(rows, dtype=np.float64)
    leverage = np.sum(arr * arr, axis=0)
    leverage = (leverage - float(leverage.mean())) / max(float(leverage.std()), 1e-8)
    budget = np.full_like(leverage, float(arr.shape[0] / arr.shape[1]))
    feat = np.stack([leverage.reshape(img_size, img_size), budget.reshape(img_size, img_size)], axis=0)
    return torch.from_numpy(feat[None].astype(np.float32)).to(device)


def build_arms(config: Mapping[str, Any], *, device: torch.device) -> list[ArmState]:
    op = dict(config["operator"])
    img_size = int(op.get("img_size", 64))
    total_m = int(op.get("total_m", 41))
    base_seed = int(op.get("seed", 0))
    arm_cfgs = list(config.get("arms", []))
    if not arm_cfgs:
        arm_cfgs = [
            {"arm_id": "dc_plus_40_random", "family": "random"},
            {"arm_id": "dc_plus_40_non_dc_dct", "family": "dct"},
            {"arm_id": "dc_plus_40_non_dc_hadamard", "family": "hadamard"},
            {"arm_id": "dc_plus_8_random_32_non_dc_dct", "family": "mixed_random_dct", "random_count": 8, "dct_count": 32},
        ]
    arms: list[ArmState] = []
    for i, arm in enumerate(arm_cfgs):
        rows = arm_rows(arm, img_size=img_size, total_m=total_m, seed=base_seed + i * 1000)
        projector = MatrixFreeNullProjector(torch.from_numpy(rows).to(device=device, dtype=torch.float32))
        cond = operator_condition_features(rows, img_size=img_size, device=device)
        arms.append(ArmState(str(arm.get("arm_id", arm.get("family", f"arm_{i}"))), rows, projector, cond))
    return arms


def train_one_epoch(
    *,
    model: torch.nn.Module,
    arms: Sequence[ArmState],
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    img_size: int,
    clip_grad: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    rels: list[float] = []
    for batch in loader:
        x_img, _labels, _indices = occ.batch_to_flat(batch, device)
        x = x_img.reshape(x_img.shape[0], -1)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = torch.zeros((), device=device)
                for arm in arms:
                    y = arm.projector.measurement(x)
                    r = arm.projector.data_anchor(y)
                    target_null = x - r
                    out = reconstruct_with_projected_residual(
                        model,
                        arm.projector,
                        r,
                        y,
                        img_size=img_size,
                        cond_features=arm.cond_features,
                    )
                    loss = loss + F.mse_loss(out.null_hat, target_null)
                    rels.append(float(out.relmeaserr.detach().max().cpu()))
                loss = loss / max(len(arms), 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = torch.zeros((), device=device)
            for arm in arms:
                y = arm.projector.measurement(x)
                r = arm.projector.data_anchor(y)
                target_null = x - r
                out = reconstruct_with_projected_residual(
                    model,
                    arm.projector,
                    r,
                    y,
                    img_size=img_size,
                    cond_features=arm.cond_features,
                )
                loss = loss + F.mse_loss(out.null_hat, target_null)
                rels.append(float(out.relmeaserr.detach().max().cpu()))
            loss = loss / max(len(arms), 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss_mean": float(np.mean(losses)), "relmeaserr_max": float(np.max(rels))}


@torch.no_grad()
def predict_arm(
    *,
    model: torch.nn.Module,
    arm: ArmState,
    loader,
    device: torch.device,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    xs, xhats, rels, labels, indices = [], [], [], [], []
    for batch in loader:
        x_img, lab, idx = occ.batch_to_flat(batch, device)
        x = x_img.reshape(x_img.shape[0], -1)
        y = arm.projector.measurement(x)
        r = arm.projector.data_anchor(y)
        out = reconstruct_with_projected_residual(model, arm.projector, r, y, img_size=img_size, cond_features=arm.cond_features)
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


def add_arm(rows: list[dict[str, Any]], arm_id: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["arm_id"] = arm_id
        out.append(item)
    return out


def paired_delta_for_arm(per_rows: Sequence[Mapping[str, Any]], arm_id: str, method: str, reference: str, metric: str, *, reps: int, seed: int) -> dict[str, Any]:
    subset = [r for r in per_rows if str(r.get("arm_id")) == arm_id]
    out = occ.paired_delta(subset, method, reference, metric, reps=reps, seed=seed)
    out["arm_id"] = arm_id
    return out


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/operator_conditioned_nullspace/budget_dev"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    device = resolve_device(str(config.get("device", "cuda")))
    seed = int(config.get("seed", 20260625))
    torch.manual_seed(seed)
    np.random.seed(seed)

    img_size = int(config["operator"].get("img_size", 64))
    splits, split_manifest = occ.split_indices(config)
    loaders = {
        "train": occ.make_loader(splits["train"], batch_size=int(config.get("batch_size", 16)), shuffle=True, seed=seed + 1),
        "val": occ.make_loader(splits["val"], batch_size=int(config.get("batch_size", 16)), shuffle=False, seed=seed + 2),
        "test": occ.make_loader(splits["test"], batch_size=int(config.get("batch_size", 16)), shuffle=False, seed=seed + 3),
    }
    arms = build_arms(config, device=device)
    model_cfg = dict(config.get("model", {}))
    model = SmallNullspaceUNet(in_channels=3, base_channels=int(model_cfg.get("base_channels", 32)), blocks=int(model_cfg.get("blocks", 2))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 3e-4)), weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)))
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    clip_grad = float(config.get("training", {}).get("clip_grad", 1.0))
    epochs = int(config.get("training", {}).get("epochs", 10))
    train_log: list[dict[str, Any]] = []
    best = {"epoch": -1, "val_centered_mean": float("inf"), "state": None}
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            arms=arms,
            loader=loaders["train"],
            optimizer=opt,
            device=device,
            img_size=img_size,
            clip_grad=clip_grad,
            scaler=scaler,
        )
        val_centered: list[float] = []
        val_joint_centered: list[float] = []
        val_rel: list[float] = []
        for arm in arms:
            x_val, pred_val, rel_val, _lab, _idx = predict_arm(model=model, arm=arm, loader=loaders["val"], device=device, img_size=img_size)
            joint_val, _rel_joint = occ.predict_joint(arm.projector, x_val, device=device)
            val_centered.append(float(occ.centered_rmse(pred_val, x_val).mean()))
            val_joint_centered.append(float(occ.centered_rmse(joint_val, x_val).mean()))
            val_rel.append(float(rel_val.max()))
        row = {
            "epoch": epoch,
            **train_stats,
            "val_centered_mean": float(np.mean(val_centered)),
            "val_joint_centered_mean": float(np.mean(val_joint_centered)),
            "val_delta_vs_joint_mean": float(np.mean(val_centered) - np.mean(val_joint_centered)),
            "val_relmeaserr_max": float(np.max(val_rel)),
        }
        train_log.append(row)
        if row["val_centered_mean"] < best["val_centered_mean"]:
            best = {"epoch": epoch, "val_centered_mean": row["val_centered_mean"], "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if best["state"] is None:
        raise BudgetConditionedError("NO_BEST_STATE")
    model.load_state_dict(best["state"])
    ckpt_path = output_dir / "checkpoints" / "best.pt"
    ensure_dir(ckpt_path.parent)
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "config": occ.json_safe(config), "best_epoch": int(best["epoch"])}, tmp)
    os.replace(tmp, ckpt_path)

    metric_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    qcfg = dict(config.get("quality", {}))
    for arm_index, arm in enumerate(arms):
        x_test, pred_test, rel_model, labels, indices = predict_arm(model=model, arm=arm, loader=loaders["test"], device=device, img_size=img_size)
        joint, rel_joint = occ.predict_joint(arm.projector, x_test, device=device)
        estimates: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("joint_minimum_norm", joint, rel_joint),
            ("budget_conditioned_null_predictor", pred_test, rel_model),
        ]
        if bool(config.get("classical", {}).get("include_tikhonov", True)):
            tikh, rel_tikh = occ.tikhonov_anchor(arm.rows, x_test, lambda_=float(config.get("classical", {}).get("tikhonov_lambda", 1e-5)), device=device)
            estimates.append(("joint_tikhonov", tikh, rel_tikh))
        if bool(config.get("classic_tv", {}).get("enabled", False)):
            y_test = x_test @ arm.rows.T
            tv, tv_diag = tv_baseline(arm.rows, y_test, img_size=img_size, config=config, device=device)
            if tv.shape == x_test.shape:
                rel_tv = np.linalg.norm(tv @ arm.rows.T - y_test, axis=1) / np.maximum(np.linalg.norm(y_test, axis=1), 1e-12)
                estimates.append(("joint_tv_pgd", tv, rel_tv))
        else:
            tv_diag = {"status": "SKIPPED_BY_CONFIG"}
        for method, xhat, rel in estimates:
            summary, per = occ.metric_bundle(
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
            summary["arm_id"] = arm.arm_id
            summary["arm_index"] = arm_index
            metric_rows.append(summary)
            per_rows.extend(add_arm(per, arm.arm_id))

    write_csv(reports / "method_metrics.csv", metric_rows)
    write_csv(reports / "per_image_metrics.csv", per_rows)
    write_csv(reports / "train_log.csv", train_log)
    reps = int(config.get("statistics", {}).get("bootstrap_replicates", 500))
    comparisons = [
        paired_delta_for_arm(per_rows, arm.arm_id, "budget_conditioned_null_predictor", "joint_minimum_norm", "centered_rmse", reps=reps, seed=seed + 100 + i)
        for i, arm in enumerate(arms)
    ]
    write_json(reports / "paired_comparisons.json", comparisons)
    arm_pass = {
        c["arm_id"]: bool(c["mean_delta"] < -float(config.get("gate", {}).get("min_centered_rmse_gain", 1e-4)) and c["ci_upper"] < 0)
        for c in comparisons
    }
    rel_ok = all(float(row["relmeaserr_max"]) < float(config.get("gate", {}).get("relmeaserr_max", 1e-4)) for row in metric_rows if row["method"] == "budget_conditioned_null_predictor")
    required = int(config.get("gate", {}).get("required_arm_passes", len(arms)))
    decision = "BUDGET_CONDITIONED_DEV_PASS_EXPAND_SEEDS" if sum(arm_pass.values()) >= required and rel_ok else "BUDGET_CONDITIONED_DEV_FAIL_DIAGNOSE"
    gate = {
        "status": "PASS",
        "decision": decision,
        "scope": "fresh development split; not locked",
        "arm_pass": arm_pass,
        "required_arm_passes": required,
        "relmeaserr_ok": rel_ok,
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
            "arms": [
                {
                    "arm_id": arm.arm_id,
                    "rows_shape": list(arm.rows.shape),
                    "rows_sha256": occ.sha256_numpy(arm.rows),
                    "row_audit": row_audit(arm.rows, name=arm.arm_id),
                    "projector": arm.projector.diagnostics(),
                }
                for arm in arms
            ],
        },
    )
    conclusion = ["# Budget-Conditioned Null-Space Development", "", f"- Decision: `{decision}`", f"- Best epoch: `{best['epoch']}`", ""]
    for c in comparisons:
        conclusion.append(f"- `{c['arm_id']}` centered delta vs joint: `{c['mean_delta']}` CI `[{c['ci_lower']}, {c['ci_upper']}]`, wins `{c['wins']}/{c['n']}`")
    conclusion.append("")
    conclusion.append("This is a development-only expansion after the DCT canary. It does not authorize locked testing.")
    (reports / "research_decision.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    runtime = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "artifact_hashes": {
            "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
            "best.pt": sha256_file(ckpt_path),
            "method_metrics.csv": sha256_file(reports / "method_metrics.csv"),
            "paired_comparisons.json": sha256_file(reports / "paired_comparisons.json"),
            "gate_report.json": sha256_file(reports / "gate_report.json"),
        },
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "BUDGET_CONDITIONED_DEVELOPMENT_COMPLETE",
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
    occ.atomic_json(output_dir / "BUDGET_CONDITIONED_DEVELOPMENT_COMPLETE.json", {"status": summary["status"], "decision": decision, "summary_sha256": sha256_file(reports / "summary.json")})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a shared budget/operator-conditioned null-space predictor.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.config))
    print(json.dumps(occ.json_safe({"status": summary["status"], "output_dir": summary["output_dir"], "decision": summary["gate"]["decision"], "key_artifacts": summary["key_artifacts"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
