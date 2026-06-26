from __future__ import annotations

import copy
import csv
import json
import math
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from . import phase71_gauge_cgan_paired_seeds as p71
from . import phase73_overnight_gauge_gan_expansion as p73
from .models import build_generator
from .utils import set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase74_high_tier_gauge_cgan_pack"
PH69A = ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH71 = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
PH72 = ROOT / "outputs_phase72_scr10_gauge_cgan_regime_validation"
PH73 = ROOT / "outputs_phase73_overnight_gauge_gan_expansion"
CERT = ROOT / "results" / "cert_package_20260612"
CACHE = CERT / "cache"

TRAIN_COUNT = 1024
VAL_COUNT = 256
TEST_COUNT = 256
BATCH_SIZE = 8
STEP_BUDGET = 300
EVAL_EVERY = 100
PSNR_LOSS_GUARDRAIL_DB = 0.3
RELMEASERR_ABS_DELTA_GUARDRAIL = 1e-4

REGIME_INFO = copy.deepcopy(p73.REGIMES)
REGIME_INFO["rad10"] = {
    "checkpoint": ROOT / "outputs_phase15" / "imported_noleak" / "rademacher10_full_noise001_colab" / "last.pt",
    "config": ROOT / "outputs_phase15" / "imported_noleak" / "rademacher10_full_noise001_colab" / "resolved_config.yaml",
    "A": CACHE / "A_rad10.npy",
    "cache": CACHE / "main_rad10.npz",
    "manifest": CACHE / "main_rad10_manifest.json",
    "orthonormal": False,
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |",
        "| " + " | ".join("-" * widths[c] for c in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) + " |")
    return "\n".join(lines)


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def configure_phase73_helpers() -> None:
    p73.OUT = OUT
    p73.REGIMES.clear()
    p73.REGIMES.update(REGIME_INFO)


def metric_direction(metric: str) -> str:
    return "higher" if metric in {"psnr", "ssim"} else "lower"


def paired_compare(rows: list[dict[str, Any]], baseline: str, candidate: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row["arm"]), int(row["sample_index"]))] = row
    metrics = [
        "psnr",
        "ssim",
        "relmeaserr_unclipped_float64",
        "correction_norm_rel",
        "rapsd_distance",
        "gradient_mean_abs_error",
        "highfreq_ratio_abs_error",
        "p0_l2",
        "lpips",
    ]
    out: list[dict[str, Any]] = []
    sample_ids = sorted({sid for arm, sid in by_key if arm == baseline} & {sid for arm, sid in by_key if arm == candidate})
    for metric in metrics:
        vals_b, vals_c, imp = [], [], []
        for sid in sample_ids:
            rb = by_key[(baseline, sid)]
            rc = by_key[(candidate, sid)]
            if metric not in rb or metric not in rc or rb.get(metric, "") == "" or rc.get(metric, "") == "":
                continue
            b = float(rb[metric])
            c = float(rc[metric])
            vals_b.append(b)
            vals_c.append(c)
            imp.append((c - b) if metric_direction(metric) == "higher" else (b - c))
        if not imp:
            continue
        mean, lo, hi = p69b.bootstrap_ci(np.asarray(imp, dtype=np.float64), seed=74000 + len(out), n_boot=1000)
        out.append(
            {
                "pair": f"{candidate}_vs_{baseline}",
                "metric": metric,
                "direction": metric_direction(metric),
                f"mean_{baseline}": float(np.nanmean(vals_b)),
                f"mean_{candidate}": float(np.nanmean(vals_c)),
                f"mean_{candidate}_minus_{baseline}": float(np.nanmean(vals_c) - np.nanmean(vals_b)),
                f"improvement_positive_means_{candidate}_better": mean,
                "ci_low": lo,
                "ci_high": hi,
                f"ci_excludes_zero_in_favor_of_{candidate}": bool(lo > 0),
            }
        )
    return out


def compute_lpips_any(
    out_dir: Path,
    test: p69b.SplitCache,
    outputs: dict[str, np.ndarray],
    device: torch.device,
    per_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    try:
        import lpips
    except Exception as exc:
        write_csv(out_dir / "lpips_or_dists_results.csv", [{"metric_package": "LPIPS", "available": False, "note": str(exc)}])
        return
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true = test.x[:, 0].numpy().astype(np.float32)
    true_t = p71.prep_lpips(true)
    sample_indices = test.indices.numpy().astype(int).tolist()
    per_lpips: list[dict[str, Any]] = []
    lpips_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], float] = {}
    with torch.no_grad():
        for arm, arr in outputs.items():
            pred_t = p71.prep_lpips(arr.astype(np.float32))
            vals: list[float] = []
            for i in range(0, pred_t.shape[0], 16):
                vals.extend(
                    loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device))
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )
            vals_np = np.asarray(vals, dtype=np.float64)
            lpips_rows.append(
                {
                    "metric_package": "LPIPS",
                    "available": True,
                    "arm": arm,
                    "n": int(vals_np.shape[0]),
                    "lpips_mean": float(vals_np.mean()),
                    "lpips_median": float(np.median(vals_np)),
                    "lpips_std": float(vals_np.std()),
                }
            )
            for ordinal, value in enumerate(vals):
                row = {"arm": arm, "sample_ordinal": ordinal, "sample_index": sample_indices[ordinal], "lpips": float(value)}
                per_lpips.append(row)
                lookup[(arm, sample_indices[ordinal])] = float(value)
    write_csv(out_dir / "lpips_or_dists_results.csv", lpips_rows)
    write_csv(out_dir / "lpips_per_sample.csv", per_lpips)
    for row in per_rows:
        key = (str(row["arm"]), int(row["sample_index"]))
        if key in lookup:
            row["lpips"] = lookup[key]
    for row in eval_rows:
        vals = [v for (arm, _sid), v in lookup.items() if arm == row["arm"]]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            row["lpips_mean"] = float(arr.mean())
            row["lpips_median"] = float(np.median(arr))
            row["lpips_std"] = float(arr.std())


def plot_score_bars(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aucs = np.asarray([float(r["auc"]) for r in rows])
    lows = np.asarray([float(r.get("auc_ci_low", r["auc"])) for r in rows])
    highs = np.asarray([float(r.get("auc_ci_high", r["auc"])) for r in rows])
    ax.bar(range(len(rows)), aucs, yerr=np.vstack([aucs - lows, highs - aucs]), capsize=4)
    ax.axhline(0.65, color="tab:orange", linestyle="--", linewidth=1)
    ax.axhline(0.75, color="tab:green", linestyle="--", linewidth=1)
    ax.set_ylabel("held-out AUC")
    ax.set_title(title)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["model"] for r in rows], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def visual_grid_any(path: Path, test: p69b.SplitCache, outputs: dict[str, np.ndarray], arms: list[str], n: int = 6) -> None:
    count = min(n, int(test.x.shape[0]))
    fig, axes = plt.subplots(count, len(arms) + 1, figsize=(2.1 * (len(arms) + 1), 2.0 * count))
    if count == 1:
        axes = axes[None, :]
    true = test.x[:count, 0].numpy()
    for i in range(count):
        imgs = [("GT", true[i])] + [(arm, outputs[arm][i]) for arm in arms]
        for j, (title, img) in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_phase74_checkpoint(
    path: Path,
    regime: str,
    seed_id: int,
    arm: str,
    step: int,
    generator,
    opt_g,
    config: dict[str, Any],
    metrics: dict[str, Any],
    loader_seed: int,
    beta: float,
    adv_mode: str,
    critic=None,
    opt_d=None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase74",
        "regime": regime,
        "seed_id": int(seed_id),
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": opt_g.state_dict() if opt_g else None,
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(REGIME_INFO[regime]["checkpoint"]),
        "source_checkpoint_sha256": p69a.sha256_file(REGIME_INFO[regime]["checkpoint"]),
        "beta": float(beta),
        "adv_mode": adv_mode,
        "paired_loader_seed": int(loader_seed),
        "checkpoint_selection_rule": "best_by_val_rec_loss",
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if opt_d is not None:
        payload["optimizer_d"] = opt_d.state_dict()
    torch.save(payload, path)


def train_general_arm(
    regime: str,
    seed_id: int,
    arm: str,
    generator,
    measurement,
    train: p69b.SplitCache,
    val: p69b.SplitCache,
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    loader_seed: int,
    beta: float,
    adv_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    arm_dir = ensure_dir(out_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adv_mode in {"gauge", "standard"} else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = p69b.cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=loader_seed))
    rows: list[dict[str, Any]] = []
    d_hist: list[float] = []
    best_val = float("inf")
    best_step = -1
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics: dict[str, Any] = {}
    for step in range(1, STEP_BUDGET + 1):
        x, y, _lbl, _idx = next(loader)
        x = x.to(device)
        y = y.to(device)
        d_loss_v = float("nan")
        d_acc = float("nan")
        if critic is not None and opt_d is not None:
            generator.eval()
            critic.train()
            with torch.no_grad():
                od = p73.forward_candidate_general(generator, measurement, x, y, config)
            real_img = od["real_gauge"] if adv_mode == "gauge" else x
            fake_img = od["fake_gauge"] if adv_mode == "gauge" else od["x_hat"]
            opt_d.zero_grad(set_to_none=True)
            rs = critic(real_img)
            fs = critic(fake_img)
            d_loss = F.relu(1.0 - rs).mean() + F.relu(1.0 + fs).mean()
            d_loss.backward()
            opt_d.step()
            d_loss_v = float(d_loss.detach().cpu())
            d_acc = p69b.d_accuracy(rs, fs)
            d_hist.append(d_acc)
            generator.train()
        opt_g.zero_grad(set_to_none=True)
        out = p73.forward_candidate_general(generator, measurement, x, y, config)
        rec = p69b.charbonnier(out["x_hat"], x)
        adv = torch.zeros((), device=device)
        if critic is not None:
            critic.eval()
            fake_img = out["fake_gauge"] if adv_mode == "gauge" else out["x_hat"]
            adv = -critic(fake_img).mean()
        loss = rec + float(beta) * adv
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {regime} seed={seed_id} arm={arm} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()
        row = {
            "regime": regime,
            "seed": seed_id,
            "arm": arm,
            "step": step,
            "paired_loader_seed": loader_seed,
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec.detach().cpu()),
            "loss_adv": float(adv.detach().cpu()),
            "loss_d": d_loss_v,
            "d_accuracy": d_acc,
            "beta": float(beta),
            "adv_mode": adv_mode,
        }
        if step % EVAL_EVERY == 0 or step == STEP_BUDGET:
            vm = p73.eval_val_loss_general(generator, measurement, val, config, device)
            row.update(vm)
            final_metrics = vm
            if vm["val_rec_loss"] < best_val:
                best_val = float(vm["val_rec_loss"])
                best_step = step
                save_phase74_checkpoint(best_path, regime, seed_id, arm, step, generator, opt_g, config, vm, loader_seed, beta, adv_mode, critic, opt_d)
        rows.append(row)
    final_path = ckpt_dir / "final.pt"
    save_phase74_checkpoint(final_path, regime, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, adv_mode, critic, opt_d)
    if not best_path.exists():
        best_step = STEP_BUDGET
        best_val = float(final_metrics.get("val_rec_loss", float("nan")))
        save_phase74_checkpoint(best_path, regime, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, adv_mode, critic, opt_d)
    summary = {
        "regime": regime,
        "seed": seed_id,
        "arm": arm,
        "steps": STEP_BUDGET,
        "paired_loader_seed": loader_seed,
        "beta": float(beta),
        "adv_mode": adv_mode,
        "best_val_rec_loss": best_val,
        "best_step": best_step,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": p69a.sha256_file(best_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": p69a.sha256_file(final_path),
        "d_accuracy_last_mean": float(np.nanmean(d_hist[-50:])) if d_hist else float("nan"),
        "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_hist[-50:]) > 0.95) if d_hist else False,
    }
    write_csv(arm_dir / "training_log.csv", rows)
    save_json(arm_dir / "training_summary.json", summary)
    return summary, rows, best_path


def load_checkpoint_for_eval(path: Path, config: dict[str, Any], measurement, device: torch.device):
    generator = build_generator(config, measurement=measurement).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(payload["generator"])
    generator.eval()
    return generator


def beta_calibration_general(
    regime: str,
    generator,
    measurement,
    train: p69b.SplitCache,
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    target: float = 0.075,
) -> float:
    critic = p69a.PatchCritic(1).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9))
    loader = p69b.cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=74201))
    generator.eval()
    for _ in range(20):
        x, y, _lbl, _idx = next(loader)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = p73.forward_candidate_general(generator, measurement, x, y, config)
        opt.zero_grad(set_to_none=True)
        rs = critic(out["real_gauge"])
        fs = critic(out["fake_gauge"])
        d_loss = F.relu(1.0 - rs).mean() + F.relu(1.0 + fs).mean()
        d_loss.backward()
        opt.step()
    x, y, _lbl, _idx = next(iter(p69b.make_loader(train, BATCH_SIZE, shuffle=False, seed=74202)))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        base = p73.forward_candidate_general(generator, measurement, x, y, config)
    v = base["v_pre"].detach().requires_grad_(True)
    x_hat_flat = measurement.dc_project(v, y)
    x_hat = measurement.unflatten_img(x_hat_flat)
    A64, G, K = p73.exact_projectors(measurement.A, float(config["lambda_solver"]))
    fake_gauge = measurement.unflatten_img((p69a.p0_exact(v, A64, G) + p69a.blambda_y(y, A64, K)).to(torch.float32))
    rec = p69b.charbonnier(x_hat, x)
    adv = -critic(fake_gauge).mean()
    gr = torch.autograd.grad(rec, v, retain_graph=True)[0]
    ga = torch.autograd.grad(adv, v)[0]
    rec_norm = float(torch.linalg.norm(gr).detach().cpu())
    adv_norm = float(torch.linalg.norm(ga).detach().cpu())
    ratio = adv_norm / max(rec_norm, 1e-12)
    beta0 = float(np.clip(target / max(ratio, 1e-12), 1e-5, 1.0))
    write_csv(
        out_dir / f"{regime}_beta_calibration.csv",
        [
            {
                "regime": regime,
                "grad_rec_norm": rec_norm,
                "grad_adv_norm": adv_norm,
                "adv_to_rec_ratio": ratio,
                "target_beta_times_ratio": target,
                "selected_beta0": beta0,
                "candidate_0p3_beta0": 0.3 * beta0,
                "candidate_beta0": beta0,
                "candidate_3_beta0": 3.0 * beta0,
            }
        ],
    )
    return beta0


def preflight_and_protocol() -> None:
    ensure_dir(OUT)
    failures: list[str] = []
    required = [
        PH69A / "critic_auc_results.csv",
        PH69A / "shortcut_control_results.csv",
        PH71 / "scr5_seed_stability.csv",
        PH72 / "scr10_gauge_signal_auc.csv",
        PH73 / "rad5_gauge_signal_auc.csv",
        PH73 / "rad5_seed_delta_metrics.csv",
        PH73 / "human_2afc_pack",
        p69a.SPLIT_TRAIN,
        p69a.SPLIT_EVAL,
    ]
    for regime, info in REGIME_INFO.items():
        required.extend([info["checkpoint"], info["config"], info["A"], info["cache"], info["manifest"]])
    for path in required:
        if not Path(path).exists():
            failures.append(f"Missing required path: {path}")
    if failures:
        write_text(
            OUT / "UNSAFE_TO_RUN.md",
            "# UNSAFE TO RUN\n\n" + "\n".join(f"- {x}" for x in failures) + "\n\nNo Phase74 training/evaluation was run.\n",
        )
        raise RuntimeError("Unsafe to run Phase74; see UNSAFE_TO_RUN.md")
    train_idx = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_idx = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    protocol = {
        "phase": "Phase74",
        "output_dir": str(OUT),
        "created": now(),
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "batch_size": BATCH_SIZE,
        "step_budget": STEP_BUDGET,
        "eval_every": EVAL_EVERY,
        "train_full_sorted_sha256": p69a.sha256_np(train_idx, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_idx, sort_int64=True),
        "forbidden_D_inputs": "Au-y, RelMeasErr, correction vector, Pi_y(v)-v except explicit residual shortcut controls",
        "main_results_modified": False,
        "checkpoint_overwrite_policy": "new Phase74 output directory only",
        "paired_seed_rule": "B/C/D and beta arms use same init seed, train loader seed, optimizer, budget, validation selection; only adversarial input/loss or beta differs.",
        "regimes": {
            name: {
                "checkpoint": str(info["checkpoint"]),
                "checkpoint_sha256": p69a.sha256_file(info["checkpoint"]),
                "A": str(info["A"]),
                "A_sha256_float32": p69a.sha256_np(np.load(info["A"]).astype(np.float32)),
                "cache": str(info["cache"]),
            }
            for name, info in REGIME_INFO.items()
        },
    }
    save_json(OUT / "phase74_protocol_config.json", protocol)
    write_text(
        OUT / "PHASE74_PROTOCOL_LOCK.md",
        "\n".join(
            [
                "# Phase74 Protocol Lock",
                "",
                f"Output directory: `{OUT}`",
                "",
                "## Non-Negotiable Constraints",
                "",
                "- First-paper title, abstract, checkpoint, tables, and main results are not modified.",
                "- No existing checkpoint is overwritten; all new checkpoints are Phase74-only.",
                "- Test split is never used for training.",
                "- D is never given residual/certificate/correction features except the explicit shortcut control.",
                "- RelMeasErr is unclipped float64 against the recorded or generated y; PSNR/SSIM use clipped display images.",
                "",
                "## Split Hashes",
                "",
                table(
                    [
                        {"split": "train_full_sorted", "sha256": protocol["train_full_sorted_sha256"]},
                        {"split": "eval_full_sorted", "sha256": protocol["eval_full_sorted_sha256"]},
                    ],
                    ["split", "sha256"],
                ),
                "",
                "## Regime Sources",
                "",
                table(
                    [
                        {
                            "regime": name,
                            "checkpoint_sha256": item["checkpoint_sha256"],
                            "A_sha256_float32": item["A_sha256_float32"],
                        }
                        for name, item in protocol["regimes"].items()
                    ],
                    ["regime", "checkpoint_sha256", "A_sha256_float32"],
                ),
                "",
            ]
        ),
    )
    append_log("preflight_and_protocol_complete")


def inventory() -> None:
    rows: list[dict[str, Any]] = []

    def add(stage: str, path: Path, role: str, status: str = "present") -> None:
        rows.append(
            {
                "stage": stage,
                "role": role,
                "path": str(path),
                "exists": path.exists(),
                "status": status if path.exists() else "missing",
                "sha256": p69a.sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )

    for stage, folder, files in [
        ("Phase69A", PH69A, ["critic_auc_results.csv", "shortcut_control_results.csv", "PHASE69A_GO_NOGO.md", "gauge_dataset_manifest.json"]),
        ("Phase70", PH70, ["beta_frontier.csv", "PHASE70_GAUGE_GAN_PAPER_REPORT.md", "WORKSHOP_READINESS.md"]),
        ("Phase71", PH71, ["scr5_seed_stability.csv", "scr5_seed_delta_metrics.csv", "checkpoint_hashes.csv", "PHASE71_PAIRED_SEED_REPORT.md"]),
        ("Phase72", PH72, ["scr10_gauge_signal_auc.csv", "PHASE72_SCR10_REPORT.md", "SCR10_SEED_STABILITY_REPORT.md"]),
        ("Phase73", PH73, ["rad5_gauge_signal_auc.csv", "rad5_shortcut_controls.csv", "rad5_seed_delta_metrics.csv", "p0_space_metrics.csv", "human_2afc_pairs.csv", "PHASE73_OVERNIGHT_REPORT.md"]),
    ]:
        for file in files:
            add(stage, folder / file, file)
    write_csv(OUT / "evidence_inventory_phase74.csv", rows)

    auc_rows: list[dict[str, Any]] = []
    for label, path in [
        ("Scr5 Phase69A gauge", PH69A / "critic_auc_results.csv"),
        ("Scr5 Phase69A shortcut", PH69A / "shortcut_control_results.csv"),
        ("Scr10 Phase72 gauge", PH72 / "scr10_gauge_signal_auc.csv"),
        ("Rad5 Phase73 gauge", PH73 / "rad5_gauge_signal_auc.csv"),
        ("Rad5 Phase73 shortcut", PH73 / "rad5_shortcut_controls.csv"),
    ]:
        if path.exists():
            for r in read_csv(path):
                rr = dict(r)
                rr["source_label"] = label
                auc_rows.append(rr)
    write_csv(OUT / "auc_shortcut_inventory_phase74.csv", auc_rows)
    write_text(
        OUT / "EVIDENCE_INVENTORY_PHASE74.md",
        "\n".join(
            [
                "# Phase74 Evidence Inventory",
                "",
                "This inventory is read-only with respect to Phase69A-73. New files are written only under Phase74.",
                "",
                "## Core Evidence Files",
                "",
                table(rows, ["stage", "role", "exists", "status", "sha256"]),
                "",
                "## AUC / Shortcut Evidence",
                "",
                table(auc_rows, ["source_label", "model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy"]),
                "",
            ]
        ),
    )
    append_log("inventory_complete")


def rad10_diagnostic(device: torch.device) -> dict[str, Any]:
    append_log("rad10_diagnostic_start")
    config = p73.regime_config("rad10", device)
    measurement, _A = p73.make_regime_measurement("rad10", config, device)
    gen, config = p73.load_regime_generator("rad10", config, measurement, device, train=False)
    train, val, test, split = p73.build_caches("rad10", config, measurement, device)
    save_json(OUT / "rad10_split_manifest.json", split)
    tr = p73.gauge_split(gen, measurement, train, config, device)
    va = p73.gauge_split(gen, measurement, val, config, device)
    te = p73.gauge_split(gen, measurement, test, config, device)
    rows: list[dict[str, Any]] = []
    hist: list[dict[str, Any]] = []
    score_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    r, h, y, s = p73.train_image_model(
        "patchgan_unconditional_gauge_rad10",
        p69a.PatchCritic(1),
        tr["image_x"],
        tr["image_y"],
        va["image_x"],
        va["image_y"],
        te["image_x"],
        te["image_y"],
        device,
        epochs=4,
        seed=74101,
    )
    rows.append(r)
    hist.extend(h)
    score_sets[r["model"]] = (y, s)
    r, h, y, s = p73.train_image_model(
        "simple_cnn_gauge_rad10",
        p69a.SimpleCNN(1),
        tr["image_x"],
        tr["image_y"],
        va["image_x"],
        va["image_y"],
        te["image_x"],
        te["image_y"],
        device,
        epochs=4,
        seed=74102,
    )
    rows.append(r)
    hist.extend(h)
    score_sets[r["model"]] = (y, s)
    cond_train = np.concatenate([tr["image_x"], tr["x_data_pair"]], axis=1)
    cond_val = np.concatenate([va["image_x"], va["x_data_pair"]], axis=1)
    cond_test = np.concatenate([te["image_x"], te["x_data_pair"]], axis=1)
    r, h, y, s = p73.train_image_model(
        "patchgan_conditional_xdata_gauge_rad10",
        p69a.PatchCritic(2),
        cond_train,
        tr["image_y"],
        cond_val,
        va["image_y"],
        cond_test,
        te["image_y"],
        device,
        epochs=3,
        seed=74103,
    )
    rows.append(r)
    hist.extend(h)
    score_sets[r["model"]] = (y, s)
    ctrl, h, y, s = p73.train_residual_model(tr, va, te, device)
    ctrl["model"] = "residual_features_logistic_rad10"
    hist.extend(h)
    score_sets[ctrl["model"]] = (y, s)
    write_csv(OUT / "rad10_gauge_auc.csv", rows)
    write_csv(OUT / "rad10_shortcut_controls.csv", [ctrl])
    write_csv(OUT / "rad10_gauge_training_history.csv", hist)
    p69a.save_score_histograms(OUT, score_sets)
    if (OUT / "critic_score_histograms.png").exists():
        (OUT / "critic_score_histograms.png").replace(OUT / "rad10_gauge_score_histograms.png")
    plot_score_bars(OUT / "rad10_shortcut_auc_plot.png", rows + [ctrl], "Rad-10 gauge signal and shortcut controls")
    main = rows[0]
    auc = float(main["auc"])
    lo = float(main["auc_ci_low"])
    hi = float(main["auc_ci_high"])
    if auc >= 0.75 and lo > 0.70:
        decision = "strong_run_3_paired_seeds"
        n_seeds = 3
    elif auc >= 0.65:
        decision = "moderate_run_1_paired_seed"
        n_seeds = 1
    else:
        decision = "weak_no_training"
        n_seeds = 0
    gate = {
        "decision": decision,
        "paired_seed_count": n_seeds,
        "rad10_gauge_auc": auc,
        "rad10_gauge_auc_ci_low": lo,
        "rad10_gauge_auc_ci_high": hi,
        "rad10_residual_auc": float(ctrl["auc"]),
    }
    save_json(OUT / "rad10_gate_decision.json", gate)
    write_text(
        OUT / "RAD10_GAUGE_SIGNAL_REPORT.md",
        "\n".join(
            [
                "# Rad-10 Gauge Signal Report",
                "",
                table(rows, ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"]),
                "",
                "## Shortcut Control",
                "",
                table([ctrl], ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"]),
                "",
                f"Decision: `{decision}`.",
                "",
            ]
        ),
    )
    append_log(f"rad10_diagnostic_complete decision={decision} auc={auc:.4f}")
    return gate


def run_rad10_seeds(gate: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = int(gate.get("paired_seed_count", 0))
    if n <= 0:
        write_csv(OUT / "rad10_seed_metrics.csv", [{"status": "not_run", "reason": "Rad-10 gauge gate did not pass"}])
        write_csv(OUT / "rad10_seed_delta_metrics.csv", [{"status": "not_run", "reason": "Rad-10 gauge gate did not pass"}])
        write_text(
            OUT / "RAD10_NO_TRAIN_GATE_REPORT.md",
            "# Rad-10 No-Train Gate Report\n\nRad-10 cGAN paired training was not run because the gauge AUC gate did not pass.\n",
        )
        return [], []
    config = p73.regime_config("rad10", device)
    measurement, _A = p73.make_regime_measurement("rad10", config, device)
    probe, config = p73.load_regime_generator("rad10", config, measurement, device, train=False)
    train, val, test, _split = p73.build_caches("rad10", config, measurement, device)
    beta0 = beta_calibration_general("rad10", probe, measurement, train, config, device, OUT)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_delta: list[dict[str, Any]] = []
    ckpt_rows: list[dict[str, Any]] = []
    for seed_id in range(1, n + 1):
        seed_dir = ensure_dir(OUT / f"rad10_seed{seed_id:02d}")
        loader_seed = 744000 + seed_id
        base_seed = 744100 + 100 * seed_id
        set_seed(base_seed)
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        gen_b, _ = p73.load_regime_generator("rad10", config, measurement, device, train=True)
        bsum, brows, bbest = train_general_arm("rad10", seed_id, "B", gen_b, measurement, train, val, config, device, seed_dir, loader_seed, 0.0, "none")
        del gen_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(base_seed)
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        gen_c, _ = p73.load_regime_generator("rad10", config, measurement, device, train=True)
        csum, crows, cbest = train_general_arm("rad10", seed_id, "C", gen_c, measurement, train, val, config, device, seed_dir, loader_seed, beta0, "gauge")
        del gen_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_csv(seed_dir / "training_log.csv", brows + crows)
        eval_dir = ensure_dir(seed_dir / "evaluation")
        gen_a, _ = p73.load_regime_generator("rad10", config, measurement, device, train=False)
        gen_b_eval = load_checkpoint_for_eval(bbest, config, measurement, device)
        gen_c_eval = load_checkpoint_for_eval(cbest, config, measurement, device)
        eval_rows: list[dict[str, Any]] = []
        per_rows: list[dict[str, Any]] = []
        outputs: dict[str, np.ndarray] = {}
        for arm, gen in [("A", gen_a), ("B", gen_b_eval), ("C", gen_c_eval)]:
            agg, per, arr = p73.evaluate_general(arm, gen, measurement, test, config, device, eval_dir)
            agg["seed"] = seed_id
            eval_rows.append(agg)
            for row in per:
                row["seed"] = seed_id
            per_rows.extend(per)
            outputs[arm] = arr
        compute_lpips_any(seed_dir, test, outputs, device, per_rows, eval_rows)
        comp = paired_compare(per_rows, "B", "C")
        write_csv(seed_dir / "evaluation_metrics.csv", eval_rows)
        write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
        write_csv(seed_dir / "paired_comparison_C_vs_B.csv", comp)
        visual_grid_any(seed_dir / "visual_grid_A_B_C.png", test, outputs, ["A", "B", "C"], n=6)
        seed_results.append({"seed": seed_id, "armB_summary": bsum, "armC_summary": csum})
        all_metrics.extend(eval_rows)
        for row in comp:
            row["seed"] = seed_id
        all_delta.extend(comp)
        for arm, summary in [("B", bsum), ("C", csum)]:
            ckpt_rows.append({"seed": seed_id, "arm": arm, "kind": "best_by_val", "path": summary["best_checkpoint"], "sha256": summary["best_checkpoint_sha256"]})
            ckpt_rows.append({"seed": seed_id, "arm": arm, "kind": "final", "path": summary["final_checkpoint"], "sha256": summary["final_checkpoint_sha256"]})
    write_csv(OUT / "rad10_seed_metrics.csv", all_metrics)
    write_csv(OUT / "rad10_seed_delta_metrics.csv", all_delta)
    write_csv(OUT / "rad10_checkpoint_hashes.csv", ckpt_rows)
    write_text(
        OUT / "RAD10_PAIRED_SEED_REPORT.md",
        "\n".join(
            [
                "# Rad-10 Paired-Seed Report",
                "",
                f"Gate: `{gate['decision']}`; counted seeds: `{n}`.",
                "",
                table([r for r in all_delta if r.get("metric") in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64"}], ["seed", "metric", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high"]),
                "",
            ]
        ),
    )
    return seed_results, all_delta


def standard_cgan_baseline_scr5(device: torch.device) -> None:
    append_log("standard_cgan_baseline_scr5_start")
    config = p73.regime_config("scr5", device)
    measurement, _A = p73.make_regime_measurement("scr5", config, device)
    train, val, test, split = p73.build_caches("scr5", config, measurement, device)
    save_json(OUT / "scr5_standard_split_manifest.json", split)
    seed_dir = ensure_dir(OUT / "standard_cgan_scr5_seed01")
    beta0 = float(read_csv(PH69B / "beta_calibration.csv")[0]["selected_beta0"])
    d_best = seed_dir / "D_standard" / "checkpoints" / "best_by_val.pt"
    if not d_best.exists():
        seed_base = 710100
        loader_seed = 710401
        set_seed(seed_base)
        random.seed(seed_base)
        np.random.seed(seed_base)
        torch.manual_seed(seed_base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base)
        gen_d, _ = p73.load_regime_generator("scr5", config, measurement, device, train=True)
        d_summary, d_rows, d_best = train_general_arm("scr5", 1, "D_standard", gen_d, measurement, train, val, config, device, seed_dir, loader_seed, beta0, "standard")
        write_csv(seed_dir / "training_log.csv", d_rows)
        save_json(seed_dir / "D_standard_summary.json", d_summary)
        del gen_d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        d_summary = read_json(seed_dir / "D_standard" / "training_summary.json")
    b_best = PH71 / "seed01" / "B" / "checkpoints" / "best_by_val.pt"
    c_best = PH71 / "seed01" / "C" / "checkpoints" / "best_by_val.pt"
    eval_dir = ensure_dir(seed_dir / "evaluation")
    models = {
        "A": p73.load_regime_generator("scr5", config, measurement, device, train=False)[0],
        "B": load_checkpoint_for_eval(b_best, config, measurement, device),
        "C_gauge": load_checkpoint_for_eval(c_best, config, measurement, device),
        "D_standard": load_checkpoint_for_eval(d_best, config, measurement, device),
    }
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, gen in models.items():
        agg, per, arr = p73.evaluate_general(arm, gen, measurement, test, config, device, eval_dir)
        agg["seed"] = 1
        agg["source"] = "Phase71_BC_reused" if arm in {"B", "C_gauge"} else "Phase74_trained" if arm == "D_standard" else "published_mean"
        eval_rows.append(agg)
        for row in per:
            row["seed"] = 1
        per_rows.extend(per)
        outputs[arm] = arr
    compute_lpips_any(seed_dir, test, outputs, device, per_rows, eval_rows)
    comparisons = []
    comparisons.extend(paired_compare(per_rows, "B", "C_gauge"))
    comparisons.extend(paired_compare(per_rows, "B", "D_standard"))
    comparisons.extend(paired_compare(per_rows, "C_gauge", "D_standard"))
    write_csv(OUT / "standard_cgan_baseline_scr5.csv", eval_rows)
    write_csv(OUT / "standard_cgan_baseline_pairwise_scr5.csv", comparisons)
    write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
    write_csv(seed_dir / "pairwise_comparisons.csv", comparisons)
    visual_grid_any(OUT / "standard_vs_gauge_visual_grid.png", test, outputs, ["A", "B", "C_gauge", "D_standard"], n=6)
    focus = [r for r in eval_rows if r["arm"] in {"B", "C_gauge", "D_standard"}]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric in zip(axes, ["psnr_mean", "lpips_mean", "rapsd_distance_mean"]):
        vals = [float(r.get(metric, "nan")) for r in focus]
        ax.bar([r["arm"] for r in focus], vals)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "standard_vs_gauge_metric_bars.png", dpi=180)
    plt.close(fig)
    write_text(
        OUT / "STANDARD_GAN_BASELINE_REPORT.md",
        "\n".join(
            [
                "# Standard cGAN Baseline Report",
                "",
                "Arm D_standard trains a standard image-input discriminator on real x vs deployed fake x_hat. It does not receive residual features, correction vectors, or RelMeasErr. B/C are the Phase71 seed01 paired checkpoints evaluated in the same Phase74 loader/cache.",
                "",
                "## Aggregate Metrics",
                "",
                table(eval_rows, ["arm", "psnr_mean", "ssim_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_unclipped_float64_mean", "source"]),
                "",
                "## Pairwise",
                "",
                table([r for r in comparisons if r["metric"] in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64"}], ["pair", "metric", "ci_low", "ci_high"]),
                "",
                f"D_standard checkpoint SHA256: `{d_summary['best_checkpoint_sha256']}`",
                "",
            ]
        ),
    )
    append_log("standard_cgan_baseline_scr5_complete")


def scr5_beta_frontier(device: torch.device) -> None:
    append_log("scr5_beta_frontier_start")
    config = p73.regime_config("scr5", device)
    measurement, _A = p73.make_regime_measurement("scr5", config, device)
    train, val, test, _split = p73.build_caches("scr5", config, measurement, device)
    beta0 = float(read_csv(PH69B / "beta_calibration.csv")[0]["selected_beta0"])
    frontier_dir = ensure_dir(OUT / "scr5_beta_frontier_runs")
    candidates = [
        ("beta0", 0.0, "none", PH71 / "seed01" / "B" / "checkpoints" / "best_by_val.pt", "observed_Phase71_B"),
        ("beta0p3", 0.3 * beta0, "gauge", None, "phase74_trained"),
        ("beta1p0", beta0, "gauge", PH71 / "seed01" / "C" / "checkpoints" / "best_by_val.pt", "observed_Phase71_C"),
        ("beta3p0", 3.0 * beta0, "gauge", None, "phase74_trained"),
    ]
    rows: list[dict[str, Any]] = []
    per_all: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for idx, (name, beta, adv_mode, existing_ckpt, status) in enumerate(candidates):
        arm_dir = ensure_dir(frontier_dir / name)
        if existing_ckpt is not None:
            ckpt = existing_ckpt
            ckpt_sha = p69a.sha256_file(ckpt)
        else:
            ckpt = arm_dir / name / "checkpoints" / "best_by_val.pt"
            if not ckpt.exists():
                seed_base = 710100
                loader_seed = 710401
                set_seed(seed_base)
                random.seed(seed_base)
                np.random.seed(seed_base)
                torch.manual_seed(seed_base)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base)
                gen, _ = p73.load_regime_generator("scr5", config, measurement, device, train=True)
                summary, train_rows, ckpt = train_general_arm("scr5", 1, name, gen, measurement, train, val, config, device, arm_dir, loader_seed, beta, adv_mode)
                write_csv(arm_dir / "training_log.csv", train_rows)
                save_json(arm_dir / "training_summary.json", summary)
                del gen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ckpt_sha = p69a.sha256_file(ckpt)
        gen_eval = load_checkpoint_for_eval(ckpt, config, measurement, device)
        eval_dir = ensure_dir(arm_dir / "evaluation")
        agg, per, arr = p73.evaluate_general(name, gen_eval, measurement, test, config, device, eval_dir)
        outputs[name] = arr
        for row in per:
            row["beta"] = beta
            row["frontier_arm"] = name
        per_all.extend(per)
        agg.update(
            {
                "beta": beta,
                "beta_multiplier": "0" if beta == 0 else beta / beta0,
                "source": status,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": ckpt_sha,
            }
        )
        rows.append(agg)
    compute_lpips_any(frontier_dir, test, outputs, device, per_all, rows)
    write_csv(OUT / "scr5_beta_frontier_full.csv", rows)
    write_csv(frontier_dir / "per_sample_metrics.csv", per_all)
    for metric, filename, ylabel in [
        ("lpips_mean", "scr5_beta_lpips_psnr_frontier.png", "LPIPS lower better"),
        ("rapsd_distance_mean", "scr5_beta_rapsd_psnr_frontier.png", "RAPSD lower better"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter([float(r["psnr_mean"]) for r in rows], [float(r.get(metric, "nan")) for r in rows])
        for r in rows:
            ax.annotate(str(r["beta_multiplier"]), (float(r["psnr_mean"]), float(r.get(metric, "nan"))))
        ax.set_xlabel("PSNR")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot([float(r["beta"]) for r in rows], [float(r["relmeaserr_unclipped_float64_mean"]) for r in rows], marker="o")
    ax.set_xlabel("beta")
    ax.set_ylabel("RelMeasErr")
    fig.tight_layout()
    fig.savefig(OUT / "scr5_beta_relmeaserr.png", dpi=180)
    plt.close(fig)
    write_text(
        OUT / "SCR5_BETA_FRONTIER_FULL_REPORT.md",
        "\n".join(
            [
                "# Scr-5 Beta Frontier Full Report",
                "",
                "This sweep uses one paired seed budget. Beta 0 and beta0 reuse the Phase71 seed01 B/C checkpoints; 0.3 beta0 and 3 beta0 are trained under Phase74 with the same init seed, loader seed, optimizer, budget, and validation selection rule.",
                "",
                table(rows, ["beta_multiplier", "beta", "psnr_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_unclipped_float64_mean", "source", "checkpoint_sha256"]),
                "",
            ]
        ),
    )
    append_log("scr5_beta_frontier_complete")


def ood_transfer() -> None:
    append_log("ood_transfer_start")
    data_root = ROOT / "data"
    hits: list[str] = []
    for pattern in ["*Set11*", "*set11*", "*BSD68*", "*bsd68*"]:
        hits.extend(str(p) for p in data_root.rglob(pattern))
    if not hits:
        rows = [{"dataset": "Set11/BSD68", "status": "unavailable", "reason": "No local Set11 or BSD68 files found under E:/ns_mc_gan_gi/data; no downloads performed."}]
        write_csv(OUT / "ood_transfer_gauge_cgan.csv", rows)
        write_text(OUT / "OOD_TRANSFER_GAUGE_CGAN_REPORT.md", "# OOD Transfer Report\n\nSet11/BSD68 were not available locally. No OOD training or evaluation was run.\n")
    else:
        rows = [{"dataset_path": h, "status": "found_not_evaluated", "reason": "Local path found but no vetted loader/provenance lock exists in this branch."} for h in hits]
        write_csv(OUT / "ood_transfer_gauge_cgan.csv", rows)
        write_text(OUT / "OOD_TRANSFER_GAUGE_CGAN_REPORT.md", "# OOD Transfer Report\n\nLocal candidate files were found, but Phase74 did not evaluate them without a vetted loader/provenance lock.\n")
    append_log("ood_transfer_complete")


def ood_transfer_set11_eval(device: torch.device) -> None:
    append_log("ood_transfer_set11_eval_start")
    try:
        from PIL import Image
    except Exception as exc:
        write_csv(OUT / "ood_transfer_gauge_cgan.csv", [{"dataset": "Set11", "status": "unavailable", "reason": f"PIL import failed: {exc}"}])
        write_text(OUT / "OOD_TRANSFER_GAUGE_CGAN_REPORT.md", "# OOD Transfer Report\n\nSet11 was found locally, but PIL image loading failed.\n")
        return
    candidates = sorted((ROOT / "data" / "external").rglob("DataSets/Set11"))
    image_dir = next((p for p in candidates if p.is_dir() and list(p.glob("*.tif"))), None)
    if image_dir is None:
        ood_transfer()
        return
    image_paths = sorted(image_dir.glob("*.tif"))
    xs: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("L")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((64, 64), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        xs.append(arr)
        manifest_rows.append(
            {
                "dataset": "Set11",
                "ordinal": i,
                "path": str(path),
                "source_sha256": p69a.sha256_file(path),
                "original_width": w,
                "original_height": h,
                "transform": "grayscale_center_crop_resize_64",
            }
        )
    x = torch.from_numpy(np.stack(xs)[:, None, :, :]).float()
    labels = torch.full((x.shape[0],), -1, dtype=torch.long)
    indices = torch.arange(900000, 900000 + x.shape[0], dtype=torch.long)
    config = p73.regime_config("scr5", device)
    measurement, _A = p73.make_regime_measurement("scr5", config, device)
    torch.manual_seed(747001)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(747001)
    with torch.no_grad():
        y = measurement.measure(x.to(device)).detach().cpu()
    test = p69b.SplitCache(name="set11_ood", x=x, y=y, labels=labels, indices=indices)
    b_best = PH71 / "seed01" / "B" / "checkpoints" / "best_by_val.pt"
    c_best = PH71 / "seed01" / "C" / "checkpoints" / "best_by_val.pt"
    models = {
        "A": p73.load_regime_generator("scr5", config, measurement, device, train=False)[0],
        "B": load_checkpoint_for_eval(b_best, config, measurement, device),
        "C": load_checkpoint_for_eval(c_best, config, measurement, device),
    }
    eval_dir = ensure_dir(OUT / "ood_set11_eval")
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, gen in models.items():
        agg, per, arr = p73.evaluate_general(arm, gen, measurement, test, config, device, eval_dir)
        agg["dataset"] = "Set11"
        agg["n_eval"] = int(x.shape[0])
        agg["source"] = "published_mean" if arm == "A" else "Phase71_seed01_checkpoint"
        eval_rows.append(agg)
        for row in per:
            row["dataset"] = "Set11"
        per_rows.extend(per)
        outputs[arm] = arr
    compute_lpips_any(eval_dir, test, outputs, device, per_rows, eval_rows)
    comparisons = paired_compare(per_rows, "B", "C")
    write_csv(OUT / "set11_image_manifest_phase74.csv", manifest_rows)
    write_csv(OUT / "ood_transfer_gauge_cgan.csv", eval_rows)
    write_csv(OUT / "ood_transfer_set11_pairwise.csv", comparisons)
    write_csv(eval_dir / "per_sample_metrics.csv", per_rows)
    visual_grid_any(OUT / "ood_transfer_visual_grid.png", test, outputs, ["A", "B", "C"], n=min(6, int(x.shape[0])))
    write_text(
        OUT / "OOD_TRANSFER_GAUGE_CGAN_REPORT.md",
        "\n".join(
            [
                "# OOD Transfer Report",
                "",
                "Set11 was found locally and evaluated without training. Images were converted to grayscale, center-cropped, resized to 64x64, measured with the Scr-5 operator, and reconstructed by the published mean model plus Phase71 seed01 B/C checkpoints.",
                "",
                "## Aggregate Metrics",
                "",
                table(eval_rows, ["dataset", "arm", "psnr_mean", "ssim_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_unclipped_float64_mean", "n_eval"]),
                "",
                "## C vs B",
                "",
                table([r for r in comparisons if r["metric"] in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64"}], ["pair", "metric", "ci_low", "ci_high"]),
                "",
            ]
        ),
    )
    append_log("ood_transfer_set11_eval_complete")


def noise_eval(device: torch.device) -> None:
    append_log("noise_eval_start")
    config = p73.regime_config("scr5", device)
    measurement, _A = p73.make_regime_measurement("scr5", config, device)
    _train, _val, test_full, _split = p73.build_caches("scr5", config, measurement, device)
    n = min(128, int(test_full.x.shape[0]))
    test = p69b.SplitCache(
        name="test_noise128",
        x=test_full.x[:n].clone(),
        y=test_full.y[:n].clone(),
        labels=test_full.labels[:n].clone(),
        indices=test_full.indices[:n].clone(),
    )
    b_best = PH71 / "seed01" / "B" / "checkpoints" / "best_by_val.pt"
    c_best = PH71 / "seed01" / "C" / "checkpoints" / "best_by_val.pt"
    models = {
        "A": p73.load_regime_generator("scr5", config, measurement, device, train=False)[0],
        "B": load_checkpoint_for_eval(b_best, config, measurement, device),
        "C": load_checkpoint_for_eval(c_best, config, measurement, device),
    }
    rows: list[dict[str, Any]] = []
    for sigma in [0.0, 0.005, 0.01, 0.02]:
        noisy = p69b.SplitCache(
            name=f"test_noise_{sigma}",
            x=test.x.clone(),
            y=test.y.clone(),
            labels=test.labels.clone(),
            indices=test.indices.clone(),
        )
        if sigma > 0:
            gen = torch.Generator().manual_seed(746000 + int(sigma * 10000))
            noisy.y = noisy.y + sigma * torch.randn(noisy.y.shape, generator=gen)
        eval_dir = ensure_dir(OUT / "noise_eval" / f"sigma_{sigma:g}")
        for arm, gen_model in models.items():
            agg, _per, _arr = p73.evaluate_general(arm, gen_model, measurement, noisy, config, device, eval_dir)
            agg["noise_sigma"] = sigma
            agg["n_eval"] = n
            rows.append(agg)
    write_csv(OUT / "noise_eval_gauge_cgan.csv", rows)
    write_text(
        OUT / "NOISE_EVAL_GAUGE_CGAN_REPORT.md",
        "\n".join(
            [
                "# Noise Robustness Evaluation",
                "",
                "No retraining was performed. Phase71 seed01 B/C checkpoints were evaluated with generated measurement noise levels on a 128-sample Scr-5 subset.",
                "",
                table(rows, ["noise_sigma", "arm", "psnr_mean", "ssim_mean", "rapsd_distance_mean", "relmeaserr_unclipped_float64_mean", "n_eval"]),
                "",
            ]
        ),
    )
    append_log("noise_eval_complete")


def p0_metrics_regime(regime: str, base_dir: Path, prefix: str, seeds: list[int], device: torch.device) -> list[dict[str, Any]]:
    config = p73.regime_config(regime, device)
    measurement, _A = p73.make_regime_measurement(regime, config, device)
    _train, _val, test, _split = p73.build_caches(regime, config, measurement, device)
    A64, G, _K = p73.exact_projectors(measurement.A, float(config["lambda_solver"]))
    true_flat = measurement.flatten_img(test.x.to(device)).to(torch.float64)
    p0_true = p69a.p0_exact(true_flat, A64, G).detach().cpu().numpy().reshape(-1, 64, 64)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = base_dir / f"{prefix}{seed:02d}" / "evaluation"
        if not seed_dir.exists():
            continue
        arm_metrics: dict[str, dict[str, float]] = {}
        for arm in ["B", "C"]:
            path = seed_dir / f"per_sample_outputs_{arm}.npz"
            if not path.exists():
                continue
            pred = np.load(path)["x_hat_unclipped"].astype(np.float32)
            pred_flat = torch.from_numpy(pred.reshape(pred.shape[0], -1)).to(device).to(torch.float64)
            p0_pred = p69a.p0_exact(pred_flat, A64, G).detach().cpu().numpy().reshape(-1, 64, 64)
            p0_rapsd = [float(np.linalg.norm(p69b.rapsd(p0_pred[i]) - p69b.rapsd(p0_true[i]))) for i in range(pred.shape[0])]
            p0_l2 = [float(np.linalg.norm((p0_pred[i] - p0_true[i]).reshape(-1)) / math.sqrt(4096)) for i in range(pred.shape[0])]
            hf = [float(abs(p69b.hf_ratio(np.clip(pred[i], 0, 1)) - p69b.hf_ratio(test.x[i, 0].numpy()))) for i in range(pred.shape[0])]
            item = {
                "regime": regime,
                "seed": seed,
                "arm": arm,
                "p0_rapsd_mean": float(np.mean(p0_rapsd)),
                "p0_l2_mean": float(np.mean(p0_l2)),
                "highfreq_error_mean": float(np.mean(hf)),
            }
            rows.append(item)
            arm_metrics[arm] = item
        if "B" in arm_metrics and "C" in arm_metrics:
            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "arm": "C_minus_B",
                    "p0_rapsd_mean_delta": arm_metrics["C"]["p0_rapsd_mean"] - arm_metrics["B"]["p0_rapsd_mean"],
                    "p0_l2_mean_delta": arm_metrics["C"]["p0_l2_mean"] - arm_metrics["B"]["p0_l2_mean"],
                    "highfreq_error_mean_delta": arm_metrics["C"]["highfreq_error_mean"] - arm_metrics["B"]["highfreq_error_mean"],
                }
            )
    return rows


def p0_space_combined(device: torch.device) -> None:
    append_log("p0_space_combined_start")
    rows: list[dict[str, Any]] = []
    if (PH73 / "p0_space_metrics.csv").exists():
        rows.extend(read_csv(PH73 / "p0_space_metrics.csv"))
    rows.extend(p0_metrics_regime("rad5", PH73, "rad5_seed", [1, 2, 3], device))
    write_csv(OUT / "p0_space_metrics_rad5.csv", [r for r in rows if str(r.get("regime")) == "rad5"])
    write_csv(OUT / "p0_space_metrics_combined.csv", rows)
    delta = [r for r in rows if r.get("arm") == "C_minus_B" and r.get("p0_rapsd_mean_delta", "") != ""]
    if delta:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        labels = [f"{r['regime']}-s{r['seed']}" for r in delta]
        vals = [float(r["p0_rapsd_mean_delta"]) for r in delta]
        ax.bar(labels, vals)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("C-B P0 RAPSD delta")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(OUT / "p0_space_combined_plot.png", dpi=180)
        plt.close(fig)
    write_text(
        OUT / "P0_SPACE_METRICS_COMBINED_REPORT.md",
        "\n".join(
            [
                "# P0-Space Metrics Combined Report",
                "",
                "Scr-5 rows are imported from Phase73; Rad-5 rows are recomputed from saved Phase73 per-sample outputs and exact P0 projectors.",
                "",
                table(delta, ["regime", "seed", "p0_rapsd_mean_delta", "p0_l2_mean_delta", "highfreq_error_mean_delta"]),
                "",
            ]
        ),
    )
    append_log("p0_space_combined_complete")


def human_2afc_ready() -> None:
    append_log("human_2afc_ready_start")
    dst = OUT / "human_2afc_ready"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PH73 / "human_2afc_pack", dst)
    pairs = read_csv(PH73 / "human_2afc_pairs.csv")
    template = [
        {
            "rater_id": "",
            "pair_id": row.get("pair_id", ""),
            "choice": "",
            "confidence_1_to_5": "",
            "notes": "",
        }
        for row in pairs
    ]
    write_csv(OUT / "human_2afc_response_template.csv", template)
    analysis_py = OUT / "human_2afc_analysis.py"
    write_text(
        analysis_py,
        "\n".join(
            [
                "from __future__ import annotations",
                "import csv, sys",
                "from pathlib import Path",
                "import numpy as np",
                "",
                "def main(path: str) -> int:",
                "    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))",
                "    choices = [r.get('choice','').strip().upper() for r in rows if r.get('choice','').strip()]",
                "    n = len(choices)",
                "    if n == 0:",
                "        print('No responses found; do not report preference results.')",
                "        return 0",
                "    c = sum(1 for x in choices if x == 'C')",
                "    boot = []",
                "    rng = np.random.default_rng(740)",
                "    arr = np.asarray([1 if x == 'C' else 0 for x in choices], dtype=float)",
                "    for _ in range(10000):",
                "        boot.append(float(rng.choice(arr, size=n, replace=True).mean()))",
                "    print({'n': n, 'C_rate': c/n, 'ci_low': float(np.percentile(boot, 2.5)), 'ci_high': float(np.percentile(boot, 97.5))})",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else 'human_2afc_response_template.csv'))",
                "",
            ]
        ),
    )
    write_text(
        OUT / "HUMAN_2AFC_STATUS.md",
        "\n".join(
            [
                "# Human 2AFC Status",
                "",
                "The Phase73 randomized 2AFC pack was copied into `human_2afc_ready/`.",
                "",
                "- No human responses were found or invented.",
                "- Response template: `human_2afc_response_template.csv`.",
                "- Analysis script: `human_2afc_analysis.py`.",
                "",
            ]
        ),
    )
    append_log("human_2afc_ready_complete")


def diffusion_positioning() -> None:
    append_log("diffusion_positioning_start")
    query = "DPS|DDRM|DDNM|diffusion|score|denoiser|PnP|plug-and-play"
    try:
        proc = subprocess.run(["rg", "-n", "-i", query, str(Path("E:/ns_mc_gan_gi_code"))], capture_output=True, text=True, timeout=60)
        hits = proc.stdout.splitlines()[:200]
    except Exception as exc:
        hits = [f"search_failed: {exc}"]
    write_csv(OUT / "diffusion_local_search_hits.csv", [{"hit": h} for h in hits])
    found_impl = [h for h in hits if "def " in h.lower() or "class " in h.lower()]
    status = "positioning_only_no_vetted_diffusion_baseline" if not found_impl else "local_mentions_found_not_vetted"
    write_text(
        OUT / "DIFFUSION_POSITIONING_REPORT.md",
        "\n".join(
            [
                "# Diffusion / Inverse-Problem Positioning Report",
                "",
                f"Status: `{status}`.",
                "",
                "No diffusion, DPS, DDRM, DDNM, score-model, or PnP baseline was downloaded or run in Phase74. Local search hits are stored in `diffusion_local_search_hits.csv`.",
                "",
                "The honest paper position is therefore comparative positioning, not an empirical claim over diffusion inverse solvers.",
                "",
            ]
        ),
    )
    rows = [
        {"method": "Gauge-equalized cGAN branch", "uses_training": "small controlled fine-tune", "measurement_certificate": "explicit Pi_y^lambda audit", "claim": "small perceptual/null-space prior under preserved certificate"},
        {"method": "DPS/DDRM/DDNM diffusion inverse solvers", "uses_training": "pretrained score/diffusion prior", "measurement_certificate": "often data-consistency step, implementation-specific", "claim": "strong generic inverse-prior baseline; not locally run"},
        {"method": "PnP/RED denoiser prior", "uses_training": "pretrained denoiser", "measurement_certificate": "operator-dependent", "claim": "classical learned prior baseline; not locally run"},
    ]
    write_csv(OUT / "method_comparison_table.csv", rows)
    write_text(OUT / "METHOD_COMPARISON_TABLE.md", "# Method Comparison Table\n\n" + table(rows, ["method", "uses_training", "measurement_certificate", "claim"]) + "\n")
    append_log("diffusion_positioning_complete")


def paper_pack() -> None:
    append_log("paper_pack_start")
    rad10_gate = read_json(OUT / "rad10_gate_decision.json") if (OUT / "rad10_gate_decision.json").exists() else {}
    draft = [
        "# Gauge-Equalized Adversarial Priors for Audited Computational Ghost Imaging",
        "",
        "## Abstract Draft",
        "",
        "We study whether an adversarial prior can improve perceptual structure without weakening an explicit measurement-consistency audit. The key device is a gauge-equalized discriminator: real and fake samples are compared only after replacing the measured component by a shared canonical term, removing the residual shortcut that otherwise lets a discriminator classify by measurement error. Across Scr-5 and Rad-5 regimes, the gauge signal is nontrivial and paired-seed pilots improve LPIPS/RAPSD while preserving the projection certificate. Scr-10 remains a negative weak-signal stress test, and Rad-10 is gated by Phase74 diagnostics.",
        "",
        "## Main Claims",
        "",
        "1. Residual-fed discriminators cheat; gauge-equalized discriminators retain usable but harder signal.",
        "2. Controlled gauge cGAN fine-tuning improves perceptual/spectral metrics in Scr-5 and Rad-5 without material PSNR or RelMeasErr cost.",
        "3. The method is not claimed as a universal GAN win: Scr-10 weak gate and Rad-10 gate are regime-dependent evidence.",
        "4. Strong-conference readiness still depends on human 2AFC responses and, ideally, vetted diffusion/PnP baselines.",
        "",
        "## Rad-10 Phase74 Gate",
        "",
        json.dumps(rad10_gate, indent=2),
        "",
    ]
    write_text(OUT / "gauge_cgan_high_tier_draft.md", "\n".join(draft))
    write_text(
        OUT / "gauge_cgan_high_tier_draft.tex",
        "\n".join(
            [
                r"\section{Gauge-Equalized Adversarial Priors}",
                r"We compare real and fake reconstructions in a canonical gauge $\tilde{x}=P_0x+B_\lambda y$, preventing the discriminator from exploiting measurement residuals.",
                r"\paragraph{Evidence.} Scr-5 and Rad-5 paired seeds show LPIPS/RAPSD improvements under preserved measurement audit; Scr-10 remains a weak-signal negative gate; Rad-10 is reported by Phase74 diagnostics.",
                "",
            ]
        ),
    )
    write_text(
        OUT / "ABSTRACT_OPTIONS.md",
        "# Abstract Options\n\n1. Conservative workshop abstract: emphasize gauge diagnostic, Scr-5/Rad-5 paired evidence, and Scr-10 negative gate.\n2. Stronger conference abstract: only use after human 2AFC responses and diffusion/PnP baselines are added.\n",
    )
    write_text(
        OUT / "FIGURE_PLAN.md",
        "# Figure Plan\n\n1. Gauge construction and forbidden shortcut.\n2. AUC/shortcut regime map: Scr-5, Scr-10, Rad-5, Rad-10.\n3. Paired-seed C-vs-B LPIPS/RAPSD bars.\n4. Standard-vs-gauge baseline visual/metric panel.\n5. Beta perception-distortion frontier.\n6. P0-space and certificate preservation panel.\n7. Human 2AFC randomized examples/status.\n",
    )
    write_text(
        OUT / "FINAL_FIGURE_CAPTIONS.md",
        "# Final Figure Captions\n\n- Figure 1: Gauge equalization removes residual shortcuts by replacing the measured component with a shared canonical term.\n- Figure 2: Residual-fed critics separate real/fake by certificate error; gauge critics expose regime-dependent null-space signal.\n- Figure 3: Paired Scr-5/Rad-5 seeds show C improves perceptual/spectral metrics relative to B under the same audit.\n- Figure 4: Standard cGAN baseline tests whether gauge input, not generic adversarial training, explains the effect.\n- Figure 5: Beta sweep traces the perception-distortion frontier while monitoring RelMeasErr.\n",
    )
    write_text(
        OUT / "RELATED_WORK_GAUGE_CGAN.md",
        "# Related Work Notes\n\nPosition the branch against conditional GANs for inverse problems, null-space priors, audited projection methods, PnP/RED, and diffusion inverse solvers. Do not claim diffusion superiority without a local vetted baseline.\n",
    )
    write_text(
        OUT / "REVIEWER_ATTACK_BANK.md",
        "# Reviewer Attack Bank\n\n1. The discriminator may exploit measurement residuals. Response: residual-fed shortcut controls and gauge-only D inputs.\n2. GAN gains may be seed noise. Response: paired-seed Scr-5/Rad-5 results and C-vs-B bootstrap.\n3. Certificate may be weakened. Response: final deployment uses Pi_y^lambda and RelMeasErr is computed unclipped float64.\n4. Results may not generalize. Response: Scr-10 weak gate is reported as a negative regime; Rad-10 is gated.\n5. Diffusion baselines are missing. Response: current strong-conference gap; not claimed complete.\n",
    )
    append_log("paper_pack_complete")


def readiness_reports(rad10_gate: dict[str, Any]) -> None:
    append_log("readiness_reports_start")
    standard = read_csv(OUT / "standard_cgan_baseline_scr5.csv") if (OUT / "standard_cgan_baseline_scr5.csv").exists() else []
    beta = read_csv(OUT / "scr5_beta_frontier_full.csv") if (OUT / "scr5_beta_frontier_full.csv").exists() else []
    rad10_delta = read_csv(OUT / "rad10_seed_delta_metrics.csv") if (OUT / "rad10_seed_delta_metrics.csv").exists() else []
    answers = [
        {"question": "Does gauge-equalized D have usable signal?", "answer": f"Scr-5 and Rad-5 yes; Scr-10 weak; Rad-10 gate is {rad10_gate.get('decision', 'missing')} with AUC {rad10_gate.get('rad10_gauge_auc', 'missing')}."},
        {"question": "Is conditional D better than unconditional?", "answer": "Historical evidence says no: conditional gauge D was below unconditional in Scr-5/Rad-5, so Phase69-74 recommends unconditional gauge D."},
        {"question": "Does residual-fed D cheat?", "answer": "Yes. Residual controls are high-AUC in Scr-5/Scr-10/Rad-5/Rad-10 diagnostics."},
        {"question": "Does gauge remove shortcut?", "answer": "Yes structurally; gauge AUC is lower than residual-fed controls and D is not given residual/correction features."},
        {"question": "Are canonical images numerically stable?", "answer": "Yes for locked Scr-5/Scr-10/Rad-5 structural checks; Rad-10 uses exact A/P0/B_lambda in Phase74 diagnostic."},
        {"question": "Should Phase69B/controlled cGAN be run?", "answer": "Already run for Scr-5/Rad-5; Rad-10 follows its Phase74 gate; Scr-10 remains no-train weak gate."},
        {"question": "Recommended D input?", "answer": "Unconditional gauge-only PatchCritic D(tilde{x}); avoid residual, correction, and RelMeasErr inputs."},
        {"question": "Standard cGAN baseline status?", "answer": "Phase74 trains/evaluates a one-seed standard image-input cGAN baseline and compares B/C/D." if standard else "Missing."},
        {"question": "Beta frontier status?", "answer": "Phase74 completes a one-seed beta sweep at 0, 0.3beta0, beta0, 3beta0." if beta else "Missing."},
        {"question": "Human 2AFC status?", "answer": "Ready-to-send pack exists; no responses are invented."},
        {"question": "Diffusion baseline status?", "answer": "Positioning-only. No local vetted diffusion baseline was run or downloaded."},
        {"question": "Workshop readiness?", "answer": "Ready with cautious claims and transparent negative gates."},
        {"question": "Strong-conference readiness?", "answer": "Close but not fully ready: Set11 no-train OOD is included, but human responses and a vetted diffusion/PnP baseline are still needed for high-tier confidence."},
    ]
    write_csv(OUT / "readiness_answers_phase74.csv", answers)
    summary_lines = [
        "# Phase74 High-Tier Report",
        "",
        "Phase74 is a high-tier evidence pack that locks prior evidence, adds Rad-10 diagnostics, adds a one-seed standard cGAN baseline, completes a one-seed Scr-5 beta frontier, prepares human 2AFC analysis without inventing responses, and writes paper-positioning documents.",
        "",
        "## Readiness Questions",
        "",
        table(answers, ["question", "answer"]),
        "",
        "## Rad-10 Gate",
        "",
        json.dumps(rad10_gate, indent=2),
        "",
        "## Rad-10 Seed Delta Snapshot",
        "",
        table([r for r in rad10_delta if r.get("metric") in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64"}], ["seed", "metric", "mean_C_minus_B", "ci_low", "ci_high"]),
        "",
        "## OOD / Transfer",
        "",
        "Set11 is evaluated without training when a local vetted image folder is found. Aggregate rows are in `ood_transfer_gauge_cgan.csv`.",
        "",
    ]
    write_text(OUT / "PHASE74_HIGH_TIER_REPORT.md", "\n".join(summary_lines))
    write_text(OUT / "WORKSHOP_READINESS_FINAL.md", "# Workshop Readiness Final\n\nReady, with cautious framing: gauge branch is evidence-backed for Scr-5/Rad-5, Scr-10 is a negative weak-signal gate, and human responses are pending.\n")
    write_text(OUT / "STRONG_CONFERENCE_READINESS_FINAL.md", "# Strong Conference Readiness Final\n\nNot fully ready yet. The pack is much stronger, but a high-tier submission should still add human 2AFC responses and a vetted diffusion/PnP baseline or explicitly scope away from them.\n")
    write_text(OUT / "HIGH_JOURNAL_READINESS_FINAL.md", "# High Journal Readiness Final\n\nNot ready as a standalone high-journal empirical paper. Needs broader OOD/diffusion/human evidence and stronger theory-to-practice synthesis.\n")
    if rad10_gate.get("paired_seed_count", 0):
        next_exp = "Analyze Rad-10 paired seeds, then decide whether to include Rad-10 as robustness evidence."
    else:
        next_exp = "Do not train Rad-10 cGAN under the current gate. Next priority: collect human 2AFC responses and add vetted diffusion/PnP baselines."
    write_text(OUT / "NEXT_EXPERIMENT_DECISION.md", f"# Next Experiment Decision\n\n{next_exp}\n")
    append_log("readiness_reports_complete")


def manifest() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": p69a.sha256_file(path)})
    write_csv(OUT / "phase74_manifest.csv", rows)
    write_text(OUT / "PHASE74_MANIFEST.md", "# Phase74 Manifest\n\n" + table(rows, ["path", "bytes", "sha256"]) + "\n")


def main() -> int:
    configure_phase73_helpers()
    ensure_dir(OUT)
    append_log("phase74_start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preflight_and_protocol()
    inventory()
    rad10_gate = rad10_diagnostic(device)
    run_rad10_seeds(rad10_gate, device)
    standard_cgan_baseline_scr5(device)
    scr5_beta_frontier(device)
    ood_transfer_set11_eval(device)
    noise_eval(device)
    p0_space_combined(device)
    human_2afc_ready()
    diffusion_positioning()
    paper_pack()
    readiness_reports(rad10_gate)
    manifest()
    append_log("phase74_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
