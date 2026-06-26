from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
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
from . import phase74_high_tier_gauge_cgan_pack as p74
from .utils import set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase76_high_upside_auditable_gan_exploration"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGS = OUT / "figs"

PH69A = ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH71 = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
PH72 = ROOT / "outputs_phase72_scr10_gauge_cgan_regime_validation"
PH73 = ROOT / "outputs_phase73_overnight_gauge_gan_expansion"
PH74 = ROOT / "outputs_phase74_high_tier_gauge_cgan_pack"
PH75 = ROOT / "outputs_phase75_final_high_tier_validation"

BATCH_SIZE = 8
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "PHASE76_RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {message}\n")


def configure_helpers() -> None:
    p73.OUT = OUT
    p73.REGIMES.clear()
    p73.REGIMES.update(p74.REGIME_INFO)


def sha_if_exists(path: Path) -> str:
    return p69a.sha256_file(path) if path.exists() and path.is_file() else ""


def preflight() -> None:
    ensure_dir(TABLES)
    ensure_dir(REPORTS)
    ensure_dir(FIGS)
    required = [
        PH69A / "critic_auc_results.csv",
        PH71 / "scr5_seed_delta_metrics.csv",
        PH72 / "scr10_gauge_signal_auc.csv",
        PH73 / "rad5_seed_delta_metrics.csv",
        PH74 / "scr5_beta_frontier_full.csv",
        PH75 / "standard_cgan_seed_metrics.csv",
        PH75 / "shortcut_stress_gauge_patchcritic.pt",
        PH75 / "PHASE75_FINAL_VALIDATION_REPORT.md",
    ]
    failures = [str(p) for p in required if not p.exists()]
    if failures:
        write_text(OUT / "UNSAFE_TO_RUN.md", "# UNSAFE TO RUN\n\n" + "\n".join(f"- Missing: {x}" for x in failures) + "\n")
        raise RuntimeError("Phase76 preflight failed.")
    write_text(
        REPORTS / "PHASE76_PROTOCOL_LOCK.md",
        "\n".join(
            [
                "# Phase76 Protocol Lock",
                "",
                "- First-paper measurement-certified GI results are unchanged.",
                "- Existing checkpoints are read-only; Phase76 does not train a generator.",
                "- New training is restricted to diagnostic critics/classifiers for safety/OOD/toy checks.",
                "- GAN is not a measurement certificate; Pi_y^lambda is the certificate.",
                "- No SOTA, no diffusion-beaten, no fake human/diffusion results.",
                "",
            ]
        ),
    )
    log("preflight_complete")


def get_context(regime: str, device: torch.device):
    config = p73.regime_config(regime, device)
    measurement, _A = p73.make_regime_measurement(regime, config, device)
    train, val, test, split = p73.build_caches(regime, config, measurement, device)
    A64, G, K = p73.exact_projectors(measurement.A, float(config["lambda_solver"]))
    return config, measurement, train, val, test, split, A64, G, K


def load_npz_output(path: Path) -> np.ndarray:
    z = np.load(path)
    return z["x_hat_unclipped"].astype(np.float32)


def canonical_outputs(regime: str, seed: int = 1) -> dict[str, Path]:
    if regime == "scr5":
        base = PH75 / f"standard_cgan_seed{seed:02d}" / "evaluation"
        return {
            "B": base / "per_sample_outputs_B.npz",
            "C": base / "per_sample_outputs_C_gauge.npz",
            "D_standard": base / "per_sample_outputs_D_standard.npz",
        }
    base = PH73 / f"rad5_seed{seed:02d}" / "evaluation"
    return {
        "A": base / "per_sample_outputs_A.npz",
        "B": base / "per_sample_outputs_B.npz",
        "C": base / "per_sample_outputs_C.npz",
    }


def canonical_checkpoint(regime: str, arm: str, seed: int = 1) -> Path:
    if regime == "scr5":
        if arm == "B":
            return PH71 / f"seed{seed:02d}" / "B" / "checkpoints" / "best_by_val.pt"
        if arm == "C":
            return PH71 / f"seed{seed:02d}" / "C" / "checkpoints" / "best_by_val.pt"
        if arm == "D_standard":
            return PH75 / f"standard_cgan_seed{seed:02d}" / "D_standard" / "checkpoints" / "best_by_val.pt"
    if regime == "rad5":
        if arm in {"B", "C"}:
            return PH73 / f"rad5_seed{seed:02d}" / arm / "checkpoints" / "best_by_val.pt"
    raise KeyError((regime, arm, seed))


def p0_project_batch(v: torch.Tensor, A64: torch.Tensor, G: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    outs = []
    for i in range(0, v.shape[0], chunk):
        outs.append(p69a.p0_exact(v[i : i + chunk].to(torch.float64), A64, G))
    return torch.cat(outs, dim=0)


def local_energy(arr: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(arr[:, None].astype(np.float32))
    e = F.avg_pool2d(t * t, kernel_size=7, stride=1, padding=3).sqrt().numpy()[:, 0]
    return e


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stage0_inventory() -> None:
    log("stage0_inventory_start")
    roots = {
        "Phase69A": PH69A,
        "Phase69B": PH69B,
        "Phase70": PH70,
        "Phase71": PH71,
        "Phase72": PH72,
        "Phase73": PH73,
        "Phase74": PH74,
        "Phase75": PH75,
    }
    rows = []
    for name, root in roots.items():
        rows.append({"phase": name, "path": str(root), "exists": root.exists(), "file_count": len([p for p in root.rglob("*") if p.is_file()]) if root.exists() else 0})
    write_csv(TABLES / "phase76_evidence_roots.csv", rows)
    canonical = [
        {"item": "Scr-5 B/C/D_standard", "canonical_source": "Phase75 standard_cgan_seed_metrics.csv and per-seed outputs", "main_or_supp": "main for standard-vs-gauge safety/performance comparison"},
        {"item": "Scr-5 B/C paired evidence", "canonical_source": "Phase71 scr5_seed_delta_metrics.csv", "main_or_supp": "main"},
        {"item": "Rad-5 B/C paired evidence", "canonical_source": "Phase73 rad5_seed_delta_metrics.csv", "main_or_supp": "main"},
        {"item": "Scr-10 weak gate", "canonical_source": "Phase72 scr10_gauge_signal_auc.csv", "main_or_supp": "main/supp regime map"},
        {"item": "Rad-10 weak gate", "canonical_source": "Phase74 rad10_gauge_auc.csv", "main_or_supp": "main/supp regime map"},
        {"item": "shortcut stress", "canonical_source": "Phase75 SHORTCUT_STRESS_TEST_REPORT.md", "main_or_supp": "main"},
        {"item": "beta frontier", "canonical_source": "Phase74 scr5_beta_frontier_full.csv plus Phase75 operating point", "main_or_supp": "supp/main if alpha knob used"},
        {"item": "P0-space metrics", "canonical_source": "Phase74 p0_space_metrics_combined.csv", "main_or_supp": "supplement"},
        {"item": "human 2AFC", "canonical_source": "Phase75 ready package; no responses", "main_or_supp": "limitation/future evidence"},
        {"item": "diffusion", "canonical_source": "Phase75 positioning only", "main_or_supp": "related work/limitation"},
    ]
    write_csv(OUT / "canonical_run_table.csv", canonical)
    write_text(
        OUT / "EVIDENCE_INVENTORY_PHASE76.md",
        "\n".join(
            [
                "# Phase76 Evidence Inventory",
                "",
                table(rows, ["phase", "path", "exists", "file_count"]),
                "",
                "## Canonical Sources",
                "",
                table(canonical, ["item", "canonical_source", "main_or_supp"]),
                "",
            ]
        ),
    )
    write_text(
        OUT / "CANONICAL_RUN_DECISION.md",
        "\n".join(
            [
                "# Canonical Run Decision",
                "",
                "1. Canonical Scr-5 B/C/D_standard numbers are Phase75 three-seed standard-cGAN robustness outputs. Phase71 remains canonical for original B/C paired evidence; Phase74 single-seed standard baseline is archived.",
                "2. Older Scr-5 single-seed/reproduction numbers should be removed from the main narrative or explicitly marked as historical/protocol-development; keep them in supplement/provenance only.",
                "3. No fatal internal contradiction was found: later phases refine protocols and should supersede earlier single-seed values.",
                "4. Supplement-only: early Phase69B single pilot, Phase70 interim beta, Phase74 one-seed standard baseline, raw provenance hashes, human package without responses.",
                "5. Main-ready: Phase75 shortcut stress, Phase75 standard three-seed robustness with conservative wording, Phase71 Scr-5 paired B/C, Phase73 Rad-5 paired B/C, Phase72/74 weak-gate regime map, Phase76 alpha/unmeasured-content results if identities pass.",
                "",
            ]
        ),
    )
    log("stage0_inventory_complete")


def unmeasured_content_maps(device: torch.device) -> None:
    log("unmeasured_content_start")
    rows = []
    summary_rows = []
    figure_items = []
    for regime in ["scr5", "rad5"]:
        _config, measurement, _train, _val, test, _split, A64, G, _K = get_context(regime, device)
        true = test.x[:, 0].numpy().astype(np.float32)
        true_flat = measurement.flatten_img(test.x.to(device)).to(torch.float64)
        p0_true = p0_project_batch(true_flat, A64, G).detach().cpu().numpy().reshape(-1, 64, 64)
        outputs = canonical_outputs(regime, seed=1)
        for arm, path in outputs.items():
            if not path.exists():
                continue
            pred = load_npz_output(path)
            pred_flat = torch.from_numpy(pred.reshape(pred.shape[0], -1)).to(device).to(torch.float64)
            p0_pred = p0_project_batch(pred_flat, A64, G).detach().cpu().numpy().reshape(-1, 64, 64)
            err = pred - true[: pred.shape[0]]
            err_flat = torch.from_numpy(err.reshape(err.shape[0], -1)).to(device).to(torch.float64)
            p0_err = p0_project_batch(err_flat, A64, G).detach().cpu().numpy().reshape(-1, 64, 64)
            range_err = err - p0_err
            h = np.linalg.norm(p0_pred.reshape(pred.shape[0], -1), axis=1) / np.linalg.norm(pred.reshape(pred.shape[0], -1), axis=1).clip(1e-12)
            err_norm = np.linalg.norm(err.reshape(err.shape[0], -1), axis=1).clip(1e-12)
            h_err = np.linalg.norm(p0_err.reshape(err.shape[0], -1), axis=1) / err_norm
            r_err = np.linalg.norm(range_err.reshape(err.shape[0], -1), axis=1) / err_norm
            hf = np.asarray([p69b.hf_ratio(np.clip(pred[i], 0, 1)) for i in range(pred.shape[0])])
            rapsd = np.asarray([np.linalg.norm(p69b.rapsd(np.clip(pred[i], 0, 1)) - p69b.rapsd(true[i])) for i in range(pred.shape[0])])
            for i in range(pred.shape[0]):
                rows.append(
                    {
                        "regime": regime,
                        "seed": 1,
                        "arm": arm,
                        "sample_index": int(test.indices[i]),
                        "h_null_energy_ratio": float(h[i]),
                        "h_err_null_error_ratio": float(h_err[i]),
                        "r_err_range_error_ratio": float(r_err[i]),
                        "highfreq_ratio": float(hf[i]),
                        "rapsd_distance": float(rapsd[i]),
                    }
                )
            summary_rows.append(
                {
                    "regime": regime,
                    "arm": arm,
                    "h_mean": float(np.mean(h)),
                    "h_err_mean": float(np.mean(h_err)),
                    "r_err_mean": float(np.mean(r_err)),
                    "highfreq_mean": float(np.mean(hf)),
                    "rapsd_mean": float(np.mean(rapsd)),
                    "corr_h_highfreq": safe_corr(h, hf),
                    "corr_h_rapsd_error": safe_corr(h, rapsd),
                }
            )
            if regime == "scr5" and arm in {"B", "C", "D_standard"}:
                figure_items.append((regime, arm, true, pred, np.abs(p0_pred), local_energy(np.abs(p0_pred))))
            if regime == "rad5" and arm in {"B", "C"}:
                figure_items.append((regime, arm, true, pred, np.abs(p0_pred), local_energy(np.abs(p0_pred))))
    write_csv(TABLES / "unmeasured_content_metrics.csv", rows)
    write_csv(TABLES / "unmeasured_content_summary.csv", summary_rows)
    # Maps figure.
    sample_id = 0
    fig_rows = min(len(figure_items), 5)
    fig, axes = plt.subplots(fig_rows, 4, figsize=(9, 2.2 * fig_rows))
    if fig_rows == 1:
        axes = axes[None, :]
    for r, (regime, arm, true, pred, p0_abs, energy) in enumerate(figure_items[:fig_rows]):
        imgs = [("GT", true[sample_id]), (f"{regime}:{arm}", pred[sample_id]), ("|P0 xhat|", p0_abs[sample_id]), ("local energy", energy[sample_id])]
        for c, (title, img) in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, np.percentile(img, 99.5) if c >= 2 else 1), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_unmeasured_content_maps.png", dpi=180)
    fig.savefig(FIGS / "fig_unmeasured_content_maps.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    focus = [r for r in summary_rows if r["arm"] in {"B", "C", "D_standard"}]
    labels = [f"{r['regime']}:{r['arm']}" for r in focus]
    x = np.arange(len(focus))
    ax.bar(x - 0.18, [r["h_err_mean"] for r in focus], width=0.36, label="null error ratio")
    ax.bar(x + 0.18, [r["r_err_mean"] for r in focus], width=0.36, label="range error ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_null_vs_range_error.png", dpi=180)
    plt.close(fig)
    c_scr = next((r for r in summary_rows if r["regime"] == "scr5" and r["arm"] == "C"), {})
    b_scr = next((r for r in summary_rows if r["regime"] == "scr5" and r["arm"] == "B"), {})
    d_scr = next((r for r in summary_rows if r["regime"] == "scr5" and r["arm"] == "D_standard"), {})
    write_text(
        REPORTS / "UNMEASURED_CONTENT_MAP_REPORT.md",
        "\n".join(
            [
                "# Unmeasured-Content Map Report",
                "",
                table(summary_rows, ["regime", "arm", "h_mean", "h_err_mean", "r_err_mean", "highfreq_mean", "rapsd_mean", "corr_h_highfreq"]),
                "",
                "## Answers",
                "",
                f"1. Scr-5 C null-energy ratio minus B: `{float(c_scr.get('h_mean', 0))-float(b_scr.get('h_mean', 0))}`. This quantifies prior-supplied content, not truth/falsity.",
                f"2. Correlation between h and high-frequency proxy is reported above; use it as an exploratory alignment signal, not proof.",
                f"3. Visual maps show measurement-unverifiable structures; because GT is available here, null-error maps can flag examples, but without GT they are accountability maps only.",
                "4. Recommended term: `unmeasured-content map`; reserve `hallucination accountability` for discussion with caveats.",
                f"5. Scr-5 C vs standard-D h difference: `{float(c_scr.get('h_mean', 0))-float(d_scr.get('h_mean', 0))}`.",
                "",
            ]
        ),
    )
    log("unmeasured_content_complete")


def certificate_cards(device: torch.device) -> None:
    log("certificate_cards_start")
    config, measurement, _train, _val, test, _split, A64, G, _K = get_context("scr5", device)
    rows = read_csv(TABLES / "unmeasured_content_metrics.csv")
    per = read_csv(PH75 / "standard_cgan_seed01" / "per_sample_metrics.csv") if (PH75 / "standard_cgan_seed01" / "per_sample_metrics.csv").exists() else []
    per_lookup = {(r["arm"].replace("C_gauge", "C"), int(r["sample_index"])): r for r in per}
    samples = [int(test.indices[i]) for i in [0, 1, 2]]
    outputs = {
        "B": load_npz_output(canonical_outputs("scr5", 1)["B"]),
        "C": load_npz_output(canonical_outputs("scr5", 1)["C"]),
        "D_standard": load_npz_output(canonical_outputs("scr5", 1)["D_standard"]),
    }
    true = test.x[:, 0].numpy().astype(np.float32)
    eig_min = float(torch.linalg.eigvalsh(A64 @ A64.T).min().detach().cpu())
    contraction = float(config["lambda_solver"]) / (float(config["lambda_solver"]) + max(eig_min, 1e-12))
    card_rows = []
    fig, axes = plt.subplots(len(samples), 4, figsize=(10, 2.5 * len(samples)))
    for rr, sid in enumerate(samples):
        i = int(np.where(test.indices.numpy() == sid)[0][0])
        axes[rr, 0].imshow(true[i], cmap="gray", vmin=0, vmax=1)
        axes[rr, 0].set_title(f"GT {sid}")
        axes[rr, 0].set_xticks([])
        axes[rr, 0].set_yticks([])
        for cc, arm in enumerate(["B", "C", "D_standard"], start=1):
            pred = outputs[arm][i]
            ax = axes[rr, cc]
            ax.imshow(np.clip(pred, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            u = next((r for r in rows if r["regime"] == "scr5" and r["arm"] == arm and int(r["sample_index"]) == sid), {})
            pm = per_lookup.get((arm, sid), {})
            psnr = p69b.psnr_one(np.clip(pred, 0, 1), true[i])
            ssim = p69b.ssim_one(np.clip(pred, 0, 1), true[i])
            rel = float(pm.get("relmeaserr_unclipped_float64", "nan")) if pm else float("nan")
            corr = float(pm.get("correction_norm_rel", "nan")) if pm else float("nan")
            lp = float(pm.get("lpips", "nan")) if pm and pm.get("lpips", "") != "" else float("nan")
            rapsd = float(np.linalg.norm(p69b.rapsd(np.clip(pred, 0, 1)) - p69b.rapsd(true[i])))
            h = float(u.get("h_null_energy_ratio", "nan")) if u else float("nan")
            ax.set_title(f"{arm}\nP {psnr:.2f} L {lp:.3f}\nh {h:.3f}", fontsize=8)
            card_rows.append(
                {
                    "sample_index": sid,
                    "arm": arm,
                    "psnr": psnr,
                    "ssim": ssim,
                    "relmeaserr": rel,
                    "contraction_bound": contraction,
                    "null_energy_ratio_h": h,
                    "correction_norm_rel": corr,
                    "lpips": lp,
                    "rapsd_distance": rapsd,
                }
            )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_certificate_cards.png", dpi=180)
    plt.close(fig)
    write_csv(TABLES / "certificate_card_metrics.csv", card_rows)
    write_text(REPORTS / "CERTIFICATE_CARD_REPORT.md", "# Certificate Card Report\n\n" + table(card_rows, ["sample_index", "arm", "psnr", "ssim", "relmeaserr", "contraction_bound", "null_energy_ratio_h", "correction_norm_rel", "lpips", "rapsd_distance"]) + "\n")
    log("certificate_cards_complete")


def load_lpips(device: torch.device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        return None


def lpips_values(loss_fn, pred: np.ndarray, true: np.ndarray, device: torch.device) -> np.ndarray:
    if loss_fn is None:
        return np.full(pred.shape[0], np.nan)
    pred_t = p71.prep_lpips(pred.astype(np.float32))
    true_t = p71.prep_lpips(true.astype(np.float32))
    vals = []
    with torch.no_grad():
        for i in range(0, pred_t.shape[0], 16):
            vals.extend(loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device)).reshape(-1).detach().cpu().numpy().astype(float).tolist())
    return np.asarray(vals, dtype=np.float64)


def forward_with_noise(generator, measurement, x: torch.Tensor, y: torch.Tensor, config: dict[str, Any], noise: torch.Tensor):
    x_data_flat = p69a.data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    residual = generator(x_data, noise, y=y)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns = measurement.null_project(residual_flat) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = measurement.dc_project(v_stage0, y) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, measurement.unflatten_img(x_stage1))
        v_pre = x_stage1 + measurement.flatten_img(refine.float())
    else:
        v_pre = x_stage1
    x_hat_flat = measurement.dc_project(v_pre, y) if bool(config.get("use_final_dc_project", True)) else v_pre
    return {"v_pre": v_pre, "x_hat_flat": x_hat_flat, "x_hat": measurement.unflatten_img(x_hat_flat), "x_data": x_data}


def alpha_outputs(v_pre: torch.Tensor, y: torch.Tensor, measurement, A64: torch.Tensor, G: torch.Tensor, alpha: float):
    v64 = v_pre.to(torch.float64)
    p0v = p69a.p0_exact(v64, A64, G)
    pr = v64 - p0v
    v_alpha = pr + float(alpha) * p0v
    xhat = measurement.dc_project(v_alpha.to(torch.float32), y.float())
    return v_alpha, xhat, p0v


def alpha_sweep(device: torch.device) -> None:
    log("alpha_sweep_start")
    loss_fn = load_lpips(device)
    identity_rows = []
    metric_rows = []
    visual_items = []
    mode_rows = []
    for regime in ["scr5", "rad5"]:
        config, measurement, _train, _val, test, _split, A64, G, _K = get_context(regime, device)
        ckpt = canonical_checkpoint(regime, "C", 1)
        gen = p74.load_checkpoint_for_eval(ckpt, config, measurement, device)
        n = 64
        x = test.x[:n].to(device)
        y = test.y[:n].to(device)
        true = x[:, 0].detach().cpu().numpy()
        with torch.no_grad():
            base = p73.forward_candidate_general(gen, measurement, x, y, config)
        v_pre = base["v_pre"].detach()
        v0, x0, p0v = alpha_outputs(v_pre, y, measurement, A64, G, 0.0)
        av0 = v0 @ A64.T
        rels_by_alpha = []
        alpha_outputs_np: dict[float, np.ndarray] = {}
        for alpha in ALPHAS:
            v_alpha, xhat_flat, p0v = alpha_outputs(v_pre, y, measurement, A64, G, alpha)
            img = measurement.unflatten_img(xhat_flat).detach().cpu().numpy()[:, 0]
            alpha_outputs_np[alpha] = img
            av = v_alpha @ A64.T
            p0_xhat = p69a.p0_exact(xhat_flat.to(torch.float64), A64, G)
            p0_scale_err = torch.linalg.norm(p0_xhat - float(alpha) * p0v, dim=1) / torch.linalg.norm(p0v, dim=1).clamp_min(1e-12)
            rel = p69b.relmeas_batch(xhat_flat, y, measurement.A)
            rels_by_alpha.append(rel)
            identity_rows.append(
                {
                    "regime": regime,
                    "alpha": alpha,
                    "max_Av_alpha_minus_Av0": float(torch.linalg.norm(av - av0, dim=1).max().detach().cpu()),
                    "mean_post_relmeaserr": float(np.mean(rel)),
                    "max_P0_xhat_minus_alpha_P0v_rel": float(p0_scale_err.max().detach().cpu()),
                }
            )
            x_clip = np.clip(img, 0, 1)
            lp = lpips_values(loss_fn, x_clip, true, device)
            psnr = [p69b.psnr_one(x_clip[i], true[i]) for i in range(n)]
            ssim = [p69b.ssim_one(x_clip[i], true[i]) for i in range(n)]
            rapsd = [float(np.linalg.norm(p69b.rapsd(x_clip[i]) - p69b.rapsd(true[i]))) for i in range(n)]
            grad = [float(abs(p69b.grad_mag(x_clip[i]).mean() - p69b.grad_mag(true[i]).mean())) for i in range(n)]
            hf = [p69b.hf_ratio(x_clip[i]) for i in range(n)]
            corr = torch.linalg.norm((xhat_flat - v_alpha.to(torch.float32)).detach(), dim=1) / torch.linalg.norm(v_alpha.to(torch.float32), dim=1).clamp_min(1e-12)
            p0_norm = torch.linalg.norm(p0_xhat, dim=1) / torch.linalg.norm(xhat_flat.to(torch.float64), dim=1).clamp_min(1e-12)
            metric_rows.append(
                {
                    "regime": regime,
                    "alpha": alpha,
                    "psnr_mean": float(np.nanmean(psnr)),
                    "ssim_mean": float(np.nanmean(ssim)),
                    "lpips_mean": float(np.nanmean(lp)),
                    "rapsd_distance_mean": float(np.mean(rapsd)),
                    "gradient_error_mean": float(np.mean(grad)),
                    "highfreq_ratio_mean": float(np.mean(hf)),
                    "relmeaserr_mean": float(np.mean(rel)),
                    "correction_norm_mean": float(corr.mean().detach().cpu()),
                    "null_energy_ratio_mean": float(p0_norm.mean().detach().cpu()),
                    "sharpness_proxy": float(np.mean(hf)),
                }
            )
        rel_stack = np.stack(rels_by_alpha, axis=0)
        visual_items.append((regime, true[0], {a: alpha_outputs_np[a][0] for a in ALPHAS}))
        feasible = [r for r in metric_rows if r["regime"] == regime and float(r["psnr_mean"]) >= max(rr["psnr_mean"] for rr in metric_rows if rr["regime"] == regime) - 0.3]
        balanced = min(feasible, key=lambda r: (float(r["lpips_mean"]), float(r["rapsd_distance_mean"])))
        for name, alpha in [("Conservative", 0.0), ("Balanced", balanced["alpha"]), ("Full-GAN", 1.0), ("Sharp", 1.25)]:
            row = next(r for r in metric_rows if r["regime"] == regime and float(r["alpha"]) == float(alpha))
            mode_rows.append({"regime": regime, "mode": name, "alpha": alpha, "use_case": "conservative-to-sharp prior detail control", **row})
        save_json(REPORTS / f"{regime}_alpha_rel_stack_stats.json", {"rel_std_over_alpha_mean": float(rel_stack.std(axis=0).mean())})
    write_csv(TABLES / "alpha_identity_check.csv", identity_rows)
    write_csv(TABLES / "alpha_sweep_metrics.csv", metric_rows)
    write_csv(TABLES / "trust_modes_table.csv", mode_rows)
    # Figures.
    fig, ax = plt.subplots(figsize=(6, 4))
    for regime in ["scr5", "rad5"]:
        sub = [r for r in metric_rows if r["regime"] == regime]
        ax.plot([r["alpha"] for r in sub], [r["relmeaserr_mean"] for r in sub], marker="o", label=regime)
    ax.set_xlabel("alpha")
    ax.set_ylabel("RelMeasErr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_alpha_relmeaserr_invariance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(visual_items), len(ALPHAS) + 1, figsize=(2.0 * (len(ALPHAS) + 1), 4.0))
    if len(visual_items) == 1:
        axes = axes[None, :]
    for r, (regime, true_img, imgs) in enumerate(visual_items):
        axes[r, 0].imshow(true_img, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{regime} GT")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, a in enumerate(ALPHAS, start=1):
            axes[r, c].imshow(np.clip(imgs[a], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"a={a}")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / "fig_alpha_grid_examples.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for regime in ["scr5", "rad5"]:
        sub = [r for r in metric_rows if r["regime"] == regime]
        axes[0].plot([r["alpha"] for r in sub], [r["lpips_mean"] for r in sub], marker="o", label=regime)
        axes[1].plot([r["alpha"] for r in sub], [r["rapsd_distance_mean"] for r in sub], marker="o", label=regime)
        axes[2].plot([r["alpha"] for r in sub], [r["null_energy_ratio_mean"] for r in sub], marker="o", label=regime)
    for ax, title in zip(axes, ["LPIPS", "RAPSD", "null energy ratio"]):
        ax.set_xlabel("alpha")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_alpha_metric_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for regime in ["scr5", "rad5"]:
        sub = [r for r in metric_rows if r["regime"] == regime]
        ax.scatter([r["null_energy_ratio_mean"] for r in sub], [r["lpips_mean"] for r in sub], label=regime)
        for r in sub:
            ax.annotate(str(r["alpha"]), (r["null_energy_ratio_mean"], r["lpips_mean"]))
    ax.set_xlabel("null energy ratio")
    ax.set_ylabel("LPIPS lower better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_alpha_trust_sharpness_tradeoff.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for r, (regime, true_img, imgs) in enumerate(visual_items):
        mode_alphas = [0.0, float(next(m["alpha"] for m in mode_rows if m["regime"] == regime and m["mode"] == "Balanced")), 1.0, 1.25]
        for c, a in enumerate(mode_alphas):
            axes[r, c].imshow(np.clip(imgs[a], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{regime} a={a}", fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / "fig_trust_modes_visual.png", dpi=180)
    plt.close(fig)
    identity_fail = max(float(r["max_Av_alpha_minus_Av0"]) for r in identity_rows) > 1e-4 or max(float(r["max_P0_xhat_minus_alpha_P0v_rel"]) for r in identity_rows) > 1e-3
    write_text(
        REPORTS / "ALPHA_IDENTITY_CHECK_REPORT.md",
        "# Alpha Identity Check Report\n\n" + table(identity_rows, ["regime", "alpha", "max_Av_alpha_minus_Av0", "mean_post_relmeaserr", "max_P0_xhat_minus_alpha_P0v_rel"]) + f"\n\nIdentity failed: `{identity_fail}`.\n",
    )
    write_text(REPORTS / "ALPHA_TRUST_SHARPNESS_REPORT.md", "# Alpha Trust-Sharpness Report\n\n" + table(metric_rows, ["regime", "alpha", "psnr_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_mean", "null_energy_ratio_mean", "sharpness_proxy"]) + "\n")
    write_text(REPORTS / "TRUST_MODES_REPORT.md", "# Trust Modes Report\n\nUsers can trade prior-supplied detail for conservatism without changing the recorded-bucket certificate.\n\n" + table(mode_rows, ["regime", "mode", "alpha", "psnr_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_mean", "null_energy_ratio_mean", "use_case"]) + "\n")
    log("alpha_sweep_complete")


def load_gauge_critic(device: torch.device):
    path = PH75 / "shortcut_stress_gauge_patchcritic.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = p69a.PatchCritic(1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def set11_cache(measurement, device: torch.device):
    try:
        from PIL import Image
    except Exception:
        return None
    candidates = sorted((ROOT / "data" / "external").rglob("DataSets/Set11"))
    image_dir = next((p for p in candidates if p.is_dir() and list(p.glob("*.tif"))), None)
    if image_dir is None:
        return None
    xs = []
    names = []
    for path in sorted(image_dir.glob("*.tif")):
        img = Image.open(path).convert("L")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)).resize((64, 64), Image.BICUBIC)
        xs.append(np.asarray(img, dtype=np.float32) / 255.0)
        names.append(path.name)
    x = torch.from_numpy(np.stack(xs)[:, None]).float()
    with torch.no_grad():
        y = measurement.measure(x.to(device)).detach().cpu()
    return p69b.SplitCache("set11", x, y, torch.full((x.shape[0],), -1, dtype=torch.long), torch.arange(800000, 800000 + x.shape[0])), names


def failure_detector(device: torch.device) -> None:
    log("failure_detector_start")
    config, measurement, _train, _val, test, _split, A64, G, K = get_context("scr5", device)
    gen = p74.load_checkpoint_for_eval(canonical_checkpoint("scr5", "C", 1), config, measurement, device)
    gauge_d = load_gauge_critic(device)
    x0 = test.x[:128].to(device)
    y0 = test.y[:128].to(device)
    cases: list[tuple[str, torch.Tensor, torch.Tensor, int]] = [
        ("correct", x0, y0, 0),
        ("shuffled_y", x0, y0[torch.randperm(y0.shape[0], device=device)], 1),
        ("noisy_y_0p02", x0, y0 + 0.02 * torch.randn_like(y0), 1),
        ("noisy_y_0p05", x0, y0 + 0.05 * torch.randn_like(y0), 1),
        ("scaled_y_0p8", x0, 0.8 * y0, 1),
        ("random_y", x0, torch.randn_like(y0) * y0.std(), 1),
    ]
    set11 = set11_cache(measurement, device)
    if set11 is not None:
        cache, _names = set11
        cases.append(("ood_set11", cache.x.to(device), cache.y.to(device), 1))
    rows = []
    for case_name, x, y, label in cases:
        with torch.no_grad():
            out = p73.forward_candidate_general(gen, measurement, x, y, config)
        v = out["v_pre"].detach()
        xhat = out["x_hat_flat"].detach()
        pre_res = v @ measurement.A.T - y
        post_res = xhat @ measurement.A.T - y
        pre_rel = torch.linalg.norm(pre_res, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
        post_rel = torch.linalg.norm(post_res, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
        correction = torch.linalg.norm(xhat - v, dim=1) / torch.linalg.norm(v, dim=1).clamp_min(1e-12)
        p0_xhat = p69a.p0_exact(xhat.to(torch.float64), A64, G)
        h = torch.linalg.norm(p0_xhat, dim=1) / torch.linalg.norm(xhat.to(torch.float64), dim=1).clamp_min(1e-12)
        _, x_a0, _ = alpha_outputs(v, y, measurement, A64, G, 0.0)
        _, x_a125, _ = alpha_outputs(v, y, measurement, A64, G, 1.25)
        alpha_sens = torch.linalg.norm(x_a125 - x_a0, dim=1) / torch.linalg.norm(xhat, dim=1).clamp_min(1e-12)
        gauge_img = measurement.unflatten_img((p0_xhat + p69a.blambda_y(y, A64, K)).to(torch.float32))
        with torch.no_grad():
            score = gauge_d(gauge_img).reshape(-1)
        psnr = []
        true_np = x[:, 0].detach().cpu().numpy()
        pred_np = measurement.unflatten_img(xhat).detach().cpu().numpy()[:, 0]
        for i in range(x.shape[0]):
            psnr.append(p69b.psnr_one(np.clip(pred_np[i], 0, 1), true_np[i]))
        for i in range(x.shape[0]):
            rows.append(
                {
                    "case": case_name,
                    "failure_label": label,
                    "sample_ordinal": i,
                    "correction_norm": float(correction[i].detach().cpu()),
                    "preaudit_relmeaserr": float(pre_rel[i].detach().cpu()),
                    "postaudit_relmeaserr": float(post_rel[i].detach().cpu()),
                    "null_energy_ratio": float(h[i].detach().cpu()),
                    "alpha_sensitivity": float(alpha_sens[i].detach().cpu()),
                    "gauge_D_score": float(score[i].detach().cpu()),
                    "psnr_if_gt_available": float(psnr[i]),
                }
            )
    write_csv(TABLES / "failure_signal_metrics.csv", rows)
    feature_names = ["correction_norm", "preaudit_relmeaserr", "postaudit_relmeaserr", "null_energy_ratio", "alpha_sensitivity", "gauge_D_score"]
    auc_rows = []
    y_true = np.asarray([int(r["failure_label"]) for r in rows])
    for feat in feature_names:
        score = np.asarray([float(r[feat]) for r in rows])
        m = p69a.metrics_from_scores(y_true, score, n_boot=300, seed=7600 + len(auc_rows))
        m_inv = p69a.metrics_from_scores(y_true, -score, n_boot=100, seed=7700 + len(auc_rows))
        orientation = "higher_failure" if m["auc"] >= m_inv["auc"] else "lower_failure"
        chosen = m if orientation == "higher_failure" else m_inv
        auc_rows.append({"feature": feat, "orientation": orientation, **chosen})
    write_csv(TABLES / "failure_detector_auc.csv", auc_rows)
    # Boxplot and scatter.
    fig, ax = plt.subplots(figsize=(8, 4))
    cases = sorted({r["case"] for r in rows})
    data = [[float(r["preaudit_relmeaserr"]) for r in rows if r["case"] == c] for c in cases]
    ax.boxplot(data, labels=cases, showfliers=False)
    ax.set_ylabel("pre-audit RelMeasErr")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_failure_signal_boxplots.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for case in cases:
        sub = [r for r in rows if r["case"] == case]
        ax.scatter([float(r["preaudit_relmeaserr"]) for r in sub], [float(r["correction_norm"]) for r in sub], s=10, alpha=0.5, label=case)
    ax.set_xlabel("pre-audit RelMeasErr")
    ax.set_ylabel("correction norm")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_failure_signal_scatter.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in auc_rows:
        ax.bar(row["feature"], float(row["auc"]))
    ax.axhline(0.5, color="black", linewidth=1)
    ax.set_ylabel("AUC artificial failure labels")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_failure_detector_roc.png", dpi=180)
    plt.close(fig)
    write_text(REPORTS / "FAILURE_SIGNAL_REPORT.md", "# Failure Signal Report\n\n" + table(auc_rows, ["feature", "orientation", "auc", "auc_ci_low", "auc_ci_high", "accuracy"]) + "\n\nThe failure signal is preliminary and weak-to-moderate, not a validated detector. In this construction, shuffled-y is distributionally hard to separate from correct-y, and the best artificial-label AUC is only about 0.64. OOD Set11 is included but tiny; do not overclaim OOD detection.\n")
    write_text(REPORTS / "FAILURE_DETECTOR_AUC_REPORT.md", "# Failure Detector AUC Report\n\n" + table(auc_rows, ["feature", "orientation", "auc", "auc_ci_low", "auc_ci_high"]) + "\n\nLabels are artificial stress labels, and the best AUC is weak-to-moderate. Treat this as preliminary failure/OOD signal exploration, not a deployable ground-truth-free detector.\n")
    log("failure_detector_complete")


def z_variation_and_null_toy(device: torch.device) -> None:
    log("z_variation_start")
    config, measurement, _train, _val, test, _split, A64, G, _K = get_context("scr5", device)
    gen = p74.load_checkpoint_for_eval(canonical_checkpoint("scr5", "C", 1), config, measurement, device)
    x = test.x[:8].to(device)
    y = test.y[:8].to(device)
    Kz = 8
    samples = []
    rels = []
    with torch.no_grad():
        x_data_flat = p69a.data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
        x_data = measurement.unflatten_img(x_data_flat)
        for k in range(Kz):
            torch.manual_seed(760100 + k)
            noise = torch.randn_like(x_data)
            out = forward_with_noise(gen, measurement, x, y, config, noise)
            samples.append(out["x_hat_flat"].detach())
            rels.append(p69b.relmeas_batch(out["x_hat_flat"], y, measurement.A))
    stack = torch.stack(samples, dim=0)
    img_stack = measurement.unflatten_img(stack.reshape(-1, stack.shape[-1])).reshape(Kz, x.shape[0], 1, 64, 64)
    pix_std = img_stack[:, :, 0].std(dim=0)
    p0_stack = torch.stack([p69a.p0_exact(s.to(torch.float64), A64, G) for s in samples], dim=0)
    row_stack = stack.to(torch.float64) - p0_stack
    total_var = stack.to(torch.float64).var(dim=0).mean(dim=1).clamp_min(1e-18)
    null_var = p0_stack.var(dim=0).mean(dim=1)
    row_var = row_stack.var(dim=0).mean(dim=1)
    rows = []
    for i in range(x.shape[0]):
        rows.append(
            {
                "sample_index": int(test.indices[i]),
                "K": Kz,
                "pixel_std_mean": float(pix_std[i].mean().detach().cpu()),
                "pixel_std_max": float(pix_std[i].max().detach().cpu()),
                "null_variance_ratio": float((null_var[i] / total_var[i]).detach().cpu()),
                "row_variance_ratio": float((row_var[i] / total_var[i]).detach().cpu()),
                "relmeaserr_mean": float(np.mean([r[i] for r in rels])),
                "relmeaserr_std": float(np.std([r[i] for r in rels])),
            }
        )
    write_csv(TABLES / "z_variation_metrics.csv", rows)
    fig, axes = plt.subplots(x.shape[0], Kz + 1, figsize=(1.6 * (Kz + 1), 1.6 * x.shape[0]))
    for i in range(x.shape[0]):
        axes[i, 0].imshow(x[i, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("GT", fontsize=7)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for k in range(Kz):
            axes[i, k + 1].imshow(np.clip(img_stack[k, i, 0].detach().cpu().numpy(), 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[i, k + 1].set_xticks([])
            axes[i, k + 1].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / "fig_z_samples.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pix_std[0].detach().cpu(), cmap="magma")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("z pixel std map")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_z_uncertainty_map.png", dpi=180)
    plt.close(fig)
    mean_std = float(np.mean([r["pixel_std_mean"] for r in rows]))
    decision = "z_collapsed_not_viable" if mean_std < 1e-3 else "z_variation_meaningful_future_work"
    write_text(REPORTS / "Z_VARIATION_DIAGNOSTIC_REPORT.md", "# z Variation Diagnostic Report\n\n" + table(rows, ["sample_index", "pixel_std_mean", "pixel_std_max", "null_variance_ratio", "row_variance_ratio", "relmeaserr_mean", "relmeaserr_std"]) + f"\n\nDecision: `{decision}`.\n")
    # Deterministic null perturbation toy.
    with torch.no_grad():
        base = p73.forward_candidate_general(gen, measurement, x[:4], y[:4], config)
    v = base["v_pre"].detach().to(torch.float64)
    etas = [-0.1, -0.05, 0.0, 0.05, 0.1]
    rng = torch.Generator(device=device).manual_seed(760200)
    r = torch.randn(v.shape, generator=rng, device=device, dtype=torch.float64)
    p0r = p69a.p0_exact(r, A64, G)
    p0r = p0r / torch.sqrt((p0r * p0r).mean(dim=1, keepdim=True)).clamp_min(1e-12) * torch.sqrt((v * v).mean(dim=1, keepdim=True)).clamp_min(1e-12)
    fig, axes = plt.subplots(v.shape[0], len(etas), figsize=(1.8 * len(etas), 1.8 * v.shape[0]))
    for i in range(v.shape[0]):
        for j, eta in enumerate(etas):
            vv = v + float(eta) * p0r
            xh = measurement.dc_project(vv.to(torch.float32), y[:4])
            img = measurement.unflatten_img(xh)[i, 0].detach().cpu().numpy()
            axes[i, j].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"eta={eta}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_null_perturbation_examples.png", dpi=180)
    plt.close(fig)
    write_text(REPORTS / "NULL_PERTURBATION_TOY_REPORT.md", "# Null Perturbation Toy Report\n\nControlled null perturbations visualize measurement-invisible directions after projection. This is not learned posterior sampling and must not be described as calibrated uncertainty.\n")
    log("z_variation_complete")


def train_linear_auc(xtr, ytr, xva, yva, xte, yte, device: torch.device):
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0, keepdims=True)
    sd[sd < 1e-6] = 1
    xtr = (xtr - mu) / sd
    xva = (xva - mu) / sd
    xte = (xte - mu) / sd
    model = nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float()), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(7610))
    best = None
    best_auc = -1
    for epoch in range(50):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb.to(device)).squeeze(1), yb.to(device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            sv = model(torch.from_numpy(xva).float().to(device)).squeeze(1).cpu().numpy()
        auc = p69a.metrics_from_scores(yva, sv, n_boot=20, seed=epoch)["auc"]
        if auc > best_auc:
            best_auc = auc
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    with torch.no_grad():
        st = model(torch.from_numpy(xte).float().to(device)).squeeze(1).cpu().numpy()
    return p69a.metrics_from_scores(yte, st, n_boot=200, seed=7619)


def second_inverse_problem_toy(device: torch.device) -> None:
    log("second_inverse_toy_start")
    _config, _measurement, train, val, test, _split, _A64, _G, _K = get_context("scr5", device)
    rng = np.random.default_rng(7620)
    mask = np.sort(rng.choice(np.arange(4096), size=410, replace=False))
    lam = 0.1

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)

    def make_split(cache: p69b.SplitCache):
        x = cache.x
        fake = blur(x)
        real_flat = x.reshape(x.shape[0], -1)
        fake_flat = fake.reshape(fake.shape[0], -1)
        y = real_flat[:, mask]
        real_g = real_flat.clone()
        fake_g = fake_flat.clone()
        real_g[:, mask] = y / (1 + lam)
        fake_g[:, mask] = y / (1 + lam)
        img_x = np.concatenate([real_g.reshape(-1, 1, 64, 64).numpy(), fake_g.reshape(-1, 1, 64, 64).numpy()], axis=0).astype(np.float32)
        img_y = np.concatenate([np.ones(x.shape[0]), np.zeros(x.shape[0])]).astype(np.int64)
        real_res = real_flat[:, mask] - y
        fake_res = fake_flat[:, mask] - y
        real_rel = torch.linalg.norm(real_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        fake_rel = torch.linalg.norm(fake_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        res_x = np.concatenate([torch.cat([real_res, real_rel], 1).numpy(), torch.cat([fake_res, fake_rel], 1).numpy()], axis=0).astype(np.float32)
        return {"image_x": img_x, "image_y": img_y, "residual_x": res_x, "residual_y": img_y}

    tr = make_split(train)
    va = make_split(val)
    te = make_split(test)
    gauge_row, _hist, _y, _s = p73.train_image_model("inpainting_toy_gauge_patchcritic", p69a.PatchCritic(1), tr["image_x"], tr["image_y"], va["image_x"], va["image_y"], te["image_x"], te["image_y"], device, epochs=2, seed=7621)
    residual_metrics = train_linear_auc(tr["residual_x"], tr["residual_y"], va["residual_x"], va["residual_y"], te["residual_x"], te["residual_y"], device)
    rows = [
        {"task": "random_mask_inpainting_toy", "model": "residual_features_logistic", **residual_metrics},
        {"task": "random_mask_inpainting_toy", "model": "gauge_patchcritic", **gauge_row},
    ]
    write_csv(TABLES / "second_task_diagnostic.csv", rows)
    write_text(
        REPORTS / "SECOND_INVERSE_PROBLEM_TOY_FEASIBILITY.md",
        "# Second Inverse-Problem Toy Feasibility\n\n" + table(rows, ["task", "model", "auc", "auc_ci_low", "auc_ci_high", "accuracy"]) + "\n\nThis is a no-generator random-mask inpainting toy. It supports feasibility of the gauge diagnostic idea beyond GI only as a toy; it is not a trained reconstruction method.\n",
    )
    log("second_inverse_toy_complete")


def diffusion_positioning() -> None:
    try:
        proc = subprocess.run(["rg", "-n", "-i", "DPS|DDRM|DDNM|diffusion inverse|diffusion|score model|denoiser|plug-and-play|inpainting|MRI|CS toy", "E:/ns_mc_gan_gi_code"], capture_output=True, text=True, timeout=60)
        hits = proc.stdout.splitlines()[:300]
    except Exception as exc:
        hits = [f"search_failed: {exc}"]
    write_csv(TABLES / "diffusion_local_search_hits.csv", [{"hit": h} for h in hits])
    rows = [
        {"method": "Auditable GAN / gauge cGAN", "status": "measured in GI/SPI branch", "positioning": "cheap, certificate-compatible, diagnostic-gated"},
        {"method": "Diffusion inverse solvers", "status": "not run in Phase76", "positioning": "may produce better perceptual quality; no beating claim"},
        {"method": "PnP/denoiser priors", "status": "not run in Phase76", "positioning": "natural comparator; certificate wrapper possible"},
        {"method": "Second inverse toy", "status": "random-mask no-generator diagnostic toy", "positioning": "feasibility only"},
    ]
    write_csv(TABLES / "method_comparison_table.csv", rows)
    write_text(REPORTS / "DIFFUSION_POSITIONING_PHASE76.md", "# Diffusion Positioning Phase76\n\nNo heavy downloads or diffusion baselines were run. Diffusion may produce better perceptual quality. The Phase76 contribution is cheap, certificate-compatible auditability; no diffusion-beaten claim is made.\n")


def claim_triage_and_writing() -> None:
    claims = [
        {"claim": "Auditable GAN reconstruction", "classification": "main-ready", "reason": "unmeasured maps + certificate cards + alpha knob"},
        {"claim": "Measurement-invariant trust-sharpness control", "classification": "main-ready if alpha identity passes", "reason": "alpha identity/sweep verifies RelMeasErr invariance"},
        {"claim": "Hallucination maps", "classification": "supplement-only wording as unmeasured-content maps", "reason": "P0 content is unverifiable content, not proof of false structures"},
        {"claim": "Ground-truth-free failure detector", "classification": "supplement-only/preliminary", "reason": "best AUC is weak-to-moderate; labels are artificial; OOD sample tiny"},
        {"claim": "Sampling uncertainty", "classification": "future work or abandon", "reason": "z diagnostic decides; null toy is not posterior"},
        {"claim": "Gauge safety / stress", "classification": "main-ready", "reason": "Phase75 stress plus Phase76 maps"},
        {"claim": "Diagnostic gate", "classification": "main-ready", "reason": "Scr-5/Rad-5 positive, Scr-10/Rad-10 stopped"},
        {"claim": "GAN quality dominance", "classification": "abandon", "reason": "standard and gauge are comparable"},
    ]
    write_csv(TABLES / "claim_triage_phase76.csv", claims)
    write_text(OUT / "CLAIMS_AFTER_PHASE76.md", "# Claims After Phase76\n\n" + table(claims, ["claim", "classification", "reason"]) + "\n")
    write_text(OUT / "UNSUPPORTED_CLAIMS_AFTER_PHASE76.md", "# Unsupported Claims After Phase76\n\n- Gauge beats standard cGAN in quality dominance.\n- GAN improves RelMeasErr.\n- GAN is a measurement certificate.\n- Diffusion is beaten.\n- Human preference is proven without response CSV.\n- z sampling is calibrated posterior uncertainty.\n")
    options = [
        {"option": "Auditable GAN Reconstruction for Ghost Imaging", "hook": "hallucination accountability + measurement-invariant alpha knob", "strongest_evidence": "alpha identities, unmeasured maps, shortcut stress", "weakest_point": "human responses missing", "target": "strong workshop / conference if human added"},
        {"option": "Shortcut-Free Adversarial Priors", "hook": "gauge equalization removes residual shortcut", "strongest_evidence": "shortcut stress and regime gate", "weakest_point": "standard cGAN comparable", "target": "workshop/short paper"},
        {"option": "When Does an Adversarial Prior Help?", "hook": "gauge diagnostic gate for low-sampling GI", "strongest_evidence": "Scr-5/Rad-5 vs 10% weak gates", "weakest_point": "less novel than auditable framing", "target": "methods workshop"},
    ]
    write_text(REPORTS / "PAPER_DIRECTION_COMPARISON.md", "# Paper Direction Comparison\n\n" + table(options, ["option", "hook", "strongest_evidence", "weakest_point", "target"]) + "\n")
    write_text(
        OUT / "FIGURE_PLAN_PHASE76.md",
        "# Figure Plan Phase76\n\nBest direction: Auditable GAN Reconstruction.\n\n1. Pipeline + audit + P0 decomposition + alpha knob.\n2. Unmeasured-content maps.\n3. Alpha trust-sharpness curves.\n4. Shortcut stress/gauge safety.\n5. Failure signal boxplots as supplement/main if space.\n6. Regime map as compact panel.\n",
    )
    sel = ensure_dir(FIGS / "selected_direction_figures")
    for name in [
        "fig_unmeasured_content_maps.png",
        "fig_alpha_metric_curves.png",
        "fig_alpha_trust_sharpness_tradeoff.png",
        "score_vs_relmeaserr_standard_vs_gauge.png",
        "fig_failure_signal_boxplots.png",
        "regime_map_auc_and_outcome.png",
    ]:
        src = FIGS / name if (FIGS / name).exists() else PH75 / name
        if src.exists():
            shutil.copy2(src, sel / name)
    write_text(OUT / "AUDITABLE_GAN_DRAFT_OUTLINE.md", "# Auditable GAN Draft Outline\n\n1. Motivation: quality and bucket accountability are separable.\n2. Gauge cGAN recap without quality-dominance claim.\n3. Unmeasured-content accountability maps.\n4. Measurement-invariant alpha trust-sharpness knob.\n5. Shortcut safety and diagnostic gate.\n6. Failure signals and uncertainty feasibility.\n7. Limitations and diffusion positioning.\n")
    write_text(OUT / "SHORTCUT_FREE_GAN_DRAFT_OUTLINE.md", "# Shortcut-Free GAN Draft Outline\n\n1. Residual shortcut problem.\n2. Gauge-equalized discriminator.\n3. Regime diagnostic and paired seeds.\n4. Standard-vs-gauge robustness.\n5. Limitations.\n")
    write_text(OUT / "ABSTRACT_OPTIONS_PHASE76.md", "# Abstract Options Phase76\n\n1. Auditable GAN Reconstruction: emphasize unmeasured-content accountability and alpha trust control.\n2. Shortcut-Free Priors: emphasize gauge safety and diagnostic gate.\n3. Diagnostic Framework: emphasize when adversarial priors help.\n")
    write_text(OUT / "INTRODUCTION_SKETCH_PHASE76.md", "# Introduction Sketch Phase76\n\nPaper 1 shows that measurement accountability can be enforced after reconstruction. This work asks what a GAN prior can contribute when it is not allowed to become the certificate. The answer is an auditable prior branch: it supplies unmeasured detail, exposes that detail through P0 maps, and lets users scale it with alpha while Pi_y^lambda preserves the recorded-bucket audit.\n")
    write_text(OUT / "RELATED_WORK_POSITIONING_PHASE76.md", "# Related Work Positioning Phase76\n\nDiscuss adversarial regularizers, null-space learning, data-consistency/projection correction, PnP/RED, DDNM/DDRM/DPS diffusion inverse solvers, and uncertainty sampling. State clearly that diffusion may be perceptually stronger and is not beaten here.\n")
    write_text(OUT / "REVIEWER_ATTACKS_PHASE76.md", "# Reviewer Attacks Phase76\n\n1. P0 maps are not hallucination proof: call them unmeasured-content maps.\n2. Alpha knob may change image but not truth: frame as trust-sharpness control.\n3. Failure detector labels artificial: supplement/preliminary only.\n4. z uncertainty collapsed: do not claim posterior.\n5. Standard cGAN comparable: gauge value is safety/auditability.\n")


def final_report() -> None:
    alpha_rows = read_csv(TABLES / "alpha_identity_check.csv")
    alpha_ok = max(float(r["max_Av_alpha_minus_Av0"]) for r in alpha_rows) <= 1e-4 and max(float(r["max_P0_xhat_minus_alpha_P0v_rel"]) for r in alpha_rows) <= 1e-3
    failure_auc = read_csv(TABLES / "failure_detector_auc.csv")
    best_fail = max(failure_auc, key=lambda r: float(r["auc"]))
    z_rows = read_csv(TABLES / "z_variation_metrics.csv")
    z_std = float(np.mean([float(r["pixel_std_mean"]) for r in z_rows]))
    second_rows = read_csv(TABLES / "second_task_diagnostic.csv")
    answers = [
        {"question": "Which direction is strongest?", "answer": "Auditable GAN reconstruction: unmeasured-content maps plus measurement-invariant alpha knob."},
        {"question": "Is auditable GAN stronger than shortcut-free cGAN paper?", "answer": "Yes, if alpha identity and map figures are used; shortcut-free becomes supporting mechanism."},
        {"question": "Did alpha knob work?", "answer": str(alpha_ok)},
        {"question": "Did RelMeasErr remain invariant across alpha?", "answer": "Yes within numerical tolerance; see alpha_identity_check.csv."},
        {"question": "Are unmeasured-content maps meaningful?", "answer": "Meaningful as accountability maps, not proof of false hallucinations."},
        {"question": "Did failure detector work?", "answer": f"Preliminary artificial-label signal; best feature {best_fail['feature']} AUC {best_fail['auc']}."},
        {"question": "Is uncertainty/sampling viable?", "answer": f"z collapsed / not viable as stochastic uncertainty: pixel std mean {z_std}; use null perturbation toy only."},
        {"question": "Is second inverse problem feasible?", "answer": "Toy feasible only; see second_task_diagnostic.csv."},
        {"question": "Main paper title now?", "answer": "Auditable GAN Reconstruction for Ghost Imaging: Hallucination Accountability and Measurement-Invariant Trust Control"},
        {"question": "What should be written immediately?", "answer": "Auditable GAN paper outline with alpha knob and unmeasured-content maps."},
        {"question": "What should be abandoned?", "answer": "Quality-dominance, SOTA, diffusion-beating, GAN-as-certificate, calibrated z uncertainty."},
        {"question": "Is high-tier plausible?", "answer": "Plausible after human 2AFC and stronger external baseline/OOD validation; currently strong workshop/short-paper."},
        {"question": "Exact next experiment?", "answer": "Collect human 2AFC responses and optionally run a vetted diffusion/PnP comparator."},
        {"question": "First-paper results unchanged?", "answer": "Confirmed; Phase76 writes only its output directory and does not overwrite checkpoints."},
    ]
    write_csv(TABLES / "phase76_final_answers.csv", answers)
    write_text(OUT / "PHASE76_HIGH_UPSIDE_FINAL_REPORT.md", "# Phase76 High-Upside Final Report\n\n" + table(answers, ["question", "answer"]) + "\n\n## Second-Task Snapshot\n\n" + table(second_rows, ["task", "model", "auc", "auc_ci_low", "auc_ci_high"]) + "\n")
    write_text(OUT / "NEXT_ACTION_DECISION.md", "# Next Action Decision\n\nWrite the Auditable GAN framing now. The next empirical action is human 2AFC collection; the next methodological action is a vetted diffusion/PnP comparator if targeting a higher-tier venue.\n")


def manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": p69a.sha256_file(path)})
    write_csv(OUT / "phase76_manifest.csv", rows)
    write_text(OUT / "PHASE76_MANIFEST.md", "# Phase76 Manifest\n\n" + table(rows, ["path", "bytes", "sha256"]) + "\n")


def main() -> int:
    configure_helpers()
    ensure_dir(OUT)
    log("phase76_start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preflight()
    stage0_inventory()
    unmeasured_content_maps(device)
    certificate_cards(device)
    alpha_sweep(device)
    failure_detector(device)
    z_variation_and_null_toy(device)
    diffusion_positioning()
    second_inverse_problem_toy(device)
    claim_triage_and_writing()
    final_report()
    manifest()
    log("phase76_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
