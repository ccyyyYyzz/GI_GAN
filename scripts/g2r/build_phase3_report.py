"""Assemble the Phase 3 three-arm report: trajectory tables + D margin curves.

Reads each arm's gate_trajectory.csv and train_log.json from the run output
dirs and writes results/g2r_pilot_phase3/PHASE3_REPORT.md in the repo.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RUNS_ROOT = Path(r"E:\ns_mc_gan_gi\outputs_g2r")
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "g2r_pilot_phase3"

ARMS = [
    ("g2r_pilot_scr5_adv1e-3", "1e-3"),
    ("g2r_pilot_scr5_adv3e-3", "3e-3"),
    ("g2r_pilot_scr5_adv1e-2", "1e-2"),
]
MARGIN_WINDOW = 1000  # average d_real/d_fake over windows of this many steps


def trajectory_table(run_dir: Path) -> list[str]:
    rows = list(csv.DictReader((run_dir / "gate_trajectory.csv").open(encoding="utf-8")))
    lines = [
        "| step | N | std_med | PSNR(mean) | PSNR(sample) | G-CAL gap | edge_rho | NVR | relmeas_med | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        gap = float(r["psnr_mean_db"]) - float(r["psnr_sample_db"])
        lines.append(
            f"| {r['step']} | {r['n_images']} | {float(r['std_median']):.4f} | {float(r['psnr_mean_db']):.2f} "
            f"| {float(r['psnr_sample_db']):.2f} | {gap:.2f} | {float(r['edge_spearman']):.3f} "
            f"| {float(r['nvr']):.3f} | {float(r['relmeas_median_f64']):.2e} | {r['gates_passed']}/6 |"
        )
    return lines


def margin_table(run_dir: Path) -> list[str]:
    log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    buckets: dict[int, list[tuple[float, float]]] = {}
    for row in log:
        b = (int(row["step"]) - 1) // MARGIN_WINDOW
        buckets.setdefault(b, []).append((float(row["d_real_mean"]), float(row["d_fake_mean"])))
    lines = [
        "| steps | d_real (mean) | d_fake (mean) | margin | grad-norm anomalies |",
        "|---:|---:|---:|---:|:---|",
    ]
    for b in sorted(buckets):
        vals = buckets[b]
        dr = sum(v[0] for v in vals) / len(vals)
        df = sum(v[1] for v in vals) / len(vals)
        anomalies = [
            row["step"]
            for row in log
            if (int(row["step"]) - 1) // MARGIN_WINDOW == b
            and (
                not _finite(row["g_grad_norm"]) or not _finite(row["d_grad_norm"])
                or float(row["d_grad_norm"]) > 50
            )
        ]
        note = f"steps {anomalies}" if anomalies else ""
        lines.append(
            f"| {b * MARGIN_WINDOW + 1}-{(b + 1) * MARGIN_WINDOW} | {dr:+.3f} | {df:+.3f} "
            f"| {dr - df:+.3f} | {note} |"
        )
    return lines


def _finite(v) -> bool:
    try:
        f = float(v)
        return f == f and abs(f) != float("inf")
    except (TypeError, ValueError):
        return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = ["# Phase 3 pilot — three-arm report (scr5, K=4, 20000 steps)\n"]
    for run_id, omega in ARMS:
        run_dir = RUNS_ROOT / run_id
        if not (run_dir / "gate_trajectory.csv").exists():
            parts.append(f"## omega_adv = {omega} — MISSING ({run_dir})\n")
            continue
        summary = json.loads((run_dir / "smoke_summary.json").read_text(encoding="utf-8"))
        parts.append(f"## omega_adv = {omega} ({run_id})\n")
        parts.append(
            f"Final: **{summary['final_gates_passed']}/6 gates** — "
            + ", ".join(f"{k} {v}" for k, v in summary["final_gates"].items())
            + f"; collapse_detected={summary['collapse_detected']}; "
            f"roundtrip diff {summary['roundtrip_max_abs_diff']}\n"
        )
        parts.append("### Gate trajectory (fixed val N=128, K=8, seed-pinned)\n")
        parts.extend(trajectory_table(run_dir))
        parts.append("\n### Discriminator real/fake margin (window-averaged hinge logits)\n")
        parts.extend(margin_table(run_dir))
        parts.append("")
    report = "\n".join(parts) + "\n"
    out_path = OUT_DIR / "PHASE3_REPORT.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
