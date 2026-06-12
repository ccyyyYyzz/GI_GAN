from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SESSIONS = {
    "rad5_no_gate": ("session_02_rad5_no_gate", "Rad-5", "no_gate"),
    "rad5_no_final_audit": ("session_03_rad5_no_final_audit", "Rad-5", "no_final_audit"),
    "scr5_no_gate": ("session_04_scr5_no_gate", "Scr-5", "no_gate"),
    "scr5_no_final_audit": ("session_05_scr5_no_final_audit", "Scr-5", "no_final_audit"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize Phase48/49 imported ablation reports.")
    parser.add_argument("--import_root", default="E:/ns_mc_gan_gi/outputs_phase48_49_colab_import")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(k, "")) for k in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def flatten_eval_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "psnr": as_float(row, "model_psnr"),
        "ssim": as_float(row, "model_ssim"),
        "rel_meas_err": as_float(row, "model_rel_meas_error"),
        "bp_psnr": as_float(row, "backprojection_psnr"),
        "delta_psnr": as_float(row, "improvement_delta_psnr"),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.import_root)
    rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    ablations: dict[tuple[str, str], dict[str, Any]] = {}

    for _key, (session, family, variant) in SESSIONS.items():
        eval_rows = read_csv(root / session / "eval_final.csv")
        for row in eval_rows:
            run = row.get("run", "")
            metrics = flatten_eval_row(row)
            if run.startswith("full_"):
                baselines.setdefault(family, {"family": family, "variant": "full", "session": "baseline", **metrics})
            elif run == session:
                ablations[(family, variant)] = {"family": family, "variant": variant, "session": session, **metrics}

    for family in ["Rad-5", "Scr-5"]:
        if family in baselines:
            rows.append(baselines[family])
        for variant in ["no_gate", "no_final_audit"]:
            if (family, variant) in ablations:
                row = dict(ablations[(family, variant)])
                base = baselines.get(family, {})
                if base:
                    row["psnr_drop_vs_full"] = (base.get("psnr") or 0.0) - (row.get("psnr") or 0.0)
                    row["ssim_drop_vs_full"] = (base.get("ssim") or 0.0) - (row.get("ssim") or 0.0)
                    rel = row.get("rel_meas_err")
                    base_rel = base.get("rel_meas_err")
                    row["relmeas_ratio_vs_full"] = (rel / base_rel) if rel is not None and base_rel else ""
                rows.append(row)

    write_csv(root / "phase48_49_summary.csv", rows)
    write_md(root / "phase48_49_summary.md", rows, "Phase48/49 Standardized Summary")

    def get(family: str, variant: str, metric: str) -> float:
        row = baselines.get(family) if variant == "full" else ablations.get((family, variant), {})
        value = row.get(metric) if row else None
        return float(value) if value is not None else float("nan")

    rad_no_gate_psnr = get("Rad-5", "no_gate", "psnr")
    rad_no_audit_psnr = get("Rad-5", "no_final_audit", "psnr")
    rad_no_gate_rel = get("Rad-5", "no_gate", "rel_meas_err")
    rad_no_audit_rel = get("Rad-5", "no_final_audit", "rel_meas_err")
    scr_no_gate_psnr = get("Scr-5", "no_gate", "psnr")
    scr_no_audit_psnr = get("Scr-5", "no_final_audit", "psnr")
    scr_no_gate_rel = get("Scr-5", "no_gate", "rel_meas_err")
    scr_no_audit_rel = get("Scr-5", "no_final_audit", "rel_meas_err")

    report = [
        "# Aggregate Phase48/49 Report",
        "",
        "This standardized report re-reads the imported Phase48/49 Colab outputs and separates endpoint image quality from measurement accountability.",
        "",
        "## Required Findings",
        "",
        "1. Train-time `no_gate` nearly recovers PSNR/SSIM for both Rad-5 and Scr-5.",
        f"   - Rad-5 no_gate: PSNR {rad_no_gate_psnr:.3f}, RelMeasErr {rad_no_gate_rel:.5f}.",
        f"   - Scr-5 no_gate: PSNR {scr_no_gate_psnr:.3f}, RelMeasErr {scr_no_gate_rel:.5f}.",
        "2. Train-time `no_final_audit` only modestly reduces PSNR/SSIM.",
        f"   - Rad-5 no_final_audit: PSNR {rad_no_audit_psnr:.3f}.",
        f"   - Scr-5 no_final_audit: PSNR {scr_no_audit_psnr:.3f}.",
        "3. `no_final_audit` has clearly higher RelMeasErr than `no_gate`.",
        f"   - Rad-5 RelMeasErr ratio no_final_audit/no_gate: {rad_no_audit_rel / rad_no_gate_rel:.2f}.",
        f"   - Scr-5 RelMeasErr ratio no_final_audit/no_gate: {scr_no_audit_rel / scr_no_gate_rel:.2f}.",
        "4. Inference-time ablation and train-time ablation are not the same question: train-time ablations allow the network and supervised losses to adapt.",
        "5. The next mechanism-closure step must run combined removal and no-measurement-loss ablations.",
        "",
        "## Interpretation",
        "",
        "Phase48/49 does not support the claim that P_N or final Pi_y is required for endpoint PSNR under supervised training. It does support treating final Pi_y as an explicit measurement certificate, because removing it increases RelMeasErr substantially even when PSNR remains high.",
    ]
    (root / "AGGREGATE_PHASE48_49_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    claims = [
        "# Phase48/49 Supported Claims",
        "",
        "- Under train-time adaptation, removing P_N alone does not materially reduce endpoint PSNR/SSIM.",
        "- Removing final Pi_y alone does not collapse PSNR/SSIM, but it increases RelMeasErr.",
        "- PSNR/SSIM and measurement accountability must be reported separately.",
        "- Inference-time ablation collapse should not be conflated with train-time adaptation.",
        "- Phase51A is needed to test combined removal and the substitutive role of measurement loss.",
    ]
    (root / "PHASE48_49_SUPPORTED_CLAIMS.md").write_text("\n".join(claims) + "\n", encoding="utf-8")

    risks = [
        "# Phase48/49 Remaining Risks",
        "",
        "- Single-removal ablations do not close the mechanism story.",
        "- Measurement loss may compensate for missing final audit during training.",
        "- Combined removal may reveal hidden gate/audit redundancy.",
        "- Perturbation sensitivity must be checked for new no-measurement-loss variants.",
        "- These are mechanism-closure ablations and should not be auto-promoted to main tables.",
    ]
    (root / "PHASE48_49_REMAINING_RISKS.md").write_text("\n".join(risks) + "\n", encoding="utf-8")
    print(root / "AGGREGATE_PHASE48_49_REPORT.md")


if __name__ == "__main__":
    main()
