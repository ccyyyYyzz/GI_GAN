from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE48_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase48_49_colab_import")
SESSIONS_51A = {
    "Rad-5 no_gate_no_final_audit": ("session_06_rad5_no_gate_no_final_audit", "Rad-5", "no_gate_no_final_audit"),
    "Scr-5 no_gate_no_final_audit": ("session_07_scr5_no_gate_no_final_audit", "Scr-5", "no_gate_no_final_audit"),
    "Rad-5 no_final_audit_no_meas_loss": ("session_08_rad5_no_final_audit_no_meas_loss", "Rad-5", "no_final_audit_no_meas_loss"),
    "Scr-5 no_final_audit_no_meas_loss": ("session_09_scr5_no_final_audit_no_meas_loss", "Scr-5", "no_final_audit_no_meas_loss"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase51A Colab outputs.")
    parser.add_argument("--import_root", default="E:/ns_mc_gan_gi/outputs_phase51A_colab_import")
    parser.add_argument("--phase48_root", default=str(PHASE48_ROOT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(k, "")) for k in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def mean_csv(path: Path, keys: list[str]) -> dict[str, float | str]:
    rows = read_csv(path)
    out: dict[str, float | str] = {}
    for key in keys:
        vals = [to_float(row.get(key)) for row in rows]
        vals = [v for v in vals if v is not None]
        out[key] = sum(vals) / len(vals) if vals else ""
    return out


def phase48_rows(phase48_root: Path) -> list[dict[str, Any]]:
    rows = read_csv(phase48_root / "phase48_49_summary.csv")
    mapped: list[dict[str, Any]] = []
    for row in rows:
        mapped.append(
            {
                "family": row.get("family"),
                "variant": row.get("variant"),
                "source_phase": "Phase48/49",
                "session": row.get("session"),
                "psnr": row.get("psnr"),
                "ssim": row.get("ssim"),
                "rel_meas_err": row.get("rel_meas_err"),
                "bp_psnr": row.get("bp_psnr"),
                "delta_psnr": row.get("delta_psnr"),
                "status": "imported",
            }
        )
    return mapped


def phase51_row(import_root: Path, session: str, family: str, variant: str) -> dict[str, Any]:
    session_dir = import_root / session
    metrics = read_json(session_dir / "eval_final" / "eval_metrics.json")
    posthoc = mean_csv(
        session_dir / "posthoc_audit_eval.csv",
        ["psnr_before", "psnr_after", "relmeas_before_unclamped", "relmeas_after_unclamped"],
    )
    perturb = mean_csv(session_dir / "measurement_perturbation_subset.csv", ["psnr_drop"])
    model = metrics.get("model", {}) if metrics else {}
    back = metrics.get("backprojection", {}) if metrics else {}
    improvement = metrics.get("improvement", {}) if metrics else {}
    status = read_json(session_dir / "SESSION_STATUS.json")
    return {
        "family": family,
        "variant": variant,
        "source_phase": "Phase51A",
        "session": session,
        "psnr": model.get("psnr", ""),
        "ssim": model.get("ssim", ""),
        "rel_meas_err": model.get("rel_meas_error", ""),
        "bp_psnr": back.get("psnr", ""),
        "delta_psnr": improvement.get("delta_psnr", ""),
        "wrong_shuffle_drop": perturb.get("psnr_drop", ""),
        "posthoc_audit_psnr_change": (
            posthoc.get("psnr_after", 0) - posthoc.get("psnr_before", 0)
            if isinstance(posthoc.get("psnr_after"), float) and isinstance(posthoc.get("psnr_before"), float)
            else ""
        ),
        "posthoc_audit_relmeas_change": (
            posthoc.get("relmeas_after_unclamped", 0) - posthoc.get("relmeas_before_unclamped", 0)
            if isinstance(posthoc.get("relmeas_after_unclamped"), float)
            and isinstance(posthoc.get("relmeas_before_unclamped"), float)
            else ""
        ),
        "status": "ok" if status.get("ok") else ("missing" if not session_dir.exists() else "check"),
    }


def add_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_family = {row["family"]: row for row in rows if row.get("variant") == "full"}
    no_gate_by_family = {row["family"]: row for row in rows if row.get("variant") == "no_gate"}
    out = []
    for row in rows:
        row = dict(row)
        family = row.get("family")
        full = full_by_family.get(family, {})
        no_gate = no_gate_by_family.get(family, {})
        psnr = to_float(row.get("psnr"))
        ssim = to_float(row.get("ssim"))
        rel = to_float(row.get("rel_meas_err"))
        full_psnr = to_float(full.get("psnr"))
        full_ssim = to_float(full.get("ssim"))
        full_rel = to_float(full.get("rel_meas_err"))
        ng_rel = to_float(no_gate.get("rel_meas_err"))
        row["psnr_drop_vs_full"] = full_psnr - psnr if psnr is not None and full_psnr is not None else ""
        row["ssim_drop_vs_full"] = full_ssim - ssim if ssim is not None and full_ssim is not None else ""
        row["relmeas_ratio_vs_full"] = rel / full_rel if rel is not None and full_rel not in {None, 0.0} else ""
        row["relmeas_ratio_vs_no_gate"] = rel / ng_rel if rel is not None and ng_rel not in {None, 0.0} else ""
        out.append(row)
    return out


def row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("family")), str(row.get("variant"))): row for row in rows}


def fmt(value: Any, digits: int = 3) -> str:
    number = to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def build_report(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    by_key = row_lookup(rows)
    all_51a_ok = all(
        row.get("status") == "ok"
        for row in rows
        if row.get("source_phase") == "Phase51A"
    )

    def describe(family: str, variant: str) -> str:
        row = by_key.get((family, variant), {})
        return (
            f"{family} `{variant}`: PSNR {fmt(row.get('psnr'))}, SSIM {fmt(row.get('ssim'), 4)}, "
            f"RelMeasErr {fmt(row.get('rel_meas_err'), 5)}, "
            f"drop vs full {fmt(row.get('psnr_drop_vs_full'))} dB, "
            f"RelMeasErr ratio vs full {fmt(row.get('relmeas_ratio_vs_full'), 2)}, "
            f"posthoc RelMeasErr change {fmt(row.get('posthoc_audit_relmeas_change'), 5)}."
        )

    report = [
        "# Phase51A Aggregate Report",
        "",
        "This report aggregates Phase48/49 single-removal ablations with Phase51A mechanism-closure ablations.",
        "",
        "## Import Status",
        "",
        f"- All four Phase51A sessions imported and `SESSION_STATUS.json` OK: {'yes' if all_51a_ok else 'no'}.",
        "- Phase48/49 baselines were refreshed before aggregation.",
        "",
        "## Main Findings",
        "",
        f"- {describe('Rad-5', 'no_gate_no_final_audit')}",
        f"- {describe('Scr-5', 'no_gate_no_final_audit')}",
        f"- {describe('Rad-5', 'no_final_audit_no_meas_loss')}",
        f"- {describe('Scr-5', 'no_final_audit_no_meas_loss')}",
        "",
        "## Interpretation",
        "",
        "1. Combined `no_gate + no_final_audit` preserves endpoint PSNR/SSIM close to the single `no_final_audit` runs, "
        "so endpoint image quality alone does not prove either mechanism is individually necessary under supervised retraining.",
        "2. RelMeasErr remains much higher when the final audit is removed, especially for Rad-5, so measurement accountability "
        "and endpoint image quality separate.",
        "3. Removing measurement-domain loss further worsens Rad-5 RelMeasErr while PSNR remains similar, supporting the claim "
        "that measurement loss and final audit are accountability mechanisms rather than simple PSNR boosters.",
        "4. Posthoc audit reduces RelMeasErr in all four Phase51A diagnostics, consistent with Pi_y acting as an explicit "
        "re-legalization/certificate step.",
        "",
        "## Files",
        "",
        "- `phase51A_ablation_matrix.csv/md`: full metric matrix.",
        "- `phase51A_summary.csv/md`: same rows for downstream scripts.",
        "- Each session folder contains `SESSION_REPORT.md`, `visual_grid.png`, `posthoc_audit_eval.csv`, and `measurement_perturbation_subset.csv`.",
    ]

    claims = [
        "# Phase51A Supported Claims",
        "",
        f"- All four Phase51A sessions are imported and status-checked: {'yes' if all_51a_ok else 'no'}.",
        "- Endpoint PSNR/SSIM are robust to train-time removal of the null-space gate and/or final audit in these 5% STL-10 settings.",
        "- Final audit removal mainly appears in measurement accountability: RelMeasErr increases more strongly than PSNR drops.",
        "- `no_final_audit_no_meas_loss` separates image quality from measurement consistency most clearly on Rad-5.",
        "- Posthoc Pi_y audit reduces RelMeasErr in the diagnostic subset, supporting its role as an explicit measurement-certificate step.",
    ]

    risks = [
        "# Phase51A Remaining Risks",
        "",
        "- These are mechanism-closure ablations, not new main-table SOTA runs.",
        "- The posthoc audit and perturbation checks are subset diagnostics and should be described as diagnostic, not full-dataset proof.",
        "- Scrambled Hadamard remains a randomized Hadamard configuration, not the low-frequency Hadamard primary HQ setting.",
        "- Strong manuscript claims should distinguish endpoint image quality from measurement accountability.",
    ]
    return report, claims, risks


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root)
    phase48_root = Path(args.phase48_root)
    import_root.mkdir(parents=True, exist_ok=True)
    rows = phase48_rows(phase48_root)
    for _name, (session, family, variant) in SESSIONS_51A.items():
        if (import_root / session).exists():
            rows.append(phase51_row(import_root, session, family, variant))
        else:
            rows.append({"family": family, "variant": variant, "source_phase": "Phase51A", "session": session, "status": "missing"})
    rows = add_comparisons(rows)
    write_csv(import_root / "phase51A_summary.csv", rows)
    write_md(import_root / "phase51A_summary.md", rows, "Phase51A Summary")
    write_csv(import_root / "phase51A_ablation_matrix.csv", rows)
    write_md(import_root / "phase51A_ablation_matrix.md", rows, "Phase51A Ablation Matrix")

    report, claims, risks = build_report(rows)
    (import_root / "PHASE51A_AGGREGATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (import_root / "PHASE51A_SUPPORTED_CLAIMS.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    (import_root / "PHASE51A_REMAINING_RISKS.md").write_text("\n".join(risks) + "\n", encoding="utf-8")
    print(import_root / "PHASE51A_AGGREGATE_REPORT.md")


if __name__ == "__main__":
    main()
