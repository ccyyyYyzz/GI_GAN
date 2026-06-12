from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SESSIONS = {
    "session_20_exact_null_mi_pretest": ("exact_null_mi_pretest", "exact_null_mi_pretest_results.csv"),
    "session_21_soft_leakage_and_shortcut_audit": ("soft_leakage_and_shortcut_audit", "shortcut_audit_results.csv"),
    "session_22_feasible_hallucination_figure": ("feasible_hallucination_figure", "feasible_hallucination_metrics.csv"),
    "session_23_exact_null_critic_evaluator": ("exact_null_critic_evaluator", "critic_evaluator_scores.csv"),
    "session_24_optional_gan_and_posterior_sampling": ("optional_gan_and_posterior_sampling", "posterior_sampling_metrics.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase53C Colab outputs.")
    parser.add_argument("--import_root", default="E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def mean(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else ""


def max_value(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
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


def fmt(value: Any, digits: int = 3) -> str:
    v = to_float(value)
    return "n/a" if v is None else f"{v:.{digits}f}"


def summarize(import_root: Path, session: str, kind: str, csv_name: str) -> dict[str, Any]:
    session_dir = import_root / session
    rows = read_csv(session_dir / csv_name)
    status = read_json(session_dir / "SESSION_STATUS.json")
    out: dict[str, Any] = {
        "session": session,
        "kind": kind,
        "status": "ok" if status.get("ok") else ("missing" if not session_dir.exists() else "check"),
        "rows": len(rows),
    }
    if kind == "exact_null_mi_pretest":
        out.update({"max_auc": max_value(rows, "auc"), "mean_auc": mean(rows, "auc"), "mean_mi_nats": mean(rows, "infoNCE_mi_lower_nats")})
    elif kind == "soft_leakage_and_shortcut_audit":
        leak = read_csv(session_dir / "soft_leakage_results.csv")
        out.update({"mean_eval_auc": mean(rows, "eval_auc"), "max_soft_leakage": max_value(leak, "mean_leakage_ratio"), "max_recover_Au_R2": max_value(leak, "recover_Au_R2")})
    elif kind == "feasible_hallucination_figure":
        out.update({"mean_cross_relmeas": mean(rows, "cross_relmeas"), "mean_ours_relmeas": mean(rows, "ours_relmeas")})
    elif kind == "exact_null_critic_evaluator":
        out.update({"mean_critic_score": mean(rows, "critic_score_mean"), "mean_relmeas": mean(rows, "rel_meas_err")})
    elif kind == "optional_gan_and_posterior_sampling":
        gan = read_csv(session_dir / "optional_gan_results.csv")
        out.update({"mean_variance_null_ratio": mean(rows, "variance_null_ratio_mean"), "gan_status_rows": ";".join(sorted(set(r.get("status", "") for r in gan)))})
    return out


def build_reports(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str], list[str]]:
    by_kind = {row["kind"]: row for row in rows}
    all_ok = all(row.get("status") == "ok" for row in rows)
    e1 = by_kind.get("exact_null_mi_pretest", {})
    e2 = by_kind.get("soft_leakage_and_shortcut_audit", {})
    e3 = by_kind.get("feasible_hallucination_figure", {})
    e4 = by_kind.get("exact_null_critic_evaluator", {})
    e5 = by_kind.get("optional_gan_and_posterior_sampling", {})
    aggregate = [
        "# Phase53C Aggregate Report",
        "",
        f"- All five sessions imported and status OK: {'yes' if all_ok else 'no'}.",
        "- All outputs are exploratory / innovation screening.",
        "",
        "## Required Question Answers",
        "",
        f"1. Exact-null critic anchor-null dependence: max AUC {fmt(e1.get('max_auc'))}, mean MI lower bound {fmt(e1.get('mean_mi_nats'))}.",
        "2. Scr > Rad and 10% > 5% trend: inspect `exact_null_mi_pretest_results.csv` by family/sampling.",
        f"3. Soft P_N leakage: max mean leakage {fmt(e2.get('max_soft_leakage'), 5)}, max recover-Au R2 {fmt(e2.get('max_recover_Au_R2'))}.",
        f"4. Full-input D shortcut: mean shortcut eval AUC {fmt(e2.get('mean_eval_auc'))}.",
        f"5. Feasible hallucination ambiguity: cross-feasible mean RelMeasErr {fmt(e3.get('mean_cross_relmeas'), 5)}.",
        "6. Critic vs RelMeasErr: inspect `critic_evaluator_scores.csv` for feasible hallucinations.",
        f"7. Optional GAN status: {e5.get('gan_status_rows', 'n/a')}.",
        f"8. Posterior variance null ratio: {fmt(e5.get('mean_variance_null_ratio'), 5)}.",
        "9. Main-text candidates: theory notes and feasible hallucination figure if visually clear.",
        "10. Supplement candidates: exact-null critic AUC/MI and shortcut audit unless very strong.",
        "11. Old gate/audit PSNR-essential claims should remain abandoned.",
        "12. Pivot: anchor information law + exact-null critic is a possible supplement, not yet a main headline.",
        "13. Next step: sampling-sweep E1 extension only if Session20 supports nontrivial AUC/MI.",
    ]
    supported = [
        "# Phase53C Supported New Claims",
        "",
        "- Analytic Pi_y is the measurement certificate; D is only a learned plausibility critic.",
        "- Exact P0 critic input is algebraically blind to row-space residual under the projector checks.",
        "- Feasible hallucination diagnostics can illustrate same-bucket ambiguity.",
        "- Anchor information law provides a falsifiable explanation for weak/strong critic performance.",
    ]
    failed = [
        "# Phase53C Failed Or Forbidden Claims",
        "",
        "- Do not claim adversarial certificate.",
        "- Do not claim D proves measurement consistency or improves RelMeasErr.",
        "- Do not claim final main results are GAN-trained.",
        "- Do not claim P_N or Pi_y are PSNR-essential; train-time ablations contradict that.",
        "- Do not use soft P_N^lambda as exact-blind critic input.",
    ]
    pivot = [
        "# Phase53C Paper Pivot Plan",
        "",
        "Keep the strict no-leak main results unchanged.",
        "Use Phase53C as exploratory support for a two-track story: analytic measurement certification plus learned null-space plausibility.",
        "Promote only the theory note and feasible hallucination figure unless the AUC/MI and evaluator results are strong.",
    ]
    return aggregate, supported, failed, pivot


def main() -> None:
    args = parse_args()
    root = Path(args.import_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [summarize(root, s, kind, csv_name) for s, (kind, csv_name) in SESSIONS.items()]
    write_csv(root / "phase53C_summary.csv", rows)
    write_md(root / "phase53C_summary.md", rows, "Phase53C Summary")
    aggregate, supported, failed, pivot = build_reports(rows)
    (root / "PHASE53C_AGGREGATE_REPORT.md").write_text("\n".join(aggregate) + "\n", encoding="utf-8")
    (root / "PHASE53C_SUPPORTED_NEW_CLAIMS.md").write_text("\n".join(supported) + "\n", encoding="utf-8")
    (root / "PHASE53C_FAILED_CLAIMS.md").write_text("\n".join(failed) + "\n", encoding="utf-8")
    (root / "PHASE53C_PAPER_PIVOT_PLAN.md").write_text("\n".join(pivot) + "\n", encoding="utf-8")
    print(root / "PHASE53C_AGGREGATE_REPORT.md")


if __name__ == "__main__":
    main()
