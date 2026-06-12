from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SESSIONS = {
    "session_20_blind_critic_pretest": {
        "type": "blind_critic_pretest",
        "csv": "blind_critic_pretest_results.csv",
        "question": "Does blind critic distinguish audited cross-pair feasible hallucinations?",
    },
    "session_21_shortcut_audit": {
        "type": "shortcut_audit",
        "csv": "shortcut_audit_results.csv",
        "question": "Does full-input D use residual shortcuts?",
    },
    "session_22_feasible_hallucination_dataset": {
        "type": "feasible_hallucination_dataset",
        "csv": "feasible_hallucination_metrics.csv",
        "question": "Does feasible hallucination figure show physical indistinguishability?",
    },
    "session_23_blind_critic_gan_pilot": {
        "type": "blind_critic_gan_pilot",
        "csv": "blind_critic_gan_results.csv",
        "question": "Does blind critic GAN improve perceptual/null-space plausibility at controlled PSNR loss?",
    },
    "session_24_posterior_sampling_pilot": {
        "type": "posterior_sampling_pilot",
        "csv": "posterior_sampling_metrics.csv",
        "question": "Does posterior sampling diversity lie in null-space?",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate imported Phase53B Colab outputs.")
    parser.add_argument("--import_root", default="E:/ns_mc_gan_gi/outputs_phase53B_blind_null_critic_import")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
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


def mean(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else ""


def max_val(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def fmt(value: Any, digits: int = 3) -> str:
    v = to_float(value)
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def summarize_session(import_root: Path, session: str, meta: dict[str, str]) -> dict[str, Any]:
    session_dir = import_root / session
    status = read_json(session_dir / "SESSION_STATUS.json")
    rows = read_csv(session_dir / meta["csv"])
    row: dict[str, Any] = {
        "session": session,
        "type": meta["type"],
        "status": "ok" if status.get("ok") else ("missing" if not session_dir.exists() else "check"),
        "question": meta["question"],
        "rows": len(rows),
    }
    if meta["type"] == "blind_critic_pretest":
        row.update({"mean_auc": mean(rows, "auc"), "max_auc": max_val(rows, "auc")})
    elif meta["type"] == "shortcut_audit":
        full = [r for r in rows if r.get("model") == "D_full"]
        blind = [r for r in rows if r.get("model") == "D_blind"]
        row.update({"D_full_mean_eval_auc": mean(full, "eval_auc"), "D_blind_mean_eval_auc": mean(blind, "eval_auc")})
    elif meta["type"] == "feasible_hallucination_dataset":
        row.update({"mean_cross_relmeas": mean(rows, "cross_relmeas"), "mean_ours_relmeas": mean(rows, "ours_relmeas")})
    elif meta["type"] == "blind_critic_gan_pilot":
        row.update({"mean_psnr": mean(rows, "psnr"), "mean_relmeas": mean(rows, "rel_meas_error")})
    elif meta["type"] == "posterior_sampling_pilot":
        row.update({"mean_variance_nullspace_ratio": mean(rows, "variance_nullspace_ratio_mean"), "mean_coverage": mean(rows, "coverage_fraction")})
    return row


def build_reports(summary_rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str], list[str]]:
    by_type = {row["type"]: row for row in summary_rows}
    all_ok = all(row.get("status") == "ok" for row in summary_rows)
    pre = by_type.get("blind_critic_pretest", {})
    shortcut = by_type.get("shortcut_audit", {})
    fig = by_type.get("feasible_hallucination_dataset", {})
    gan = by_type.get("blind_critic_gan_pilot", {})
    post = by_type.get("posterior_sampling_pilot", {})
    aggregate = [
        "# Phase53B Aggregate Report",
        "",
        f"- All five sessions imported and status OK: {'yes' if all_ok else 'no'}.",
        "- Framing: certify the measured subspace analytically; criticize only the unmeasured/null-space completion.",
        "- All Phase53B results are exploratory / innovation screening unless explicitly promoted later.",
        "",
        "## Question Answers",
        "",
        f"1. Blind critic separability: max AUC {fmt(pre.get('max_auc'))}, mean AUC {fmt(pre.get('mean_auc'))}.",
        f"2. Shortcut audit: D_full mean eval AUC {fmt(shortcut.get('D_full_mean_eval_auc'))}; D_blind mean eval AUC {fmt(shortcut.get('D_blind_mean_eval_auc'))}.",
        f"3. Feasible hallucination: mean cross-feasible RelMeasErr {fmt(fig.get('mean_cross_relmeas'), 5)}, mean ours RelMeasErr {fmt(fig.get('mean_ours_relmeas'), 5)}.",
        f"4. Blind critic GAN pilot: mean PSNR {fmt(gan.get('mean_psnr'))}, mean RelMeasErr {fmt(gan.get('mean_relmeas'), 5)}.",
        f"5. Posterior sampling: mean variance measurement visibility {fmt(post.get('mean_variance_nullspace_ratio'), 5)}, mean coverage {fmt(post.get('mean_coverage'))}.",
        "",
        "## Interpretation Rules",
        "",
        "- D is a critic / learned plausibility test, not a certificate.",
        "- Analytic Pi_y is the measurement certificate.",
        "- Do not claim GAN improves PSNR or RelMeasErr unless the pilot directly proves it.",
        "- Full-input D belongs only to shortcut diagnostics because it can exploit residual features.",
    ]
    supported = [
        "# Phase53B Supported New Claims",
        "",
        "- The proposed certified-blind critic framing cleanly separates analytic measurement certification from learned null-space plausibility testing.",
        "- Feasible hallucination is a physically meaningful diagnostic: row-space consistency alone cannot validate null-space semantics.",
        "- If imported metrics support it, blind critic separability can be reported as an exploratory supplement.",
    ]
    failed = [
        "# Phase53B Failed Or Forbidden Claims",
        "",
        "- Do not claim an adversarial certificate.",
        "- Do not claim D proves measurement consistency.",
        "- Do not claim final main results are GAN-trained.",
        "- Do not claim GAN improves PSNR, RelMeasErr, FID, KID, or LPIPS without direct evidence.",
        "- Do not use full-input residual-aware D as the main method.",
    ]
    pivot = [
        "# Phase53B Paper Pivot Plan",
        "",
        "Recommended framing: measurement-audited neural completion with an optional exploratory certified-blind null-space critic.",
        "",
        "Main text can use the theory note and feasible hallucination figure if the figure is clear.",
        "Critic/GAN results should remain supplement unless Session20 and Session23 show strong, stable benefits.",
    ]
    return aggregate, supported, failed, pivot


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root)
    import_root.mkdir(parents=True, exist_ok=True)
    summary_rows = [summarize_session(import_root, session, meta) for session, meta in SESSIONS.items()]
    write_csv(import_root / "phase53B_summary.csv", summary_rows)
    write_md(import_root / "phase53B_summary.md", summary_rows, "Phase53B Summary")
    aggregate, supported, failed, pivot = build_reports(summary_rows)
    (import_root / "PHASE53B_AGGREGATE_REPORT.md").write_text("\n".join(aggregate) + "\n", encoding="utf-8")
    (import_root / "PHASE53B_SUPPORTED_NEW_CLAIMS.md").write_text("\n".join(supported) + "\n", encoding="utf-8")
    (import_root / "PHASE53B_FAILED_CLAIMS.md").write_text("\n".join(failed) + "\n", encoding="utf-8")
    (import_root / "PHASE53B_PAPER_PIVOT_PLAN.md").write_text("\n".join(pivot) + "\n", encoding="utf-8")
    print(import_root / "PHASE53B_AGGREGATE_REPORT.md")


if __name__ == "__main__":
    main()
