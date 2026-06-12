from __future__ import annotations

from .phase55_common import PHASE55_ROOT, fmt, mean, max_value, read_csv_rows, read_text, to_float


def find_row(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    return next((r for r in rows if r.get("row") == name), {})


def main() -> None:
    out = PHASE55_ROOT
    comp = read_csv_rows(out / "phase53C_vs_53D_comparison.csv")
    split = read_csv_rows(out / "split_leakage_audit.csv")
    gan_decision = read_text(out / "optional_gan_decision.md")
    max_auc_row = find_row(comp, "max AUC")
    posthoc_row = find_row(comp, "posthoc audit RelMeasErr recovery")
    feas_row = find_row(comp, "feasible hallucination RelMeasErr")
    split_risk = split[0].get("risk", "unknown") if split else "unknown"
    optional_level = "exploratory/supplement only"
    if "Decision: **supplement only**" in gan_decision:
        optional_level = "supplement only"
    if "Decision: **do not cite**" in gan_decision:
        optional_level = "do not cite"
    report = [
        "# Phase55 Cross-Audit and Paper Decision",
        "",
        "Scope: read-only cross-audit of Phase53C, Phase53D, Phase48/49, and Phase51A outputs. No model training was run.",
        "",
        "## Executive Decision",
        "",
        "- Phase53C and Phase53D are partially consistent: both support exact P0 / soft-leakage diagnostics and feasible ambiguity, but their AUC patterns differ because Phase53C uses deep critics while Phase53D uses local PCA/linear CPU classifiers.",
        f"- Phase53C max AUC: {max_auc_row.get('Phase53C', 'n/a')}; Phase53D max AUC: {max_auc_row.get('Phase53D', 'n/a')}.",
        f"- AUC 0.992 does **not** pass strict split/leakage audit yet: `{split_risk}`.",
        f"- Optional GAN decision: {optional_level}.",
        "- Main text now can use the measurement certificate / re-legalization claim and the PSNR-accountability separation.",
        "- Exact-null critic should be held to supplement/provisional status until group-split repeat is done.",
        "",
        "## Key Evidence",
        "",
        f"- Feasible hallucination: {feas_row.get('Phase53C', 'n/a')} in Phase53C; {feas_row.get('Phase53D', 'n/a')} in Phase53D.",
        f"- Posthoc certificate: {posthoc_row.get('Phase53D', 'n/a')} in Phase53D.",
        "- Baseline and split audit: no saved Phase53C image-id group split or shuffled-label/random-anchor baseline was found.",
        "- Optional GAN lacks LPIPS/FID/KID/perceptual improvement metrics and is Scr-5 pilot only.",
        "",
        "## Required Final Answers",
        "",
        "1. Phase55 output directory: `E:/ns_mc_gan_gi/outputs_phase55_cross_audit`.",
        "2. Phase53C vs Phase53D consistency: partially consistent, but AUC gap must be explained by model class and split-risk differences.",
        f"3. AUC 0.992 split/leakage audit: not passed; status `{split_risk}`.",
        f"4. Optional GAN citation: {optional_level}; not main contribution.",
        "5. Feasible hallucination: main-text candidate with caveat that cross RelMeasErr remains higher than ours.",
        "6. Posterior sampling: supplement only.",
        "7. Supported/unsupported claims are listed in `PHASE55_SUPPORTED_CLAIMS.md` and `PHASE55_UNSUPPORTED_CLAIMS.md`.",
        "8. Next experiment: only group-split exact-null critic repeat if you want to claim critic signal.",
        "9. No training was run.",
    ]
    (out / "PHASE55_CROSS_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "PHASE55_CROSS_AUDIT_REPORT.md")


if __name__ == "__main__":
    main()

