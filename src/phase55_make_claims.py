from __future__ import annotations

from .phase55_common import PHASE53C_ROOT, PHASE53D_ROOT, PHASE55_ROOT, classify_optional_gan, fmt, mean, read_csv_rows, to_float


def main() -> None:
    out = PHASE55_ROOT
    out.mkdir(parents=True, exist_ok=True)
    gan = read_csv_rows(PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "optional_gan_results.csv")
    decision, reason = classify_optional_gan(gan)
    ran = [r for r in gan if str(r.get("status", "")).startswith("ran")]
    psnr = mean([r.get("psnr") for r in ran])
    ssim = mean([r.get("ssim") for r in ran])
    rel = mean([r.get("rel_meas_error") for r in ran])
    optional = [
        "# Phase55 Optional GAN Decision",
        "",
        f"Decision: **{decision}**.",
        "",
        f"- Reason: {reason}.",
        f"- Available metrics: PSNR {fmt(psnr,3)}, SSIM {fmt(ssim,3)}, RelMeasErr {fmt(rel,5)}.",
        "- LPIPS/FID/KID/perceptual improvement metrics were not found.",
        "- The run is a Scr-5 pilot only; it should not be presented as the final method or main contribution.",
    ]
    (out / "optional_gan_decision.md").write_text("\n".join(optional) + "\n", encoding="utf-8")
    supported = [
        "# Phase55 Supported Claims",
        "",
        "- Image PSNR and measurement accountability are separable.",
        "- `Pi_y` acts as a measurement certificate / re-legalizer.",
        "- Post-hoc audit can greatly reduce RelMeasErr with negligible PSNR cost.",
        "- Soft `P_N^lambda` leaks row-space; exact `P0` is required for null-space critic diagnostics.",
        "- Feasible / near-feasible hallucination shows bucket consistency alone cannot certify semantic correctness.",
        "- Exact-null critic detects anchor-null compatibility only as a conditional claim after group-split audit confirms no memorization.",
        "- Posterior variation lies mostly in null/weakly measured directions only as supplement-level evidence from Session24.",
    ]
    (out / "PHASE55_SUPPORTED_CLAIMS.md").write_text("\n".join(supported) + "\n", encoding="utf-8")
    unsupported = [
        "# Phase55 Unsupported Or Risky Claims",
        "",
        "- `P_N` is necessary for train-time PSNR.",
        "- final audit is necessary for retrained PSNR.",
        "- GAN is the main final method.",
        "- discriminator is a certificate.",
        "- full measurement-conditioned D is novel if it sees residuals.",
        "- stronger BP PSNR implies stronger exact-null AUC.",
        "- Phase53C AUC 0.992 is claim-ready without saved group split / permutation-baseline audit.",
        "- Cross-feasible images are fully measurement-indistinguishable when their RelMeasErr remains higher than ours.",
    ]
    (out / "PHASE55_UNSUPPORTED_CLAIMS.md").write_text("\n".join(unsupported) + "\n", encoding="utf-8")
    main_plan = [
        "# Phase55 Main Text Insertion Plan",
        "",
        "Recommended main-text insertions:",
        "1. Measurement certificate paragraph: `Pi_y` is a post-hoc re-legalizer, not a train-time PSNR engine.",
        "2. Ablation sentence: PSNR and measurement accountability separate in Phase48/51A.",
        "3. Feasible hallucination figure only if caption states cross-feasible RelMeasErr is low but still higher than ours.",
        "",
        "Not recommended for main text yet:",
        "- Phase53C exact-null critic AUC 0.992, because split audit is unknown.",
        "- Optional GAN pilot, because it lacks perceptual metric evidence and is not the final method.",
    ]
    (out / "PHASE55_MAIN_TEXT_INSERTION_PLAN.md").write_text("\n".join(main_plan) + "\n", encoding="utf-8")
    supp_plan = [
        "# Phase55 Supplement Insertion Plan",
        "",
        "- Phase53C/53D cross-audit comparison table.",
        "- Exact P0 and soft leakage checks.",
        "- E1-mini local PCA/linear preflight table.",
        "- Shortcut and memorization-risk audit.",
        "- Posterior sampling maps and variance-null-ratio plot.",
        "- Optional GAN pilot as exploratory supplement only.",
    ]
    (out / "PHASE55_SUPPLEMENT_INSERTION_PLAN.md").write_text("\n".join(supp_plan) + "\n", encoding="utf-8")
    next_action = [
        "# Phase55 Next Action",
        "",
        "Decision: **Only run group-split exact-null critic repeat** before making any exact-null critic claim.",
        "",
        "Rationale:",
        "- Phase53C AUC 0.992 is strong but imported outputs do not include train/eval image-id group split indices.",
        "- Shuffled-label and random-anchor permutation baselines were not found.",
        "- Optional GAN lacks perceptual gain metrics, so do not continue GAN now.",
        "- Posthoc audit remains strong, so write the measurement certificate claim now.",
        "- Feasible hallucination is useful but should carry a caveat because cross RelMeasErr remains higher than ours.",
    ]
    (out / "PHASE55_NEXT_ACTION.md").write_text("\n".join(next_action) + "\n", encoding="utf-8")
    print(out / "PHASE55_NEXT_ACTION.md")


if __name__ == "__main__":
    main()

