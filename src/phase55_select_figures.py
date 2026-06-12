from __future__ import annotations

from pathlib import Path

from .phase55_common import PHASE53C_ROOT, PHASE53D_ROOT, PHASE55_ROOT, copy_if_exists, write_rows


def main() -> None:
    out = PHASE55_ROOT
    fig_dir = out / "figures_selected"
    rows: list[dict] = []

    def add(source: Path, name: str, recommendation: str, reason: str, caveat: str) -> None:
        dest = copy_if_exists(source, fig_dir, name)
        rows.append(
            {
                "figure_path": str(dest) if dest else str(source),
                "exists": bool(dest),
                "main_or_supp": recommendation,
                "reason": reason,
                "known_caveat": caveat,
                "source_path": str(source),
            }
        )

    add(
        PHASE53C_ROOT / "session_22_feasible_hallucination_figure" / "feasible_hallucination_grid.png",
        "phase53C_feasible_hallucination_grid.png",
        "main candidate",
        "Shows same-bucket ambiguity and motivates null-space plausibility.",
        "Cross RelMeasErr is low but still higher than ours; caption must state this.",
    )
    add(
        PHASE53D_ROOT / "feasible_hallucination_grid.png",
        "phase53D_feasible_hallucination_grid.png",
        "supplement",
        "Independent local reproduction of feasible hallucination diagnostic.",
        "Small local sample count.",
    )
    add(
        PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "uncertainty_maps.png",
        "phase53C_posterior_uncertainty_maps.png",
        "supplement",
        "Visualizes posterior variation / uncertainty maps.",
        "Posterior sampling is exploratory and not the main method.",
    )
    add(
        PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "sample_grid.png",
        "phase53C_posterior_sample_grid.png",
        "supplement",
        "Shows posterior sample diversity.",
        "No perceptual metric gain claim.",
    )
    add(
        PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "variance_null_ratio.png",
        "phase53C_variance_null_ratio.png",
        "supplement",
        "Supports null/weakly measured posterior variation as a diagnostic.",
        "Do not overclaim without broader posterior validation.",
    )
    add(
        PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / "auc_by_family.png",
        "phase53C_auc_by_family.png",
        "supplement",
        "Deep exact-null critic AUC by family.",
        "Requires group-split repeat before claim-ready use.",
    )
    add(
        PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / "infoNCE_mi_by_family.png",
        "phase53C_infoNCE_mi_by_family.png",
        "supplement",
        "MI lower-bound diagnostic by family.",
        "Critic split leakage risk unresolved.",
    )
    add(
        PHASE53C_ROOT / "session_21_soft_leakage_and_shortcut_audit" / "leakage_factor_by_lambda.png",
        "phase53C_soft_leakage_by_lambda.png",
        "supplement",
        "Demonstrates soft P_N leakage.",
        "Diagnostic only; exact P0 is required for critic inputs.",
    )
    add(
        PHASE53D_ROOT / "posthoc_lambda_tradeoff.png",
        "phase53D_posthoc_certificate_tradeoff.png",
        "main candidate",
        "Shows posthoc certificate tradeoff and small PSNR cost.",
        "Local diagnostic sample size; pair with table.",
    )
    add(
        PHASE53D_ROOT / "posthoc_psnr_vs_relmeaserr.png",
        "phase53D_posthoc_psnr_vs_relmeaserr.png",
        "main/supp candidate",
        "Supports PSNR vs measurement accountability separation.",
        "Use with Phase48/51A ablation matrix.",
    )
    write_rows(out, "figure_inventory", rows, "Phase55 Figure Inventory")
    print(out / "figure_inventory.md")


if __name__ == "__main__":
    main()

