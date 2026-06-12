from __future__ import annotations

import json

from .phase15_common import write_csv
from .phase16_common import PHASE16, PHASE15_SUPP, copy_to_legacy, ensure_dir, registry_rows


OUT = PHASE16 / "_report"
SUPPORTED = OUT / "PHASE16_SUPPORTED_CLAIMS.md"
SUPPORTED_CSV = OUT / "phase16_supported_claims.csv"
UNSUPPORTED_CSV = OUT / "phase16_unsupported_claims.csv"


SUPPORTED_ROWS = [
    {
        "claim": "All Phase15 main-table numbers are strict no-leak imported results.",
        "support": "Phase15 noleak_registry.csv plus Phase16 aggregate status.",
        "recommended_wording": "We report strict no-leak reconstructions evaluated on held-out test data.",
        "caveat": "Only use rows from imported_noleak / noleak_registry, not old exploratory runs.",
    },
    {
        "claim": "Rademacher exact-A re-evaluation is now safe.",
        "support": "Phase15R cache rebuild patch and Phase16 safe_exactA / exactA_reeval audit.",
        "recommended_wording": "For Rademacher sensing, evaluation reloads the exported exact operator and rebuilds the cached solver before inference.",
        "caveat": "Do not cite pre-fix local Rademacher re-evaluations.",
    },
    {
        "claim": "The learned model improves over physical backprojection.",
        "support": "Phase16 attribution_final and registry delta_psnr/delta_ssim.",
        "recommended_wording": "The learned inverse consistently improves PSNR/SSIM over the linear backprojection initialization.",
        "caveat": "Separate this from claims about measurement design.",
    },
    {
        "claim": "Rademacher and scrambled Hadamard are complementary sensing regimes.",
        "support": "Rademacher weak backprojection but strong learned final; scrambled stronger backprojection and similar final quality.",
        "recommended_wording": "Random Rademacher measurements are harder for direct inversion but remain recoverable by the learned prior.",
        "caveat": "Do not claim one dominates on every metric.",
    },
    {
        "claim": "Noise and perturbation checks support measurement-conditioned inference.",
        "support": "Phase16 noise_sweep and measurement_perturbation.",
        "recommended_wording": "Performance degrades under measurement corruption, indicating dependence on the measured signal.",
        "caveat": "This is diagnostic robustness, not adversarial robustness.",
    },
    {
        "claim": "Traditional inverse-problem controls are included.",
        "support": "Backprojection, adjoint, and small-subset TV-PGD rows.",
        "recommended_wording": "We compare to linear inverses and a TV-regularized iterative baseline.",
        "caveat": "TV-PGD is small-subset and should be labeled as such.",
    },
]

UNSUPPORTED_ROWS = [
    {
        "claim": "State-of-the-art performance.",
        "reason": "No broad external SOTA benchmark sweep is included.",
        "safe_replacement": "Strong no-leak reconstruction performance under the reported sensing setups.",
    },
    {
        "claim": "Low-frequency Hadamard 5% is a high-quality STL-10 setting.",
        "reason": "Earlier rescue rows showed this setting was negative or auxiliary.",
        "safe_replacement": "Use scrambled Hadamard/Rademacher for the primary STL-10 claims.",
    },
    {
        "claim": "TV-PGD is fully optimized and conclusively dominated.",
        "reason": "TV-PGD is run as a time-bounded small-subset reviewer-defense baseline.",
        "safe_replacement": "The reported TV-PGD control is below the learned model under the tested settings.",
    },
    {
        "claim": "The method is robust to arbitrary measurement corruption.",
        "reason": "Only finite Gaussian, shuffled, and wrong-sample perturbations are tested.",
        "safe_replacement": "The method is robust within the tested noise sweep and sensitive to severe measurement mismatch.",
    },
    {
        "claim": "Old exploratory or leaked results support the final paper.",
        "reason": "Those runs are superseded by strict no-leak imports and Phase16 audits.",
        "safe_replacement": "Use only Phase15 no-leak and Phase16 supplementary outputs.",
    },
]


def main() -> None:
    ensure_dir(OUT)
    registry = registry_rows()
    write_csv(SUPPORTED_CSV, SUPPORTED_ROWS, ["claim", "support", "recommended_wording", "caveat"])
    write_csv(UNSUPPORTED_CSV, UNSUPPORTED_ROWS, ["claim", "reason", "safe_replacement"])
    lines = [
        "# Phase16 supported and unsupported writing claims",
        "",
        "## Main no-leak rows to cite",
        "",
        "|method_id|psnr|ssim|backproj_psnr|delta_psnr|",
        "|---|---|---|---|---|",
    ]
    for row in registry:
        lines.append(f"|{row.get('method_id', '')}|{row.get('psnr', '')}|{row.get('ssim', '')}|{row.get('backproj_psnr', '')}|{row.get('delta_psnr', '')}|")
    lines.extend(["", "## Supported claims", ""])
    for row in SUPPORTED_ROWS:
        lines.extend(
            [
                f"### {row['claim']}",
                "",
                f"- Support: {row['support']}",
                f"- Suggested wording: {row['recommended_wording']}",
                f"- Caveat: {row['caveat']}",
                "",
            ]
        )
    lines.extend(["## Unsupported or risky claims", ""])
    for row in UNSUPPORTED_ROWS:
        lines.extend(
            [
                f"### {row['claim']}",
                "",
                f"- Why not: {row['reason']}",
                f"- Safer replacement: {row['safe_replacement']}",
                "",
            ]
        )
    SUPPORTED.write_text("\n".join(lines), encoding="utf-8")
    copied = [copy_to_legacy(path) for path in [SUPPORTED, SUPPORTED_CSV, UNSUPPORTED_CSV] if path.exists()]
    ensure_dir(PHASE15_SUPP)
    print(json.dumps({"supported_claims": str(SUPPORTED), "legacy_copies": [str(p) for p in copied]}, indent=2))


if __name__ == "__main__":
    main()
