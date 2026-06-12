from __future__ import annotations

import json

from .phase15r_common import RADEMACHER_METHODS, REPRO_DEBUG, evaluate_variant, write_rows_all_formats


FIELDS = [
    "method_id",
    "variant",
    "psnr",
    "ssim",
    "mse",
    "backproj_psnr",
    "backproj_ssim",
    "rel_meas_err",
    "colab_psnr",
    "colab_ssim",
    "diff_psnr",
    "diff_ssim",
    "checkpoint_used",
    "A_source",
    "A_sha",
    "model_mode",
    "refiner",
    "split",
    "noise_map_mode",
    "output_range_mode",
    "status",
    "missing_key_count",
    "unexpected_key_count",
    "override_mode",
    "cache_rebuilt",
]


VARIANTS = [
    {"variant": "exactA_best_hq_default", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "unsafe_old_chol", "state_mode": "ema"},
    {"variant": "exactA_best_hq_rebuiltK", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema"},
    {"variant": "exactA_best_hq_raw_generator", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "raw"},
    {"variant": "exactA_best_hq_ema_generator", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema"},
    {"variant": "exactA_best_hq_no_refiner", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "enable_refiner": False},
    {"variant": "exactA_best_hq_with_refiner", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "enable_refiner": True},
    {"variant": "exactA_last_default", "checkpoint": "last.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema"},
    {"variant": "generatedA_seed_default", "checkpoint": "best_hq.pt", "a_source": "generated", "state_mode": "ema"},
    {"variant": "exactA_zero_noise_map", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "noise_map_mode": "zero"},
    {"variant": "exactA_fixed_noise_seed", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "noise_map_mode": "fixed"},
    {"variant": "exactA_val_split", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "split": "train"},
    {"variant": "exactA_test_split", "checkpoint": "best_hq.pt", "a_source": "exact", "a_override_mode": "safe_rebuild", "state_mode": "ema", "split": "test"},
]


def main() -> None:
    rows = []
    for method in RADEMACHER_METHODS:
        for variant in VARIANTS:
            try:
                rows.append(evaluate_variant(method["method_id"], variant))
            except Exception as exc:
                row = {field: "" for field in FIELDS}
                row.update(
                    {
                        "method_id": method["method_id"],
                        "variant": variant["variant"],
                        "checkpoint_used": variant.get("checkpoint", ""),
                        "A_source": variant.get("a_source", ""),
                        "model_mode": variant.get("state_mode", ""),
                        "refiner": variant.get("enable_refiner", True),
                        "split": variant.get("split", "test"),
                        "noise_map_mode": variant.get("noise_map_mode", "default"),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
                rows.append(row)
    write_rows_all_formats(REPRO_DEBUG / "eval_variants", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "eval_variants.csv")}, indent=2))


if __name__ == "__main__":
    main()
