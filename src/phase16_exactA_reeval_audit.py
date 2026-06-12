from __future__ import annotations

import json

from .phase15_common import read_json, sha256_file
from .phase16_common import PHASE16, evaluate_model, method_dir, write_all


OUT = PHASE16 / "exactA_reeval"
FIELDS = [
    "method_id",
    "original_psnr",
    "original_ssim",
    "reeval_psnr",
    "reeval_ssim",
    "abs_diff_psnr",
    "abs_diff_ssim",
    "original_backproj_psnr",
    "reeval_backproj_psnr",
    "abs_diff_backproj_psnr",
    "exact_A_sha256",
    "checkpoint_sha256",
    "exact_A_loaded",
    "cache_rebuilt",
    "status",
    "notes",
]


def main() -> None:
    rows = []
    for method_id in ["rademacher5_hq_noise001_colab", "rademacher10_full_noise001_colab"]:
        original = read_json(method_dir(method_id) / "eval_metrics.json")
        summary, _ = evaluate_model(method_id, limit=2000, state_mode="ema", noise_map_mode="fixed")
        diff_psnr = abs(summary["psnr"] - float(original["model"]["psnr"]))
        diff_ssim = abs(summary["ssim"] - float(original["model"]["ssim"]))
        diff_bp = abs(summary["backproj_psnr"] - float(original["backprojection"]["psnr"]))
        status = "reproduced" if diff_psnr <= 0.02 and diff_ssim <= 0.002 else "mismatch"
        rows.append(
            {
                "method_id": method_id,
                "original_psnr": original["model"]["psnr"],
                "original_ssim": original["model"]["ssim"],
                "reeval_psnr": summary["psnr"],
                "reeval_ssim": summary["ssim"],
                "abs_diff_psnr": diff_psnr,
                "abs_diff_ssim": diff_ssim,
                "original_backproj_psnr": original["backprojection"]["psnr"],
                "reeval_backproj_psnr": summary["backproj_psnr"],
                "abs_diff_backproj_psnr": diff_bp,
                "exact_A_sha256": sha256_file(method_dir(method_id) / "measurement_operator_exact.pt"),
                "checkpoint_sha256": sha256_file(method_dir(method_id) / "last.pt"),
                "exact_A_loaded": summary["exact_A_loaded"],
                "cache_rebuilt": summary["cache_rebuilt"],
                "status": status,
                "notes": "safe set_A_override rebuilds K/cholesky cache",
            }
        )
    write_all(OUT / "exactA_reeval_results", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(OUT / "exactA_reeval_results.csv")}, indent=2))


if __name__ == "__main__":
    main()
