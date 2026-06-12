from __future__ import annotations

import json
from pathlib import Path

from .phase15r_common import RADEMACHER_METHODS, REPRO_DEBUG, method_dir, write_rows_all_formats
from .phase15_common import sha256_file


ARTIFACTS = [
    "eval_metrics.json",
    "best_hq.pt",
    "best_score.pt",
    "best_ssim.pt",
    "best_psnr.pt",
    "last.pt",
    "measurement_operator_exact.pt",
    "measurement_operator_exact_manifest.json",
    "resolved_config.yaml",
    "RUN_REPORT.md",
    "per_epoch_metrics.csv",
    "convergence_summary.md",
    "eval_samples/recon_grid.png",
    "sha256_manifest.json",
]

FIELDS = ["method_id", "artifact", "path", "exists", "size", "sha256", "notes"]


def main() -> None:
    rows = []
    for method in RADEMACHER_METHODS:
        root = method_dir(method["method_id"])
        rows.append(
            {
                "method_id": method["method_id"],
                "artifact": "imported output dir",
                "path": str(root),
                "exists": root.exists(),
                "size": "",
                "sha256": "",
                "notes": "Phase 15 imported local copy; cloud original may still exist in Google Drive/Colab.",
            }
        )
        for rel in ARTIFACTS:
            path = root / rel
            rows.append(
                {
                    "method_id": method["method_id"],
                    "artifact": rel,
                    "path": str(path),
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() and path.is_file() else "",
                    "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                    "notes": "found" if path.exists() else "missing",
                }
            )
    write_rows_all_formats(REPRO_DEBUG / "artifact_inventory", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "artifact_inventory.csv")}, indent=2))


if __name__ == "__main__":
    main()
