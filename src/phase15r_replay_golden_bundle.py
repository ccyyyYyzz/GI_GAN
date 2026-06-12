from __future__ import annotations

import json
from pathlib import Path

import torch

from .phase15r_common import REPRO_DEBUG, torch_load, write_rows_all_formats


FIELDS = [
    "bundle",
    "exists",
    "method_id",
    "sample_count",
    "A_tensor_sha256",
    "checkpoint_sha256",
    "x_shape",
    "y_shape",
    "x_data_shape",
    "x_metric_shape",
    "status",
    "notes",
]


def main() -> None:
    bundle_dir = REPRO_DEBUG / "golden_bundles"
    rows = []
    paths = sorted(bundle_dir.glob("*_golden_bundle.pt")) if bundle_dir.exists() else []
    if not paths:
        rows.append(
            {
                "bundle": str(bundle_dir),
                "exists": False,
                "method_id": "",
                "sample_count": "",
                "A_tensor_sha256": "",
                "checkpoint_sha256": "",
                "x_shape": "",
                "y_shape": "",
                "x_data_shape": "",
                "x_metric_shape": "",
                "status": "pending_golden_bundle",
                "notes": "Run the generated Colab script and download bundles here.",
            }
        )
    for path in paths:
        try:
            payload = torch_load(path, "cpu")
            rows.append(
                {
                    "bundle": str(path),
                    "exists": True,
                    "method_id": payload.get("method_id", ""),
                    "sample_count": payload.get("sample_count", ""),
                    "A_tensor_sha256": payload.get("A_tensor_sha256", ""),
                    "checkpoint_sha256": payload.get("checkpoint_sha256", ""),
                    "x_shape": tuple(payload["x"].shape) if torch.is_tensor(payload.get("x")) else "",
                    "y_shape": tuple(payload["y"].shape) if torch.is_tensor(payload.get("y")) else "",
                    "x_data_shape": tuple(payload["x_data"].shape) if torch.is_tensor(payload.get("x_data")) else "",
                    "x_metric_shape": tuple(payload["x_metric"].shape) if torch.is_tensor(payload.get("x_metric")) else "",
                    "status": "bundle_loaded",
                    "notes": "Full tensor-stage replay is pending; bundle presence is verified.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "bundle": str(path),
                    "exists": True,
                    "method_id": "",
                    "sample_count": "",
                    "A_tensor_sha256": "",
                    "checkpoint_sha256": "",
                    "x_shape": "",
                    "y_shape": "",
                    "x_data_shape": "",
                    "x_metric_shape": "",
                    "status": "failed",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    write_rows_all_formats(REPRO_DEBUG / "golden_bundle_replay_results", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "golden_bundle_replay_results.csv")}, indent=2))


if __name__ == "__main__":
    main()
