from __future__ import annotations

import json
import math

import torch

from .phase15r_common import (
    RADEMACHER_METHODS,
    REPRO_DEBUG,
    exact_A_path,
    infer_rademacher_normalization,
    load_exact_A,
    method_dir,
    read_yaml,
    tensor_sha256,
    torch_load,
    write_rows_all_formats,
)
from .phase15_common import sha256_file


FIELDS = [
    "method_id",
    "file_exists",
    "loadable",
    "object_type",
    "keys",
    "A_tensor_key",
    "shape",
    "dtype",
    "device",
    "min",
    "max",
    "mean",
    "std",
    "unique_abs_values",
    "row_norm_mean",
    "row_norm_std",
    "col_norm_mean",
    "col_norm_std",
    "row_mean_mean",
    "row_mean_std",
    "expected_m",
    "expected_n",
    "expected_sampling_ratio",
    "config_sampling_ratio",
    "config_pattern_type",
    "config_matrix_normalization",
    "inferred_normalization",
    "A_sha256",
    "tensor_sha256",
    "notes",
]


def inspect_one(method: dict) -> dict:
    method_id = method["method_id"]
    path = exact_A_path(method_id)
    row = {
        "method_id": method_id,
        "file_exists": path.exists(),
        "loadable": False,
        "object_type": "",
        "keys": "",
        "A_tensor_key": "",
        "shape": "",
        "dtype": "",
        "device": "",
        "min": "",
        "max": "",
        "mean": "",
        "std": "",
        "unique_abs_values": "",
        "row_norm_mean": "",
        "row_norm_std": "",
        "col_norm_mean": "",
        "col_norm_std": "",
        "row_mean_mean": "",
        "row_mean_std": "",
        "expected_m": method["expected_m"],
        "expected_n": method["expected_n"],
        "expected_sampling_ratio": method["sampling_ratio"],
        "config_sampling_ratio": "",
        "config_pattern_type": "",
        "config_matrix_normalization": "",
        "inferred_normalization": "",
        "A_sha256": sha256_file(path) if path.exists() else "",
        "tensor_sha256": "",
        "notes": "",
    }
    cfg = read_yaml(method_dir(method_id) / "resolved_config.yaml")
    row["config_sampling_ratio"] = cfg.get("sampling_ratio", "")
    row["config_pattern_type"] = cfg.get("pattern_type", "")
    row["config_matrix_normalization"] = cfg.get("matrix_normalization", "")
    if not path.exists():
        row["notes"] = "missing exact A"
        return row
    try:
        obj = torch_load(path, "cpu")
        row["object_type"] = type(obj).__name__
        if isinstance(obj, dict):
            row["keys"] = ";".join(str(k) for k in obj.keys())
            for key, value in obj.items():
                if torch.is_tensor(value) and value.ndim == 2:
                    row["A_tensor_key"] = str(key)
                    break
        A = load_exact_A(method_id, "cpu")
        row["loadable"] = True
        row["shape"] = f"{tuple(A.shape)}"
        row["dtype"] = str(A.dtype)
        row["device"] = str(A.device)
        row["min"] = float(A.min())
        row["max"] = float(A.max())
        row["mean"] = float(A.mean())
        row["std"] = float(A.std(unbiased=False))
        abs_values = A.abs().flatten()
        unique = torch.unique(abs_values)
        if unique.numel() <= 12:
            row["unique_abs_values"] = ";".join(f"{float(x):.10g}" for x in unique)
        else:
            q = torch.quantile(abs_values, torch.tensor([0.0, 0.5, 1.0]))
            row["unique_abs_values"] = f"many|min={float(q[0]):.10g};median={float(q[1]):.10g};max={float(q[2]):.10g}"
        row_norm = A.norm(dim=1)
        col_norm = A.norm(dim=0)
        row_mean = A.mean(dim=1)
        row["row_norm_mean"] = float(row_norm.mean())
        row["row_norm_std"] = float(row_norm.std(unbiased=False))
        row["col_norm_mean"] = float(col_norm.mean())
        row["col_norm_std"] = float(col_norm.std(unbiased=False))
        row["row_mean_mean"] = float(row_mean.mean())
        row["row_mean_std"] = float(row_mean.std(unbiased=False))
        row["inferred_normalization"] = infer_rademacher_normalization(A)
        row["tensor_sha256"] = tensor_sha256(A)
        shape_ok = tuple(A.shape) == (method["expected_m"], method["expected_n"])
        norm_ok = row["inferred_normalization"] == "legacy_sqrt_m_or_row_norm_sqrt_n_over_m"
        row["notes"] = f"shape_ok={shape_ok}; legacy_norm_like={norm_ok}"
    except Exception as exc:
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
    return row


def main() -> None:
    rows = [inspect_one(method) for method in RADEMACHER_METHODS]
    write_rows_all_formats(REPRO_DEBUG / "exactA_inspection", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "exactA_inspection.csv")}, indent=2))


if __name__ == "__main__":
    main()
