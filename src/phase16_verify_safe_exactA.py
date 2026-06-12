from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from .measurement import GhostMeasurementOperator
from .phase16_common import PHASE16, ensure_dir, load_exact_A, method_config, make_measurement
from .phase15_common import write_json


OUT = PHASE16 / "safe_exactA_verification"


def main() -> None:
    ensure_dir(OUT)
    rows = []
    warnings = []
    has_api = hasattr(GhostMeasurementOperator, "set_A_override")
    for method_id in ["rademacher5_hq_noise001_colab", "rademacher10_full_noise001_colab"]:
        try:
            config = method_config(method_id, limit=8)
            device = torch.device(config["device"])
            measurement = make_measurement(config, device)
            old_A = measurement.A.detach().clone()
            old_chol = getattr(measurement, "_chol", None)
            A = load_exact_A(method_id, device)
            stats = measurement.set_A_override(A, metadata={"test": "phase16"}, rebuild_cache=True)
            chol = getattr(measurement, "_chol", None)
            cache_rebuilt = chol is not None and (old_chol is None or chol.data_ptr() != old_chol.data_ptr())
            v = torch.randn(2, measurement.n, device=device)
            y = measurement.A_forward(v)
            at = measurement.AT_forward(y)
            solved = measurement.solve_K(y)
            x_data = measurement.data_solution(y)
            ns = measurement.null_project(v)
            dc = measurement.dc_project(v, y)
            ok = (
                has_api
                and tuple(A.shape) == tuple(measurement.A.shape)
                and cache_rebuilt
                and y.shape[1] == measurement.m
                and at.shape[1] == measurement.n
                and solved.shape[1] == measurement.m
                and x_data.shape[1] == measurement.n
                and ns.shape[1] == measurement.n
                and dc.shape[1] == measurement.n
            )
            rows.append(
                {
                    "method_id": method_id,
                    "exact_A_loaded": True,
                    "set_A_override_exists": has_api,
                    "cache_rebuilt": cache_rebuilt,
                    "same_shape": tuple(A.shape) == tuple(measurement.A.shape),
                    "uses_cholesky": stats.get("uses_cholesky", ""),
                    "status": "pass" if ok else "fail",
                    "notes": "safe exact-A override tested across all core operator calls",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method_id": method_id,
                    "exact_A_loaded": False,
                    "set_A_override_exists": has_api,
                    "cache_rebuilt": False,
                    "same_shape": False,
                    "uses_cholesky": "",
                    "status": "fail",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    phase16_files = sorted(Path(__file__).resolve().parent.glob("phase16_*.py"))
    for path in phase16_files:
        if path.name == "phase16_verify_safe_exactA.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "unsafe_old_chol" in text:
            warnings.append(f"unsafe_old_chol appears in {path.name}")
    status = "fail" if any(row["status"] == "fail" for row in rows) else ("warning" if warnings else "pass")
    payload = {"status": status, "rows": rows, "warnings": warnings}
    write_json(OUT / "safe_exactA_verification.json", payload)
    lines = ["# Safe Exact-A Verification", "", f"Status: `{status}`", ""]
    for row in rows:
        lines.append(f"- {row['method_id']}: {row['status']} cache_rebuilt={row['cache_rebuilt']} notes={row['notes']}")
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    (OUT / "safe_exactA_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
