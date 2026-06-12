from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def torch_load(path: str | Path, map_location: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_from_exact_payload(payload: Any) -> torch.Tensor:
    if isinstance(payload, dict):
        for key in ["A", "matrix", "measurement_matrix"]:
            value = payload.get(key)
            if torch.is_tensor(value) and value.ndim == 2:
                return value
        for value in payload.values():
            if torch.is_tensor(value) and value.ndim == 2:
                return value
    if torch.is_tensor(payload) and payload.ndim == 2:
        return payload
    raise TypeError(f"Could not find a 2-D measurement tensor in payload type {type(payload).__name__}.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def apply_measurement_override_from_config(
    config: dict[str, Any],
    measurement: Any,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a saved exact-A tensor when a config requests strict exact-A reuse."""
    exact_required = bool(config.get("exact_A_required", False))
    exact_path = (
        config.get("measurement_operator_exact_path")
        or config.get("exact_A_path")
        or (config.get("phase25_measurement_lock") or {}).get("exact_A_path")
        or (config.get("phase26_measurement_lock") or {}).get("exact_A_path")
    )
    if not exact_path:
        if exact_required:
            raise FileNotFoundError("exact_A_required=true but no measurement_operator_exact_path was provided.")
        return {"exact_A_required": False, "exact_A_loaded": False}

    path = Path(str(exact_path))
    if not path.exists():
        if exact_required:
            raise FileNotFoundError(f"Required exact-A file is missing: {path}")
        return {"exact_A_required": False, "exact_A_loaded": False, "exact_A_path": str(path)}

    payload = torch_load(path, map_location=device)
    A = tensor_from_exact_payload(payload).to(device=device, dtype=torch.float32)
    if not hasattr(measurement, "set_A_override"):
        raise RuntimeError("Measurement operator has no safe set_A_override API.")
    stats = measurement.set_A_override(
        A,
        metadata={
            "source": str(path),
            "override_mode": "safe_rebuild",
            "tensor_sha256": tensor_sha256(A),
        },
        rebuild_cache=True,
    )
    return {
        **dict(stats),
        "exact_A_required": exact_required,
        "exact_A_loaded": True,
        "exact_A_path": str(path),
        "override_mode": "safe_rebuild",
        "tensor_sha256": tensor_sha256(A),
    }
