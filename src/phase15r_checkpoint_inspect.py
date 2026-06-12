from __future__ import annotations

import json

import torch

from .models import build_generator
from .phase15r_common import (
    RADEMACHER_METHODS,
    REPRO_DEBUG,
    base_config_for,
    checkpoint_candidates,
    make_measurement,
    method_dir,
    torch_load,
    write_rows_all_formats,
)
from .phase15_common import sha256_file


FIELDS = [
    "method_id",
    "checkpoint",
    "exists",
    "sha256",
    "checkpoint_keys",
    "contains_generator",
    "contains_generator_ema",
    "contains_refiner",
    "contains_discriminator",
    "contains_config",
    "contains_best_metrics",
    "epoch",
    "model_type",
    "parameter_count",
    "parameter_norm",
    "refiner_parameter_count",
    "ema_parameter_count",
    "missing_key_count_raw",
    "unexpected_key_count_raw",
    "missing_key_count_ema",
    "unexpected_key_count_ema",
    "load_strict_possible_raw",
    "load_strict_possible_ema",
    "eval_default_uses",
    "eval_loads_refiner",
    "notes",
]


def state_param_stats(state: dict | None) -> tuple[int, float]:
    if not isinstance(state, dict):
        return 0, 0.0
    count = 0
    norm_sq = 0.0
    for value in state.values():
        if torch.is_tensor(value):
            count += value.numel()
            norm_sq += float(value.float().norm().item() ** 2)
    return count, norm_sq ** 0.5


def inspect_checkpoint(method_id: str, checkpoint_path) -> dict:
    row = {field: "" for field in FIELDS}
    row.update({"method_id": method_id, "checkpoint": checkpoint_path.name, "exists": checkpoint_path.exists()})
    if not checkpoint_path.exists():
        row["notes"] = "missing"
        return row
    row["sha256"] = sha256_file(checkpoint_path)
    payload = torch_load(checkpoint_path, "cpu")
    keys = list(payload.keys()) if isinstance(payload, dict) else []
    row["checkpoint_keys"] = ";".join(keys)
    row["contains_generator"] = isinstance(payload, dict) and "generator" in payload
    row["contains_generator_ema"] = isinstance(payload, dict) and "generator_ema" in payload and payload.get("generator_ema") is not None
    row["contains_refiner"] = isinstance(payload, dict) and ("refiner" in payload or any(str(k).startswith("refiner.") for k in (payload.get("generator") or {}).keys()))
    row["contains_discriminator"] = isinstance(payload, dict) and "discriminator" in payload
    row["contains_config"] = isinstance(payload, dict) and isinstance(payload.get("config"), dict)
    row["contains_best_metrics"] = isinstance(payload, dict) and ("best_metrics" in payload or "metrics" in payload)
    row["epoch"] = payload.get("epoch", "") if isinstance(payload, dict) else ""
    config = base_config_for(method_id, checkpoint_path)
    row["model_type"] = config.get("model_type", "")
    device = torch.device("cpu")
    measurement = make_measurement(config, device)
    model = build_generator(config, measurement=measurement).to(device)
    row["eval_loads_refiner"] = hasattr(model, "refine")
    raw_state = payload.get("generator") if isinstance(payload, dict) else payload
    ema_state = payload.get("generator_ema") if isinstance(payload, dict) else None
    count, norm = state_param_stats(raw_state)
    row["parameter_count"] = count
    row["parameter_norm"] = norm
    row["refiner_parameter_count"] = sum(v.numel() for k, v in raw_state.items() if str(k).startswith("refiner.") and torch.is_tensor(v)) if isinstance(raw_state, dict) else 0
    ema_count, _ = state_param_stats(ema_state)
    row["ema_parameter_count"] = ema_count
    for label, state in [("raw", raw_state), ("ema", ema_state)]:
        if not isinstance(state, dict):
            row[f"missing_key_count_{label}"] = ""
            row[f"unexpected_key_count_{label}"] = ""
            row[f"load_strict_possible_{label}"] = False
            continue
        result = model.load_state_dict(state, strict=False)
        row[f"missing_key_count_{label}"] = len(result.missing_keys)
        row[f"unexpected_key_count_{label}"] = len(result.unexpected_keys)
        row[f"load_strict_possible_{label}"] = len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0
    row["eval_default_uses"] = "generator_ema if present, otherwise generator"
    notes = []
    if row["contains_generator_ema"]:
        notes.append("EMA present")
    if row["contains_refiner"]:
        notes.append("refiner state present")
    if not row["load_strict_possible_ema"] and row["contains_generator_ema"]:
        notes.append("EMA strict load mismatch")
    if not row["load_strict_possible_raw"]:
        notes.append("raw strict load mismatch")
    row["notes"] = "; ".join(notes)
    return row


def main() -> None:
    rows = []
    for method in RADEMACHER_METHODS:
        root = method_dir(method["method_id"])
        for checkpoint in checkpoint_candidates(root):
            rows.append(inspect_checkpoint(method["method_id"], checkpoint))
    write_rows_all_formats(REPRO_DEBUG / "checkpoint_inspection", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "checkpoint_inspection.csv")}, indent=2))


if __name__ == "__main__":
    main()
