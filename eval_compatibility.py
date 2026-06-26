from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from src.compatibility_data import compute_train_normalization, load_rad5_96_components, save_json, write_csv
from src.compatibility_eval import e1_gate, evaluate_critic_split
from src.compatibility_model import CompatibilityCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase-1 compatibility critic on Rad-5/96.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--checkpoint", required=True, help="Compatibility critic checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--device", default=None, help="Override config device.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = str(path)
    return cfg


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.device:
        cfg["device"] = args.device
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    device = resolve_device(str(cfg.get("device", "cuda")))
    measurement, _rad_config, splits, _split_info = load_rad5_96_components(cfg, output_dir=out, device=device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    normalization = payload.get("normalization") or compute_train_normalization(splits["train"])
    model = CompatibilityCritic(
        embed_dim=int(cfg.get("embed_dim", 128)),
        base_channels=int(cfg.get("base_channels", 24)),
        temperature=float(cfg.get("temperature", 0.07)),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    metrics = {}
    for split_name in ["val", "test"]:
        split_metrics, rows = evaluate_critic_split(
            model,
            splits[split_name],
            normalization,
            device=device,
            seed=int(cfg.get("seed", 1)) + (700 if split_name == "val" else 800),
            donors_per_anchor=int(cfg.get("donors_per_anchor", 32)),
            batch_size=int(cfg.get("eval_batch_size", 128)),
        )
        metrics[split_name] = split_metrics
        write_csv(out / f"{split_name}_per_image.csv", rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config_path": str(args.config),
        "output_dir": str(out),
        "measurement": {"img_size": measurement.img_size, "m": measurement.m, "n": measurement.n},
        "metrics": metrics,
        "e1_gate_on_validation": e1_gate(metrics["val"]),
    }
    save_json(out / "compatibility_eval_summary.json", summary)
    print(json.dumps({"summary": str(out / "compatibility_eval_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
