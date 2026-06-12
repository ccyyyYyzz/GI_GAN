from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Convert continuous learned patterns to binary STE logits.")
    parser.add_argument("--continuous_checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--target_mode", default="learned_balanced_binary_ste")
    parser.add_argument("--target_transmission", type=float, default=0.5)
    parser.add_argument("--logit_abs_scale", type=float, default=2.0)
    return parser.parse_args()


def topk_binary(P_soft: torch.Tensor, target_transmission: float) -> torch.Tensor:
    if P_soft.ndim != 2:
        raise ValueError("Pattern logits must have shape [m, n].")
    n = P_soft.shape[1]
    k = max(0, min(n, int(round(float(target_transmission) * n))))
    if k == 0:
        return torch.zeros_like(P_soft)
    if k == n:
        return torch.ones_like(P_soft)
    indices = torch.topk(P_soft, k=k, dim=1).indices
    P_hard = torch.zeros_like(P_soft)
    return P_hard.scatter(1, indices, 1.0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.continuous_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "pattern_bank" not in checkpoint:
        raise RuntimeError("continuous_checkpoint does not contain pattern_bank.")
    pattern_state = dict(checkpoint["pattern_bank"])
    if "logits" not in pattern_state:
        raise RuntimeError("pattern_bank state_dict does not contain logits.")

    config = dict(checkpoint.get("config", {}))
    tau = float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0)))
    logits = pattern_state["logits"].detach().float()
    P_soft = torch.sigmoid(logits / max(tau, 1e-6))
    P_hard = topk_binary(P_soft, args.target_transmission)
    new_logits = (P_hard * 2.0 - 1.0) * abs(float(args.logit_abs_scale))
    pattern_state["logits"] = new_logits

    config.update(
        {
            "pattern_mode": args.target_mode,
            "pattern_init": "converted_from_continuous",
            "pattern_logit_abs_init": float(args.logit_abs_scale),
            "target_transmission": float(args.target_transmission),
            "balanced_target_transmission": float(args.target_transmission),
        }
    )
    checkpoint["pattern_bank"] = pattern_state
    checkpoint["config"] = config
    checkpoint["conversion"] = {
        "source_checkpoint": args.continuous_checkpoint,
        "target_mode": args.target_mode,
        "target_transmission": float(args.target_transmission),
        "logit_abs_scale": float(args.logit_abs_scale),
    }

    out = Path(args.output_checkpoint)
    ensure_dir(out.parent)
    torch.save(checkpoint, out)
    print(f"Saved converted checkpoint to: {out}")


if __name__ == "__main__":
    main()
