from __future__ import annotations

from pathlib import Path


DEFAULT_CHECKPOINT_ORDER = (
    "best_hq.pt",
    "best_score.pt",
    "best_ssim.pt",
    "best_psnr.pt",
    "best_mse.pt",
    "last.pt",
)


def find_best_checkpoint(
    output_dir: str | Path,
    preferred: tuple[str, ...] = DEFAULT_CHECKPOINT_ORDER,
) -> Path | None:
    """Return the first preferred checkpoint that exists in an experiment directory."""
    root = Path(output_dir)
    for name in preferred:
        path = root / name
        if path.exists():
            return path
    checkpoints = sorted(root.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None
