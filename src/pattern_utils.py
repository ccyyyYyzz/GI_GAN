from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .utils import ensure_dir


def save_pattern_grid(
    P: torch.Tensor,
    path: str | Path,
    img_size: int,
    max_patterns: int = 32,
    binarize_for_display: bool = False,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    patterns = P.detach().float().cpu()
    if patterns.ndim != 2 or patterns.shape[1] != img_size * img_size:
        raise ValueError(f"Expected P with shape [m, {img_size * img_size}].")
    if binarize_for_display:
        patterns = (patterns > 0.5).float()
    patterns = patterns[: max(1, min(max_patterns, patterns.shape[0]))]
    images = patterns.reshape(patterns.shape[0], img_size, img_size)

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        cols = min(8, images.shape[0])
        rows = math.ceil(images.shape[0] / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes_list = np.asarray(axes, dtype=object).reshape(-1).tolist()
        for idx, ax in enumerate(axes_list):
            ax.axis("off")
            if idx < images.shape[0]:
                ax.imshow(images[idx].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        title = (
            f"mean={patterns.mean():.3f}, std={patterns.std(unbiased=False):.3f}, "
            f"min={patterns.min():.3f}, max={patterns.max():.3f}"
        )
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        _fallback_pattern_grid(images, patterns, path)
    return path


def _fallback_pattern_grid(images: torch.Tensor, patterns: torch.Tensor, path: Path) -> None:
    from PIL import Image, ImageDraw

    cell = 72
    cols = min(8, images.shape[0])
    rows = math.ceil(images.shape[0] / cols)
    canvas = Image.new("RGB", (cols * cell, rows * cell + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 4),
        f"mean={patterns.mean():.3f}, std={patterns.std(unbiased=False):.3f}, "
        f"min={patterns.min():.3f}, max={patterns.max():.3f}",
        fill="black",
    )
    for idx, image in enumerate(images):
        arr = (image.clamp(0.0, 1.0).numpy() * 255).astype("uint8")
        tile = Image.fromarray(arr, mode="L").resize((cell, cell))
        x = (idx % cols) * cell
        y = 34 + (idx // cols) * cell
        canvas.paste(tile.convert("RGB"), (x, y))
    canvas.save(path)


def save_pattern_stats_json(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path
