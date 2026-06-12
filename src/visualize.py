from pathlib import Path

import torch


def save_recon_grid(
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_hat: torch.Tensor,
    path: str | Path,
    max_items: int = 4,
    title: str | None = None,
) -> None:
    """Save rows of GT | Backprojection | NS-MC-GAN | Abs Error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = x.detach().clamp(0.0, 1.0).cpu()
    x_data = x_data.detach().clamp(0.0, 1.0).cpu()
    x_hat = x_hat.detach().clamp(0.0, 1.0).cpu()
    error = torch.abs(x_hat - x).clamp(0.0, 1.0)

    if _torch_numpy_bridge_available():
        try:
            _save_with_matplotlib(x, x_data, x_hat, error, path, max_items, title)
            return
        except Exception as exc:
            _save_with_torchvision(x, x_data, x_hat, error, path, max_items, exc, title)
            return
    _save_with_torchvision(
        x,
        x_data,
        x_hat,
        error,
        path,
        max_items,
        RuntimeError("Torch NumPy bridge is unavailable."),
        title,
    )


def _torch_numpy_bridge_available() -> bool:
    try:
        torch.zeros(1).cpu().numpy()
        return True
    except Exception:
        return False


def _save_with_matplotlib(
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_hat: torch.Tensor,
    error: torch.Tensor,
    path: Path,
    max_items: int,
    title: str | None,
) -> None:
    import matplotlib.pyplot as plt

    rows = min(max_items, x.shape[0])
    titles = ["GT", "Backprojection", "NS-MC-GAN", "Abs Error"]
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=10)

    for row in range(rows):
        images = [x[row, 0], x_data[row, 0], x_hat[row, 0], error[row, 0]]
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_axis_off()
            if row == 0:
                axes[row, col].set_title(titles[col])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_with_torchvision(
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_hat: torch.Tensor,
    error: torch.Tensor,
    path: Path,
    max_items: int,
    reason: Exception,
    title: str | None,
) -> None:
    try:
        from PIL import Image, ImageDraw

        rows = min(max_items, x.shape[0])
        row_images = []
        for row in range(rows):
            row_images.append(torch.cat([x[row], x_data[row], x_hat[row], error[row]], dim=2))
        grid = torch.cat(row_images, dim=1).squeeze(0)
        grid_u8 = (grid * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous()
        height, width = grid_u8.shape
        image = Image.frombytes("L", (width, height), bytes(grid_u8.reshape(-1).tolist()))
        if title:
            strip_h = 18
            titled = Image.new("L", (width, height + strip_h), color=255)
            titled.paste(image, (0, strip_h))
            draw = ImageDraw.Draw(titled)
            draw.text((4, 2), title[:160], fill=0)
            image = titled
        image.save(path)
    except Exception:
        fallback = path.with_suffix(".pgm")
        rows = min(max_items, x.shape[0])
        row_images = []
        for row in range(rows):
            row_images.append(torch.cat([x[row], x_data[row], x_hat[row], error[row]], dim=2))
        grid = torch.cat(row_images, dim=1).squeeze(0)
        grid_u8 = (grid * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous()
        height, width = grid_u8.shape
        with open(fallback, "wb") as f:
            f.write(f"P5\n# matplotlib fallback failed: {reason}\n{width} {height}\n255\n".encode())
            f.write(bytes(grid_u8.reshape(-1).tolist()))
