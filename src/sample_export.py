from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from .metrics import batch_metrics
from .utils import ensure_dir
from .visualize import save_recon_grid


def _save_gray_png(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image

    image = tensor.detach().clamp(0.0, 1.0).cpu()
    if image.ndim == 3:
        image = image[0]
    image_u8 = (image * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous()
    Image.frombytes("L", (image_u8.shape[1], image_u8.shape[0]), bytes(image_u8.reshape(-1).tolist())).save(path)


def save_eval_samples_individual(
    output_dir: str | Path,
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_hat: torch.Tensor,
    measurement,
    y: torch.Tensor,
    *,
    start_index: int = 0,
    max_items: int | None = None,
) -> list[dict]:
    sample_dir = ensure_dir(Path(output_dir) / "eval_samples_individual")
    rows = []
    count = x.shape[0] if max_items is None else min(x.shape[0], int(max_items))
    for idx in range(count):
        sample_id = start_index + idx
        prefix = f"sample_{sample_id:03d}"
        gt_path = sample_dir / f"{prefix}_gt.png"
        backproj_path = sample_dir / f"{prefix}_backproj.png"
        recon_path = sample_dir / f"{prefix}_recon.png"
        abs_error_path = sample_dir / f"{prefix}_abs_error.png"
        metrics_path = sample_dir / f"{prefix}_metrics.json"
        gt = x[idx : idx + 1]
        back = x_data[idx : idx + 1]
        recon = x_hat[idx : idx + 1]
        err = torch.abs(recon - gt)
        _save_gray_png(gt[0], gt_path)
        _save_gray_png(back[0], backproj_path)
        _save_gray_png(recon[0], recon_path)
        _save_gray_png(err[0], abs_error_path)
        metrics = batch_metrics(recon, gt, measurement, y[idx : idx + 1])
        metrics_payload = {
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "mse": metrics.get("mse"),
            "rel_meas_err": metrics.get("rel_meas_error"),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "sample_id": sample_id,
                "psnr": metrics_payload["psnr"],
                "ssim": metrics_payload["ssim"],
                "mse": metrics_payload["mse"],
                "rel_meas_err": metrics_payload["rel_meas_err"],
                "gt_path": str(gt_path),
                "backproj_path": str(backproj_path),
                "recon_path": str(recon_path),
                "abs_error_path": str(abs_error_path),
            }
        )
    return rows


def write_per_sample_csv(output_dir: str | Path, rows: list[dict]) -> Path:
    path = Path(output_dir) / "eval_samples_individual" / "per_sample_metrics.csv"
    ensure_dir(path.parent)
    fields = [
        "sample_id",
        "psnr",
        "ssim",
        "mse",
        "rel_meas_err",
        "gt_path",
        "backproj_path",
        "recon_path",
        "abs_error_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def save_example_grid(rows: list[dict], path: str | Path, title: str, max_items: int = 8) -> None:
    from PIL import Image

    selected = rows[:max_items]
    gt, back, recon, err = [], [], [], []
    for row in selected:
        gt.append(_load_png_tensor(row["gt_path"]))
        back.append(_load_png_tensor(row["backproj_path"]))
        recon.append(_load_png_tensor(row["recon_path"]))
        err.append(_load_png_tensor(row["abs_error_path"]))
    if not selected:
        return
    save_recon_grid(
        torch.stack(gt, dim=0),
        torch.stack(back, dim=0),
        torch.stack(recon, dim=0),
        path,
        max_items=len(selected),
        title=title,
    )


def _load_png_tensor(path: str | Path) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("L")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    return data.view(1, img.height, img.width).float().div(255.0)
