from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .eval import make_measurement
from .phase15r_common import tensor_from_exact_payload, tensor_sha256, torch_load
from .utils import apply_experiment_defaults, load_config, set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase44_operator_centered"
PROV = OUT / "provenance"
IMPORTED = ROOT / "outputs_phase15" / "imported_noleak"
COMPONENTS = ROOT / "outputs_phase41_inkscape_signal_trace" / "components"

METHODS = [
    {
        "method": "Rad-5",
        "method_id": "rademacher5_hq_noise001_colab",
        "component_dir": COMPONENTS / "rad5",
        "exact_a": True,
    },
    {
        "method": "Scr-5",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "component_dir": COMPONENTS / "scr5",
        "exact_a": False,
    },
    {
        "method": "Rad-10",
        "method_id": "rademacher10_full_noise001_colab",
        "component_dir": None,
        "exact_a": True,
    },
    {
        "method": "Scr-10",
        "method_id": "scrambled_hadamard10_full_noise001_colab",
        "component_dir": None,
        "exact_a": False,
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_gray_image(path: Path, size: int = 64) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def crop_largest_image_region(cell: Image.Image) -> Image.Image:
    gray = np.asarray(cell.convert("L"), dtype=np.uint8)
    mask = gray < 248
    # Drop sparse title text by requiring dense non-white support.
    row_frac = mask.mean(axis=1)
    col_frac = mask.mean(axis=0)

    def longest_dense_segment(frac: np.ndarray, threshold: float) -> tuple[int, int]:
        best = (0, len(frac))
        best_len = 0
        start = None
        for i, value in enumerate(frac):
            if value >= threshold and start is None:
                start = i
            if (value < threshold or i == len(frac) - 1) and start is not None:
                end = i if value < threshold else i + 1
                if end - start > best_len:
                    best = (start, end)
                    best_len = end - start
                start = None
        return best

    y0, y1 = longest_dense_segment(row_frac, 0.18)
    x0, x1 = longest_dense_segment(col_frac, 0.18)
    if y1 - y0 < 80 or x1 - x0 < 80:
        w, h = cell.size
        side = min(w, h) * 0.70
        x0 = int((w - side) / 2)
        y0 = int(h * 0.24)
        x1 = int(x0 + side)
        y1 = int(y0 + side)
    side = min(x1 - x0, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(cell.size[0], x0 + side)
    y1 = min(cell.size[1], y0 + side)
    return cell.crop((x0, y0, x1, y1))


def extract_from_recon_grid(grid_path: Path, col: int, rows: int = 8, cols: int = 4) -> np.ndarray:
    image = Image.open(grid_path).convert("RGB")
    w, h = image.size
    cell_w = w / cols
    cell_h = h / rows
    x0 = int(round(col * cell_w))
    x1 = int(round((col + 1) * cell_w))
    y0 = 0
    y1 = int(round(cell_h))
    cell = image.crop((x0, y0, x1, y1))
    cropped = crop_largest_image_region(cell)
    cropped = cropped.resize((64, 64), Image.Resampling.BICUBIC).convert("L")
    return np.asarray(cropped, dtype=np.float32) / 255.0


def representative_images(method: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    comp_dir = method.get("component_dir")
    if comp_dir and Path(comp_dir).exists():
        gt_path = Path(comp_dir) / "ground_truth.png"
        recon_path = Path(comp_dir) / "final_audited.png"
        if gt_path.exists() and recon_path.exists():
            return load_gray_image(gt_path), load_gray_image(recon_path), str(comp_dir)

    method_dir = IMPORTED / method["method_id"]
    individual = method_dir / "eval_samples_individual"
    gt_path = individual / "sample_000_gt.png"
    recon_path = individual / "sample_000_recon.png"
    if gt_path.exists() and recon_path.exists():
        return load_gray_image(gt_path), load_gray_image(recon_path), str(individual)

    grid = method_dir / "eval_samples" / "recon_grid.png"
    if not grid.exists():
        raise FileNotFoundError(f"No representative sample found for {method['method_id']}")
    return extract_from_recon_grid(grid, 0), extract_from_recon_grid(grid, 2), str(grid)


def make_method_measurement(method: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    method_dir = IMPORTED / method["method_id"]
    cfg_path = method_dir / "resolved_config.yaml"
    config = apply_experiment_defaults(load_config(cfg_path))
    config["dataset_root"] = str(ROOT / "data")
    config["device"] = "cpu"
    device = torch.device("cpu")
    set_seed(int(config.get("seed", 42)))
    measurement = make_measurement(config, device)
    exact_info: dict[str, Any] = {"exact_A_required": bool(method.get("exact_a")), "exact_A_loaded": False}
    if method.get("exact_a"):
        exact_path = method_dir / "measurement_operator_exact.pt"
        payload = torch_load(exact_path, device)
        A = tensor_from_exact_payload(payload).to(device=device, dtype=torch.float32)
        if not hasattr(measurement, "set_A_override"):
            raise RuntimeError("Measurement operator does not support safe exact-A override.")
        override = measurement.set_A_override(
            A,
            metadata={"source": "phase44_provenance", "tensor_sha256": tensor_sha256(A)},
            rebuild_cache=True,
        )
        exact_info.update({"exact_A_loaded": True, "exact_A_path": str(exact_path), **override})
    else:
        A = measurement.get_current_A()
        exact_info.update({"exact_A_loaded": False, "operator_sha256": tensor_sha256(A)})
    return measurement, exact_info


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).reshape(1, 1, 64, 64)


def norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((numerator.norm() / denominator.norm().clamp_min(1e-12)).detach().cpu())


def mse_psnr_ssim_proxy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mse = float(np.mean((x - y) ** 2))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    # A tiny global SSIM proxy for provenance metadata; manuscript metrics remain unchanged.
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cxy = float(((x - mux) * (y - muy)).mean())
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mux * muy + c1) * (2 * cxy + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    return mse, psnr, float(ssim)


def display_linear(arr: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    arr = arr.astype(np.float32)
    if symmetric:
        scale = float(np.max(np.abs(arr)))
        if scale <= 1e-12:
            return np.full_like(arr, 0.5)
        return np.clip(0.5 + 0.5 * arr / scale, 0.0, 1.0)
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if hi - lo <= 1e-12:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "method_id",
        "sample_source",
        "measured_norm_ratio",
        "learned_null_norm_ratio",
        "audit_rel_meas_error",
        "gt_vs_final_mse",
        "gt_vs_final_psnr",
        "gt_vs_final_ssim_proxy",
        "exact_A_required",
        "exact_A_loaded",
        "exact_A_path",
        "operator_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "measured_norm_ratio",
        "learned_null_norm_ratio",
        "audit_rel_meas_error",
        "gt_vs_final_psnr",
        "gt_vs_final_ssim_proxy",
        "exact_A_loaded",
    ]
    lines = [
        "# Phase 44 Provenance Decomposition",
        "",
        "This is an evaluation-only, regularized soft decomposition of one representative sample per main STL-10 method.",
        "For Rademacher methods, the exported exact measurement operator is loaded with a safe cache rebuild.",
        "The decomposition uses \(x_{\\rm measured}=B_\\lambda A\\hat{x}\) and \(x_{\\rm learned}=(I-B_\\lambda A)\\hat{x}\).",
        "",
        "|" + "|".join(fields) + "|",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    lines += [
        "",
        "Interpretation: because \(B_\\lambda\) is regularized, these components are not a hard orthogonal subspace split.",
        "They show how much of the displayed representative reconstruction lies in the operator-recoverable component versus the regularized complement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_grid(items: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(items), 5, figsize=(11.5, 2.4 * len(items)), squeeze=False)
    titles = ["GT", "final xhat", "measured component", "learned/null component", "absolute error"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
    for row_idx, item in enumerate(items):
        panels = [
            (item["gt"], "gray", 0.0, 1.0),
            (item["xhat"], "gray", 0.0, 1.0),
            (display_linear(item["measured"]), "gray", 0.0, 1.0),
            (display_linear(item["learned"], symmetric=True), "coolwarm", 0.0, 1.0),
            (np.abs(item["xhat"] - item["gt"]), "magma", 0.0, float(max(0.12, np.percentile(np.abs(item["xhat"] - item["gt"]), 99)))),
        ]
        for col, (image, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col]
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            if col == 0:
                ax.text(-0.05, 0.5, item["method"], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
        metrics = item["metrics"]
        axes[row_idx, 1].text(
            0.02,
            -0.12,
            f"meas {metrics['measured_norm_ratio']:.2f}, null {metrics['learned_null_norm_ratio']:.2f}, audit {metrics['audit_rel_meas_error']:.2e}",
            transform=axes[row_idx, 1].transAxes,
            fontsize=8,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(PROV / "provenance_grid.png", dpi=300)
    fig.savefig(PROV / "provenance_grid.pdf")
    plt.close(fig)


def main() -> None:
    ensure_dir(PROV)
    rows: list[dict[str, Any]] = []
    grid_items: list[dict[str, Any]] = []
    for method in METHODS:
        gt, xhat, source = representative_images(method)
        measurement, exact_info = make_method_measurement(method)
        gt_t = to_tensor(gt)
        xhat_t = to_tensor(xhat)
        with torch.no_grad():
            gt_flat = measurement.flatten_img(gt_t)
            xhat_flat = measurement.flatten_img(xhat_t)
            measured_flat = measurement.data_solution(measurement.A_forward(xhat_flat), mode="ridge_pinv")
            learned_flat = xhat_flat - measured_flat
            y_ref = measurement.A_forward(gt_flat)
            audit_rel = norm_ratio(measurement.A_forward(xhat_flat) - y_ref, y_ref)
        measured = measurement.unflatten_img(measured_flat).reshape(64, 64).detach().cpu().numpy()
        learned = measurement.unflatten_img(learned_flat).reshape(64, 64).detach().cpu().numpy()
        mse, psnr, ssim = mse_psnr_ssim_proxy(gt, xhat)
        metrics = {
            "method": method["method"],
            "method_id": method["method_id"],
            "sample_source": source,
            "measured_norm_ratio": norm_ratio(measured_flat, xhat_flat),
            "learned_null_norm_ratio": norm_ratio(learned_flat, xhat_flat),
            "audit_rel_meas_error": audit_rel,
            "gt_vs_final_mse": mse,
            "gt_vs_final_psnr": psnr,
            "gt_vs_final_ssim_proxy": ssim,
            "exact_A_required": exact_info.get("exact_A_required", False),
            "exact_A_loaded": exact_info.get("exact_A_loaded", False),
            "exact_A_path": exact_info.get("exact_A_path", ""),
            "operator_sha256": exact_info.get("tensor_sha256", exact_info.get("operator_sha256", "")),
        }
        rows.append(metrics)
        grid_items.append({"method": method["method"], "gt": gt, "xhat": xhat, "measured": measured, "learned": learned, "metrics": metrics})

    write_csv_rows(PROV / "provenance_metrics.csv", rows)
    write_md_rows(PROV / "provenance_metrics.md", rows)
    (PROV / "provenance_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_grid(grid_items)
    print(
        {
            "csv": str(PROV / "provenance_metrics.csv"),
            "md": str(PROV / "provenance_metrics.md"),
            "png": str(PROV / "provenance_grid.png"),
            "pdf": str(PROV / "provenance_grid.pdf"),
            "methods": len(rows),
        }
    )


if __name__ == "__main__":
    main()
