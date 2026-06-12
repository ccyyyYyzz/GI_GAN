from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from .datasets import get_val_dataloader
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config
from .metrics import batch_metrics, relative_measurement_error
from .models import build_generator
from .utils import (
    apply_experiment_defaults,
    load_config,
    reconstruct_from_measurements,
    resolve_device,
    set_seed,
)


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase41_inkscape_signal_trace"
COMP = OUT / "components"
PHASE15 = ROOT / "outputs_phase15" / "imported_noleak"
PHASE16_ABLATION = (
    ROOT
    / "outputs_phase16"
    / "supplementary_experiments"
    / "inference_ablation"
    / "real_inference_ablation_results.csv"
)
DATA_ROOT = ROOT / "data"

SAMPLE_ID = 2

METHODS: dict[str, dict[str, Any]] = {
    "rad5": {
        "method_id": "rademacher5_hq_noise001_colab",
        "label": "Rad-5",
        "dir": PHASE15 / "rademacher5_hq_noise001_colab",
        "exact_a_path": PHASE15 / "rademacher5_hq_noise001_colab" / "measurement_operator_exact.pt",
        "exact_a_used": True,
        "result_badge": "22.316 / 0.635",
        "ablation_badge": "22.202 vs 19.399",
    },
    "scr5": {
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "label": "Scr-5",
        "dir": PHASE15 / "scrambled_hadamard5_hq_noise001_colab",
        "exact_a_path": None,
        "exact_a_used": False,
        "result_badge": "22.271 / 0.632",
        "ablation_badge": "22.155 vs 6.352",
    },
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tensor_to_gray_image(x: torch.Tensor, path: Path) -> None:
    arr = x.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    im = Image.fromarray((arr * 255).round().astype("uint8"), mode="L").convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize((256, 256), Image.Resampling.NEAREST).save(path)


def tensor_to_signed_image(x: torch.Tensor, path: Path) -> None:
    arr = x.detach().float().cpu().squeeze().numpy()
    scale = float(max(abs(arr.min()), abs(arr.max()), 1e-8))
    scale = max(scale, float(torch.as_tensor(arr).abs().quantile(0.995).item()), 1e-8)
    arr = (arr / scale).clip(-1, 1)
    rgb = torch.zeros((arr.shape[0], arr.shape[1], 3), dtype=torch.float32)
    t = torch.from_numpy(arr).float()
    pos = torch.clamp(t, 0, 1)
    neg = torch.clamp(-t, 0, 1)
    rgb[..., 0] = 1.0 - 0.12 * neg
    rgb[..., 1] = 1.0 - 0.52 * pos - 0.35 * neg
    rgb[..., 2] = 1.0 - 0.82 * pos
    im = Image.fromarray((rgb.clamp(0, 1).numpy() * 255).round().astype("uint8"), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize((256, 256), Image.Resampling.NEAREST).save(path)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(31, 35, 40), anchor=None) -> None:
    draw.text(xy, text, fill=fill, anchor=anchor)


def make_pattern_preview(A: torch.Tensor, img_size: int, path: Path, count: int = 6) -> None:
    rows = min(count, A.shape[0])
    tile = 104
    pad = 18
    label_h = 28
    cols = 3
    w = cols * tile + (cols + 1) * pad
    h = 2 * (tile + label_h) + 3 * pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx in range(rows):
        pat = A[idx].detach().float().cpu().reshape(img_size, img_size)
        pmin = float(pat.min())
        pmax = float(pat.max())
        pat = (pat - pmin) / max(pmax - pmin, 1e-8)
        im = Image.fromarray((pat.numpy() * 255).round().astype("uint8"), mode="L").convert("RGB")
        x = pad + (idx % cols) * (tile + pad)
        y = pad + (idx // cols) * (tile + label_h + pad)
        canvas.paste(im.resize((tile, tile), Image.Resampling.NEAREST), (x, y))
        draw.rounded_rectangle((x, y, x + tile, y + tile), radius=8, outline=(190, 200, 212), width=2)
        draw_text(draw, (x + tile // 2, y + tile + 18), f"pattern {idx}", fill=(95, 107, 122), anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_vector_plot(values: torch.Tensor, path: Path, title: str, kind: str = "bars") -> None:
    vals = values.detach().float().cpu().flatten()
    if vals.numel() > 96:
        stride = math.ceil(vals.numel() / 96)
        vals = vals[::stride][:96]
    w, h = 420, 190
    im = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=16, outline=(205, 213, 224), width=2)
    draw_text(draw, (18, 18), title)
    x0, y0, x1, y1 = 30, 54, w - 26, h - 30
    draw.line((x0, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=(205, 213, 224), width=1)
    vmax = max(float(vals.abs().max()), 1e-9)
    if kind == "line":
        pts = []
        for i, val in enumerate(vals.tolist()):
            x = x0 + (x1 - x0) * i / max(1, vals.numel() - 1)
            y = (y0 + y1) / 2 - (float(val) / vmax) * (y1 - y0) * 0.45
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=(31, 119, 180), width=3)
    else:
        bar_w = max(2, int((x1 - x0) / max(1, vals.numel())))
        mid = (y0 + y1) // 2
        for i, val in enumerate(vals.tolist()):
            x = x0 + i * (x1 - x0) / max(1, vals.numel())
            y = mid - (float(val) / vmax) * (y1 - y0) * 0.45
            color = (31, 119, 180) if val >= 0 else (217, 121, 4)
            draw.rectangle((int(x), int(min(y, mid)), int(x + bar_w), int(max(y, mid))), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def make_relmeaserr_bar(path: Path, pre: float, post: float) -> None:
    w, h = 420, 180
    im = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=16, outline=(205, 213, 224), width=2)
    draw_text(draw, (18, 18), "RelMeasErr audit")
    vmax = max(pre, post, 1e-8)
    for i, (label, value, color) in enumerate(
        [("pre-audit", pre, (217, 121, 4)), ("post-audit", post, (35, 139, 69))]
    ):
        y = 62 + 48 * i
        draw_text(draw, (22, y + 11), label, fill=(95, 107, 122))
        draw.rounded_rectangle((130, y, 330, y + 24), radius=6, fill=(241, 245, 249))
        draw.rounded_rectangle((130, y, 130 + max(3, int(200 * value / vmax)), y + 24), radius=6, fill=color)
        draw_text(draw, (344, y + 12), f"{value:.3g}", anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def make_metric_bar(path: Path, labels: list[str], psnr: list[float], ssim: list[float]) -> None:
    w, h = 500, 210
    im = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=16, outline=(205, 213, 224), width=2)
    draw_text(draw, (18, 18), "Sample PSNR / SSIM")
    max_psnr = max(psnr + [1.0])
    for i, label in enumerate(labels):
        y = 62 + i * 42
        draw_text(draw, (22, y + 11), label, fill=(95, 107, 122))
        draw.rounded_rectangle((128, y, 318, y + 18), radius=5, fill=(241, 245, 249))
        draw.rounded_rectangle((128, y, 128 + int(190 * psnr[i] / max_psnr), y + 18), radius=5, fill=(31, 119, 180))
        draw_text(draw, (330, y + 9), f"{psnr[i]:.2f} dB", anchor="lm")
        draw_text(draw, (430, y + 9), f"{ssim[i]:.3f}", anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_method(method_key: str, meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = meta["dir"] / "resolved_config.yaml"
    checkpoint_path = meta["dir"] / "last.pt"
    config = apply_experiment_defaults(load_config(config_path))
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        config = apply_experiment_defaults(merged)
    config.update(
        {
            "dataset_root": str(DATA_ROOT),
            "device": str(device),
            "batch_size": SAMPLE_ID + 1,
            "num_workers": 0,
            "limit_val_samples": max(16, SAMPLE_ID + 1),
            "use_null_project": True,
            "use_dc_project": True,
            "output_range_mode": "clamp_eval_only",
        }
    )
    if meta["exact_a_path"] is not None:
        config["exact_A_required"] = True
        config["measurement_operator_exact_path"] = str(meta["exact_a_path"])
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    exact_a_info = apply_measurement_override_from_config(config, measurement, device)
    generator = build_generator(config, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("generator_ema") or checkpoint["generator"]
    else:
        state = checkpoint
    generator.load_state_dict(state)
    generator.eval()
    return config, {
        "device": device,
        "checkpoint": checkpoint_path,
        "measurement": measurement,
        "generator": generator,
        "exact_a_info": exact_a_info,
    }


def export_method(method_key: str, meta: dict[str, Any], ablation_rows: list[dict[str, str]]) -> dict[str, Any]:
    config, runtime = load_method(method_key, meta)
    device = runtime["device"]
    measurement = runtime["measurement"]
    generator = runtime["generator"]
    method_dir = COMP / method_key
    method_dir.mkdir(parents=True, exist_ok=True)

    loader = get_val_dataloader(
        dataset_root=str(DATA_ROOT),
        img_size=int(config["img_size"]),
        batch_size=SAMPLE_ID + 1,
        num_workers=0,
        limit_val_samples=max(16, SAMPLE_ID + 1),
        seed=int(config["seed"]),
        pin_memory=False,
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    batch = next(iter(loader))[0].to(device)
    x = batch[SAMPLE_ID : SAMPLE_ID + 1]
    with torch.no_grad():
        y = measurement.measure(x)
        torch.manual_seed(4100)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(4100)
        x_hat, x_data, extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=True,
            use_dc_project=True,
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )
        torch.manual_seed(4100)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(4100)
        x_no_audit, _, no_audit_extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=True,
            use_dc_project=False,
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )

    raw_residual = extras.get("raw_residual")
    filtered_residual = extras.get("filtered_residual")
    pre_audit = no_audit_extras.get("x_hat_unclamped", x_no_audit)
    final_unclamped = extras.get("x_hat_unclamped", x_hat)
    pre_rel = relative_measurement_error(measurement, pre_audit, y)
    post_rel = relative_measurement_error(measurement, final_unclamped, y)
    x_data_metrics = batch_metrics(x_data, x, measurement, y)
    pre_metrics = batch_metrics(pre_audit.clamp(0, 1), x, measurement, y)
    final_metrics = batch_metrics(x_hat, x, measurement, y)

    tensor_to_gray_image(x, method_dir / "ground_truth.png")
    tensor_to_gray_image(x_data, method_dir / "x_data.png")
    tensor_to_signed_image(raw_residual, method_dir / "raw_residual.png")
    tensor_to_signed_image(filtered_residual, method_dir / "filtered_residual.png")
    tensor_to_gray_image(pre_audit, method_dir / "pre_audit.png")
    tensor_to_gray_image(x_hat, method_dir / "final_audited.png")
    tensor_to_signed_image((x_hat - x).abs(), method_dir / "abs_error_final.png")
    make_pattern_preview(measurement.A, int(config["img_size"]), method_dir / "pattern_preview.png")
    make_vector_plot(y[0], method_dir / "bucket_vector.png", "bucket vector y")
    residual_pre = measurement.A_forward(measurement.flatten_img(pre_audit.float())) - y
    make_vector_plot(residual_pre[0], method_dir / "measurement_residual_pre.png", "A x_pre - y", kind="line")
    make_vector_plot(extras["measurement_residual_post"][0], method_dir / "measurement_residual_post.png", "A x_hat - y", kind="line")
    make_relmeaserr_bar(method_dir / "relmeaserr_bar.png", pre_rel, post_rel)
    make_metric_bar(
        method_dir / "psnr_ssim_bar.png",
        ["x_data", "pre-audit", "final"],
        [x_data_metrics["psnr"], pre_metrics["psnr"], final_metrics["psnr"]],
        [x_data_metrics["ssim"], pre_metrics["ssim"], final_metrics["ssim"]],
    )
    save_json(
        method_dir / "sample_manifest.json",
        {
            "method_key": method_key,
            "method_id": meta["method_id"],
            "sample_id": SAMPLE_ID,
            "checkpoint_path": str(runtime["checkpoint"]),
            "exact_A_info": runtime["exact_a_info"],
            "components": sorted(p.name for p in method_dir.glob("*.png")),
            "metrics": {
                "x_data": x_data_metrics,
                "pre_audit": {**pre_metrics, "rel_meas_error_unclamped": pre_rel},
                "final": {**final_metrics, "rel_meas_error_unclamped": post_rel},
            },
            "pre_audit_note": "pre_audit.png is the same checkpoint evaluated with final measurement-consistency projection disabled; this is an inference ablation, not a separately trained network.",
        },
    )

    return {
        "method_id": meta["method_id"],
        "sample_id": SAMPLE_ID,
        "psnr_x_data": x_data_metrics["psnr"],
        "ssim_x_data": x_data_metrics["ssim"],
        "psnr_pre_audit": pre_metrics["psnr"],
        "ssim_pre_audit": pre_metrics["ssim"],
        "psnr_final": final_metrics["psnr"],
        "ssim_final": final_metrics["ssim"],
        "relmeaserr_pre": pre_rel,
        "relmeaserr_post": post_rel,
        "raw_residual_available": raw_residual is not None,
        "filtered_residual_available": filtered_residual is not None,
        "exact_A_used": bool(runtime["exact_a_info"].get("exact_A_loaded", False)),
        "checkpoint_path": str(runtime["checkpoint"]),
        "status": "completed_eval_only",
    }


def write_manifest(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 41 Signal-Trace Component Manifest",
        "",
        "These components were exported by eval-only forward passes from existing strict no-leak checkpoints. No training, new benchmark, PCA/oracle run, architecture pilot, or sampling-scaling experiment was run.",
        "",
        f"Representative validation sample id: `{SAMPLE_ID}`.",
        "",
        "## Exported Methods",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['method_id']}`: x_data PSNR {float(row['psnr_x_data']):.3f}, "
            f"pre-audit PSNR {float(row['psnr_pre_audit']):.3f}, final PSNR {float(row['psnr_final']):.3f}, "
            f"RelMeasErr pre/post {float(row['relmeaserr_pre']):.3g}/{float(row['relmeaserr_post']):.3g}."
        )
    lines += [
        "",
        "## Component Files per Method",
        "",
        "`ground_truth.png`, `pattern_preview.png`, `bucket_vector.png`, `x_data.png`, `raw_residual.png`, `filtered_residual.png`, `pre_audit.png`, `final_audited.png`, `abs_error_final.png`, `measurement_residual_pre.png`, `measurement_residual_post.png`, `relmeaserr_bar.png`, and `psnr_ssim_bar.png`.",
        "",
        "`pre_audit.png` is the same checkpoint evaluated with the final measurement-consistency projection disabled. It is an inference ablation used to expose the audit path, not a separately trained unconstrained network.",
        "",
        "Rademacher methods use the imported exact-A artifact through the safe cache-rebuild override path.",
    ]
    (COMP / "component_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if COMP.exists():
        shutil.rmtree(COMP)
    COMP.mkdir(parents=True, exist_ok=True)
    ablation_rows = read_csv(PHASE16_ABLATION)
    del ablation_rows
    rows = []
    for method_key, meta in METHODS.items():
        rows.append(export_method(method_key, meta, []))
    fields = [
        "method_id",
        "sample_id",
        "psnr_x_data",
        "ssim_x_data",
        "psnr_pre_audit",
        "ssim_pre_audit",
        "psnr_final",
        "ssim_final",
        "relmeaserr_pre",
        "relmeaserr_post",
        "raw_residual_available",
        "filtered_residual_available",
        "exact_A_used",
        "checkpoint_path",
        "status",
    ]
    write_csv(COMP / "component_metrics.csv", rows, fields)
    write_manifest(rows)
    print({"components": str(COMP), "rows": len(rows)})


if __name__ == "__main__":
    main()
