from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .datasets import get_val_dataloader
from .measurement import GhostMeasurementOperator
from .metrics import batch_metrics
from .utils import ensure_dir, resolve_device, save_json, set_seed


OUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase9/sanity_hadamard")


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check Hadamard measurement operators.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--dataset_root", default="E:/ns_mc_gan_gi/data")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def aa_stats(A: torch.Tensor) -> dict[str, float]:
    gram = A @ A.T
    diag = torch.diagonal(gram)
    offdiag = gram - torch.diag_embed(diag)
    return {
        "aa_t_diag_mean": float(diag.mean().detach().cpu()),
        "aa_t_diag_std": float(diag.std(unbiased=False).detach().cpu()),
        "aa_t_offdiag_abs_mean": float(offdiag.abs().sum().detach().cpu() / max(1, A.shape[0] * (A.shape[0] - 1))),
        "aa_t_offdiag_abs_max": float(offdiag.abs().max().detach().cpu()) if A.shape[0] > 1 else 0.0,
    }


def make_op(
    *,
    img_size: int,
    sampling_ratio: float,
    pattern_type: str,
    device: torch.device,
    seed: int,
    include_dc: bool = True,
    row_order: str = "sequency",
    matrix_normalization: str = "orthonormal_rows",
    backprojection_mode: str = "hadamard_zero_filled",
) -> GhostMeasurementOperator:
    return GhostMeasurementOperator(
        img_size=img_size,
        sampling_ratio=sampling_ratio,
        pattern_type=pattern_type,
        noise_std=0.0,
        lambda_dc=1e-3,
        backprojection_mode=backprojection_mode,
        matrix_normalization=matrix_normalization,
        hadamard_include_dc=include_dc,
        hadamard_row_order=row_order,
        device=device,
        seed=seed,
    )


def metric_row(
    op: GhostMeasurementOperator,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    ratio: float,
    label: str,
    include_dc,
) -> dict:
    try:
        x_data = op.unflatten_img(op.data_solution(y, mode=mode))
        metrics = batch_metrics(x_data, x, op, y)
        status = "ok"
    except Exception as exc:
        metrics = {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "rel_meas_error": float("nan")}
        status = f"error: {exc}"
    return {
        "sampling_ratio": ratio,
        "pattern_label": label,
        "pattern_type": op.pattern_type,
        "matrix_normalization": op.matrix_normalization,
        "hadamard_include_dc": include_dc,
        "backprojection_mode": mode,
        "mse": metrics["mse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "rel_meas_error": metrics.get("rel_meas_error", ""),
        "status": status,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    set_seed(args.seed)

    full_op = make_op(
        img_size=args.img_size,
        sampling_ratio=1.0,
        pattern_type="hadamard",
        device=device,
        seed=args.seed,
        include_dc=True,
        row_order="natural",
    )
    x = torch.rand(4, 1, args.img_size, args.img_size, device=device)
    x_flat = full_op.flatten_img(x)
    y = full_op.A_forward(x_flat)
    rec = full_op.hadamard_zero_filled_solution(y)
    rel_error = float((rec - x_flat).norm(dim=1).div(x_flat.norm(dim=1).clamp_min(1e-12)).mean().detach().cpu())
    full_pass = rel_error < 1e-5

    partial_op = make_op(
        img_size=args.img_size,
        sampling_ratio=0.10,
        pattern_type="lowfreq_hadamard",
        device=device,
        seed=args.seed,
        include_dc=True,
    )
    lowfreq_skip_op = make_op(
        img_size=args.img_size,
        sampling_ratio=0.10,
        pattern_type="lowfreq_hadamard",
        device=device,
        seed=args.seed,
        include_dc=False,
    )
    full_aa = aa_stats(full_op.A)
    partial_aa = aa_stats(partial_op.A)
    selected = partial_op.measurement_metadata or {}
    selected_rows = selected.get("selected_rows", [])

    loader = get_val_dataloader(
        dataset_root=args.dataset_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=0,
        limit_val_samples=args.batch_size,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        dataset_name="stl10",
    )
    batch = next(iter(loader))
    stl_x = batch[0].to(device, non_blocking=True)

    quality_rows = []
    specs = [
        ("rademacher", True, "sequency", "legacy_sqrt_m", "ridge_pinv", "rademacher"),
        ("lowfreq_hadamard", True, "sequency", "orthonormal_rows", "hadamard_zero_filled", "lowfreq_include_dc"),
        ("lowfreq_hadamard", False, "sequency", "orthonormal_rows", "hadamard_zero_filled", "lowfreq_skip_dc"),
        ("scrambled_hadamard", True, "sequency", "orthonormal_rows", "hadamard_zero_filled", "scrambled_hadamard"),
        ("hadamard", True, "natural", "orthonormal_rows", "hadamard_zero_filled", "hadamard_sequential"),
    ]
    for ratio in [0.05, 0.10]:
        for pattern, include_dc, row_order, norm, primary_mode, label in specs:
            op = make_op(
                img_size=args.img_size,
                sampling_ratio=ratio,
                pattern_type=pattern,
                device=device,
                seed=args.seed,
                include_dc=include_dc,
                row_order=row_order,
                matrix_normalization=norm,
                backprojection_mode=primary_mode,
            )
            y_stl = op.measure(stl_x)
            for mode in ["ridge_pinv", "adjoint"]:
                quality_rows.append(metric_row(op, stl_x, y_stl, mode, ratio, label, include_dc))
            if pattern != "rademacher":
                quality_rows.append(metric_row(op, stl_x, y_stl, "hadamard_zero_filled", ratio, label, include_dc))

    write_csv(quality_rows, output_dir / "backproj_quality_table.csv")
    write_markdown_table(quality_rows, output_dir / "backproj_quality_table.md")

    def lookup(ratio: float, label: str, mode: str, metric: str) -> float:
        for row in quality_rows:
            if row["sampling_ratio"] == ratio and row["pattern_label"] == label and row["backprojection_mode"] == mode:
                return float(row.get(metric, float("nan")))
        return float("nan")

    warnings = []
    for ratio in [0.05, 0.10]:
        low_psnr = lookup(ratio, "lowfreq_include_dc", "hadamard_zero_filled", "psnr")
        rad_psnr = lookup(ratio, "rademacher", "ridge_pinv", "psnr")
        if low_psnr + 3.0 < rad_psnr:
            warnings.append(
                f"potential_bug_or_unhelpful_ordering: lowfreq_include_dc PSNR {low_psnr:.3f} is much worse than rademacher ridge {rad_psnr:.3f} at ratio={ratio}."
            )

    summary = {
        "status": "passed" if full_pass else "failed",
        "full_sampling_exact_passed": full_pass,
        "full_sampling_rel_error": rel_error,
        "full_sampling_threshold": 1e-5,
        "full_aa_t": full_aa,
        "partial_aa_t_10pct_lowfreq_include_dc": partial_aa,
        "lowfreq_selected_row_check": {
            "include_dc": bool(selected.get("dc_row_selected", False)),
            "selected_rows_count": len(selected_rows),
            "expected_m": partial_op.m,
            "first_selected_rows": selected_rows[:16],
            "first_sign_changes": selected.get("selected_sign_changes_preview", []),
        },
        "lowfreq_skip_dc_selected_row_check": {
            "include_dc": bool((lowfreq_skip_op.measurement_metadata or {}).get("dc_row_selected", False)),
            "selected_rows_count": len((lowfreq_skip_op.measurement_metadata or {}).get("selected_rows", [])),
        },
        "warnings": warnings,
    }
    save_json(summary, output_dir / "hadamard_sanity.json")

    lines = [
        "# Hadamard Sanity",
        "",
        f"- status: {summary['status']}",
        f"- full_sampling_exact_passed: {full_pass}",
        f"- full_sampling_rel_error: {rel_error:.8g}",
        f"- full AA^T diag mean/std: {full_aa['aa_t_diag_mean']:.6g} / {full_aa['aa_t_diag_std']:.6g}",
        f"- full AA^T offdiag abs mean/max: {full_aa['aa_t_offdiag_abs_mean']:.6g} / {full_aa['aa_t_offdiag_abs_max']:.6g}",
        f"- partial AA^T diag mean/std: {partial_aa['aa_t_diag_mean']:.6g} / {partial_aa['aa_t_diag_std']:.6g}",
        f"- partial AA^T offdiag abs mean/max: {partial_aa['aa_t_offdiag_abs_mean']:.6g} / {partial_aa['aa_t_offdiag_abs_max']:.6g}",
        f"- lowfreq includes DC: {summary['lowfreq_selected_row_check']['include_dc']}",
        f"- lowfreq selected rows count: {summary['lowfreq_selected_row_check']['selected_rows_count']}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(["", "## Backprojection Table", "", "See `backproj_quality_table.md`."])
    (output_dir / "hadamard_sanity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Hadamard sanity status: {summary['status']}")
    print(f"full_sampling_rel_error={rel_error:.8g}")
    print(f"Outputs written to: {output_dir}")
    if not full_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
