from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .checkpoint_utils import find_best_checkpoint
from .phase11_common import ROOT10, ROOT11, as_float, ensure_dir, plot_bar, read_json, write_csv_rows, write_md_table


METHODS = [
    ("hadamard10_full_noise001", Path("configs/phase10/hadamard10_full_noise001.yaml"), ROOT10 / "hadamard10_full_noise001"),
    ("rademacher10_full_noise001", Path("configs/phase10/rademacher10_full_noise001.yaml"), ROOT10 / "rademacher10_full_noise001"),
    ("scrambled_hadamard10_full_noise001", Path("configs/phase10/scrambled_hadamard10_full_noise001.yaml"), ROOT10 / "scrambled_hadamard10_full_noise001"),
    ("hadamard5_medium_noise001", Path("configs/phase10/hadamard5_medium_noise001.yaml"), ROOT10 / "hadamard5_medium_noise001"),
]
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]
FIELDS = ["method", "noise_std", "model_psnr", "model_ssim", "model_mse", "model_rel_meas_err", "backproj_psnr", "backproj_ssim", "status"]


def eval_one(method: str, config_path: Path, checkpoint: Path, root: Path, noise: float) -> dict:
    out = ensure_dir(root / f"noise_{noise:g}".replace(".", "p"))
    cmd = [
        sys.executable,
        "-m",
        "src.eval",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(out),
        "--noise_std",
        str(noise),
        "--device",
        "cuda",
    ]
    status = "completed"
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        status = "failed"
    metrics = read_json(out / "eval_metrics.json")
    model = metrics.get("model", {})
    back = metrics.get("backprojection", {})
    return {
        "method": method,
        "noise_std": noise,
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "model_rel_meas_err": model.get("rel_meas_error", ""),
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "status": status if metrics else "failed",
    }


def plot_noise(rows: list[dict], root: Path, key: str, name: str) -> None:
    try:
        import matplotlib.pyplot as plt

        completed = [row for row in rows if row.get("status") == "completed" and as_float(row.get(key)) is not None]
        fig, ax = plt.subplots(figsize=(6, 4))
        if completed:
            xs = [float(row["noise_std"]) for row in completed]
            ys = [float(row[key]) for row in completed]
            ax.plot(xs, ys, marker="o")
        ax.set_xlabel("noise_std")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / f"noise_sweep_{key.split('_')[-1]}.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        (root / f"noise_sweep_{key}.txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def main() -> None:
    base = ensure_dir(ROOT11 / "noise_sweep")
    summary_rows = []
    for method, config_path, output_dir in METHODS:
        root = ensure_dir(base / method)
        checkpoint = find_best_checkpoint(output_dir)
        rows = []
        if checkpoint is None:
            rows = [{field: "" for field in FIELDS} | {"method": method, "noise_std": noise, "status": "missing_checkpoint"} for noise in NOISE_LEVELS]
        else:
            rows = [eval_one(method, config_path, checkpoint, root, noise) for noise in NOISE_LEVELS]
        write_csv_rows(rows, root / "noise_sweep_metrics.csv", FIELDS)
        write_md_table(rows, root / "noise_sweep_metrics.md", FIELDS)
        plot_noise(rows, root, "model_psnr", "PSNR")
        plot_noise(rows, root, "model_ssim", "SSIM")
        best = max((row for row in rows if as_float(row.get("model_psnr")) is not None), key=lambda row: float(row["model_psnr"]), default=None)
        summary_rows.append(
            {
                "method": method,
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "best_noise_std": "" if best is None else best.get("noise_std"),
                "best_model_psnr": "" if best is None else best.get("model_psnr"),
                "best_model_ssim": "" if best is None else best.get("model_ssim"),
                "status": "completed" if checkpoint is not None else "missing_checkpoint",
            }
        )
    fields = ["method", "checkpoint", "best_noise_std", "best_model_psnr", "best_model_ssim", "status"]
    write_csv_rows(summary_rows, ROOT11 / "noise_sweep_summary.csv", fields)
    write_md_table(summary_rows, ROOT11 / "noise_sweep_summary.md", fields)
    print(f"Phase 11 noise sweep summary written to: {ROOT11 / 'noise_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
