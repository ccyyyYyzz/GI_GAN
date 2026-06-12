from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .utils import ensure_dir


DEFAULT_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase10/noise_sweep_hadamard10")
DEFAULT_CHECKPOINT = Path("E:/ns_mc_gan_gi/outputs_phase10/hadamard10_full_noise001/best_hq.pt")
DEFAULT_CONFIG = Path("configs/phase10/hadamard10_full_noise001.yaml")
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 10 Hadamard checkpoint under a noise sweep.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(rows: list[dict], root: Path) -> None:
    csv_path = root / "noise_sweep_results.csv"
    fields = ["noise_std", "model_psnr", "model_ssim", "model_mse", "model_rel_meas_err", "backproj_psnr", "backproj_ssim", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["|noise_std|model_psnr|model_ssim|model_rel_meas_err|backproj_psnr|status|", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"|{row.get('noise_std')}|{row.get('model_psnr')}|{row.get('model_ssim')}|"
            f"{row.get('model_rel_meas_err')}|{row.get('backproj_psnr')}|{row.get('status')}|"
        )
    (root / "noise_sweep_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(rows, root / "noise_sweep_psnr.png", "model_psnr", "PSNR")
    plot(rows, root / "noise_sweep_ssim.png", "model_ssim", "SSIM")


def plot(rows: list[dict], path: Path, key: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt

        values = [(float(row["noise_std"]), float(row[key])) for row in rows if row.get("status") == "completed" and row.get(key) not in {"", None}]
        fig, ax = plt.subplots(figsize=(6, 4))
        if values:
            xs, ys = zip(*values)
            ax.plot(xs, ys, marker="o")
        ax.set_xlabel("noise_std")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        rows = [
            {
                "noise_std": noise,
                "model_psnr": "",
                "model_ssim": "",
                "model_mse": "",
                "model_rel_meas_err": "",
                "backproj_psnr": "",
                "backproj_ssim": "",
                "status": "missing_checkpoint",
            }
            for noise in NOISE_LEVELS
        ]
        write_outputs(rows, root)
        print(f"Checkpoint missing, wrote missing noise sweep to: {root}")
        return
    rows = []
    for noise in NOISE_LEVELS:
        out = ensure_dir(root / f"noise_{noise:g}".replace(".", "p"))
        cmd = [
            sys.executable,
            "-m",
            "src.eval",
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(out),
            "--noise_std",
            str(noise),
            "--device",
            args.device,
        ]
        if args.limit_val_samples is not None:
            cmd.extend(["--limit_val_samples", str(args.limit_val_samples)])
        status = "completed"
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            status = "failed"
        metrics = read_metrics(out / "eval_metrics.json")
        model = metrics.get("model", {})
        back = metrics.get("backprojection", {})
        rows.append(
            {
                "noise_std": noise,
                "model_psnr": model.get("psnr", ""),
                "model_ssim": model.get("ssim", ""),
                "model_mse": model.get("mse", ""),
                "model_rel_meas_err": model.get("rel_meas_error", ""),
                "backproj_psnr": back.get("psnr", ""),
                "backproj_ssim": back.get("ssim", ""),
                "status": status if metrics else "failed",
            }
        )
    write_outputs(rows, root)
    print(f"Noise sweep written to: {root}")


if __name__ == "__main__":
    main()
