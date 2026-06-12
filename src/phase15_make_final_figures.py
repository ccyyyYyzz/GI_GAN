from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .phase15_common import PHASE15, ensure_dir, load_registry, numeric, read_csv, threshold_for, write_json


OUT_DIR = PHASE15 / "paper_figures_final"


def short_label(method_id: str) -> str:
    mapping = {
        "mnist_hadamard5_full_colab": "MNIST H5",
        "fashion_hadamard5_full_colab": "Fashion H5",
        "scrambled_hadamard5_hq_noise001_colab": "Scr-H5",
        "rademacher5_hq_noise001_colab": "Rad-5",
        "scrambled_hadamard10_full_noise001_colab": "Scr-H10",
        "rademacher10_full_noise001_colab": "Rad-10",
    }
    return mapping.get(method_id, method_id.replace("_colab", ""))


def save_current(name: str) -> None:
    ensure_dir(OUT_DIR)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=220)
    plt.savefig(OUT_DIR / f"{name}.pdf")
    plt.close()


def bar_chart(rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, name: str, threshold: float | None = None) -> None:
    labels = [short_label(row["method_id"]) for row in rows]
    values = [numeric(row[metric]) for row in rows]
    colors = ["#2f6f9f" if "rademacher" not in row["method_id"] else "#bf6f33" for row in rows]
    fig, ax = plt.subplots(figsize=(max(5.2, 1.2 * len(rows)), 3.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
    if threshold is not None:
        ax.axhline(threshold, color="#b3261e", linestyle="--", linewidth=1.2, label=f"threshold {threshold:g}")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        if value == value:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    save_current(name)


def bp_vs_model(rows: list[dict[str, Any]]) -> None:
    labels = [short_label(row["method_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    bp = [numeric(row["backproj_psnr"]) for row in rows]
    model = [numeric(row["psnr"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - width / 2, bp, width, label="Backprojection", color="#8aa6b1", edgecolor="#222222", linewidth=0.5)
    ax.bar(x + width / 2, model, width, label="NS-MC-GAN", color="#3b7a57", edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Final strict no-leak: backprojection vs model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("fig_bp_vs_model_final")


def threshold_summary(rows: list[dict[str, Any]]) -> None:
    labels = [short_label(row["method_id"]) for row in rows]
    psnr_margins = []
    ssim_margins = []
    for row in rows:
        _, thr_psnr, thr_ssim = threshold_for(row["dataset"], numeric(row["sampling_ratio"]))
        psnr_margins.append(numeric(row["psnr"]) - thr_psnr)
        ssim_margins.append((numeric(row["ssim"]) - thr_ssim) * 20.0)
    data = np.array([psnr_margins, ssim_margins])
    fig, ax = plt.subplots(figsize=(8.0, 2.7))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-2.0, vmax=4.0)
    ax.set_yticks([0, 1], ["PSNR margin", "SSIM margin x20"])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_title("Threshold margin summary")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save_current("fig_threshold_summary_final")


def simple_domains(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["dataset"] in {"MNIST", "Fashion-MNIST"}]
    labels = [short_label(row["method_id"]) for row in selected]
    psnr = [numeric(row["psnr"]) for row in selected]
    ssim = [numeric(row["ssim"]) for row in selected]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    ax2 = ax1.twinx()
    x = np.arange(len(selected))
    ax1.bar(x - 0.18, psnr, 0.34, label="PSNR", color="#2f6f9f", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x + 0.18, ssim, 0.34, label="SSIM", color="#9a6f9f", edgecolor="#222222", linewidth=0.5)
    ax1.axhline(25.0, color="#2f6f9f", linestyle="--", linewidth=1.0)
    ax2.axhline(0.80, color="#9a6f9f", linestyle="--", linewidth=1.0)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("SSIM")
    ax1.set_title("Simple-domain final no-leak sanity")
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    save_current("fig_simple_domains_final")


def main() -> None:
    ensure_dir(OUT_DIR)
    rows = load_registry()
    stl5 = [row for row in rows if row["dataset"] == "STL-10" and abs(numeric(row["sampling_ratio"]) - 0.05) < 1e-9]
    stl10 = [row for row in rows if row["dataset"] == "STL-10" and abs(numeric(row["sampling_ratio"]) - 0.10) < 1e-9]

    bar_chart(stl5, "psnr", "STL-10 5% final PSNR", "PSNR (dB)", "fig_stl10_5pct_final_psnr", 20.0)
    bar_chart(stl5, "ssim", "STL-10 5% final SSIM", "SSIM", "fig_stl10_5pct_final_ssim", 0.60)
    bar_chart(stl10, "psnr", "STL-10 10% final PSNR", "PSNR (dB)", "fig_stl10_10pct_final_psnr", 22.0)
    bar_chart(stl10, "ssim", "STL-10 10% final SSIM", "SSIM", "fig_stl10_10pct_final_ssim", 0.65)
    bp_vs_model(rows)
    bar_chart(rows, "delta_psnr", "Final strict no-leak PSNR gain", "Delta PSNR (dB)", "fig_delta_psnr_final")
    threshold_summary(rows)
    simple_domains(rows)

    manifest = {
        "output_dir": str(OUT_DIR),
        "figures": sorted(p.name for p in OUT_DIR.glob("*.png")),
        "note": "Each PNG has a matching PDF file.",
    }
    write_json(OUT_DIR / "paper_figures_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
