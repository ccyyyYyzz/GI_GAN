from __future__ import annotations

from .phase12_common import PHASE12, as_float, load_registry
from .utils import ensure_dir


OUT = PHASE12 / "paper_figures"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.pdf")


def label_bars(ax, bars, digits=2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.{digits}f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    ensure_dir(OUT)
    import matplotlib.pyplot as plt

    rows = load_registry()
    by_id = {row["method_id"]: row for row in rows}
    stl_ids = ["stl10_hadamard10_local_full", "stl10_rademacher10_colab_full", "stl10_scrambled10_colab_full"]
    labels = ["Lowfreq\nHadamard", "Rademacher", "Scrambled\nHadamard"]
    psnr = [as_float(by_id[i].get("psnr")) or 0 for i in stl_ids]
    ssim = [as_float(by_id[i].get("ssim")) or 0 for i in stl_ids]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, psnr, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(22.0, color="black", linestyle="--", linewidth=1, label="HQ PSNR threshold")
    ax.set_ylabel("Model PSNR")
    ax.set_title("STL-10 10% Reconstruction PSNR")
    label_bars(ax, bars)
    ax.legend()
    save(fig, "fig_stl10_10pct_psnr_bar")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ssim, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(0.65, color="black", linestyle="--", linewidth=1, label="HQ SSIM threshold")
    ax.set_ylabel("Model SSIM")
    ax.set_title("STL-10 10% Reconstruction SSIM")
    label_bars(ax, bars, digits=3)
    ax.legend()
    save(fig, "fig_stl10_10pct_ssim_bar")
    plt.close(fig)
    delta_ids = stl_ids + ["stl10_hadamard5_local_medium"]
    delta_labels = ["H10", "R10", "S10", "H5"]
    back = [as_float(by_id[i].get("backproj_psnr")) or 0 for i in delta_ids]
    model = [as_float(by_id[i].get("psnr")) or 0 for i in delta_ids]
    x = list(range(len(delta_ids)))
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar([v - 0.18 for v in x], back, width=0.36, label="Backprojection")
    b2 = ax.bar([v + 0.18 for v in x], model, width=0.36, label="Model")
    ax.set_xticks(x, delta_labels)
    ax.set_ylabel("PSNR")
    ax.set_title("Backprojection vs Model")
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.legend()
    save(fig, "fig_backproj_vs_model_delta")
    plt.close(fig)
    pref = [row for row in rows if str(row.get("preferred_for_paper")).lower() == "true"]
    fig, ax = plt.subplots(figsize=(8, max(3, len(pref) * 0.45)))
    mat = []
    ylabels = []
    for row in pref:
        psnr_ok = 1 if (as_float(row.get("psnr")) or 0) >= (22 if row["threshold_type"] == "stl10_10pct" else 20 if row["threshold_type"] == "stl10_5pct" else 25) else 0
        ssim_ok = 1 if (as_float(row.get("ssim")) or 0) >= (0.65 if row["threshold_type"] == "stl10_10pct" else 0.60 if row["threshold_type"] == "stl10_5pct" else 0.80) else 0
        hq_ok = 1 if str(row.get("threshold_reached")).lower() == "true" else 0
        mat.append([psnr_ok, ssim_ok, hq_ok])
        ylabels.append(row["method_id"].replace("_", " "))
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2], ["PSNR", "SSIM", "HQ"])
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title("Threshold Map")
    save(fig, "fig_threshold_map")
    plt.close(fig)
    repro_pairs = [
        ("STL-10 H5", by_id["stl10_hadamard5_local_medium"], by_id["stl10_hadamard5_colab_medium"]),
        ("Fashion H5", by_id["fashion_hadamard5_local"], by_id["fashion_hadamard5_colab_full"]),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(repro_pairs)))
    local = [as_float(p[1].get("psnr")) or 0 for p in repro_pairs]
    colab = [as_float(p[2].get("psnr")) or 0 for p in repro_pairs]
    ax.bar([v - 0.18 for v in x], local, width=0.36, label="Local")
    ax.bar([v + 0.18 for v in x], colab, width=0.36, label="Colab")
    ax.set_xticks(x, [p[0] for p in repro_pairs])
    ax.set_ylabel("PSNR")
    ax.set_title("Local vs Colab Reproducibility")
    ax.legend()
    save(fig, "fig_reproducibility_local_colab")
    plt.close(fig)
    simple_ids = ["mnist_hadamard5_colab_full", "fashion_hadamard5_colab_full"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["MNIST 5%", "Fashion 5%"], [as_float(by_id[i].get("psnr")) or 0 for i in simple_ids], color=["#4c78a8", "#e45756"])
    ax.axhline(25.0, color="black", linestyle="--", linewidth=1, label="HQ PSNR threshold")
    ax.set_ylabel("Model PSNR")
    ax.set_title("Simple Domain 5% Results")
    label_bars(ax, bars)
    ax.legend()
    save(fig, "fig_simple_domain_results")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
