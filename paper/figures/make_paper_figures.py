from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE
CERT = Path(r"E:\ns_mc_gan_gi\results\cert_package_20260612")
TABLES = CERT / "tables"
CACHE = CERT / "cache"

T4 = TABLES / "T4_pairs.csv"
T5 = TABLES / "T5_rho.csv"
MAIN_RAD5 = CACHE / "main_rad5.npz"
A_RAD5 = CACHE / "A_rad5.npy"

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#6B7280",
    "light_blue": "#DCE8F5",
    "light_orange": "#FBE6CB",
    "light_green": "#DDEED9",
    "light_gray": "#F1F3F5",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec=None, fontsize=9, lw=1.2, rounded=True):
    x, y = xy
    w, h = wh
    boxstyle = "round,pad=0.02,rounding_size=0.025" if rounded else "square,pad=0.02"
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=boxstyle,
        linewidth=lw,
        edgecolor=ec or COLORS["gray"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, x0, y0, x1, y1, color=None, lw=1.4):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color or COLORS["gray"], shrinkA=4, shrinkB=4),
    )


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, (0.035, 0.58), (0.09, 0.19), r"object" "\n" r"$x$", COLORS["light_gray"], fontsize=10)

    # Forward model with sensing matrix and bucket signal strip.
    add_box(ax, (0.175, 0.55), (0.18, 0.25), "", COLORS["light_blue"], ec=COLORS["blue"])
    ax.text(0.265, 0.745, r"forward model", ha="center", va="center", fontsize=9)
    ax.text(0.23, 0.64, r"$A$", ha="center", va="center", fontsize=16, color=COLORS["blue"])
    strip_x, strip_y = 0.275, 0.605
    for k, val in enumerate([0.25, 0.65, 0.45, 0.85, 0.35, 0.72]):
        ax.add_patch(
            patches.Rectangle(
                (strip_x + 0.011 * k, strip_y),
                0.008,
                0.09 * val,
                linewidth=0,
                facecolor=COLORS["blue"],
                alpha=0.85,
            )
        )
    ax.text(0.314, 0.645, r"$y$", ha="center", va="center", fontsize=12, color=COLORS["blue"])
    ax.text(0.265, 0.575, r"$y=Ax+\varepsilon$", ha="center", va="center", fontsize=9)
    ax.text(0.265, 0.51, r"low sampling, $m\ll n$", ha="center", va="top", fontsize=8, color=COLORS["gray"])

    add_box(
        ax,
        (0.42, 0.57),
        (0.16, 0.21),
        r"Generator" "\n" r"$G_\theta$ (GAN)",
        COLORS["light_orange"],
        ec=COLORS["orange"],
        fontsize=10,
    )
    ax.text(0.615, 0.715, r"$v_\theta$", ha="center", va="center", fontsize=11)
    add_box(
        ax,
        (0.66, 0.57),
        (0.14, 0.21),
        r"Audit" "\n" r"$\Pi_y^\lambda$",
        COLORS["light_blue"],
        ec=COLORS["blue"],
        fontsize=10,
    )
    add_box(ax, (0.855, 0.58), (0.09, 0.19), r"output" "\n" r"$\hat{x}$", COLORS["light_green"], ec=COLORS["green"], fontsize=10)

    add_arrow(ax, 0.125, 0.675, 0.175, 0.675)
    add_arrow(ax, 0.355, 0.675, 0.42, 0.675)
    add_arrow(ax, 0.58, 0.675, 0.66, 0.675)
    add_arrow(ax, 0.80, 0.675, 0.855, 0.675)

    ax.plot([0.90, 0.90], [0.58, 0.41], color=COLORS["gray"], lw=1.0)
    add_arrow(ax, 0.90, 0.42, 0.39, 0.32, color=COLORS["green"])
    ax.text(0.64, 0.39, r"range-null decomposition", ha="center", va="bottom", fontsize=9, color=COLORS["gray"])

    add_box(
        ax,
        (0.23, 0.13),
        (0.24, 0.16),
        r"$P_{\mathrm{R}}\hat{x}$" "\n" r"measurement-determined" "\n" r"row space",
        COLORS["light_blue"],
        ec=COLORS["blue"],
        fontsize=7.8,
    )
    add_box(
        ax,
        (0.53, 0.13),
        (0.24, 0.16),
        r"$P_0\hat{x}$" "\n" r"prior-supplied" "\n" r"null space",
        COLORS["light_orange"],
        ec=COLORS["orange"],
        fontsize=7.8,
    )
    ax.text(0.50, 0.21, r"$\oplus$", ha="center", va="center", fontsize=18, color=COLORS["gray"])
    ax.text(0.50, 0.075, r"$\hat{x}=P_{\mathrm{R}}\hat{x}\oplus P_0\hat{x}$, with $P_{\mathrm{R}}\hat{x}\perp P_0\hat{x}$",
            ha="center", va="center", fontsize=9)

    save_both(fig, "pipeline_range_null_schematic")


def read_t5_summary() -> dict[str, str]:
    summary = {}
    if not T5.exists():
        return summary
    with T5.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        key = f"{row['ensemble']}_{row['convention']}"
        summary[key] = row
    return summary


def figure_theorem_bound() -> None:
    s = np.linspace(0.0, 0.95, 400)
    y = -10.0 * np.log10(1.0 - s)

    fig, ax = plt.subplots(figsize=(3.55, 2.8))
    ax.plot(s, y, ls="--", lw=2.1, color=COLORS["blue"])
    ax.text(
        0.42,
        3.15,
        r"$\Delta\mathrm{PSNR}_{\max}=-10\log_{10}(1-s)$",
        color=COLORS["blue"],
        fontsize=7.6,
        rotation=18,
        ha="center",
        va="center",
    )

    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 13.2)
    ax.set_xlabel(r"$s=\|P_{\mathrm{R}}e\|^2/\|e\|^2$")
    ax.set_ylabel(r"audit $\Delta$PSNR upper bound (dB)")
    ax.grid(alpha=0.22, lw=0.7)
    save_both(fig, "theorem2_psnr_bound")


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load_pair_row() -> dict[str, str]:
    if not T4.exists():
        raise FileNotFoundError(f"Missing T4 pairs table: {T4}")
    with T4.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["ensemble"] == "rad5" and row["i"] == "1789" and row["j"] == "935":
            return row
    rad5 = [r for r in rows if r["ensemble"] == "rad5"]
    if not rad5:
        raise RuntimeError("No rad5 rows found in T4_pairs.csv")
    return min(rad5, key=lambda r: abs(float(r["RelMeasErr_truth_vs_yi"]) - 5.5e-3))


def figure_feasible_hallucination() -> None:
    row = load_pair_row()
    i = int(row["i"])
    j = int(row["j"])

    A = np.load(A_RAD5).astype(np.float64)
    d = np.load(MAIN_RAD5)
    x = d["x"].astype(np.float64)
    y = d["y"].astype(np.float64)
    labels = d["labels"]

    AAt = A @ A.T
    cho = np.linalg.cholesky(AAt)

    def adag(w: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(cho, w.T)
        z = np.linalg.solve(cho.T, z)
        return z.T @ A

    u = x[j] - adag((x[j] @ A.T - y[i])[None, :])[0]
    rm_truth = rel(x[i] @ A.T, y[i])
    rm_u = rel(u @ A.T, y[i])

    ci = STL10_CLASSES[int(labels[i])]
    cj = STL10_CLASSES[int(labels[j])]

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.85))
    for ax, img, title, subtitle in [
        (
            axes[0],
            x[i],
            rf"(a) truth $x_i$ ({ci})",
            rf"$\mathrm{{RelMeasErr}}={rm_truth:.3e}$",
        ),
        (
            axes[1],
            np.clip(u, 0, 1),
            rf"(b) feasible wrong image $u_{{ij}}$ ({cj})",
            rf"$\mathrm{{RelMeasErr}}={rm_u:.3e}$",
        ),
    ]:
        ax.imshow(img.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, pad=5)
        ax.text(
            0.5,
            -0.08,
            subtitle,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8.3,
            color=COLORS["gray"],
        )

    fig.suptitle(
        rf"Rad-5 pair $i={i}$, $j={j}$: wrong image matches $y_i$ more tightly than truth",
        fontsize=9.5,
        y=0.99,
    )
    save_both(fig, "feasible_hallucination_pair")

    print("feasible_hallucination_pair")
    print(f"  ensemble=rad5 i={i} j={j} class_i={ci} class_j={cj}")
    print(f"  RelMeasErr truth={rm_truth:.6e} table={float(row['RelMeasErr_truth_vs_yi']):.6e}")
    print(f"  RelMeasErr u={rm_u:.6e} table={float(row['RelMeasErr_u_vs_yi']):.6e}")


def main() -> None:
    setup_style()
    figure_pipeline()
    figure_theorem_bound()
    figure_feasible_hallucination()
    print(f"saved figures under {OUT}")


if __name__ == "__main__":
    main()
