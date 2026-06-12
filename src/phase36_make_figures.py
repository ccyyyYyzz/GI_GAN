from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase36_conventional_gi_aligned"
FIG_DIR = OUT / "figures"
ATTR_CSV = ROOT / "outputs_phase16" / "supplementary_experiments" / "attribution" / "attribution_final.csv"

BLUE = "#1F77B4"
LIGHT_BLUE = "#E8F2FB"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FFF3DB"
GREEN = "#238B45"
LIGHT_GREEN = "#E8F5EC"
RED = "#C93C3C"
GRAY = "#5F6368"
DARK = "#202124"
LIGHT_GRAY = "#F6F7F8"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = GRAY, lw: float = 1.3) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10.5,
            lw=lw,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def card(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str = "",
    fc: str = "white",
    ec: str = "#CBD5E1",
    color: str = DARK,
    fs: float = 8.5,
    lw: float = 1.15,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.025",
            fc=fc,
            ec=ec,
            lw=lw,
        )
    )
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def synthetic_image(kind: str, seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    body = np.exp(-((xx + 0.10) ** 2 / 0.22 + (yy - 0.04) ** 2 / 0.33))
    wing = 0.55 * np.exp(-((xx - 0.30) ** 2 / 0.16 + (yy + 0.10) ** 2 / 0.20))
    tail = 0.35 * np.exp(-((xx + 0.45) ** 2 / 0.06 + (yy + 0.22) ** 2 / 0.08))
    img = np.clip(body + wing + tail, 0, 1)
    if kind == "gi":
        out = img.copy()
        for _ in range(5):
            out = (out + np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5
        out = 0.45 * out + 0.35 * rng.normal(size=out.shape)
        return np.clip(out, 0, 1)
    if kind == "data":
        out = img.copy()
        for _ in range(3):
            out = (out + np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5
        out = 0.68 * out + 0.18 * rng.normal(size=out.shape)
        return np.clip(out, 0, 1)
    if kind == "ours":
        return np.clip(0.93 * img + 0.06 * rng.normal(size=img.shape), 0, 1)
    return img


def show_thumb(ax: plt.Axes, img: np.ndarray, x: float, y: float, w: float, h: float, label: str, ec: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(x, x + w, y, y + h), zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.1, zorder=3))
    ax.text(x + w / 2, y - 0.025, label, ha="center", va="top", fontsize=7.8, color=ec)


def draw_patterns_and_buckets(ax: plt.Axes, x: float, y: float, scale: float = 0.085) -> None:
    rng = np.random.default_rng(36)
    vals = [0.30, 0.78, 0.52]
    for i in range(3):
        arr = rng.choice([0.04, 0.96], size=(5, 5))
        yy = y - i * 0.018
        xx = x + i * 0.015
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(xx, xx + scale, yy, yy + scale), zorder=2)
        ax.add_patch(Rectangle((xx, yy), scale, scale, fill=False, ec=BLUE, lw=0.7, zorder=3))
        ax.text(xx + scale + 0.020, yy + scale * 0.50, f"{vals[i]:.2f}", fontsize=7.4, color=BLUE, va="center")
    ax.text(x + scale * 0.95, y + scale + 0.030, "patterns + buckets", ha="center", fontsize=7.9, color=BLUE)


def draw_filter(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    pts = [
        (x, y + h),
        (x + w, y + h),
        (x + 0.63 * w, y + 0.48 * h),
        (x + 0.63 * w, y),
        (x + 0.37 * w, y),
        (x + 0.37 * w, y + 0.48 * h),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.35))
    for frac in (0.45, 0.55, 0.65):
        ax.plot([x + frac * w, x + frac * w], [y + 0.16 * h, y + 0.78 * h], color=ORANGE, lw=0.7)
    ax.text(x + w / 2, y + 0.64 * h, "P_N", ha="center", va="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    ax.text(x + w / 2, y - 0.030, "residual filter", ha="center", fontsize=7.4, color=ORANGE)


def draw_audit(ax: plt.Axes, cx: float, cy: float, r: float = 0.040) -> None:
    ax.add_patch(Circle((cx, cy), r, fc=LIGHT_GREEN, ec=GREEN, lw=1.35))
    ax.plot([cx - 0.018, cx - 0.006, cx + 0.025], [cy + 0.002, cy - 0.015, cy + 0.022], color=GREEN, lw=1.7)
    ax.text(cx, cy - 0.064, "bucket audit", ha="center", fontsize=7.4, color=GREEN)


def row_label(ax: plt.Axes, y: float, letter: str, title: str, color: str) -> None:
    ax.text(0.025, y + 0.132, letter, fontsize=10.8, color=color, fontweight="bold", ha="left", va="center")
    ax.text(0.065, y + 0.132, title, fontsize=9.4, color=DARK, fontweight="bold", ha="left", va="center")


def figure1() -> None:
    gi = synthetic_image("gi", 2)
    data = synthetic_image("data", 3)
    ours = synthetic_image("ours", 4)

    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("From conventional GI correlation to measurement-audited neural completion", y=0.980, fontsize=11.8, fontweight="bold")

    panels = [
        (0.025, 0.145, 0.295, 0.760, "A", "Conventional GI\nraw correlation", BLUE),
        (0.352, 0.145, 0.295, 0.760, "B", "Regularized\ndata solution", BLUE),
        (0.680, 0.145, 0.295, 0.760, "C", "Measurement-audited\nneural completion", GREEN),
    ]
    for x, y, w, h, letter, title, color in panels:
        card(ax, (x, y), w, h, fc="white", ec="#D7DCE2", lw=1.1)
        ax.text(x + 0.018, y + h - 0.055, letter, fontsize=10.5, color=color, fontweight="bold", ha="left", va="center")
        ax.text(x + 0.060, y + h - 0.055, title, fontsize=8.5, color=DARK, fontweight="bold", ha="left", va="center", linespacing=0.95)

    # Column A
    x0, y0, w, h, *_ = panels[0]
    draw_patterns_and_buckets(ax, x0 + 0.045, y0 + 0.500, 0.062)
    card(ax, (x0 + 0.060, y0 + 0.345), w - 0.120, 0.070, r"$\hat{x}_{\rm GI}=A^Ty$", fc=LIGHT_BLUE, ec=BLUE, color=BLUE, fs=8.8)
    ax.text(x0 + w / 2, y0 + 0.305, r"$=\sum_i y_i a_i$", ha="center", fontsize=8.2, color=BLUE)
    show_thumb(ax, gi, x0 + 0.092, y0 + 0.120, 0.115, 0.115, "", BLUE)
    ax.text(x0 + w / 2, y0 + 0.085, "GI/BP image", ha="center", fontsize=7.9, color=BLUE)
    ax.text(x0 + w / 2, y0 + 0.045, "raw bucket coefficients", ha="center", fontsize=7.4, color=GRAY)

    # Column B
    x0, y0, w, h, *_ = panels[1]
    draw_patterns_and_buckets(ax, x0 + 0.045, y0 + 0.500, 0.062)
    card(ax, (x0 + 0.045, y0 + 0.370), w - 0.090, 0.064, r"$q=(AA^T+\lambda I)^{-1}y$", fc=LIGHT_BLUE, ec=BLUE, color=BLUE, fs=7.8)
    card(ax, (x0 + 0.065, y0 + 0.285), w - 0.130, 0.058, r"$x_{\rm data}=A^Tq$", fc=LIGHT_BLUE, ec=BLUE, color=BLUE, fs=8.4)
    show_thumb(ax, data, x0 + 0.092, y0 + 0.120, 0.115, 0.115, "", BLUE)
    ax.text(x0 + w / 2, y0 + 0.085, "x_data", ha="center", fontsize=7.9, color=BLUE)
    ax.text(x0 + w / 2, y0 + 0.045, "decorrelated coefficients", ha="center", fontsize=7.4, color=GRAY)

    # Column C
    x0, y0, w, h, *_ = panels[2]
    show_thumb(ax, data, x0 + 0.025, y0 + 0.515, 0.090, 0.090, "x_data", BLUE)
    card(ax, (x0 + 0.145, y0 + 0.535), 0.070, 0.050, r"$G_\theta$", fc=LIGHT_ORANGE, ec=ORANGE, color=ORANGE, fs=8.0)
    draw_filter(ax, x0 + 0.240, y0 + 0.503, 0.065, 0.092)
    arrow(ax, (x0 + 0.118, y0 + 0.560), (x0 + 0.142, y0 + 0.560), ORANGE)
    arrow(ax, (x0 + 0.217, y0 + 0.560), (x0 + 0.237, y0 + 0.560), ORANGE)
    card(ax, (x0 + 0.050, y0 + 0.340), w - 0.100, 0.062, r"$x_{\rm data}+P_N(G_\theta)$", fc="white", ec=GRAY, color=DARK, fs=7.6)
    draw_audit(ax, x0 + w / 2, y0 + 0.270, r=0.038)
    show_thumb(ax, ours, x0 + 0.092, y0 + 0.070, 0.115, 0.115, "", GREEN)
    ax.text(x0 + w / 2, y0 + 0.038, "x_hat;  A x_hat ~= y", ha="center", fontsize=7.8, color=GREEN)
    arrow(ax, (x0 + w / 2, y0 + 0.338), (x0 + w / 2, y0 + 0.312), GREEN)
    arrow(ax, (x0 + w / 2, y0 + 0.232), (x0 + w / 2, y0 + 0.190), GREEN)

    arrow(ax, (0.323, 0.525), (0.348, 0.525), GRAY, lw=1.2)
    arrow(ax, (0.650, 0.525), (0.675, 0.525), GRAY, lw=1.2)
    ax.text(
        0.500,
        0.055,
        r"$A^Ty\;\rightarrow\;A^T(AA^T+\lambda I)^{-1}y\;\rightarrow\;\Pi_y[x_{\rm data}+P_N(G_\theta)]$",
        ha="center",
        fontsize=8.9,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#B8C0CC", "lw": 1.0},
    )

    fig.subplots_adjust(left=0.020, right=0.985, top=0.920, bottom=0.035)
    save(fig, "fig1_conventional_gi_anchor")


def load_attr() -> list[dict[str, str]]:
    with ATTR_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def val(row: dict[str, str], key: str) -> float:
    return float(row[key])


def figure4() -> None:
    rows = load_attr()
    by_id = {row["method_id"]: row for row in rows}
    primary = [
        ("rademacher5_hq_noise001_colab", "Rad-5", ORANGE),
        ("scrambled_hadamard5_hq_noise001_colab", "Scr-5", GREEN),
        ("rademacher10_full_noise001_colab", "Rad-10", ORANGE),
        ("scrambled_hadamard10_full_noise001_colab", "Scr-10", GREEN),
    ]
    diagnostic = [
        ("stl10_hadamard5_local_medium", "Lowfreq-5"),
        ("stl10_hadamard10_local_full", "Lowfreq-10"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))

    ax = axes[0]
    xs = np.arange(len(primary))
    bp = [val(by_id[mid], "backproj_psnr") for mid, _label, _color in primary]
    model = [val(by_id[mid], "model_psnr") for mid, _label, _color in primary]
    labels = [label for _mid, label, _color in primary]
    ax.bar(xs - 0.18, bp, 0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="GI/BP")
    ax.bar(xs + 0.18, model, 0.36, color=BLUE, edgecolor=BLUE, label="Ours")
    ax.set_title("(a) BP vs model", loc="left", fontsize=9.6, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR (dB)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    for mid, label, color in primary:
        x = val(by_id[mid], "backproj_psnr")
        y = val(by_id[mid], "delta_psnr")
        ax.scatter([x], [y], s=74, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        dy = -0.55 if label in {"Rad-5", "Scr-5"} else 0.25
        ax.text(x + 0.30, y + dy, label, fontsize=8.3, color=color)
    for mid, label in diagnostic:
        row = by_id[mid]
        x = val(row, "backproj_psnr")
        y = val(row, "delta_psnr")
        ax.scatter([x], [y], s=72, facecolors="white", edgecolors=GRAY, linewidth=1.4, zorder=3)
        dy = 0.62 if label == "Lowfreq-10" else 0.20
        ax.text(x + 0.35, y + dy, label, fontsize=7.7, color=GRAY)
    ax.set_title("(b) Regime map", loc="left", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("Backprojection PSNR (dB)")
    ax.set_ylabel("Delta PSNR (dB)")
    ax.set_xlim(5.5, 21.0)
    ax.set_ylim(3.2, 18.3)
    ax.grid(alpha=0.25)

    fig.subplots_adjust(left=0.080, right=0.985, top=0.880, bottom=0.185, wspace=0.270)
    save(fig, "fig4_measurement_attribution_v36")


def main() -> None:
    setup()
    figure1()
    figure4()
    print(
        {
            "output_dir": str(OUT),
            "figure1": str(FIG_DIR / "fig1_conventional_gi_anchor.pdf"),
            "figure4": str(FIG_DIR / "fig4_measurement_attribution_v36.pdf"),
        }
    )


if __name__ == "__main__":
    main()
