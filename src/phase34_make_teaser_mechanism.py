from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase34_mechanism_teaser"
FIG_DIR = OUT / "figures"
ATTR_CSV = ROOT / "outputs_phase16" / "supplementary_experiments" / "attribution" / "attribution_final.csv"

BLUE = "#1F77B4"
LIGHT_BLUE = "#E8F2FB"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FFF3DB"
GREEN = "#238B45"
LIGHT_GREEN = "#E8F5EC"
RED = "#C93C3C"
LIGHT_RED = "#FCE8E8"
GRAY = "#5F6368"
LIGHT_GRAY = "#F6F7F8"
DARK = "#202124"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
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


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY, lw: float = 1.3) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
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
    fs: float = 8.8,
    lw: float = 1.15,
    pad: float = 0.012,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size=0.035",
            fc=fc,
            ec=ec,
            lw=lw,
        )
    )
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def panel_frame(ax: plt.Axes, letter: str, title: str, color: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.025),
            0.970,
            0.925,
            boxstyle="round,pad=0.010,rounding_size=0.030",
            fc="white",
            ec="#D7DCE2",
            lw=1.1,
        )
    )
    ax.text(0.045, 0.900, f"{letter}", fontsize=10.2, fontweight="bold", color=color, ha="left", va="center")
    ax.text(0.120, 0.900, title, fontsize=8.4, fontweight="bold", color=DARK, ha="left", va="center", linespacing=0.95)


def synthetic_image(kind: str, seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    body = np.exp(-((xx + 0.18) ** 2 / 0.20 + (yy - 0.05) ** 2 / 0.38))
    wing = 0.55 * np.exp(-((xx - 0.28) ** 2 / 0.13 + (yy + 0.10) ** 2 / 0.18))
    tail = 0.35 * np.exp(-((xx + 0.45) ** 2 / 0.05 + (yy + 0.22) ** 2 / 0.07))
    img = np.clip(body + wing + tail, 0, 1)
    if kind == "gt":
        return img
    if kind == "bp":
        out = img.copy()
        for _ in range(8):
            out = (
                out
                + np.roll(out, 1, 0)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 1)
            ) / 5.0
        out = 0.55 * out + 0.28 * rng.normal(size=out.shape)
        return np.clip(out, 0, 1)
    if kind == "free":
        shifted = np.roll(img, 7, axis=1)
        shifted = np.roll(shifted, -5, axis=0)
        return np.clip(0.92 * shifted + 0.10 * rng.normal(size=img.shape), 0, 1)
    if kind == "ours":
        out = img.copy()
        out = 0.92 * out + 0.08 * rng.normal(size=out.shape)
        return np.clip(out, 0, 1)
    raise ValueError(kind)


def show_thumb(ax: plt.Axes, img: np.ndarray, x: float, y: float, w: float, h: float, label: str, ec: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(x, x + w, y, y + h), zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.1, zorder=3))
    ax.text(x + w / 2, y - 0.040, label, ha="center", va="top", fontsize=7.9, color=ec)


def draw_patterns(ax: plt.Axes, x: float, y: float, size: float) -> None:
    rng = np.random.default_rng(34)
    for i in range(3):
        arr = rng.choice([0.05, 0.95], size=(6, 6))
        dx = 0.018 * i
        dy = -0.020 * i
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(x + dx, x + dx + size, y + dy, y + dy + size), zorder=2)
        ax.add_patch(Rectangle((x + dx, y + dy), size, size, fill=False, ec=BLUE, lw=0.8, zorder=3))
    ax.text(x + size / 2 + 0.020, y + size + 0.035, "known patterns", ha="center", fontsize=8.5, color=BLUE)


def draw_bucket(ax: plt.Axes, x: float, y: float, w: float = 0.080, h: float = 0.210) -> None:
    card(ax, (x, y), w, h, fc=LIGHT_BLUE, ec=BLUE)
    bars = [0.055, 0.100, 0.150, 0.078]
    for i, height in enumerate(bars):
        bx = x + 0.014 + i * 0.015
        ax.add_patch(Rectangle((bx, y + 0.025), 0.009, height, fc=BLUE, ec="none"))
    ax.text(x + w / 2, y - 0.040, "bucket y", ha="center", fontsize=7.9, color=BLUE)


def draw_sieve(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    pts = [
        (x, y + h),
        (x + w, y + h),
        (x + 0.63 * w, y + 0.48 * h),
        (x + 0.63 * w, y),
        (x + 0.37 * w, y),
        (x + 0.37 * w, y + 0.48 * h),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.4))
    for frac in (0.45, 0.55, 0.65):
        ax.plot([x + frac * w, x + frac * w], [y + 0.16 * h, y + 0.78 * h], color=ORANGE, lw=0.7, alpha=0.65)
    ax.text(x + w / 2, y + 0.64 * h, "P_N", ha="center", va="center", fontsize=10.2, color=ORANGE, fontweight="bold")
    ax.text(x + w / 2, y - 0.040, "residual filter", ha="center", fontsize=8.6, color=ORANGE)


def draw_audit(ax: plt.Axes, cx: float, cy: float, r: float = 0.060) -> None:
    ax.add_patch(Circle((cx, cy), r, fc=LIGHT_GREEN, ec=GREEN, lw=1.45))
    ax.plot([cx - 0.027, cx - 0.008, cx + 0.034], [cy + 0.002, cy - 0.020, cy + 0.030], color=GREEN, lw=2.0)
    ax.text(cx, cy - 0.095, "Pi_y final audit", ha="center", fontsize=8.6, color=GREEN)


def write_rationale() -> None:
    text = """# Figure 1 Redesign Rationale

## Why the previous Figure 1 was not enough

- It only listed acquisition, data solution, residual projection, and final projection modules.
- It did not show a visual failure mode, so readers could not see why the extra constraints are needed.
- It lacked a real-image or image-like anchor connecting bucket measurements to reconstruction quality.
- The roles of \(P_N\) and \(\Pi_y\) were easy to confuse because both appeared as ordinary pipeline blocks.
- It used too many formulas for a first figure and not enough intuition.

## Goal of the new Figure 1

- Show the problem: low-sampling bucket measurements leave many possible images.
- Show the naive failure: a neural inverse can look plausible while drifting away from the bucket signal.
- Show the proposed solution: use a data solution, filter the learned residual, and audit the completed image.

## Reading order

Low sampling -> physical inverse incomplete -> unconstrained network can drift -> \(P_N\) residual filter -> \(\Pi_y\) audit -> final reconstruction.

## Scope

The figure is a computational forward-model schematic and reconstruction-mechanism teaser. It is not a hardware optical setup, and it deliberately avoids lasers, lenses, DMDs, cameras, and other laboratory components.
"""
    (OUT / "FIGURE1_REDESIGN_RATIONALE.md").write_text(text, encoding="utf-8")


def figure1_teaser() -> None:
    gt = synthetic_image("gt", seed=1)
    bp = synthetic_image("bp", seed=2)
    free = synthetic_image("free", seed=3)
    ours = synthetic_image("ours", seed=4)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    fig.suptitle(
        "Measurement-audited neural completion for low-sampling ghost imaging",
        y=0.982,
        fontsize=11.7,
        fontweight="bold",
    )
    fig.text(0.5, 0.945, "computational forward model and reconstruction mechanism", ha="center", fontsize=8.4, color=GRAY)

    panels = [
        ("A", "Low-sampling\nbucket measurement", BLUE),
        ("B", "Physical inverse:\nfaithful but incomplete", BLUE),
        ("C", "Unconstrained neural\ninverse can drift", RED),
        ("D", "Neural residual is filtered", ORANGE),
        ("E", "Completed image is audited", GREEN),
        ("F", "Output", GREEN),
    ]
    for ax, (letter, title, color) in zip(axes.flat, panels):
        panel_frame(ax, letter, title, color)

    ax = axes[0, 0]
    draw_patterns(ax, 0.070, 0.560, 0.145)
    show_thumb(ax, gt, 0.355, 0.575, 0.155, 0.155, "object x", BLUE)
    draw_bucket(ax, 0.720, 0.545, w=0.070, h=0.185)
    arrow(ax, (0.260, 0.640), (0.355, 0.640), BLUE)
    arrow(ax, (0.515, 0.640), (0.710, 0.640), BLUE)
    ax.text(0.500, 0.405, r"$y=Ax+\epsilon,\quad m\ll n$", ha="center", fontsize=9.6, color=BLUE)
    ax.text(0.500, 0.245, "few measurements,\nmany possible images", ha="center", fontsize=9.1, color=GRAY)

    ax = axes[0, 1]
    draw_bucket(ax, 0.085, 0.580, w=0.070, h=0.155)
    card(ax, (0.315, 0.612), 0.155, 0.052, "inverse", fc=LIGHT_BLUE, ec=BLUE, fs=7.8, color=BLUE)
    show_thumb(ax, bp, 0.710, 0.575, 0.150, 0.150, "x_data / BP", BLUE)
    arrow(ax, (0.180, 0.650), (0.270, 0.650), BLUE)
    arrow(ax, (0.515, 0.650), (0.700, 0.650), BLUE)
    ax.text(0.500, 0.420, r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=8.0, color=BLUE)
    ax.text(0.500, 0.265, "measurement-tied,\nbut incomplete", ha="center", fontsize=9.1, color=GRAY)

    ax = axes[0, 2]
    show_thumb(ax, bp, 0.075, 0.590, 0.140, 0.140, "x_data", BLUE)
    card(ax, (0.342, 0.625), 0.130, 0.064, "free\nnet", fc=LIGHT_ORANGE, ec=ORANGE, color=ORANGE, fs=7.8)
    show_thumb(ax, free, 0.660, 0.590, 0.140, 0.140, "x_free", RED)
    arrow(ax, (0.230, 0.660), (0.325, 0.660), ORANGE)
    arrow(ax, (0.485, 0.660), (0.645, 0.660), ORANGE)
    ax.text(0.500, 0.430, "bucket mismatch", ha="center", fontsize=9.4, color=RED, fontweight="bold")
    ax.text(0.500, 0.370, "A x_free != y", ha="center", fontsize=8.8, color=RED)
    card(ax, (0.240, 0.245), 0.520, 0.062, "plausible, weak audit", fc=LIGHT_RED, ec=RED, color=RED, fs=7.8)

    ax = axes[1, 0]
    show_thumb(ax, bp, 0.070, 0.595, 0.125, 0.125, "x_data", BLUE)
    card(ax, (0.275, 0.628), 0.110, 0.058, "G", fc=LIGHT_ORANGE, ec=ORANGE, color=ORANGE, fs=8.2)
    card(ax, (0.500, 0.628), 0.075, 0.058, "r", fc=LIGHT_ORANGE, ec=ORANGE, color=ORANGE, fs=8.4)
    draw_sieve(ax, 0.700, 0.585, 0.115, 0.140)
    arrow(ax, (0.205, 0.640), (0.270, 0.640), ORANGE)
    arrow(ax, (0.390, 0.640), (0.495, 0.640), ORANGE)
    arrow(ax, (0.580, 0.640), (0.695, 0.640), ORANGE)
    ax.text(0.500, 0.365, "fill weakly observed or\nunobserved directions", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(0.500, 0.255, "do not overwrite\nmeasured component", ha="center", fontsize=8.5, color=GRAY)

    ax = axes[1, 1]
    card(ax, (0.075, 0.632), 0.300, 0.062, r"$x_{\rm data}+P_N(r_\theta)$", fc="white", ec=GRAY, fs=7.8)
    draw_audit(ax, 0.555, 0.665, r=0.052)
    show_thumb(ax, ours, 0.755, 0.595, 0.135, 0.135, r"$\hat{x}$", GREEN)
    arrow(ax, (0.395, 0.665), (0.500, 0.665), GREEN)
    arrow(ax, (0.625, 0.665), (0.755, 0.665), GREEN)
    ax.text(0.500, 0.425, r"$A\hat{x}\approx y$", ha="center", fontsize=9.8, color=GREEN)
    ax.text(0.500, 0.315, "final bucket-measurement check", ha="center", fontsize=8.4, color=GREEN)
    ax.text(0.500, 0.220, "network proposal\nmust pass the buckets", ha="center", fontsize=8.2, color=GREEN)

    ax = axes[1, 2]
    xs = [0.095, 0.395, 0.695]
    labels = ["GT", "BP", "Ours"]
    imgs = [gt, bp, ours]
    ecs = [GRAY, BLUE, GREEN]
    for x, label, img, ec in zip(xs, labels, imgs, ecs):
        show_thumb(ax, img, x, 0.595, 0.150, 0.150, label, ec)
    arrow(ax, (0.285, 0.670), (0.385, 0.670), GRAY)
    arrow(ax, (0.585, 0.670), (0.690, 0.670), GREEN)
    ax.text(0.500, 0.405, "learned completion\n+ physical audit", ha="center", fontsize=8.8, color=GREEN)
    ax.text(0.500, 0.270, "image quality up\nRelMeasErr controlled", ha="center", fontsize=8.4, color=GRAY)

    fig.subplots_adjust(left=0.030, right=0.985, top=0.900, bottom=0.035, wspace=0.080, hspace=0.150)
    save(fig, "fig1_mechanism_teaser_v34")


def figure_s1_equations() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Algebraic decomposition of measurement-consistent reconstruction", fontsize=12.3, fontweight="bold", y=0.965)

    card(
        ax,
        (0.135, 0.705),
        0.730,
        0.110,
        r"$\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))]$",
        fc="white",
        ec=GREEN,
        color=DARK,
        fs=13.0,
        lw=1.3,
    )
    components = [
        (0.060, 0.470, 0.190, 0.105, r"$x_{\rm data}$", "measured\ncomponent", LIGHT_BLUE, BLUE),
        (0.300, 0.470, 0.190, 0.105, r"$G_\theta$", "learned\nresidual", LIGHT_ORANGE, ORANGE),
        (0.540, 0.470, 0.170, 0.105, r"$P_N$", "residual\nfilter", LIGHT_ORANGE, ORANGE),
        (0.760, 0.470, 0.170, 0.105, r"$\Pi_y$", "measurement\naudit", LIGHT_GREEN, GREEN),
    ]
    for x, y, w, h, formula, desc, fc, ec in components:
        card(ax, (x, y), w, h, formula, fc=fc, ec=ec, color=ec, fs=11.5)
        ax.text(x + w / 2, y - 0.035, desc, ha="center", va="top", fontsize=8.7, color=ec)

    formulas = [
        (r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", BLUE),
        (r"$P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av$", ORANGE),
        (r"$\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y)$", GREEN),
    ]
    y0 = 0.275
    for i, (formula, color) in enumerate(formulas):
        card(ax, (0.140, y0 - 0.102 * i), 0.720, 0.070, formula, fc="white", ec=color, color=color, fs=10.0)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.900, bottom=0.050)
    save(fig, "figS1_equation_decomposition_v34")


def load_attr() -> list[dict[str, str]]:
    with ATTR_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def figure4_attribution() -> None:
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
    bp = [num(by_id[mid], "backproj_psnr") for mid, _label, _color in primary]
    model = [num(by_id[mid], "model_psnr") for mid, _label, _color in primary]
    labels = [label for _mid, label, _color in primary]
    ax.bar(xs - 0.18, bp, 0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="GI/BP")
    ax.bar(xs + 0.18, model, 0.36, color=BLUE, edgecolor=BLUE, label="Ours")
    ax.set_title("(a) Initialization and final quality", loc="left", fontsize=9.6, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR (dB)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    for mid, label, color in primary:
        x = num(by_id[mid], "backproj_psnr")
        y = num(by_id[mid], "delta_psnr")
        ax.scatter([x], [y], s=74, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        dx = 0.28
        dy = 0.25 if "Scr" in label else -0.45
        if label == "Scr-5":
            dy = -0.55
        ax.text(x + dx, y + dy, label, fontsize=8.4, color=color)
    for mid, label in diagnostic:
        row = by_id[mid]
        x = num(row, "backproj_psnr")
        y = num(row, "delta_psnr")
        ax.scatter([x], [y], s=72, facecolors="white", edgecolors=GRAY, linewidth=1.4, zorder=3)
        if label == "Lowfreq-10":
            ax.text(x + 0.35, y + 0.62, label, fontsize=7.7, color=GRAY)
        else:
            ax.text(x + 0.35, y + 0.20, label, fontsize=7.7, color=GRAY)
    ax.set_title("(b) Neural gain regime map", loc="left", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("Backprojection PSNR (dB)")
    ax.set_ylabel("\u0394PSNR (dB)")
    ax.set_xlim(5.5, 21.0)
    ax.set_ylim(3.2, 18.3)
    ax.grid(alpha=0.25)

    fig.subplots_adjust(left=0.080, right=0.985, top=0.880, bottom=0.185, wspace=0.270)
    save(fig, "fig4_measurement_attribution_v34")


def main() -> None:
    ensure_dirs()
    setup()
    write_rationale()
    figure1_teaser()
    figure_s1_equations()
    figure4_attribution()
    print(
        {
            "output_dir": str(OUT),
            "figure1": str(FIG_DIR / "fig1_mechanism_teaser_v34.pdf"),
            "figure_s1": str(FIG_DIR / "figS1_equation_decomposition_v34.pdf"),
            "figure4": str(FIG_DIR / "fig4_measurement_attribution_v34.pdf"),
        }
    )


if __name__ == "__main__":
    main()
