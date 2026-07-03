#!/usr/bin/env python3
"""Publication-style schematic for Method B:
radial binary optimization of a target-encoded Fresnel zone plate (FZP).

Self-contained: depends only on numpy + matplotlib. No external data files.
Outputs (written next to this script):
    method_B_radial_binary_optimization.svg
    method_B_radial_binary_optimization.png
    method_B_radial_binary_optimization_CN.png   (only if a CJK font is found)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Shared synthetic data (deterministic; no files needed)
# ----------------------------------------------------------------------------
RNG = np.random.default_rng(7)

# Panel A -- axial intensity profiles -----------------------------------------
z = np.linspace(120.0, 180.0, 600)          # mm
PEAK1, PEAK2 = 138.0, 162.0                  # mm, dual-focus targets


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


I_target = _gauss(z, PEAK1, 2.2) + _gauss(z, PEAK2, 2.2)
I_target /= I_target.max()

# An "example simulated" profile: slightly shifted/broadened peaks, small
# side-lobes and a touch of noise -- the sort of curve an early iterate gives.
I_sim = (0.93 * _gauss(z, PEAK1 + 0.8, 2.7)
         + 0.88 * _gauss(z, PEAK2 - 0.9, 3.0)
         + 0.10 * _gauss(z, 150.0, 6.0)
         + 0.06 * _gauss(z, 128.0, 3.0))
I_sim += 0.015 * RNG.standard_normal(z.size)
I_sim = np.clip(I_sim, 0.0, None)
I_sim /= I_target.max() if I_target.max() else 1.0
I_sim /= I_sim.max()

# Panel B/C -- the radial binary vector a_m ----------------------------------
N_RINGS = 36
# A Fresnel-like alternating pattern with a couple of optimized flips so it
# looks "designed" rather than perfectly periodic.
m = np.arange(N_RINGS)
a_m = ((np.floor(np.sqrt(m + 0.5)) % 2) == 0).astype(int)
for flip in (5, 12, 23, 30):                 # optimizer-style perturbations
    a_m[flip] ^= 1


# ----------------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------------
OPAQUE = "#23272e"      # 0 -> opaque (dark)
CLEAR = "#f3f4f6"       # 1 -> transparent (light)
EDGE = "#9aa0a8"
ACCENT = "#1f6feb"
TARGET_C = "#d1495b"
SIM_C = "#1f6feb"


def find_cjk_font():
    """Return a font_manager.FontProperties for a CJK font, or None."""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Microsoft JhengHei",
        "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
        "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return font_manager.FontProperties(family=name)
    return None


# ----------------------------------------------------------------------------
# Panel drawing routines (text supplied via a label dict for EN / CN reuse)
# ----------------------------------------------------------------------------
def draw_panel_A(ax, L):
    ax.plot(z, I_target, color=TARGET_C, lw=2.4, label=L["A_target"])
    ax.plot(z, I_sim, color=SIM_C, lw=1.6, ls="--", label=L["A_sim"])
    for pk in (PEAK1, PEAK2):
        ax.axvline(pk, color="#b8bcc2", lw=0.8, ls=":", zorder=0)
    ax.annotate(f"{PEAK1:.0f} mm", xy=(PEAK1, 1.02), ha="center",
                va="bottom", fontsize=8, color="#555")
    ax.annotate(f"{PEAK2:.0f} mm", xy=(PEAK2, 1.02), ha="center",
                va="bottom", fontsize=8, color="#555")
    ax.set_xlim(120, 180)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel(L["A_x"])
    ax.set_ylabel(L["A_y"])
    ax.set_title(L["A_title"], fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def draw_panel_B(ax, L):
    ax.set_xlim(0, N_RINGS)
    ax.set_ylim(-0.6, 1.4)
    for i, v in enumerate(a_m):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                   facecolor=(CLEAR if v else OPAQUE),
                                   edgecolor=EDGE, lw=0.6))
    # index ticks every few cells
    for i in range(0, N_RINGS + 1, 6):
        ax.text(i, -0.18, str(i), ha="center", va="top", fontsize=7,
                color="#666")
    ax.text(0, 1.22, "1 = " + L["B_clear"], fontsize=8, color="#444")
    ax.text(N_RINGS, 1.22, "0 = " + L["B_opaque"], fontsize=8, color="#444",
            ha="right")
    ax.set_title(L["B_title"], fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(L["B_x"])
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_panel_C(ax, L):
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    # Draw rings from outer to inner so inner ones sit on top.
    r_edges = np.linspace(0, 1, N_RINGS + 1)
    for i in range(N_RINGS - 1, -1, -1):
        color = CLEAR if a_m[i] else OPAQUE
        # smaller radius must sit ON TOP, so zorder grows as radius shrinks
        ax.add_patch(Circle((0, 0), r_edges[i + 1], facecolor=color,
                            edgecolor="none", zorder=(N_RINGS - i)))
    # aperture rim
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#111",
                        lw=1.6, zorder=N_RINGS + 1))
    ax.set_title(L["C_title"], fontsize=11, fontweight="bold", loc="left")
    ax.axis("off")


def draw_panel_D(ax, L):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(L["D_title"], fontsize=11, fontweight="bold", loc="left")

    steps = [L["D1"], L["D2"], L["D3"], L["D4"], L["D5"], L["D6"], L["D7"]]
    # vertical positions, top to bottom
    ys = np.linspace(9.2, 0.8, len(steps))
    box_w, box_h = 7.6, 0.92
    cx = 5.0
    colors = ["#eef4ff"] * len(steps)
    colors[0] = "#ffeef0"        # target  -> reddish
    colors[-1] = "#e9f7ef"       # export  -> greenish
    centers = []
    for y, txt, col in zip(ys, steps, colors):
        box = FancyBboxPatch((cx - box_w / 2, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.1, edgecolor=ACCENT, facecolor=col)
        ax.add_patch(box)
        ax.text(cx, y, txt, ha="center", va="center", fontsize=8.4,
                color="#1c2330", wrap=True)
        centers.append(y)

    # straight down arrows
    for y0, y1 in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((cx, y0 - box_h / 2),
                                     (cx, y1 + box_h / 2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     lw=1.2, color="#444"))
    # feedback loop arrow: from "update" (step 5, idx4) back up to "simulation"
    # (step 3, idx2), on the right side.
    y_from = centers[4]
    y_to = centers[2]
    right_x = cx + box_w / 2
    loop = FancyArrowPatch((right_x, y_from), (right_x, y_to),
                           connectionstyle="arc3,rad=-0.55",
                           arrowstyle="-|>", mutation_scale=12,
                           lw=1.2, color=TARGET_C)
    ax.add_patch(loop)
    ax.text(right_x + 1.0, (y_from + y_to) / 2, L["D_loop"],
            fontsize=7.6, color=TARGET_C, rotation=90, ha="center",
            va="center")


def build_figure(L, fontprops=None):
    if fontprops is not None:
        plt.rcParams["font.family"] = fontprops.get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(11, 9.2), dpi=200)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22,
                          left=0.07, right=0.965, top=0.90, bottom=0.135)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    draw_panel_A(axA, L)
    draw_panel_B(axB, L)
    draw_panel_C(axC, L)
    draw_panel_D(axD, L)

    fig.suptitle(L["suptitle"], fontsize=15, fontweight="bold", y=0.975)

    # Objective equation (math via mathtext, font-independent)
    eq = (r"$\mathrm{minimize}\ \ E=\sum_{j}\,[\,I(z_j)-I_\mathrm{target}"
          r"(z_j)\,]^{2}+\alpha\,S_\mathrm{sidelobe}+\beta\,C_\mathrm{fabrication}$")
    fig.text(0.5, 0.075, eq, ha="center", va="center", fontsize=12.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f8fa",
                       edgecolor="#d0d4da"))

    # Caption
    fig.text(0.5, 0.022, L["caption"], ha="center", va="center",
             fontsize=10.5, style="italic", color="#333", wrap=True)
    return fig


# ----------------------------------------------------------------------------
# Label sets
# ----------------------------------------------------------------------------
EN = {
    "suptitle": "Method B — Radial Binary Optimization of a "
                "Target-Encoded Fresnel Zone Plate",
    "A_title": "A  Axial intensity",
    "A_target": r"$I_\mathrm{target}(z)$",
    "A_sim": r"$I(z)$ (simulated)",
    "A_x": "axial position z (mm)",
    "A_y": "normalized intensity",
    "B_title": "B  Binary ring variables",
    "B_x": r"binary ring variables $a_m \in \{0,1\}$",
    "B_clear": "transparent",
    "B_opaque": "opaque",
    "C_title": "C  Binary FZP mask",
    "D_title": "D  Optimization loop",
    "D1": "Target axial intensity",
    "D2": "Initialize binary mask",
    "D3": "Fresnel propagation simulation",
    "D4": "Compute objective E",
    "D5": "Update binary vector",
    "D6": "Repeat until convergence",
    "D7": "Export mask",
    "D_loop": "iterate",
    "caption": "The innovation is to treat the FZP as a binary optical field "
               "encoder rather than only a fixed-focus diffractive lens.",
}

CN = {
    "suptitle": "方法 B — 面向目标编码"
                "菲涅耳波带片的径向"
                "二值优化",
    "A_title": "A  轴向光强",
    "A_target": r"$I_\mathrm{target}(z)$",
    "A_sim": r"$I(z)$ （仿真）",
    "A_x": "轴向位置 z (mm)",
    "A_y": "归一化光强",
    "B_title": "B  二值环变量",
    "B_x": r"二值环变量 $a_m \in \{0,1\}$",
    "B_clear": "透光",
    "B_opaque": "不透光",
    "C_title": "C  二值波带片掩模",
    "D_title": "D  优化循环",
    "D1": "目标轴向光强",
    "D2": "初始化二值掩模",
    "D3": "菲涅耳传播仿真",
    "D4": "计算目标函数 E",
    "D5": "更新二值向量",
    "D6": "迭代至收敛",
    "D7": "导出掩模",
    "D_loop": "迭代",
    "caption": "创新点在于将波带片视"
               "为二值光场编码器，而"
               "非仅仅是定焦衍射透镜"
               "。",
}


def main():
    # English (required)
    fig = build_figure(EN)
    svg_path = os.path.join(HERE, "method_B_radial_binary_optimization.svg")
    png_path = os.path.join(HERE, "method_B_radial_binary_optimization.png")
    fig.savefig(svg_path, format="svg", facecolor="white")
    fig.savefig(png_path, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", svg_path)
    print("wrote", png_path)

    # Chinese (optional)
    cjk = find_cjk_font()
    if cjk is not None:
        fig = build_figure(CN, fontprops=cjk)
        cn_path = os.path.join(
            HERE, "method_B_radial_binary_optimization_CN.png")
        fig.savefig(cn_path, format="png", dpi=200, facecolor="white")
        plt.close(fig)
        print("wrote", cn_path, "(font:", cjk.get_name() + ")")
    else:
        print("no CJK font found; skipped Chinese version")


if __name__ == "__main__":
    main()
