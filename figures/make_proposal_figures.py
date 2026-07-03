#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate all six schematic figures for the FZP target-encoding proposal.

Self-contained (numpy + matplotlib only). Physics-faithful: zone radii and
binary transmittance use the formulas / parameters from the proposal
(lambda = 532 nm, R = 2 mm, f0 = 150 mm, f1 = 135 mm, f2 = 165 mm).

Outputs (PDF + PNG) into the same directory:
    fig1_paradigm        fig2_fzp_structure   fig3_targets
    fig4_sector          fig5_method_b        fig6_setup
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, Wedge, Polygon)
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Chinese font
# ---------------------------------------------------------------------------
def pick_cjk():
    cand = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
            "Source Han Sans SC", "Microsoft JhengHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in cand:
        if n in have:
            return n
    return "DejaVu Sans"

plt.rcParams["font.family"] = pick_cjk()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------------
# Physical parameters (mm)
# ---------------------------------------------------------------------------
LAM = 532e-6        # wavelength in mm (532 nm)
R = 2.0             # outer radius mm
F0 = 150.0          # nominal focal length mm
F1, F2 = 135.0, 165.0

# colour palette
DARK = (0.16, 0.18, 0.20)
WARM = (0.96, 0.78, 0.78)   # f1 transparent tint
COOL = (0.74, 0.84, 0.96)   # f2 transparent tint
ACC = "#1f6feb"
RED = "#d1495b"
GRN = "#2a9d5c"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name + ".pdf / .png")


def q_zone(rho, f):
    """half-wave zone index q_f(rho) (floor)."""
    return np.floor(2.0 * (np.sqrt(rho**2 + f**2) - f) / LAM)


# ===========================================================================
# Figure 1 -- design paradigm comparison
# ===========================================================================
def fig1_paradigm():
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, txt, fc, ec=ACC):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.03,rounding_size=0.10",
                           lw=1.3, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x, y, txt, ha="center", va="center", fontsize=11)

    def arrow(x0, x1, y, color="#444"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.4, color=color))

    # Row 1: forward design
    ax.text(0.2, 5.7, "传统正向设计（焦距驱动）", fontsize=12,
            fontweight="bold", color="#333")
    y1 = 4.7
    xs = [1.5, 4.3, 7.1, 9.9]
    labs = [r"目标焦距 $f$", r"环带半径 $r_n$",
            r"透过率 $T(x,y)$", r"光强 $I(x,y,z)$"]
    for x, t in zip(xs, labs):
        box(x, y1, 2.3, 0.95, t, "#eef4ff")
    for i in range(3):
        arrow(xs[i] + 1.15, xs[i + 1] - 1.15, y1)

    # Row 2: target-driven design
    ax.text(0.2, 3.0, "本项目：目标驱动设计（目标光场驱动）", fontsize=12,
            fontweight="bold", color="#333")
    y2 = 2.0
    xs2 = [1.7, 5.3, 8.9]
    labs2 = [r"目标光强 $I_{\mathrm{target}}(z)$",
             r"二值结构 $T(x,y)$",
             r"实测光强 $I_{\mathrm{exp}}(z)$"]
    fcs = ["#ffeef0", "#eef4ff", "#e9f7ef"]
    for x, t, fc in zip(xs2, labs2, fcs):
        box(x, y2, 2.7, 0.95, t, fc)
    arrow(xs2[0] + 1.35, xs2[1] - 1.35, y2)
    arrow(xs2[1] + 1.35, xs2[2] - 1.35, y2)
    # feedback: compare exp with target
    fb = FancyArrowPatch((xs2[2], y2 - 0.52), (xs2[0], y2 - 0.52),
                         connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                         mutation_scale=14, lw=1.3, color=RED)
    ax.add_patch(fb)
    ax.text((xs2[0] + xs2[2]) / 2, y2 - 1.62, "比较并迭代设计",
            ha="center", fontsize=10, color=RED)

    save(fig, "fig1_paradigm")


# ===========================================================================
# Figure 2 -- standard FZP structure with radius annotations
# ===========================================================================
def fig2_fzp_structure():
    N = 1100
    xs = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, xs)
    rho = np.sqrt(X**2 + Y**2)
    q = q_zone(rho, F0)
    T = (np.mod(q, 2) == 0).astype(float)     # 1 transparent, 0 opaque
    img = np.ones((N, N, 3))
    aperture = rho <= R
    opaque = aperture & (T == 0)
    img[opaque] = DARK
    img[~aperture] = (1, 1, 1)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(img, extent=[-R, R, -R, R], origin="lower")
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#111", lw=1.6))
    ax.set_aspect("equal")

    # zone radii r_n = sqrt(n*lam*f)
    def r_of(n):
        return np.sqrt(n * LAM * F0)
    # annotate r1 and r3 with arrows from centre
    for n, ang in ((1, 35), (3, 70)):
        rn = r_of(n)
        a = np.deg2rad(ang)
        ax.annotate(rf"$r_{{{n}}}$", xy=(rn * np.cos(a), rn * np.sin(a)),
                    xytext=(0, 0), fontsize=12, color=ACC,
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.4),
                    ha="center", va="center")
    # outermost ring width Delta r_N ~ lam f /(2R)
    drN = LAM * F0 / (2 * R)
    ax.annotate(r"最外圈线宽 $\Delta r_N\!\approx\!20\,\mu$m",
                xy=(R, 0.0), xytext=(R + 0.05, 1.25),
                fontsize=10.5, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                                connectionstyle="arc3,rad=-0.3"))
    ax.annotate(r"外半径 $R=2$ mm",
                xy=(-R * np.cos(np.deg2rad(40)), R * np.sin(np.deg2rad(40))),
                xytext=(-2.55, 1.7), fontsize=10.5, color="#222",
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))
    ax.text(0, -R - 0.28,
            r"$T_f=1$ 透光（白）   $T_f=0$ 遮光（黑）,   $r_n=\sqrt{n\lambda f}$",
            ha="center", fontsize=10.5, color="#333")
    ax.set_xlim(-R - 0.05, R + 1.55)
    ax.set_ylim(-R - 0.45, R + 0.05)
    ax.axis("off")
    save(fig, "fig2_fzp_structure")


# ===========================================================================
# Figure 3 -- three target axial intensity profiles
# ===========================================================================
def fig3_targets():
    z = np.linspace(120, 180, 800)
    sig = 4.0
    single = np.exp(-(z - F0) ** 2 / (2 * sig**2))
    double = (np.exp(-(z - F1) ** 2 / (2 * sig**2))
              + np.exp(-(z - F2) ** 2 / (2 * sig**2)))
    df, p = 15.0, 4
    ext = np.exp(-((z - F0) / df) ** (2 * p))

    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    data = [(single, "单焦点\n（高斯）", [F0], ACC),
            (double, "双焦点\n（双高斯峰）", [F1, F2], RED),
            (ext, "扩展焦深\n（超高斯平台）", [F0], GRN)]
    tags = ["(a)", "(b)", "(c)"]
    for ax, (y, title, peaks, c), tag in zip(axs, data, tags):
        ax.fill_between(z, 0, y, color=c, alpha=0.16)
        ax.plot(z, y, color=c, lw=2.2)
        for pk in peaks:
            ax.axvline(pk, ls=":", lw=0.9, color="#999")
            ax.text(pk, 1.06, f"{pk:.0f}", ha="center", fontsize=8.5,
                    color="#555")
        ax.set_xlim(120, 180)
        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("轴向位置 z (mm)")
        ax.text(0.03, 0.92, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("归一化光强")
    fig.tight_layout()
    save(fig, "fig3_targets")


# ===========================================================================
# Figure 4 -- deterministic sector coding aperture
# ===========================================================================
def fig4_sector():
    N = 1100
    K = 8
    xs = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, xs)
    rho = np.sqrt(X**2 + Y**2)
    th = np.mod(np.arctan2(Y, X), 2 * np.pi)
    s = np.floor(K * th / (2 * np.pi)).astype(int)          # 0..K-1
    is_f1 = (np.mod(s, 2) == 0)
    f_map = np.where(is_f1, F1, F2)
    q = np.floor(2.0 * (np.sqrt(rho**2 + f_map**2) - f_map) / LAM)
    T = (np.mod(q, 2) == 0)

    img = np.ones((N, N, 3))
    ap = rho <= R
    opaque = ap & (~T)
    trans1 = ap & T & is_f1
    trans2 = ap & T & (~is_f1)
    img[opaque] = DARK
    img[trans1] = WARM
    img[trans2] = COOL
    img[~ap] = (1, 1, 1)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.imshow(img, extent=[-R, R, -R, R], origin="lower")
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#111", lw=1.6))
    # sector boundary lines
    for kk in range(K):
        a = 2 * np.pi * kk / K
        ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], color="#444",
                lw=0.8, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-R - 0.05, R + 0.05)
    ax.set_ylim(-R - 1.15, R + 0.55)
    ax.axis("off")
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=WARM, edgecolor="#888",
                 label=r"扇区焦距 $f_1=135$ mm"),
           Patch(facecolor=COOL, edgecolor="#888",
                 label=r"扇区焦距 $f_2=165$ mm"),
           Patch(facecolor=DARK, edgecolor="#888", label="遮光半波带")]
    ax.legend(handles=leg, loc="upper center", ncol=3, fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, 0.02))
    ax.set_title(r"$K=8$ 角向扇区，奇偶扇区对应不同焦距", fontsize=11)
    save(fig, "fig4_sector")


# ===========================================================================
# Figure 5 -- radial binary optimization overview (2x2)
# ===========================================================================
def fig5_method_b():
    rng = np.random.default_rng(7)
    z = np.linspace(120, 180, 600)
    sig = 3.5
    tgt = (np.exp(-(z - F1) ** 2 / (2 * sig**2))
           + np.exp(-(z - F2) ** 2 / (2 * sig**2)))
    tgt /= tgt.max()
    sim = (0.92 * np.exp(-(z - F1 - 0.8) ** 2 / (2 * (sig + 0.4) ** 2))
           + 0.9 * np.exp(-(z - F2 + 0.7) ** 2 / (2 * (sig + 0.6) ** 2))
           + 0.09 * np.exp(-(z - 150) ** 2 / (2 * 6.0**2)))
    sim += 0.012 * rng.standard_normal(z.size)
    sim = np.clip(sim, 0, None)
    sim /= sim.max()

    M = 36
    m = np.arange(M)
    a = ((np.floor(np.sqrt(m + 0.5)) % 2) == 0).astype(int)
    for fl in (5, 12, 23, 30):
        a[fl] ^= 1

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4))
    axA, axB, axC, axD = axs.ravel()

    # A target vs simulated
    axA.plot(z, tgt, color=RED, lw=2.3, label=r"$I_{\mathrm{target}}(z)$")
    axA.plot(z, sim, color=ACC, lw=1.6, ls="--", label=r"$I(z)$ 仿真")
    axA.set_xlim(120, 180)
    axA.set_ylim(0, 1.16)
    axA.set_xlabel("轴向位置 z (mm)")
    axA.set_ylabel("归一化光强")
    axA.set_title("(a) 目标 / 仿真轴向光强", fontsize=11, loc="left")
    axA.legend(fontsize=9, frameon=False, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)

    # B ring vector
    axB.set_xlim(0, M)
    axB.set_ylim(-0.5, 1.3)
    for i, v in enumerate(a):
        axB.add_patch(Rectangle((i, 0), 1, 1,
                                facecolor=("#f1f3f5" if v else DARK),
                                edgecolor="#9aa0a8", lw=0.5))
    for i in range(0, M + 1, 6):
        axB.text(i, -0.16, str(i), ha="center", va="top", fontsize=7.5,
                 color="#666")
    axB.text(0, 1.13, "1 = 透光", fontsize=9, color="#444")
    axB.text(M, 1.13, "0 = 遮光", fontsize=9, color="#444", ha="right")
    axB.set_title(r"(b) 二值环带变量 $a_m\in\{0,1\}$", fontsize=11, loc="left")
    axB.axis("off")

    # C circular binary mask from a
    axC.set_aspect("equal")
    axC.set_xlim(-1.04, 1.04)
    axC.set_ylim(-1.04, 1.04)
    redges = np.linspace(0, 1, M + 1)
    for i in range(M - 1, -1, -1):
        col = "#f1f3f5" if a[i] else DARK
        axC.add_patch(Circle((0, 0), redges[i + 1], facecolor=col,
                             edgecolor="none", zorder=(M - i)))
    axC.add_patch(Circle((0, 0), 1.0, fill=False, ec="#111", lw=1.5,
                        zorder=M + 1))
    axC.set_title("(c) 生成的二值波带片掩膜", fontsize=11, loc="left")
    axC.axis("off")

    # D optimization loop
    axD.set_xlim(0, 10)
    axD.set_ylim(0, 10)
    axD.axis("off")
    axD.set_title("(d) 目标驱动优化迭代回路", fontsize=11, loc="left")
    steps = ["目标轴向光强 I_target(z)", "初始化二值掩膜 a",
             "菲涅耳传播仿真", "计算目标函数 J(a)",
             "翻转环带 / 模拟退火", "收敛？", "导出可制备掩膜"]
    ys = np.linspace(9.0, 1.0, len(steps))
    cols = ["#ffeef0"] + ["#eef4ff"] * 5 + ["#e9f7ef"]
    bw, bh = 7.4, 0.82
    cx = 5.0
    for y, t, c in zip(ys, steps, cols):
        axD.add_patch(FancyBboxPatch((cx - bw / 2, y - bh / 2), bw, bh,
                                     boxstyle="round,pad=0.03,rounding_size=0.1",
                                     lw=1.0, edgecolor=ACC, facecolor=c))
        axD.text(cx, y, t, ha="center", va="center", fontsize=8.6)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        axD.add_patch(FancyArrowPatch((cx, y0 - bh / 2), (cx, y1 + bh / 2),
                                      arrowstyle="-|>", mutation_scale=11,
                                      lw=1.1, color="#444"))
    # feedback from "收敛?" (idx5) back to simulation (idx2)
    rx = cx + bw / 2
    lp = FancyArrowPatch((rx, ys[5]), (rx, ys[2]),
                         connectionstyle="arc3,rad=-0.55", arrowstyle="-|>",
                         mutation_scale=11, lw=1.1, color=RED)
    axD.add_patch(lp)
    axD.text(rx + 0.85, (ys[5] + ys[2]) / 2, "否", color=RED, fontsize=9,
             rotation=90, ha="center", va="center")

    fig.tight_layout()
    save(fig, "fig5_method_b")


# ===========================================================================
# Figure 6 -- optical setup
# ===========================================================================
def fig6_setup():
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    axis_y = 2.3

    # laser
    ax.add_patch(FancyBboxPatch((0.3, axis_y - 0.45), 1.5, 0.9,
                                boxstyle="round,pad=0.02", lw=1.3,
                                edgecolor="#111", facecolor="#dfe9ff"))
    ax.text(1.05, axis_y, "激光器\n532 nm", ha="center", va="center",
            fontsize=9.5)

    # diverging then collimating lens (beam expander)
    def lens(x, h, label):
        ax.add_patch(plt.matplotlib.patches.Ellipse((x, axis_y), 0.28, 2 * h,
                     fill=True, facecolor="#cfe6d8", edgecolor="#2a7d52",
                     lw=1.4))
        ax.text(x, axis_y + h + 0.22, label, ha="center", fontsize=8.6,
                color="#226")
    # initial small beam
    ax.plot([1.8, 2.7], [axis_y, axis_y], color=RED, lw=1.4)
    lens(2.8, 0.5, "")
    # diverging
    ax.plot([2.9, 4.2], [axis_y, axis_y + 0.95], color=RED, lw=1.0)
    ax.plot([2.9, 4.2], [axis_y, axis_y - 0.95], color=RED, lw=1.0)
    lens(4.35, 1.05, "扩束/准直")
    # collimated beam
    for dy in (0.9, 0.45, 0, -0.45, -0.9):
        ax.plot([4.5, 6.7], [axis_y + dy, axis_y + dy], color=RED, lw=0.9)

    # FZP element
    ax.add_patch(Rectangle((6.7, axis_y - 1.15), 0.16, 2.3, facecolor="#222",
                           edgecolor="#000"))
    ax.text(6.78, axis_y + 1.45, "波带片\n(FZP)", ha="center", fontsize=9)

    # converging rays to focus
    foc = 9.6
    for dy in (0.9, 0.45, -0.45, -0.9):
        ax.plot([6.86, foc], [axis_y + dy, axis_y], color=RED, lw=0.9)
    ax.plot([6.86, foc], [axis_y, axis_y], color=RED, lw=0.9)
    ax.plot(foc, axis_y, "o", color=RED, ms=5)

    # CMOS movable
    ax.add_patch(FancyBboxPatch((9.85, axis_y - 0.8), 0.5, 1.6,
                                boxstyle="round,pad=0.02", lw=1.3,
                                edgecolor="#111", facecolor="#f0e0a0"))
    ax.text(10.1, axis_y + 1.15, "CMOS\n相机", ha="center", fontsize=9)
    # scan double arrow
    ax.add_patch(FancyArrowPatch((9.4, axis_y - 1.15), (11.2, axis_y - 1.15),
                                 arrowstyle="<|-|>", mutation_scale=12,
                                 lw=1.3, color="#333"))
    ax.text(10.3, axis_y - 1.55, r"沿光轴扫描  $z=120$–$180$ mm",
            ha="center", fontsize=9, color="#333")

    # acquisition
    ax.add_patch(FancyArrowPatch((10.6, axis_y + 0.6), (11.7, axis_y + 0.6),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color="#444"))
    ax.text(12.3, axis_y + 0.6, "图像采集\n与处理", ha="center", va="center",
            fontsize=9)

    # optical axis
    ax.plot([1.8, 11.3], [axis_y, axis_y], color="#888", lw=0.6, ls="-.",
            zorder=0)
    save(fig, "fig6_setup")


# ===========================================================================
# Figure 7 -- three zone-plate shapes (方案一 形状对比)
# ===========================================================================
def fig7_shapes():
    N = 900
    xs = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, xs)
    rho = np.sqrt(X**2 + Y**2)
    ap = rho <= R

    def t1d(coord):
        q = np.floor(2.0 * (np.sqrt(coord**2 + F0**2) - F0) / LAM)
        return np.mod(q, 2) == 0

    # circular: depends on rho
    qc = np.floor(2.0 * (np.sqrt(rho**2 + F0**2) - F0) / LAM)
    Tc = np.mod(qc, 2) == 0
    # linear: depends on |x| only -> vertical stripes (focal line)
    Tl = t1d(X)
    # crossed/square: product of two 1D zone plates
    Tq = t1d(X) & t1d(Y)

    masks = [(Tc, "圆形波带片", "旋转对称 → 圆对称焦斑"),
             (Tl, "线型波带片", "单方向聚焦 → 焦线"),
             (Tq, "方形 / 正交型波带片", "直角对称 → 十字状旁瓣")]

    fig, axs = plt.subplots(1, 3, figsize=(11.0, 4.2))
    tags = ["(a)", "(b)", "(c)"]
    for ax, (T, title, sub), tag in zip(axs, masks, tags):
        img = np.ones((N, N, 3))
        opaque = ap & (~T)
        img[opaque] = DARK
        img[~ap] = (1, 1, 1)
        ax.imshow(img, extent=[-R, R, -R, R], origin="lower")
        ax.add_patch(Circle((0, 0), R, fill=False, ec="#111", lw=1.4))
        ax.set_aspect("equal")
        ax.set_xlim(-R - 0.04, R + 0.04)
        ax.set_ylim(-R - 0.04, R + 0.04)
        ax.axis("off")
        ax.set_title(f"{tag} {title}", fontsize=11)
        ax.text(0, -R - 0.28, sub, ha="center", va="top", fontsize=9.5,
                color="#444")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12,
                        wspace=0.12)
    save(fig, "fig7_shapes")


# ===========================================================================
# Figure 8 -- chromatic aberration f ∝ 1/lambda
# ===========================================================================
def fig8_chromatic():
    # design: f0=150 mm at 532 nm  ->  f(lam) = f0 * lam0 / lam
    lam0, f0 = 532.0, 150.0
    bands = [(633.0, "#e03131", "红光 633 nm"),
             (532.0, "#2f9e44", "绿光 532 nm"),
             (450.0, "#1c7ed6", "蓝光 450 nm")]
    foci = [(f0 * lam0 / lam, c, lab) for lam, c, lab in bands]

    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")
    yax = 0.0

    # collimated input (gray) from left
    for dy in (1.1, 0.55, 0, -0.55, -1.1):
        ax.plot([0.3, 2.0], [yax + dy, yax + dy], color="#aaa", lw=0.8)

    # FZP element
    ax.add_patch(Rectangle((2.0, yax - 1.25), 0.16, 2.5, facecolor="#222",
                           edgecolor="#000"))
    ax.text(2.08, yax + 1.5, "波带片", ha="center", fontsize=9.5)

    # map focal length (mm) to x position
    fmin, fmax = 110.0, 185.0
    x0, x1 = 2.16, 11.2

    def fx(f):
        return x0 + (f - fmin) / (fmax - fmin) * (x1 - x0)

    # optical axis
    ax.plot([2.0, 11.4], [yax, yax], color="#888", lw=0.6, ls="-.", zorder=0)

    label_dy = [1.55, 1.55, 1.55]
    sign = [1, 1, 1]
    for i, (f, c, lab) in enumerate(foci):
        xf = fx(f)
        for dy in (1.2, 0.6, -0.6, -1.2):
            ax.plot([2.16, xf], [yax + dy, yax], color=c, lw=0.9, alpha=0.85)
        ax.plot(xf, yax, "o", color=c, ms=6, zorder=5)
        # staggered labels to avoid overlap
        yl = 1.35 if i == 0 else (1.75 if i == 1 else 1.35)
        ax.annotate(f"{lab}\n$f\\approx{f:.0f}$ mm", xy=(xf, yax),
                    xytext=(xf, yl), ha="center", fontsize=9, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                    alpha=0.5))
        # z tick below
        ax.text(xf, -1.45, f"{f:.0f}", ha="center", fontsize=8, color=c)

    ax.annotate("", xy=(fx(177), -1.15), xytext=(fx(126), -1.15),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1))
    ax.text((fx(126) + fx(177)) / 2, -1.85,
            r"焦距随波长移动：$f\propto 1/\lambda$（长波长焦距更短）",
            ha="center", fontsize=10, color="#333")
    ax.text(11.45, -1.45, "z (mm)", ha="left", fontsize=8.5, color="#555")
    save(fig, "fig8_chromatic")


def main():
    fig1_paradigm()
    fig2_fzp_structure()
    fig3_targets()
    fig4_sector()
    fig5_method_b()
    fig6_setup()
    fig7_shapes()
    fig8_chromatic()


if __name__ == "__main__":
    main()
