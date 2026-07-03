#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme 1 (BASELINE) mechanism figure -- ENGLISH labels.

Two-panel publication figure explaining how a binary-amplitude Fresnel zone
plate (FZP) focuses a collimated wave:

    (a) Focusing principle -- cross-section ray schematic.
    (b) Zone structure (top view) -- the concentric binary ring pattern.

Physics is taken verbatim from make_proposal_figures.py (the canonical,
physics-verified source): lambda = 532 nm, R = 2 mm, f0 = 150 mm,
half-wave zone index q_f(rho) = floor(2*(sqrt(rho^2+f^2)-f)/lambda),
zone radii r_n = sqrt(n*lambda*f), finest width Delta r_N ~ lambda*f/(2R).

Outputs (into this directory):
    mfig_scheme1_mechanism.pdf
    mfig_scheme1_mechanism.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# English-only rendering (no CJK fonts pulled in).
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------------
# Physical parameters (mm) -- identical to the canonical source.
# ---------------------------------------------------------------------------
LAM = 532e-6          # wavelength in mm (532 nm)
R = 2.0               # outer radius mm
F0 = 150.0            # nominal focal length mm

# Colourblind-friendly muted palette.
DARK = (0.16, 0.18, 0.20)   # opaque half-wave bars
ACC = "#1f6feb"             # blue accent (radii / focus)
RED = "#d1495b"             # red accent (finest zone)
GRN = "#2a9d5c"             # green accent (rays / phase note)
RAY = "#c2783f"             # warm ray colour for incoming light


def q_zone(rho, f):
    """Half-wave zone index q_f(rho) (floor) -- from canonical source."""
    return np.floor(2.0 * (np.sqrt(rho**2 + f**2) - f) / LAM)


def r_of(n, f=F0):
    """Zone-boundary radius r_n = sqrt(n*lambda*f + n^2*lambda^2/4)."""
    return np.sqrt(n * LAM * f + (n * LAM / 2.0) ** 2)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name + ".pdf / .png")


# ===========================================================================
# Panel (a) -- focusing principle (cross-section ray schematic)
# ===========================================================================
def panel_focusing(ax):
    # Schematic coordinate frame: plate at x = 0, focus at x = XF.
    # The x-direction is NOT to scale (schematic); y is the radial extent.
    XF = 6.2                      # focus x-position (schematic units)
    X_IN = -3.6                   # incoming-wave start
    YMAX = 2.45                   # top of the plate (slightly > R for labels)

    ax.set_xlim(X_IN - 0.95, XF + 2.1)
    ax.set_ylim(-3.62, 3.55)
    ax.axis("off")
    ax.set_aspect("auto")

    # ---- Optical axis ----
    ax.plot([X_IN, XF + 1.2], [0, 0], color="#999", lw=0.7, ls="-.", zorder=1)

    # ---- Build the 1-D radial zone profile from the real physics ----
    # Open where q even (T=1), opaque where q odd (T=0). We mirror to +/-.
    n_max = 7                     # number of zone boundaries to draw
    bounds = [0.0] + [r_of(n) for n in range(1, n_max + 1)]
    # Determine open/opaque for each annular band [bounds[k], bounds[k+1]].
    # Use the band midpoint radius -> q parity.
    bands = []
    for k in range(len(bounds) - 1):
        rmid = 0.5 * (bounds[k] + bounds[k + 1])
        q = q_zone(rmid, F0)
        is_open = (q % 2 == 0)
        bands.append((bounds[k], bounds[k + 1], is_open))

    plate_w = 0.20                # drawn thickness of the plate element
    x_plate = 0.0

    # Draw plate background (thin white element outline).
    ax.add_patch(Rectangle((x_plate - plate_w / 2, -YMAX), plate_w, 2 * YMAX,
                           facecolor="white", edgecolor="#444", lw=1.0,
                           zorder=3))

    open_band_centers = []        # radial centres of OPEN slots (for rays)
    for (r0, r1, is_open) in bands:
        for sgn in (+1, -1):
            y0 = sgn * r0
            y1 = sgn * r1
            ylo, yhi = sorted((y0, y1))
            if is_open:
                # open slot: leave white (just record centre for rays)
                yc = sgn * 0.5 * (r0 + r1)
                open_band_centers.append(yc)
            else:
                ax.add_patch(Rectangle((x_plate - plate_w / 2, ylo),
                                       plate_w, yhi - ylo,
                                       facecolor=DARK, edgecolor="none",
                                       zorder=4))
    # central band (k=0) is open (q=0 even) -> a slot straddling the axis;
    # its single centre near 0 is already added twice (+/-); dedupe near axis.
    open_band_centers = sorted(set(round(y, 4) for y in open_band_centers))

    # ---- Incoming collimated plane wave (parallel rays from the left) ----
    ray_ys = np.linspace(-YMAX + 0.15, YMAX - 0.15, 9)
    for y in ray_ys:
        ax.annotate("", xy=(x_plate - plate_w / 2 - 0.04, y),
                    xytext=(X_IN, y),
                    arrowprops=dict(arrowstyle="-|>", color=RAY, lw=1.1,
                                    alpha=0.9), zorder=2)
    ax.text(X_IN + 0.85, YMAX + 0.55, "collimated plane wave",
            ha="center", va="center", fontsize=9.5, color=RAY)

    # ---- Converging rays from each OPEN slot to the common focus F ----
    for yc in open_band_centers:
        ax.plot([x_plate + plate_w / 2, XF], [yc, 0.0],
                color=GRN, lw=1.0, alpha=0.85, zorder=5)

    # ---- Focus point F ----
    ax.plot(XF, 0.0, "o", color=ACC, ms=8, zorder=6)
    ax.annotate("focus  F", xy=(XF, 0.0), xytext=(XF + 0.35, -0.7),
                fontsize=11, color=ACC, ha="left", va="center")

    # ---- Focal-length dimension arrow (plate -> F) ----
    ydim = -2.35
    ax.annotate("", xy=(XF, ydim), xytext=(x_plate, ydim),
                arrowprops=dict(arrowstyle="<|-|>", color="#333", lw=1.2))
    ax.plot([x_plate, x_plate], [ydim - 0.12, ydim + 0.12], color="#333",
            lw=0.9)
    ax.plot([XF, XF], [ydim - 0.12, ydim + 0.12], color="#333", lw=0.9)
    ax.text((x_plate + XF) / 2, ydim - 0.30,
            r"focal length  $f = 150$ mm", ha="center", va="top",
            fontsize=10.5, color="#222")

    # ---- Outer radius R (vertical dimension at the plate) ----
    xR = x_plate - 0.62
    ax.annotate("", xy=(xR, YMAX - 0.30), xytext=(xR, 0.0),
                arrowprops=dict(arrowstyle="<|-|>", color="#444", lw=1.1))
    ax.text(xR - 0.12, (YMAX - 0.30) / 2, r"$R = 2$ mm",
            ha="right", va="center", fontsize=10, color="#222", rotation=90)

    # ---- Finest outermost zone width Delta r_N ----
    y_out = bands[-1][1]          # outermost boundary (top)
    ax.annotate(r"finest outer zone" "\n" r"$\Delta r_N \approx 20\ \mu$m",
                xy=(x_plate + plate_w / 2, y_out - 0.06),
                xytext=(x_plate + 1.20, YMAX + 1.02),
                fontsize=9.5, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))

    # ---- Plate label (top, just above the plate element) ----
    ax.text(x_plate, YMAX + 0.20, "binary-amplitude\nzone plate",
            ha="center", va="bottom", fontsize=9.5, color="#333")

    # ---- Constructive-interference / phase-matching note ----
    ax.text(XF - 0.30, 2.95,
            "light from open zones\n"
            "arrives in phase\n"
            r"(half-wave path match)" "\n"
            "$\\Rightarrow$ adds constructively",
            ha="left", va="top", fontsize=9.5, color=GRN)

    # ---- Key formulas (bottom centre, clear of other text) ----
    ax.text(0.5 * (X_IN + XF), -3.38,
            r"$r_n \approx \sqrt{n\lambda f}$"
            r"$\qquad \Delta r_N \approx \dfrac{\lambda f}{2R}$"
            r"$\qquad \mathrm{NA} \approx \dfrac{R}{f}$",
            ha="center", va="center", fontsize=11, color="#222")

    # Panel letter (axes-corner, outside the content to avoid collisions).
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=14,
            fontweight="bold", color="#111", ha="left", va="top")
    ax.set_title("Focusing principle", fontsize=12.5, pad=14)


# ===========================================================================
# Panel (b) -- zone structure (top view), adapted from fig2_fzp_structure
# ===========================================================================
def panel_structure(ax):
    N = 1100
    xs = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, xs)
    rho = np.sqrt(X**2 + Y**2)
    q = q_zone(rho, F0)
    T = (np.mod(q, 2) == 0).astype(float)        # 1 transparent, 0 opaque
    img = np.ones((N, N, 3))
    aperture = rho <= R
    opaque = aperture & (T == 0)
    img[opaque] = DARK
    img[~aperture] = (1, 1, 1)

    ax.imshow(img, extent=[-R, R, -R, R], origin="lower")
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#111", lw=1.6))
    ax.set_aspect("equal")

    # Annotate r_1 and r_3 with leader lines to clearly separated labels.
    _rn_lbl = dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                   alpha=0.78)
    for n, ang, lx, ly in ((1, -22, 1.30, -0.45), (3, -52, 1.55, -1.30)):
        rn = r_of(n)
        a = np.deg2rad(ang)
        ax.annotate(rf"$r_{{{n}}}$",
                    xy=(rn * np.cos(a), rn * np.sin(a)),
                    xytext=(lx, ly), fontsize=12.5, color=ACC,
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.3),
                    ha="center", va="center", bbox=_rn_lbl)

    # Finest outermost ring width.
    ax.annotate(r"$\Delta r_N \approx 20\ \mu$m",
                xy=(R, 0.0), xytext=(R + 0.10, 1.30),
                fontsize=10.5, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                                connectionstyle="arc3,rad=-0.3"))

    # Outer radius R.
    ax.annotate(r"$R = 2$ mm",
                xy=(-R * np.cos(np.deg2rad(40)), R * np.sin(np.deg2rad(40))),
                xytext=(-2.75, 1.75), fontsize=10.5, color="#222",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))

    # Caption line under the pattern.
    ax.text(0.10, -R - 0.32,
            r"open zones ($T=1$, white) / blocked zones ($T=0$, black),"
            r"  $r_n=\sqrt{n\lambda f}$",
            ha="center", va="top", fontsize=10, color="#333")

    ax.set_xlim(-R - 0.95, R + 1.55)
    ax.set_ylim(-R - 0.85, R + 0.55)
    ax.axis("off")

    # Panel letter.
    ax.text(-R - 0.90, R + 0.50, "(b)", fontsize=14, fontweight="bold",
            color="#111", ha="left", va="top")
    ax.set_title("Zone structure (top view)", fontsize=12.5, pad=14)


# ===========================================================================
# Assemble the two-panel figure.
# ===========================================================================
def main():
    fig = plt.figure(figsize=(11.0, 4.6))
    # Slightly wider left panel for the ray schematic.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0],
                          left=0.04, right=0.985, top=0.90, bottom=0.07,
                          wspace=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_focusing(ax_a)
    panel_structure(ax_b)
    save(fig, "mfig_scheme1_mechanism")


if __name__ == "__main__":
    main()
