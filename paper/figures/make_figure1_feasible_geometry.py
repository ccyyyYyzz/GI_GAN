from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D
import numpy as np


HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "figure1_assets"
DATA_ROOT = Path(r"E:\ns_mc_gan_gi")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/mnt/e/ns_mc_gan_gi")
CACHE = DATA_ROOT / "results" / "cert_package_20260612" / "cache"
MAIN_RAD5 = CACHE / "main_rad5.npz"
A_RAD5 = CACHE / "A_rad5.npy"
SPLIT_EVAL = CACHE / "split_eval_indices_stl10_test.npy"

I_TARGET = 1789
J_DONOR = 935
IMG_SHAPE = (64, 64)

BLUE = "#2F6FED"
BLUE_SIDE = "#17448F"
ORANGE = "#E28B2C"
ORANGE_SIDE = "#9A5618"
CYAN = "#BEE8E7"
BG = "#FAFAF8"
INK = "#1F2933"
MUTED = "#5D6975"
GREEN = "#28734E"
RED = "#8A2C20"
REAL_EDGE = "#EEF2F6"
REAL_SIDE = "#AAB4BF"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255.0


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((np.clip(pred, 0.0, 1.0) - target) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def robust01(v: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def gray_rgb(img01: np.ndarray) -> np.ndarray:
    arr = np.clip(img01, 0.0, 1.0)
    return np.repeat(arr[..., None], 3, axis=2)


def resize_rgb_bilinear(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    out_h, out_w = shape
    if (h, w) == (out_h, out_w):
        return rgb
    yy = np.linspace(0, h - 1, out_h)
    xx = np.linspace(0, w - 1, out_w)
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None, None]
    wx = (xx - x0)[None, :, None]
    top = (1.0 - wx) * rgb[y0[:, None], x0[None, :]] + wx * rgb[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx) * rgb[y1[:, None], x0[None, :]] + wx * rgb[y1[:, None], x1[None, :]]
    return (1.0 - wy) * top + wy * bottom


def trim_dark_padding(rgb: np.ndarray, dark: float = 0.08, max_dark_frac: float = 0.20) -> np.ndarray:
    arr = np.clip(rgb, 0.0, 1.0)
    lum = arr.mean(axis=2)
    h, w = lum.shape
    top, bottom, left, right = 0, h, 0, w

    while top < bottom - 2 and float((lum[top, left:right] < dark).mean()) > max_dark_frac:
        top += 1
    while bottom - 1 > top + 1 and float((lum[bottom - 1, left:right] < dark).mean()) > max_dark_frac:
        bottom -= 1
    while left < right - 2 and float((lum[top:bottom, left] < dark).mean()) > max_dark_frac:
        left += 1
    while right - 1 > left + 1 and float((lum[top:bottom, right - 1] < dark).mean()) > max_dark_frac:
        right -= 1

    cropped = arr[top:bottom, left:right]
    return resize_rgb_bilinear(cropped, arr.shape[:2])


def clean_card_rgb(rgb: np.ndarray, edge_crop: int = 4) -> np.ndarray:
    arr = trim_dark_padding(rgb)
    if edge_crop <= 0 or min(arr.shape[:2]) <= 2 * edge_crop + 1:
        return arr
    cropped = arr[edge_crop:-edge_crop, edge_crop:-edge_crop]
    return resize_rgb_bilinear(cropped, arr.shape[:2])


def tint_gray(img01: np.ndarray, color: str, strength: float = 0.72) -> np.ndarray:
    gray = gray_rgb(img01)
    tint = np.ones_like(gray) * hex_rgb(color)[None, None, :]
    return np.clip((1.0 - strength) * gray + strength * tint * (0.35 + 0.65 * img01[..., None]), 0.0, 1.0)


def orange_detail(p0: np.ndarray) -> np.ndarray:
    mag01 = robust01(np.abs(p0), 2.0, 99.2)
    base = np.ones((*mag01.shape, 3), dtype=np.float64) * np.array([1.0, 0.95, 0.87])
    tint = np.ones_like(base) * hex_rgb(ORANGE)[None, None, :]
    return np.clip((1.0 - mag01[..., None]) * base + mag01[..., None] * tint, 0.0, 1.0)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    plt.imsave(path, np.clip(rgb, 0.0, 1.0))


def rotated_corners(center: tuple[float, float], size: tuple[float, float], angle: float) -> np.ndarray:
    cx, cy = center
    w, h = size
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    th = np.deg2rad(angle)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return pts @ r.T + np.array([cx, cy])


def label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str = INK,
    size: float = 8.0,
    ha: str = "center",
    va: str = "center",
    boxed: bool = False,
    zorder: int = 50,
) -> None:
    bbox = (
        dict(boxstyle="round,pad=0.12,rounding_size=0.035", facecolor=BG, edgecolor="none", alpha=0.93)
        if boxed
        else None
    )
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, zorder=zorder, bbox=bbox)


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.15,
    mutation: float = 10.5,
    alpha: float = 0.95,
    zorder: int = 30,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def draw_layer(
    ax,
    img: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    *,
    edge: str,
    side: str,
    zorder: int,
    alpha: float = 1.0,
    thickness: tuple[float, float] = (0.085, -0.09),
    edge_width: float = 1.05,
    side_alpha: tuple[float, float] = (0.38, 0.30),
) -> None:
    corners = rotated_corners(center, size, angle)
    offset = np.array(thickness)
    # A small side wall on the bottom and right edges makes each layer read as a slab.
    ax.add_patch(
        patches.Polygon(
            [corners[0], corners[1], corners[1] + offset, corners[0] + offset],
            closed=True,
            facecolor=side,
            edgecolor="none",
            alpha=side_alpha[0],
            zorder=zorder - 2,
        )
    )
    ax.add_patch(
        patches.Polygon(
            [corners[1], corners[2], corners[2] + offset, corners[1] + offset],
            closed=True,
            facecolor=side,
            edgecolor="none",
            alpha=side_alpha[1],
            zorder=zorder - 2,
        )
    )
    cx, cy = center
    w, h = size
    trans = Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    image = ax.imshow(
        np.clip(img, 0.0, 1.0),
        extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
        transform=trans,
        interpolation="bilinear",
        zorder=zorder,
        alpha=alpha,
    )
    image.set_clip_path(patches.Polygon(corners, closed=True, transform=ax.transData))
    ax.add_patch(
        patches.Polygon(corners, closed=True, facecolor="none", edgecolor=edge, linewidth=edge_width, zorder=zorder + 1)
    )


def draw_stack(
    ax,
    base_center: tuple[float, float],
    *,
    row_img: np.ndarray,
    p0_img: np.ndarray,
    top_img: np.ndarray,
    title: str,
    title_color: str,
) -> dict[str, tuple[float, float]]:
    x, y = base_center
    size = (0.93, 0.93)
    angle = -7.0
    # Layers recede upward-left from the front image: blue row-space, orange null-space, top composite.
    row_center = (x - 0.42, y + 0.34)
    p0_center = (x - 0.13, y + 0.13)
    top_center = (x + 0.22, y - 0.16)
    draw_layer(ax, row_img, row_center, size, angle, edge=BLUE, side=BLUE_SIDE, zorder=15, alpha=0.94)
    draw_layer(ax, p0_img, p0_center, size, angle, edge=ORANGE, side=ORANGE_SIDE, zorder=21, alpha=0.90)
    draw_layer(
        ax,
        top_img,
        top_center,
        size,
        angle,
        edge=REAL_EDGE,
        side=REAL_SIDE,
        zorder=28,
        alpha=1.0,
        edge_width=0.74,
        side_alpha=(0.0, 0.0),
    )
    label(ax, top_center[0], top_center[1] - 0.70, title, color=title_color, size=6.9, boxed=True)
    return {"row": row_center, "p0": p0_center, "top": top_center}


def draw_plane(ax) -> None:
    top = np.array([[0.58, 0.74], [5.62, 0.58], [5.25, 3.98], [0.95, 4.26]])
    down = np.array([0.17, -0.18])
    ax.add_patch(
        patches.Polygon(
            [top[0], top[1], top[1] + down, top[0] + down],
            closed=True,
            facecolor="#77C7CC",
            edgecolor="none",
            alpha=0.24,
            zorder=0,
        )
    )
    ax.add_patch(
        patches.Polygon(
            top,
            closed=True,
            facecolor=CYAN,
            edgecolor="#4CB7BC",
            linewidth=1.55,
            alpha=0.58,
            zorder=1,
        )
    )
    label(ax, 1.22, 4.08, r"measurement feasible set $\{x:Ax=y_i\}$", color="#277B80", size=7.6, ha="left", boxed=True)


def make_y_readout_asset(y_car: np.ndarray, y_constructed: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 1.52), dpi=260)
    fig.patch.set_facecolor("#F8FBFF")
    ax.set_facecolor("#F8FBFF")
    idx = np.arange(1, y_car.size + 1)
    ax.plot(idx, y_car, color=BLUE, linewidth=0.82, alpha=0.98, linestyle="-", label=r"$Ax_i$ solid", zorder=3)
    ax.plot(
        idx,
        y_constructed,
        color=ORANGE,
        linewidth=0.72,
        alpha=0.92,
        linestyle=(0, (2.0, 1.45)),
        label=r"$Au_{ij}$ dashed",
        zorder=4,
    )
    ax.axhline(0.0, color="#94A3B8", linewidth=0.55)
    ax.set_xlim(1, y_car.size)
    ymin = float(min(np.min(y_car), np.min(y_constructed)))
    ymax = float(max(np.max(y_car), np.max(y_constructed)))
    pad = 0.08 * max(ymax - ymin, 1e-12)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.tick_params(axis="both", labelsize=5.8, length=2.0, pad=1.0, colors=MUTED)
    ax.set_xlabel("measurement index", fontsize=6.0, color=MUTED, labelpad=1.0)
    ax.set_ylabel("bucket readout", fontsize=6.0, color=MUTED, labelpad=1.0)
    ax.legend(loc="upper right", fontsize=5.6, frameon=False, handlelength=1.6, borderpad=0.1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(BLUE)
    fig.tight_layout(pad=0.28)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    ensure_dir(ASSET_DIR)
    with np.load(MAIN_RAD5) as data:
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        labels = data["labels"].astype(int)
    split_eval = np.load(SPLIT_EVAL)
    a = np.load(A_RAD5).astype(np.float64)
    cho = np.linalg.cholesky(a @ a.T)

    def adag(w: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(cho, w.T)
        z = np.linalg.solve(cho.T, z)
        return z.T @ a

    xi = x[I_TARGET]
    xj = x[J_DONOR]
    yi = y[I_TARGET]
    uij = xj - adag((xj @ a.T - yi)[None, :])[0]
    pr_xi = adag((xi @ a.T)[None, :])[0]
    pr_yi = adag(yi[None, :])[0]
    pr_xj = adag((xj @ a.T)[None, :])[0]
    p0_xi = xi - pr_xi
    p0_xj = xj - pr_xj
    p0_uij = uij - pr_yi
    y_car = xi @ a.T
    y_constructed = uij @ a.T

    xi_img = clean_card_rgb(gray_rgb(xi.reshape(IMG_SHAPE)))
    u_img = clean_card_rgb(gray_rgb(np.clip(uij, 0.0, 1.0).reshape(IMG_SHAPE)))
    pr_xi_img = clean_card_rgb(tint_gray(robust01(pr_xi.reshape(IMG_SHAPE), 1.0, 99.0), BLUE, strength=0.78))
    pr_y_img = clean_card_rgb(tint_gray(robust01(pr_yi.reshape(IMG_SHAPE), 1.0, 99.0), BLUE, strength=0.78))
    p0_xi_img = clean_card_rgb(orange_detail(p0_xi.reshape(IMG_SHAPE)))
    p0_xj_img = clean_card_rgb(orange_detail(p0_xj.reshape(IMG_SHAPE)))

    assets = {
        "car_truth_x_i": ASSET_DIR / "x_i_car_truth.png",
        "u_ij_constructed_clipped": ASSET_DIR / "u_ij_horse_null_car_row_clipped.png",
        "P_R_x_i_AdagA_x_i": ASSET_DIR / "P_R_x_i_AdagA_x_i.png",
        "shared_rowspace_Adag_y_i": ASSET_DIR / "shared_rowspace_Adag_y_i.png",
        "P0_x_i_orange": ASSET_DIR / "P0_x_i_orange.png",
        "P0_x_j_orange": ASSET_DIR / "P0_x_j_orange.png",
        "bucket_readouts_overlay_real": ASSET_DIR / "bucket_readouts_Axi_Auij_real.png",
        "y_i_vector_npy": ASSET_DIR / "y_i_vector.npy",
        "y_i_vector_csv": ASSET_DIR / "y_i_vector.csv",
        "bucket_readouts_Axi_Auij_npy": ASSET_DIR / "bucket_readouts_Axi_Auij.npy",
        "bucket_readouts_Axi_Auij_csv": ASSET_DIR / "bucket_readouts_Axi_Auij.csv",
    }
    save_rgb(assets["car_truth_x_i"], xi_img)
    save_rgb(assets["u_ij_constructed_clipped"], u_img)
    save_rgb(assets["P_R_x_i_AdagA_x_i"], pr_xi_img)
    save_rgb(assets["shared_rowspace_Adag_y_i"], pr_y_img)
    save_rgb(assets["P0_x_i_orange"], p0_xi_img)
    save_rgb(assets["P0_x_j_orange"], p0_xj_img)
    make_y_readout_asset(y_car, y_constructed, assets["bucket_readouts_overlay_real"])
    np.save(assets["y_i_vector_npy"], yi)
    np.savetxt(assets["y_i_vector_csv"], yi, delimiter=",")
    readouts = np.stack([y_car, y_constructed, yi], axis=1)
    np.save(assets["bucket_readouts_Axi_Auij_npy"], readouts)
    np.savetxt(
        assets["bucket_readouts_Axi_Auij_csv"],
        np.column_stack([np.arange(1, yi.size + 1), readouts]),
        delimiter=",",
        header="index,A_x_i,A_u_ij,y_i",
        comments="",
    )
    y_readout_img = plt.imread(assets["bucket_readouts_overlay_real"])

    metrics = {
        "target_i": I_TARGET,
        "donor_j": J_DONOR,
        "target_original_stl10_test_index": int(split_eval[I_TARGET]),
        "donor_original_stl10_test_index": int(split_eval[J_DONOR]),
        "target_label_index": int(labels[I_TARGET]),
        "donor_label_index": int(labels[J_DONOR]),
        "m": int(yi.size),
        "n": int(a.shape[1]),
        "relmeaserr_uij_vs_yi": rel(uij @ a.T, yi),
        "relmeaserr_xi_vs_yi": rel(xi @ a.T, yi),
        "relmeaserr_xj_vs_yi": rel(xj @ a.T, yi),
        "psnr_uij_vs_xi_clipped": psnr(uij, xi),
        "psnr_xj_vs_xi": psnr(xj, xi),
        "uij_min": float(uij.min()),
        "uij_max": float(uij.max()),
        "yi_min": float(yi.min()),
        "yi_max": float(yi.max()),
        "yi_mean": float(yi.mean()),
        "relmeaserr_Axi_vs_Auij": rel(y_car, y_constructed),
        "max_abs_Axi_minus_Auij": float(np.max(np.abs(y_car - y_constructed))),
        "rmse_Axi_minus_Auij": float(np.sqrt(np.mean((y_car - y_constructed) ** 2))),
        "AP0_xi_relative_to_Axi": float(np.linalg.norm(p0_xi @ a.T) / max(np.linalg.norm(xi @ a.T), 1e-12)),
        "AP0_xj_relative_to_Axj": float(np.linalg.norm(p0_xj @ a.T) / max(np.linalg.norm(xj @ a.T), 1e-12)),
        "P0_uij_minus_P0_xj_relative": rel(p0_uij, p0_xj),
        "PR_xi_minus_Adayi_relative": rel(pr_xi, pr_yi),
        "assets": {k: str(v) for k, v in assets.items()},
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.62), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 9.2)
    ax.set_ylim(0.0, 4.62)
    ax.axis("off")

    draw_plane(ax)
    arrow(ax, (1.82, 2.02), (4.20, 3.08), color=ORANGE, lw=3.0, mutation=12, alpha=0.92, zorder=7)
    label(ax, 2.52, 3.06, r"null-space direction" "\n" r"$(AP_0=0)$", color=ORANGE_SIDE, size=6.7, boxed=True, zorder=35)

    car = draw_stack(
        ax,
        (1.78, 1.90),
        row_img=pr_y_img,
        p0_img=p0_xi_img,
        top_img=xi_img,
        title=r"$x_i$ (truth)",
        title_color=GREEN,
    )
    wrong = draw_stack(
        ax,
        (3.98, 2.86),
        row_img=pr_y_img,
        p0_img=p0_xj_img,
        top_img=u_img,
        title=r"$u_{ij}$ constructed" "\n" "consistent but wrong",
        title_color=RED,
    )
    ax.plot(
        [car["row"][0] + 0.08, wrong["row"][0] - 0.04],
        [car["row"][1] - 0.55, wrong["row"][1] - 0.55],
        color="#8DB6FF",
        linewidth=1.45,
        linestyle=(0, (2.0, 2.0)),
        zorder=18,
    )
    label(ax, 2.75, 0.78, r"shared row space $P_Rx=A^\dagger y_i$" "\n" "(same blue layer)", color=BLUE, size=6.6, boxed=True, zorder=45)
    label(ax, 4.35, 4.04, r"null space $P_0$" "\n" "(different, prior-supplied)", color=ORANGE_SIDE, size=6.6, boxed=True, zorder=45)

    add_x, add_y = 6.24, 2.20
    merge = (5.62, add_y + 0.56)
    arrow(ax, (car["top"][0] + 0.54, car["top"][1] + 0.08), merge, color=BLUE, lw=1.25, zorder=32)
    arrow(ax, (wrong["top"][0] + 0.54, wrong["top"][1] - 0.04), merge, color=BLUE, lw=1.25, zorder=32)
    ax.add_patch(patches.Circle(merge, 0.055, facecolor="#FFFFFF", edgecolor=BLUE, linewidth=1.0, zorder=34))
    arrow(ax, merge, (add_x - 0.04, add_y + 0.56), color=BLUE, lw=1.55, zorder=32)
    label(ax, 5.58, 3.14, r"$Ax_i \approx Au_{ij}=y_i$", color=BLUE, size=6.7, boxed=True, zorder=45)
    label(ax, 5.34, 2.66, r"$A$", color=BLUE, size=10.0, boxed=True, zorder=45)

    ax.imshow(y_readout_img, extent=(add_x, add_x + 1.86, add_y, add_y + 1.13), zorder=18)
    ax.add_patch(
        patches.FancyBboxPatch(
            (add_x, add_y),
            1.86,
            1.13,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="none",
            edgecolor=BLUE,
            linewidth=1.05,
            zorder=28,
        )
    )
    label(
        ax,
        add_x + 0.93,
        add_y + 1.34,
        r"bucket readouts: $Ax_i$ vs. $Au_{ij}$" "\n" rf"(real curves, $m={yi.size}$)",
        color=BLUE,
        size=6.7,
        boxed=True,
        zorder=45,
    )
    label(
        ax,
        add_x + 0.93,
        add_y - 0.20,
        "different images -> indistinguishable measurements",
        color=MUTED,
        size=5.8,
        boxed=True,
        zorder=45,
    )
    label(
        ax,
        add_x + 0.07,
        1.30,
        r"RelMeasErr: $5.4{\times}10^{-3}$ (car)"
        "\n"
        r"vs. $2.9{\times}10^{-15}$ (constructed)",
        color=INK,
        size=6.7,
        ha="left",
        boxed=True,
        zorder=45,
    )

    label(
        ax,
        4.60,
        0.18,
        "Both share the same measured row space and differ only in the unmeasured null space;\n"
        "the single-pixel measurement collapses them to the same y -- consistency does not imply correctness.",
        color=INK,
        size=6.9,
        boxed=True,
        zorder=60,
    )

    out_pdf = HERE / "figure1_feasible_geometry.pdf"
    out_png = HERE / "figure1_feasible_geometry.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03, dpi=450)
    plt.close(fig)
    metrics["figure_pdf"] = str(out_pdf)
    metrics["figure_png"] = str(out_png)
    (ASSET_DIR / "figure1_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
