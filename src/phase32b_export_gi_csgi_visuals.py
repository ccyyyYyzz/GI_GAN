from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .metrics import batch_metrics
from .phase16_common import (
    PHASE16,
    dataloader_for,
    setup_method,
    tv_loss,
)
from .phase15r_common import controlled_reconstruct
from .phase20_common import BLUE, GRAY, GREEN, LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, ORANGE, RED, setup_matplotlib


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase32b_algorithm_optics_baselines"
PROJECT = OUT / "latex_project_algorithm_optics"
FIG_DIR = OUT / "figures"
PROJECT_FIG = PROJECT / "figures"
BASELINE_DIR = OUT / "baseline_visuals"
TABLE_DIR = PROJECT / "tables"
TV_TABLE = PHASE16 / "traditional_baselines" / "tv_pgd_baseline_results.csv"

METHODS = [
    ("rademacher5_hq_noise001_colab", "Rad-5", 5),
    ("scrambled_hadamard5_hq_noise001_colab", "Scr-5", 5),
    ("rademacher10_full_noise001_colab", "Rad-10", 5),
    ("scrambled_hadamard10_full_noise001_colab", "Scr-10", 5),
]


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def arrow(ax, start, end, color=GRAY, lw=1.4, rad=0.0, scale=12, alpha=1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def panel_box(ax, xy, w, h, label: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            fc="white",
            ec="#D6D9DE",
            lw=1.05,
        )
    )
    ax.text(xy[0] + 0.020, xy[1] + h - 0.045, label, fontsize=9.8, fontweight="bold", va="center")
    ax.text(xy[0] + 0.070, xy[1] + h - 0.045, title, fontsize=8.5, fontweight="bold", va="center")


def save_figure(fig: plt.Figure, stem: str, *, dirs: tuple[Path, ...], dpi: int = 300) -> None:
    for directory in dirs:
        ensure(directory)
    for ext in ("pdf", "png", "svg"):
        path = dirs[0] / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        for directory in dirs[1:]:
            shutil.copy2(path, directory / path.name)
    plt.close(fig)


def make_forward_model_figure() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.6, 6.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.suptitle("Computational GI forward model and measurement-consistent reconstruction", y=0.985, fontsize=13, fontweight="bold")

    panels = {
        "a": (0.035, 0.575, 0.285, 0.340, "(a)", "Forward model"),
        "b": (0.360, 0.575, 0.285, 0.340, "(b)", "Low-sampling ambiguity"),
        "c": (0.680, 0.575, 0.285, 0.340, "(c)", "Data solution"),
        "d": (0.105, 0.100, 0.380, 0.335, "(d)", "Neural null-space completion"),
        "e": (0.535, 0.100, 0.380, 0.335, "(e)", "Measurement audit"),
    }
    for x, y, w, h, lab, title in panels.values():
        panel_box(ax, (x, y), w, h, lab, title)

    rng = np.random.default_rng(321)
    for idx, x0 in enumerate([0.073, 0.118, 0.163]):
        pattern = rng.choice([-1, 1], size=(7, 7))
        y0 = 0.745 - 0.018 * idx
        ax.imshow(pattern, cmap="gray", extent=(x0, x0 + 0.078, y0, y0 + 0.078), interpolation="nearest", zorder=3)
        ax.add_patch(Rectangle((x0, y0), 0.078, 0.078, fill=False, ec=BLUE, lw=0.75, zorder=4))
    ax.set_aspect("auto")
    ax.add_patch(Ellipse((0.210, 0.727), 0.086, 0.132, fc="#F8FAFC", ec=GRAY, lw=1.1))
    ax.text(0.210, 0.727, r"$x$", ha="center", va="center", fontsize=13)
    ax.add_patch(Circle((0.280, 0.727), 0.035, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.text(0.280, 0.727, r"$y$", ha="center", va="center", fontsize=12, color=BLUE)
    arrow(ax, (0.207, 0.727), (0.245, 0.727), BLUE, 1.1, scale=11)
    ax.text(0.178, 0.650, r"$y_i=\langle a_i,x\rangle+\epsilon_i$", ha="center", fontsize=9.1, color=BLUE)
    ax.text(0.178, 0.615, r"$y=Ax+\epsilon$", ha="center", fontsize=9.1, color=BLUE)
    ax.text(0.115, 0.842, "known\npatterns $a_i$", ha="center", va="top", fontsize=8.1, color=BLUE)
    ax.text(0.280, 0.667, "scalar bucket\nmeasurements", ha="center", fontsize=7.9, color=BLUE)
    ax.text(0.178, 0.592, "computational GI/SPI forward model", ha="center", fontsize=7.6, color=GRAY)

    ax.add_patch(Ellipse((0.505, 0.730), 0.210, 0.122, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.2))
    ax.plot([0.410, 0.592], [0.690, 0.770], color=GREEN, lw=2.2)
    ax.scatter([0.478], [0.721], s=65, color=BLUE, edgecolor="white", zorder=4)
    arrow(ax, (0.480, 0.722), (0.558, 0.756), GREEN, 1.55, scale=12)
    ax.text(0.505, 0.802, r"$m\ll n,\quad \mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=8.6, color=GREEN)
    ax.text(0.570, 0.774, r"$v\in\mathrm{Null}(A)$", fontsize=8.2, color=GREEN)
    ax.text(0.505, 0.635, "many images can share\nthe same bucket vector", ha="center", fontsize=8.3, color=GRAY)

    ax.add_patch(Ellipse((0.825, 0.735), 0.205, 0.122, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.2, alpha=0.86))
    ax.plot([0.735, 0.912], [0.695, 0.772], color=GREEN, lw=2.1)
    ax.scatter([0.795], [0.722], s=75, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.795, 0.684, r"$x_{\rm data}$", ha="center", fontsize=9.0, color=BLUE)
    ax.text(0.825, 0.820, r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=8.1, color=BLUE)
    ax.text(0.825, 0.628, "physical / linear\ndata solution", ha="center", fontsize=8.2, color=BLUE)

    ax.scatter([0.175], [0.260], s=70, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.175, 0.214, r"$x_{\rm data}$", ha="center", fontsize=9.0, color=BLUE)
    ax.add_patch(FancyBboxPatch((0.245, 0.232), 0.090, 0.062, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_ORANGE, ec=ORANGE, lw=1.2))
    ax.text(0.290, 0.263, r"$G_\theta(x_{\rm data},z)$", ha="center", va="center", fontsize=8.7, color=ORANGE)
    ax.add_patch(FancyBboxPatch((0.367, 0.232), 0.076, 0.062, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_GREEN, ec=GREEN, lw=1.2))
    ax.text(0.405, 0.263, r"$P_N$", ha="center", va="center", fontsize=10.2, color=GREEN)
    arrow(ax, (0.202, 0.260), (0.245, 0.260), ORANGE, 1.4, scale=12)
    arrow(ax, (0.335, 0.260), (0.367, 0.260), GREEN, 1.4, scale=12)
    arrow(ax, (0.443, 0.260), (0.485, 0.260), GREEN, 1.0, scale=11, alpha=0.55)
    ax.text(0.315, 0.155, "learned missing structure", ha="center", fontsize=8.6, color=ORANGE)
    ax.text(0.405, 0.326, "weakly observed /\nunobserved", ha="center", fontsize=8.0, color=GREEN)

    ax.add_patch(Ellipse((0.700, 0.275), 0.205, 0.122, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.2, alpha=0.88))
    ax.plot([0.607, 0.792], [0.235, 0.315], color=GREEN, lw=2.1)
    ax.scatter([0.615], [0.342], s=65, color=RED, edgecolor="white", zorder=4)
    ax.scatter([0.700], [0.272], s=78, color=GREEN, edgecolor="white", zorder=4)
    arrow(ax, (0.622, 0.334), (0.687, 0.282), GREEN, 1.6, rad=-0.15, scale=12)
    ax.text(0.612, 0.370, "candidate", ha="center", fontsize=8.5, color=RED)
    ax.text(0.700, 0.225, r"$\hat{x}$", ha="center", fontsize=9.4, color=GREEN)
    ax.add_patch(FancyBboxPatch((0.790, 0.310), 0.065, 0.052, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_GREEN, ec=GREEN, lw=1.2))
    ax.text(0.822, 0.336, r"$\Pi_y$", ha="center", va="center", fontsize=10.2, color=GREEN)
    ax.text(0.823, 0.246, r"$A\hat{x}\approx y$", ha="center", fontsize=9.2, color=GREEN)
    ax.text(0.823, 0.200, "RelMeasErr", ha="center", fontsize=8.5, color=GRAY)
    ax.text(0.700, 0.154, "output remains auditable\nagainst bucket measurements", ha="center", fontsize=8.1, color=GREEN)

    arrow(ax, (0.323, 0.727), (0.360, 0.727), BLUE, 1.0, scale=10, alpha=0.55)
    arrow(ax, (0.645, 0.727), (0.680, 0.727), BLUE, 1.0, scale=10, alpha=0.55)
    arrow(ax, (0.820, 0.575), (0.445, 0.435), ORANGE, 1.0, rad=0.18, scale=10, alpha=0.42)
    arrow(ax, (0.485, 0.260), (0.535, 0.260), GREEN, 1.0, scale=10, alpha=0.55)
    save_figure(fig, "fig1_forward_model_reconstruction", dirs=(FIG_DIR, PROJECT_FIG))


def read_best_tv_lambdas() -> dict[str, float]:
    best: dict[str, tuple[float, float]] = {}
    with TV_TABLE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("baseline") != "tv_pgd":
                continue
            method = row["method_id"]
            psnr = float(row["psnr"])
            lam = float(row["lambda_tv"])
            if method not in best or psnr > best[method][0]:
                best[method] = (psnr, lam)
    return {method: lam for method, (_psnr, lam) in best.items()}


def tensor_img(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().clamp(0, 1).cpu()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    return arr.numpy()


def run_tv_pgd(measurement: Any, y: torch.Tensor, init: torch.Tensor, *, lambda_tv: float, iterations: int = 50) -> torch.Tensor:
    z = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        pred_y = measurement.A_forward(measurement.flatten_img(z))
        fidelity = 0.5 * torch.mean((pred_y - y) ** 2)
        loss = fidelity + float(lambda_tv) * tv_loss(z)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(0.0, 1.0)
    return z.detach()


def save_tile(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    ensure(path.parent)
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def make_visual_record(method_id: str, label: str, sample_index: int, lambda_tv: float) -> dict[str, Any]:
    generator, measurement, config, _info = setup_method(method_id, limit=sample_index + 1, batch_size=sample_index + 1)
    loader = dataloader_for(config, "test")
    batch = next(iter(loader))
    x = batch[0][sample_index : sample_index + 1].to(measurement.device)
    with torch.no_grad():
        y = measurement.measure(x)
        x_hat, x_data, _extras = controlled_reconstruct(
            generator,
            measurement,
            y,
            use_null_project=True,
            use_dc_project=True,
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            noise_map_mode="fixed",
            batch_idx=0,
            seed=int(config["seed"]),
        )
    tv = run_tv_pgd(measurement, y, x_data.clamp(0, 1), lambda_tv=lambda_tv, iterations=50)
    err = torch.abs(x_hat.clamp(0, 1) - x)
    metrics = {
        "bp": batch_metrics(x_data.clamp(0, 1), x, measurement, y),
        "cstv": batch_metrics(tv, x, measurement, y),
        "ours": batch_metrics(x_hat.clamp(0, 1), x, measurement, y),
    }
    stem = label.lower().replace("-", "")
    tile_dir = BASELINE_DIR / "tiles" / stem
    for name, tensor in {
        "gt": x,
        "gi_bp": x_data,
        "csgi_cstv_pgd": tv,
        "ours": x_hat,
        "ours_abs_error": err,
    }.items():
        save_tile(tensor_img(tensor), tile_dir / f"{name}.png")
    record = {
        "method_id": method_id,
        "label": label,
        "sample_index": sample_index,
        "lambda_tv": lambda_tv,
        "gt": tensor_img(x),
        "bp": tensor_img(x_data),
        "cstv": tensor_img(tv),
        "ours": tensor_img(x_hat),
        "err": tensor_img(err),
        "bp_psnr": metrics["bp"]["psnr"],
        "bp_ssim": metrics["bp"]["ssim"],
        "cstv_psnr": metrics["cstv"]["psnr"],
        "cstv_ssim": metrics["cstv"]["ssim"],
        "ours_psnr": metrics["ours"]["psnr"],
        "ours_ssim": metrics["ours"]["ssim"],
        "tile_dir": str(tile_dir),
    }
    del generator, measurement, x, y, x_hat, x_data, tv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def draw_visual_grid(records: list[dict[str, Any]], stem: str, title: str, *, dirs: tuple[Path, ...]) -> None:
    setup_matplotlib()
    columns = [
        ("gt", "Ground truth"),
        ("bp", "GI/BP"),
        ("cstv", "CSGI / CS-TV(PGD)"),
        ("ours", "Ours"),
        ("err", "|Ours-GT|"),
    ]
    nrows = len(records)
    fig, axes = plt.subplots(nrows, len(columns), figsize=(9.8, 2.15 * nrows + 0.9))
    if nrows == 1:
        axes = axes[None, :]
    for col_idx, (_key, col_title) in enumerate(columns):
        axes[0, col_idx].set_title(col_title, fontsize=10)
    for row_idx, record in enumerate(records):
        for col_idx, (key, _col_title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            cmap = "magma" if key == "err" else "gray"
            vmax = float(np.quantile(record["err"], 0.985)) if key == "err" else 1.0
            ax.imshow(record[key], cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(record["label"], rotation=0, labelpad=28, va="center", fontsize=10)
        axes[row_idx, 1].text(0.5, -0.10, f"{record['bp_psnr']:.1f} dB", transform=axes[row_idx, 1].transAxes, ha="center", va="top", fontsize=7)
        axes[row_idx, 2].text(0.5, -0.10, f"{record['cstv_psnr']:.1f} dB", transform=axes[row_idx, 2].transAxes, ha="center", va="top", fontsize=7)
        axes[row_idx, 3].text(0.5, -0.10, f"{record['ours_psnr']:.1f} dB", transform=axes[row_idx, 3].transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.012,
        "CS-TV(PGD) is a CSGI-style lightweight visual control; quantitative conclusions use the full tables.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.08, wspace=0.03, hspace=0.12)
    save_figure(fig, stem, dirs=dirs)


def write_manifest(records: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "method_id",
        "sample_index",
        "lambda_tv",
        "bp_psnr",
        "bp_ssim",
        "cstv_psnr",
        "cstv_ssim",
        "ours_psnr",
        "ours_ssim",
        "tile_dir",
    ]
    csv_path = BASELINE_DIR / "baseline_visual_manifest.csv"
    ensure(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    lines = [
        "# Baseline Visual Manifest",
        "",
        "Visual rows use selected STL-10 test samples. GI/BP is the linear physical backprojection or correlation-like reconstruction. CSGI uses CS-TV(PGD), a TV-regularized compressed-sensing control solved by 50 steps of projected gradient descent. It is not an exhaustively optimized ADMM or FISTA benchmark.",
        "",
        "| Label | Sample | Lambda TV | GI/BP PSNR | CS-TV(PGD) PSNR | Ours PSNR | Tile dir |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['label']} | {record['sample_index']} | {record['lambda_tv']:.3g} | "
            f"{record['bp_psnr']:.3f} | {record['cstv_psnr']:.3f} | {record['ours_psnr']:.3f} | {record['tile_dir']} |"
        )
    (BASELINE_DIR / "baseline_visual_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = TABLE_DIR / "tableS8_gi_csgi_visual_subset.tex"
    ensure(table.parent)
    body = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Selected-sample GI/BP, CSGI-style CS-TV(PGD), and ours metrics for the visual comparison. CS-TV(PGD) is a lightweight visual control and is not an exhaustively optimized compressed-sensing solver.}",
        r"\label{tab:supp_gi_csgi_visual_subset}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & GI/BP PSNR & CS-TV(PGD) PSNR & Ours PSNR & CS-TV \(\lambda\) \\",
        r"\midrule",
    ]
    for record in records:
        body.append(
            f"{record['label']} & {record['bp_psnr']:.2f} & {record['cstv_psnr']:.2f} & "
            f"{record['ours_psnr']:.2f} & {record['lambda_tv']:.3g} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    ensure(FIG_DIR)
    ensure(PROJECT_FIG)
    ensure(BASELINE_DIR)
    make_forward_model_figure()
    best_lambdas = read_best_tv_lambdas()
    records = []
    for method_id, label, sample_index in METHODS:
        records.append(make_visual_record(method_id, label, sample_index, best_lambdas[method_id]))

    main_records = [record for record in records if record["label"] in {"Rad-5", "Scr-5"}]
    draw_visual_grid(
        main_records,
        "fig7_gi_csgi_ours_visual_comparison",
        "GI/BP, CSGI-style CS-TV(PGD), and ours on STL-10 5%",
        dirs=(FIG_DIR, PROJECT_FIG),
    )
    draw_visual_grid(
        records,
        "fig7_gi_csgi_ours_all_supplement",
        "Supplementary GI/BP, CSGI-style CS-TV(PGD), and ours visual comparison",
        dirs=(FIG_DIR, PROJECT_FIG),
    )
    for record in main_records:
        draw_visual_grid(
            [record],
            f"gi_csgi_ours_{record['label'].lower().replace('-', '')}",
            f"{record['label']} visual baseline comparison",
            dirs=(BASELINE_DIR,),
        )
    draw_visual_grid(
        records,
        "gi_csgi_ours_all_supplement",
        "Supplementary GI/BP, CSGI-style CS-TV(PGD), and ours visual comparison",
        dirs=(BASELINE_DIR,),
    )
    write_manifest(records)
    print(
        {
            "figure1": str(FIG_DIR / "fig1_forward_model_reconstruction.pdf"),
            "main_visual": str(FIG_DIR / "fig7_gi_csgi_ours_visual_comparison.pdf"),
            "supp_visual": str(FIG_DIR / "fig7_gi_csgi_ours_all_supplement.pdf"),
            "manifest": str(BASELINE_DIR / "baseline_visual_manifest.md"),
        }
    )


if __name__ == "__main__":
    main()
