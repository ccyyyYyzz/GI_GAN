from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image

from .phase20_common import BLUE, GRAY, GREEN, LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, ORANGE, RED, setup_matplotlib


ROOT = Path("E:/ns_mc_gan_gi")
PHASE29 = ROOT / "outputs_phase29_final_submission_polish"
PHASE29_FIG = PHASE29 / "figures"
OUT = ROOT / "outputs_phase30_submission_package"
PROJECT = OUT / "latex_project_submission"
FIG_DIR = OUT / "figures"
PROJECT_FIG = PROJECT / "figures"
EXPORT_DIR = OUT / "figures_for_submission"
HIGH_RES_DIR = OUT / "figures" / "high_res"
SOURCE_PACKAGE = OUT / "source_package"


FIGS = {
    "fig1_mechanism": ("fig1_mechanism_final", "fig1_mechanism_submission"),
    "fig2_primary_metrics": ("fig2_primary_metrics_final", "fig2_primary_metrics_submission"),
    "fig3_qualitative": ("fig3_qualitative_final", "fig3_qualitative_submission"),
    "fig4_measurement_attribution": ("fig4_measurement_attribution_final", "fig4_measurement_attribution_submission"),
    "fig5_inference_ablation": ("fig5_inference_ablation_final", "fig5_inference_ablation_submission"),
    "fig6_robustness_baselines": ("fig6_robustness_baselines_final", "fig6_robustness_baselines_submission"),
}

SUPP_FIGS = {
    "figS1_relmeaserr_ablation_submission": "figS1_relmeaserr_ablation_final",
}


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
    ax.text(xy[0] + 0.020, xy[1] + h - 0.045, label, fontsize=10.5, fontweight="bold", va="center")
    ax.text(xy[0] + 0.070, xy[1] + h - 0.045, title, fontsize=9.0, fontweight="bold", va="center")


def save_submission_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    ensure(FIG_DIR)
    ensure(PROJECT_FIG)
    for ext in ("pdf", "png", "svg"):
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        shutil.copy2(path, PROJECT_FIG / path.name)
    plt.close(fig)


def regenerate_fig1_submission() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.4, 6.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    panels = {
        "a": (0.035, 0.575, 0.285, 0.365, "(a)", "Optical acquisition"),
        "b": (0.360, 0.575, 0.285, 0.365, "(b)", "Measurement geometry"),
        "c": (0.680, 0.575, 0.285, 0.365, "(c)", "Physical data solution"),
        "d": (0.105, 0.095, 0.380, 0.355, "(d)", "Neural null-space completion"),
        "e": (0.535, 0.095, 0.380, 0.355, "(e)", "Measurement audit"),
    }
    for x, y, w, h, lab, title in panels.values():
        panel_box(ax, (x, y), w, h, lab, title)

    rng = np.random.default_rng(30)
    for idx, x0 in enumerate([0.075, 0.120, 0.165]):
        pattern = rng.choice([-1, 1], size=(7, 7))
        y0 = 0.755 - 0.020 * idx
        ax.imshow(pattern, cmap="gray", extent=(x0, x0 + 0.080, y0, y0 + 0.080), interpolation="nearest", zorder=3)
        ax.add_patch(Rectangle((x0, y0), 0.080, 0.080, fill=False, ec=BLUE, lw=0.8, zorder=4))
    ax.set_aspect("auto")
    ax.add_patch(Ellipse((0.210, 0.738), 0.086, 0.140, fc="#F8FAFC", ec=GRAY, lw=1.2))
    ax.text(0.210, 0.738, r"$x$", ha="center", va="center", fontsize=13)
    ax.add_patch(Circle((0.280, 0.738), 0.036, fc=LIGHT_BLUE, ec=BLUE, lw=1.2))
    ax.text(0.280, 0.738, r"$y$", ha="center", va="center", fontsize=12, color=BLUE)
    arrow(ax, (0.205, 0.738), (0.245, 0.738), BLUE, 1.2, scale=11)
    ax.text(0.178, 0.650, r"$y_i=\langle a_i,x\rangle+\epsilon_i$", ha="center", fontsize=9.2, color=BLUE)
    ax.text(0.115, 0.858, "structured\npatterns", ha="center", va="top", fontsize=8.3, color=BLUE)
    ax.text(0.280, 0.675, "bucket", ha="center", fontsize=8.4, color=BLUE)

    ax.add_patch(Ellipse((0.505, 0.742), 0.210, 0.122, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.2))
    ax.plot([0.410, 0.592], [0.702, 0.782], color=GREEN, lw=2.2)
    ax.scatter([0.478], [0.733], s=65, color=BLUE, edgecolor="white", zorder=4)
    arrow(ax, (0.480, 0.734), (0.558, 0.768), GREEN, 1.55, scale=12)
    ax.text(0.505, 0.815, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=9.4, color=GREEN)
    ax.text(0.573, 0.787, r"$v\in\mathrm{Null}(A)$", fontsize=8.4, color=GREEN)
    ax.text(0.505, 0.642, "many images share same y", ha="center", fontsize=8.4, color=GRAY)

    ax.add_patch(Ellipse((0.825, 0.750), 0.205, 0.122, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.2, alpha=0.86))
    ax.plot([0.735, 0.912], [0.710, 0.787], color=GREEN, lw=2.1)
    ax.scatter([0.795], [0.737], s=75, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.795, 0.699, r"$x_{\rm data}$", ha="center", fontsize=9.2, color=BLUE)
    ax.text(0.825, 0.835, r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=8.4, color=BLUE)
    ax.text(0.825, 0.640, "measured row-space\nrepresentative", ha="center", fontsize=8.4, color=BLUE)

    ax.scatter([0.175], [0.260], s=70, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.175, 0.214, r"$x_{\rm data}$", ha="center", fontsize=9.0, color=BLUE)
    ax.add_patch(FancyBboxPatch((0.245, 0.232), 0.080, 0.062, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_ORANGE, ec=ORANGE, lw=1.2))
    ax.text(0.285, 0.263, r"$G_\theta$", ha="center", va="center", fontsize=10.2, color=ORANGE)
    ax.add_patch(FancyBboxPatch((0.360, 0.232), 0.076, 0.062, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_GREEN, ec=GREEN, lw=1.2))
    ax.text(0.398, 0.263, r"$P_N$", ha="center", va="center", fontsize=10.2, color=GREEN)
    arrow(ax, (0.202, 0.260), (0.245, 0.260), ORANGE, 1.4, scale=12)
    arrow(ax, (0.325, 0.260), (0.360, 0.260), GREEN, 1.4, scale=12)
    path = MplPath([(0.450, 0.260), (0.475, 0.307), (0.486, 0.220), (0.469, 0.248), (0.450, 0.260)])
    ax.add_patch(PathPatch(path, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.2, alpha=0.9))
    ax.text(0.315, 0.155, "learned prior supplies\nmissing structure", ha="center", fontsize=8.6, color=ORANGE)
    ax.text(0.398, 0.332, "weakly observed /\nunobserved", ha="center", fontsize=8.3, color=GREEN)

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
    ax.text(0.700, 0.154, "final image remains\nauditable", ha="center", fontsize=8.5, color=GREEN)

    arrow(ax, (0.323, 0.745), (0.360, 0.745), BLUE, 1.0, scale=10, alpha=0.55)
    arrow(ax, (0.645, 0.745), (0.680, 0.745), BLUE, 1.0, scale=10, alpha=0.55)
    arrow(ax, (0.820, 0.575), (0.450, 0.450), ORANGE, 1.0, rad=0.18, scale=10, alpha=0.42)
    arrow(ax, (0.486, 0.260), (0.535, 0.260), GREEN, 1.0, scale=10, alpha=0.55)

    save_submission_figure(fig, "fig1_mechanism_submission")


def copy_submission_figures() -> None:
    ensure(FIG_DIR)
    ensure(PROJECT_FIG)
    for _short, (src_stem, dst_stem) in FIGS.items():
        for ext in ("pdf", "png", "svg"):
            src = PHASE29_FIG / f"{src_stem}.{ext}"
            if not src.exists():
                continue
            dst = FIG_DIR / f"{dst_stem}.{ext}"
            shutil.copy2(src, dst)
            shutil.copy2(dst, PROJECT_FIG / dst.name)
    regenerate_fig1_submission()
    for dst_stem, src_stem in SUPP_FIGS.items():
        for ext in ("pdf", "png", "svg"):
            src = PHASE29_FIG / f"{src_stem}.{ext}"
            if not src.exists():
                continue
            dst = FIG_DIR / f"{dst_stem}.{ext}"
            shutil.copy2(src, dst)
            shutil.copy2(dst, PROJECT_FIG / dst.name)


def run_pdftoppm(pdf: Path, out_base: Path) -> bool:
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", "600", "-singlefile", str(pdf), str(out_base)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return (out_base.with_suffix(".png")).exists()
    except Exception:
        return False


def png_to_tiff(png: Path, tiff: Path) -> bool:
    try:
        with Image.open(png) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            elif im.mode == "RGBA":
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            im.save(tiff, dpi=(600, 600), compression="tiff_lzw")
        return tiff.exists()
    except Exception:
        return False


def export_submission_figures() -> None:
    ensure(EXPORT_DIR)
    ensure(HIGH_RES_DIR)
    manifest: list[str] = [
        "# Figure Export Manifest",
        "",
        "All PNG and TIFF exports target 600 dpi. TIFF export is generated from the 600 dpi PNG when possible.",
        "",
        "| Figure | PDF | SVG | PNG 600 dpi | TIFF 600 dpi | Submission ready |",
        "|---|---|---|---|---|---|",
    ]

    for short_name, (_src_stem, submission_stem) in FIGS.items():
        pdf_src = FIG_DIR / f"{submission_stem}.pdf"
        svg_src = FIG_DIR / f"{submission_stem}.svg"
        pdf_out = EXPORT_DIR / f"{short_name}.pdf"
        svg_out = EXPORT_DIR / f"{short_name}.svg"
        png_out = EXPORT_DIR / f"{short_name}.png"
        tif_out = EXPORT_DIR / f"{short_name}.tiff"

        if pdf_src.exists():
            shutil.copy2(pdf_src, pdf_out)
        if svg_src.exists():
            shutil.copy2(svg_src, svg_out)
        png_ok = False
        if pdf_src.exists():
            png_ok = run_pdftoppm(pdf_src, EXPORT_DIR / short_name)
        if not png_ok:
            fallback = FIG_DIR / f"{submission_stem}.png"
            if fallback.exists():
                shutil.copy2(fallback, png_out)
                png_ok = True
        tif_ok = png_to_tiff(png_out, tif_out) if png_ok else False
        if short_name == "fig3_qualitative" and png_ok:
            shutil.copy2(png_out, HIGH_RES_DIR / "fig3_qualitative_submission_600dpi.png")

        manifest.append(
            "| {fig} | {pdf} | {svg} | {png} | {tiff} | {ready} |".format(
                fig=short_name,
                pdf="yes" if pdf_out.exists() else "no",
                svg="yes" if svg_out.exists() else "no",
                png="yes" if png_out.exists() else "no",
                tiff="yes" if tif_out.exists() else "no",
                ready="yes" if pdf_out.exists() and png_out.exists() else "check",
            )
        )

    (EXPORT_DIR / "FIGURE_EXPORT_MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def copy_source_package() -> None:
    if SOURCE_PACKAGE.exists():
        shutil.rmtree(SOURCE_PACKAGE)
    SOURCE_PACKAGE.mkdir(parents=True, exist_ok=True)
    for name in ("main.tex", "supplement.tex", "references.bib", "citation_audit.md", "submission_checklist.md"):
        src = PROJECT / name
        if src.exists():
            shutil.copy2(src, SOURCE_PACKAGE / name)
    for dirname in ("sections", "supplement", "tables", "figures"):
        src = PROJECT / dirname
        if src.exists():
            ignore = shutil.ignore_patterns("*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.out", "*.pdf_tex")
            shutil.copytree(src, SOURCE_PACKAGE / dirname, ignore=ignore)
    readme = """# Submission Source Package

This package contains the manuscript LaTeX source, supplement source, figures, tables, and bibliography needed to compile the submission manuscript.

It intentionally excludes checkpoints, raw data, internal logs, large training artifacts, and exploratory-analysis outputs. The trained-checkpoint manifests, exact random measurement operators, and detailed CSV tables are described in the manuscript availability statement.

Compile with latexmk if available:

```bash
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```
"""
    (SOURCE_PACKAGE / "README_SUBMISSION_SOURCE.md").write_text(readme, encoding="utf-8")
    zip_base = OUT / "source_package_submission"
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_base), "zip", SOURCE_PACKAGE)


def main() -> None:
    copy_submission_figures()
    export_submission_figures()
    copy_source_package()
    print(
        {
            "figures": str(FIG_DIR),
            "figures_for_submission": str(EXPORT_DIR),
            "manifest": str(EXPORT_DIR / "FIGURE_EXPORT_MANIFEST.md"),
            "source_package": str(SOURCE_PACKAGE),
            "source_zip": str(OUT / "source_package_submission.zip"),
        }
    )


if __name__ == "__main__":
    main()
