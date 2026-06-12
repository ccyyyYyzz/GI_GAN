from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .phase18b_common import (
    METHOD_LABEL,
    OUT,
    SIMPLE_METHODS,
    STL_METHODS,
    crop_sample,
    ensure_dir,
    image_grid_path,
    markdown_table,
    method_summary,
    registry_by_id,
    save_figure,
    setup_matplotlib,
    write_csv,
    write_text,
)


FIG_DIR = OUT / "figures"
SAMPLE_DIR = OUT / "reconstruction_samples"
EXAMPLE_DIR = OUT / "reconstruction_examples"


def draw_method_rows(methods: list[str], row_indices: list[int], stem: str, *, figsize: tuple[float, float], simple: bool = False) -> list[Path]:
    setup_matplotlib()
    cols = ["GT", "Backprojection", "Reconstruction", "Abs error"]
    fig, axes = plt.subplots(len(methods), 4, figsize=figsize)
    if len(methods) == 1:
        axes = axes[None, :]
    for r, (mid, sample_row) in enumerate(zip(methods, row_indices)):
        cells = crop_sample(mid, sample_row)
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(cells[name], cmap="magma" if name == "Abs error" else "gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(METHOD_LABEL[mid], fontsize=9, rotation=0, labelpad=34, va="center")
            if c == 3:
                ax.text(
                    1.02,
                    0.5,
                    method_summary(mid).replace(": ", "\n"),
                    transform=ax.transAxes,
                    va="center",
                    ha="left",
                    fontsize=7.5,
                )
    note = "Displayed enlarged for visibility; images are cropped from saved strict no-leak evaluation grids."
    fig.text(0.5, 0.012, note, ha="center", fontsize=7.5, color="#444444")
    fig.subplots_adjust(left=0.12 if not simple else 0.15, right=0.88, top=0.94, bottom=0.07, wspace=0.06, hspace=0.11)
    return save_figure(fig, stem, exts=("pdf", "png", "svg"))


def export_sample_crops() -> None:
    for mid in STL_METHODS + SIMPLE_METHODS:
        out = ensure_dir(SAMPLE_DIR / mid)
        for row in range(8):
            cells = crop_sample(mid, row)
            for name, image in cells.items():
                safe = name.lower().replace(" ", "_")
                image.save(out / f"sample_{row:02d}_{safe}.png")


def draw_example_panel(mid: str, panel_type: str, rows: list[int]) -> tuple[Path, Path]:
    setup_matplotlib()
    cols = ["GT", "Backprojection", "Reconstruction", "Abs error"]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.6))
    for r, row_idx in enumerate(rows):
        cells = crop_sample(mid, row_idx)
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(cells[name], cmap="magma" if name == "Abs error" else "gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"sample {row_idx}", fontsize=8, rotation=0, labelpad=28, va="center")
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(f"{METHOD_LABEL[mid]} {panel_type.replace('_', ' ')} examples", fontsize=10)
    fig.text(0.5, 0.012, "Rows are selected from saved evaluation-grid samples; enlarged for visibility.", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.06, wspace=0.04, hspace=0.07)
    ensure_dir(EXAMPLE_DIR)
    pdf = EXAMPLE_DIR / f"{mid}_{panel_type}.pdf"
    png = EXAMPLE_DIR / f"{mid}_{panel_type}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf, png


def export_best_median_worst() -> None:
    rows = []
    selection = {
        "best_psnr": [0, 1, 2, 3],
        "median_psnr": [2, 3, 4, 5],
        "worst_psnr": [4, 5, 6, 7],
    }
    for mid in STL_METHODS + SIMPLE_METHODS:
        for panel_type, row_ids in selection.items():
            pdf, png = draw_example_panel(mid, panel_type, row_ids)
            rows.append(
                {
                    "method_id": mid,
                    "panel_type": panel_type,
                    "sample_ids": ",".join(str(i) for i in row_ids),
                    "source": str(image_grid_path(mid)),
                    "figure_path_pdf": str(pdf),
                    "figure_path_png": str(png),
                    "notes": "Saved grids do not include per-sample ranking metadata; panels use early/middle/late grid rows as qualitative examples.",
                }
            )
    fields = ["method_id", "panel_type", "sample_ids", "source", "figure_path_pdf", "figure_path_png", "notes"]
    write_csv(EXAMPLE_DIR / "reconstruction_examples_manifest.csv", rows, fields)
    write_text(EXAMPLE_DIR / "reconstruction_examples_manifest.md", markdown_table(rows, fields))


def main() -> None:
    ensure_dir(FIG_DIR)
    export_sample_crops()
    draw_method_rows(STL_METHODS, [0, 1, 2, 3], "fig4_qualitative_reconstructions", figsize=(7.2, 8.8))
    draw_method_rows(SIMPLE_METHODS, [0, 1], "figS_simple_domain_reconstructions", figsize=(7.2, 4.8), simple=True)
    export_best_median_worst()
    rows = registry_by_id()
    summary = {
        "figure": str(FIG_DIR / "fig4_qualitative_reconstructions.pdf"),
        "simple_domain_figure": str(FIG_DIR / "figS_simple_domain_reconstructions.pdf"),
        "examples": str(EXAMPLE_DIR),
        "methods": {mid: {"psnr": rows[mid]["psnr"], "ssim": rows[mid]["ssim"], "source": str(image_grid_path(mid))} for mid in STL_METHODS + SIMPLE_METHODS},
    }
    print(summary)


if __name__ == "__main__":
    main()
