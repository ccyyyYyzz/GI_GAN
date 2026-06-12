from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .phase20_common import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT_GRAY,
    METHOD_LABEL,
    ORANGE,
    PURPLE,
    RED,
    STL_METHODS,
    as_float,
    ensure_dir,
    setup_matplotlib,
    table,
    write_text,
)


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase28_paragraph_polish"
FIG_DIR = OUT / "figures"
PROJECT_FIG_DIR = OUT / "latex_project_v28" / "figures"
PHASE23_FIG_DIR = ROOT / "outputs_phase23_top_journal_rewrite" / "figures"
PHASE27_FIG_DIR = ROOT / "outputs_phase27_paper_purification" / "latex_project_purified" / "figures"


def save_figure(fig: plt.Figure, stem: str, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(PROJECT_FIG_DIR)
    for ext in exts:
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        shutil.copy2(path, PROJECT_FIG_DIR / path.name)
    plt.close(fig)


def copy_figure_set(src_stem: str, dst_stem: str, src_dir: Path = PHASE23_FIG_DIR) -> list[str]:
    ensure_dir(FIG_DIR)
    ensure_dir(PROJECT_FIG_DIR)
    copied: list[str] = []
    missing: list[str] = []
    for ext in ("pdf", "png", "svg"):
        src = src_dir / f"{src_stem}.{ext}"
        if not src.exists():
            missing.append(str(src))
            continue
        dst = FIG_DIR / f"{dst_stem}.{ext}"
        shutil.copy2(src, dst)
        shutil.copy2(dst, PROJECT_FIG_DIR / dst.name)
        copied.append(str(dst))
    if missing:
        write_text(
            OUT / f"{dst_stem}_copy_warning.md",
            "# Figure Copy Warning\n\nMissing source files:\n\n" + "\n".join(f"- {path}" for path in missing),
        )
    return copied


def ensure_context_figures() -> None:
    # The manuscript still uses existing v8 figures for concept, primary metrics, and robustness.
    # Copy them into the v28 project if the clean project was just regenerated.
    ensure_dir(PROJECT_FIG_DIR)
    if not PHASE27_FIG_DIR.exists():
        return
    for path in PHASE27_FIG_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
            target = PROJECT_FIG_DIR / path.name
            if not target.exists():
                shutil.copy2(path, target)


def make_inference_ablation() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [
        ("full_model", "Full"),
        ("no_dc_project", "-MC"),
        ("no_null_project", "-Null"),
        ("stage1_only", "Stage1"),
        ("raw_weights", "Raw"),
        ("ema_weights", "EMA"),
    ]
    colors = [BLUE, RED, "#A7AEB8", ORANGE, PURPLE, GREEN]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharey=True)
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(ab[(mid, mode)]["psnr"]) for mode, _ in modes]
        x = np.arange(len(modes))
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.35, f"{yi:.1f}", ha="center", fontsize=6.8)
        ax.set_ylim(0, 26)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in modes])
        if idx % 2 == 0:
            ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.06, f"({chr(ord('a') + idx)})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "Removing the measurement-consistency projection is the clearest failure; -Null is reported as a limited metric change for these trained checkpoints.",
        ha="center",
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation_v28")


def make_relmeaserr_supplement() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-MC projection increases measurement inconsistency")
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.20)
    save_figure(fig, "figS1_relmeaserr_ablation_v28", exts=("pdf", "png", "svg"))


def copy_reselected_qualitative() -> None:
    copied = copy_figure_set("fig3_qualitative_reconstruction_v9", "fig3_qualitative_reconstruction_v28")
    warning_src = ROOT / "outputs_phase23_top_journal_rewrite" / "qualitative_selection_warning.md"
    if warning_src.exists():
        shutil.copy2(warning_src, OUT / "qualitative_selection_warning_v28.md")
    if not copied:
        # Last-resort fallback keeps the manuscript compilable and records the manual issue.
        copy_figure_set("fig3_qualitative_reconstruction_v8", "fig3_qualitative_reconstruction_v28", src_dir=PHASE27_FIG_DIR)
        write_text(
            OUT / "qualitative_selection_warning_v28.md",
            "# Qualitative Selection Warning\n\nCould not locate the reselected qualitative figure set, so the previous qualitative figure was retained.",
        )


def copy_fixed_attribution() -> None:
    copied = copy_figure_set("fig4_regime_map_v9", "fig4_measurement_attribution_v28")
    if not copied:
        copy_figure_set("fig4_measurement_attribution_v8", "fig4_measurement_attribution_v28", src_dir=PHASE27_FIG_DIR)
        write_text(
            OUT / "fig4_measurement_attribution_v28_warning.md",
            "# Figure 4 Warning\n\nCould not locate the fixed regime-map figure, so the previous attribution figure was retained.",
        )


def main() -> None:
    ensure_dir(OUT)
    ensure_dir(FIG_DIR)
    ensure_context_figures()
    copy_reselected_qualitative()
    copy_fixed_attribution()
    make_inference_ablation()
    make_relmeaserr_supplement()
    print(
        {
            "figure_dir": str(FIG_DIR),
            "latex_figure_dir": str(PROJECT_FIG_DIR),
            "figure3": str(FIG_DIR / "fig3_qualitative_reconstruction_v28.pdf"),
            "figure4": str(FIG_DIR / "fig4_measurement_attribution_v28.pdf"),
        }
    )


if __name__ == "__main__":
    main()
