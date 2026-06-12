from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .phase17_common import (
    PHASE16,
    PHASE16_TABLES,
    PHASE17,
    TITLE,
    fnum,
    main_result_rows,
    read_csv,
    tex_escape,
    write_text,
)


OUT = PHASE17 / "pdf_preview"
FIG_DIR = OUT / "figures"


FIGURES = [
    (
        "attribution_delta_psnr.png",
        PHASE16 / "attribution" / "attribution_delta_psnr.png",
        "Model gain over backprojection.",
    ),
    (
        "bp_vs_model_psnr.png",
        PHASE16 / "attribution" / "bp_vs_model_psnr.png",
        "Final model PSNR across measurement families.",
    ),
    (
        "ablation_psnr.png",
        PHASE16 / "inference_ablation" / "real_inference_ablation_psnr.png",
        "Full-model inference PSNR for STL-10 methods.",
    ),
    (
        "ablation_relmeaserr.png",
        PHASE16 / "inference_ablation" / "real_inference_ablation_relmeaserr.png",
        "Measurement error when data consistency is ablated.",
    ),
    (
        "noise_sweep_psnr.png",
        PHASE16 / "noise_sweep" / "noise_sweep_psnr.png",
        "Finite noise sweep PSNR.",
    ),
    (
        "perturbation_psnr_drop.png",
        PHASE16 / "measurement_perturbation" / "perturbation_psnr_drop.png",
        "PSNR drop under measurement perturbations.",
    ),
    (
        "traditional_baseline_psnr.png",
        PHASE16 / "traditional_baselines" / "traditional_baseline_psnr.png",
        "Best traditional baseline PSNR.",
    ),
    (
        "statistics_psnr_histograms.png",
        PHASE16 / "statistics" / "psnr_histograms.png",
        "Per-sample PSNR distributions.",
    ),
    (
        "classwise_psnr.png",
        PHASE16 / "classwise" / "classwise_psnr.png",
        "STL-10 class-wise PSNR diagnostics.",
    ),
]


def copy_figures() -> list[tuple[str, str]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, src, caption in FIGURES:
        if src.exists():
            dst = FIG_DIR / name
            shutil.copy2(src, dst)
            copied.append((name, caption))
    return copied


def table(rows: list[dict[str, Any]], fields: list[str], caption: str, label: str, *, long: bool = False) -> str:
    env = "longtable" if long else "tabular"
    cols = "l" * len(fields)
    lines = []
    if not long:
        lines.extend([r"\begin{table}[htbp]", r"\centering", r"\scriptsize", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}"])
    else:
        lines.extend([r"\scriptsize", rf"\begin{{longtable}}{{{cols}}}", rf"\caption{{{tex_escape(caption)}}}\label{{{label}}}\\"])
    if not long:
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(rf"\begin{{{env}}}{{{cols}}}")
    lines.extend([r"\toprule", " & ".join(tex_escape(f) for f in fields) + r" \\", r"\midrule"])
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(field, "")) for field in fields) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(rf"\end{{{env}}}")
    if not long:
        lines.append(r"}")
        lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def compact_exact_rows() -> list[dict[str, str]]:
    out = []
    for r in read_csv(PHASE16_TABLES["exact_a_reeval"]):
        out.append(
            {
                "method": r.get("method_id", "").replace("_hq_noise001_colab", "").replace("_full_noise001_colab", ""),
                "orig_PSNR": fnum(r.get("original_psnr")),
                "reeval_PSNR": fnum(r.get("reeval_psnr")),
                "abs_diff": fnum(r.get("abs_diff_psnr"), 4),
                "status": r.get("status", ""),
            }
        )
    return out


def compact_ablation_rows() -> list[dict[str, str]]:
    keep = {"full_model", "no_dc_project", "stage1_only", "raw_weights", "ema_weights"}
    out = []
    for r in read_csv(PHASE16_TABLES["ablation"]):
        if r.get("ablation_mode") not in keep:
            continue
        out.append(
            {
                "method": r.get("method_id", "").replace("_hq_noise001_colab", "").replace("_full_noise001_colab", ""),
                "mode": r.get("ablation_mode", ""),
                "PSNR": fnum(r.get("psnr")),
                "SSIM": fnum(r.get("ssim")),
                "dPSNR": fnum(r.get("delta_vs_full_psnr")),
            }
        )
    return out


def compact_ci_rows() -> list[dict[str, str]]:
    out = []
    for r in read_csv(PHASE16_TABLES["statistics"]):
        out.append(
            {
                "method": r.get("method_id", "").replace("_hq_noise001_colab", "").replace("_full_noise001_colab", "").replace("_full_colab", ""),
                "PSNR_mean": fnum(r.get("mean_psnr")),
                "PSNR_CI": f"[{fnum(r.get('ci95_psnr_low'))}, {fnum(r.get('ci95_psnr_high'))}]",
                "SSIM_mean": fnum(r.get("mean_ssim")),
                "SSIM_CI": f"[{fnum(r.get('ci95_ssim_low'))}, {fnum(r.get('ci95_ssim_high'))}]",
            }
        )
    return out


def make_tex(copied_figures: list[tuple[str, str]]) -> str:
    main_rows = main_result_rows()
    figure_blocks = []
    for idx in range(0, len(copied_figures), 2):
        pair = copied_figures[idx : idx + 2]
        figure_blocks.append(r"\begin{figure}[htbp]")
        figure_blocks.append(r"\centering")
        for name, caption in pair:
            figure_blocks.append(r"\begin{minipage}{0.48\textwidth}")
            figure_blocks.append(r"\centering")
            figure_blocks.append(rf"\includegraphics[width=\linewidth]{{figures/{name}}}")
            figure_blocks.append(rf"\caption*{{\scriptsize {tex_escape(caption)}}}")
            figure_blocks.append(r"\end{minipage}\hfill")
        figure_blocks.append(r"\caption{Phase16 diagnostic figures inserted into the manuscript preview. Supplementary evaluations use imported no-leak checkpoints and do not train new models.}")
        figure_blocks.append(r"\end{figure}")
        figure_blocks.append("")
    figure_tex = "\n".join(figure_blocks)

    return rf"""\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb,booktabs,geometry,graphicx,longtable,caption,float}}
\geometry{{margin=0.85in}}
\title{{{tex_escape(TITLE)}}}
\author{{Author names to be added}}
\date{{PDF preview generated from Phase15/Phase16 evidence}}
\begin{{document}}
\maketitle

\begin{{abstract}}
Ghost imaging and single-pixel imaging recover images from bucket measurements. At low sampling rates, the inverse problem is underdetermined; unconstrained neural reconstruction can hallucinate measurement-inconsistent structure. This preview summarizes a measurement-consistent null-space neural reconstruction framework using strict no-leak results and Phase16 supplementary diagnostics. It is a layout preview, not a final submission manuscript.
\end{{abstract}}

\section{{Forward Model}}
The bucket model is
\[
y = A x + \epsilon.
\]
The data solution and null-space correction are
\[
x_{{\mathrm{{data}}}} = A^T(AA^T+\lambda I)^{{-1}}y,\quad
P_N(v)=v-A^T(AA^T+\lambda I)^{{-1}}Av.
\]
The learned reconstruction is
\[
\tilde{{x}} = x_{{\mathrm{{data}}}} + P_N(G_\theta(x_{{\mathrm{{data}}}},z)),\quad
\hat{{x}}=\tilde{{x}}-A^T(AA^T+\lambda I)^{{-1}}(A\tilde{{x}}-y).
\]

\section{{Primary Strict No-Leak Results}}
{table(main_rows, ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"], "Primary strict no-leak reconstruction results from the Phase15 registry.", "tab:main")}

\section{{Exact-A Rademacher Reproducibility}}
{table(compact_exact_rows(), ["method", "orig_PSNR", "reeval_PSNR", "abs_diff", "status"], "Rademacher exact-A re-evaluation with cache-rebuilt solver.", "tab:exacta")}

\section{{Inference Ablation Preview}}
{table(compact_ablation_rows(), ["method", "mode", "PSNR", "SSIM", "dPSNR"], "Selected inference-time ablations from Phase16. TV-PGD and other controls remain in the supplementary material.", "tab:ablation", long=True)}

\section{{Bootstrap Confidence Intervals}}
{table(compact_ci_rows(), ["method", "PSNR_mean", "PSNR_CI", "SSIM_mean", "SSIM_CI"], "Per-sample bootstrap confidence intervals.", "tab:ci")}

\section{{Inserted Phase16 Figures}}
{figure_tex}

\section{{Cautions for Writing}}
Do not claim strict SOTA, universal robustness, high-quality low-frequency Hadamard 5\% on STL-10, binary learned illumination as the main contribution, GAN as the final main mechanism, or exhaustively optimized TV-PGD. TV-PGD is a small-subset reviewer-defense baseline; noise robustness is finite-range diagnostic evidence.

\end{{document}}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = copy_figures()
    write_text(OUT / "phase17_pdf_preview.tex", make_tex(copied))
    print({"output": str(OUT / "phase17_pdf_preview.tex"), "figures": len(copied)})


if __name__ == "__main__":
    main()
