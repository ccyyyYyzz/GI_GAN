from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


DRIVE_ROOT = Path("E:/ns_mc_gan_gi")
OUTPUT_ROOT = DRIVE_ROOT / "outputs_phase27_paper_purification"
PROJECT_OUT = OUTPUT_ROOT / "latex_project_purified"
INTERNAL_NOTES = OUTPUT_ROOT / "internal_notes"


BASE_CANDIDATES = [
    DRIVE_ROOT / "outputs_phase24" / "latex_project",
    DRIVE_ROOT / "outputs_phase24" / "latex_project_v10",
    DRIVE_ROOT / "outputs_phase22_submission_v8" / "latex_project_v8",
    DRIVE_ROOT / "outputs_phase20_human_written_manuscript" / "latex_project",
]

EXPLORATION_DOCS = [
    DRIVE_ROOT / "outputs_phase25" / "ARCHITECTURE_LIMIT_PLAN.md",
    DRIVE_ROOT / "outputs_phase26" / "PHASE26_LIMIT_ARCHITECTURE_REPORT.md",
    DRIVE_ROOT / "outputs_phase26" / "PHASE26_GATE_DECISION.md",
]


def find_base_project() -> Path:
    for candidate in BASE_CANDIDATES:
        if (candidate / "main.tex").exists() and (candidate / "sections").is_dir():
            return candidate
    raise FileNotFoundError("No suitable main-paper LaTeX project found.")


def reset_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUTPUT_ROOT.resolve()
    if not str(resolved).lower().startswith(str(allowed).lower()):
        raise RuntimeError(f"Refusing to remove path outside Phase27 output: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_filtered(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {
            "main.aux",
            "main.bbl",
            "main.blg",
            "main.fdb_latexmk",
            "main.fls",
            "main.log",
            "main.out",
            "main.pdf",
            "supplement.aux",
            "supplement.fdb_latexmk",
            "supplement.fls",
            "supplement.log",
            "supplement.out",
            "supplement.pdf",
            "data_csv",
            "submission_checklist.md",
            "citations_to_verify.md",
        }:
            continue
        target = dst / item.name
        if item.is_dir():
            copy_tree_filtered(item, target)
        else:
            shutil.copy2(item, target)


ABSTRACT = r"""Ghost imaging and single-pixel imaging recover spatial information from structured illumination patterns and scalar bucket measurements, but low-sampling acquisition is severely underdetermined. We formulate this regime as measurement-consistent null-space neural reconstruction. The pipeline computes a physical data solution from the forward operator, inserts a learned residual through an approximate null-space component, and applies a final projection onto the measured affine set. This decomposition keeps the reconstruction auditable against the bucket measurements while allowing learned completion of missing image structure. Under a strict leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. MNIST and Fashion-MNIST 5\% experiments provide simple-domain sanity checks, reaching 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-operator audit, measurement-family attribution, inference ablation, finite-noise tests, measurement perturbation, CS-TV comparison, and confidence intervals support measurement-dependent reconstruction rather than an unconstrained image-generation claim."""


INTRODUCTION = r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known illumination patterns and scalar bucket detector readings. Instead of measuring an image directly with a dense sensor array, the system records projections of the unknown scene onto structured patterns. This acquisition model is attractive when detector arrays are expensive, unavailable, or difficult to deploy, but it shifts the burden from direct sensing to computational reconstruction.

The central difficulty is low sampling. If the unknown image is represented as \(x\in\mathbb{R}^n\) and the system collects \(m\) bucket measurements, the measurement model is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\) is determined by the illumination patterns. In the low-sampling regime, \(m\ll n\), and the inverse problem is underdetermined. Many candidate images can explain the same measurement vector. Direct physical inverses, such as backprojection, preserve a transparent link to the measurements but often leave severe missing structure.

Deep neural networks can improve reconstruction quality in this regime, but unconstrained networks introduce a different risk. A network may generate image details that look plausible while moving away from the measured bucket signal. For computational imaging, this is not merely a perceptual issue. The reconstructed image should remain physically tied to the forward model and to the measured data.

Existing deep-learning GI/SPI methods improve reconstruction quality, but many do not explicitly decompose the measured row-space component from the unmeasured null-space component, nor do they systematically separate physical initialization quality from neural refinement gain. This distinction matters in low-sampling regimes: two measurement families may reach similar final PSNR while relying on very different balances of physical information and learned completion.

This work addresses that gap by treating low-sampling ghost imaging as measurement-consistent null-space reconstruction. The method computes a physical data solution, inserts a learned residual through an approximate null-space component, and then projects the result back to the measured affine set. This gives an auditable reconstruction path through \(x_{\rm data}\), \(P_N\), and \(\Pi_y\), rather than an unconstrained measurement-to-image mapping.

The main contributions are:
\begin{itemize}
\item a measurement-consistent null-space reconstruction wrapper for low-sampling ghost imaging;
\item strict leakage-free STL-10 evidence at 5\% and 10\% sampling for Rademacher and scrambled Hadamard measurements;
\item exact-A Rademacher re-evaluation with safe solver-cache rebuilding;
\item validation analyses including measurement-family attribution, inference ablation, finite-noise sweep, measurement perturbation, CS-TV comparison, and confidence intervals.
\end{itemize}"""


VALIDATION = r"""\section{Validation and Ablation}
\subsection{Exact-A reproducibility}
Rademacher measurements require exact-operator evaluation. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence.

\subsection{Inference-time ablation}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_v8.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
We refer to this baseline as CS-TV (PGD solver). It is a lightweight small-subset traditional baseline, not an exhaustively optimized compressed-sensing benchmark. Under the tested settings, the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_v8.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Shuffle/Wrong-y measurement perturbations, comparison against CS-TV, and bootstrap confidence intervals. These diagnostics support finite-noise stability and measurement dependence within the tested conditions.}
\label{fig:validation_summary}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. This is a measurement-design diagnostic and should not be generalized to all measurement families.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution."""


DISCUSSION = r"""\section{Discussion}
The first lesson is that low-sampling ghost imaging is measurement-constrained completion. A good reconstruction must not only look plausible but also remain tied to the measured bucket signal. The proposed pipeline achieves this by combining a data solution, null-space residual completion, and a measurement-consistency projection. The main novelty is a physics-constrained reconstruction decomposition that makes the role of measured data, missing-structure completion, and measurement projection explicit.

The second lesson is that measurement families define different initialization and neural-gain regimes. Rademacher measurements produce weak backprojections but high final quality after neural refinement. Scrambled Hadamard measurements provide stronger physical initialization and similar final quality. This suggests that physical initialization quality and learnability of the neural inverse are distinct properties.

The third lesson is that measurement audit matters more than adversarial realism. Although adversarial ideas were considered during development, the final high-quality results are driven by measurement-consistent neural reconstruction and fidelity-oriented losses. The contribution is the auditable reconstruction formulation, not adversarial generation.

Future work may explore alternative neural priors within the same measurement-consistent wrapper."""


LIMITATIONS = r"""\section{Limitations}
This study does not include a hardware optical experiment. It does not claim a strict benchmark ranking because a broad external benchmark under matched protocols is not included. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Class-wise evaluation is diagnostic rather than a claim of uniform category performance. Exact-A handling is essential for random measurements, and results should be interpreted with that audit path in place. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing."""


EXPERIMENTAL_PROTOCOL = r"""\section{Experimental Protocol}
Primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are used as simple-domain sanity checks at 5\% sampling. We use a strict leakage-free protocol, i.e., checkpoint selection and final testing are separated. For Rademacher measurements, the exact exported operators are reloaded and the solver cache is rebuilt before evaluation.

We summarize quality with predefined operational thresholds: PSNR \(\ge 20\) and SSIM \(\ge 0.60\) for STL-10 at 5\%, PSNR \(\ge 22\) and SSIM \(\ge 0.65\) for STL-10 at 10\%, and PSNR \(\ge 25\) and SSIM \(\ge 0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are operational criteria used to summarize reconstruction quality in this study, not fundamental bounds. Supplementary analyses use existing trained checkpoints and do not introduce additional training."""


SUPPLEMENT = r"""\section{Supplementary Material}
Detailed CSV files and audit manifests are provided as an accompanying data package. The tables below are curated summaries intended for a compact submission supplement.

\subsection{S1 Exact-A reproducibility}
\input{tables/tableS1_exact_a.tex}

\subsection{S2 Measurement-family attribution details}
The main text separates physical initialization from learned refinement. The accompanying data package provides the full attribution CSV used to produce the curated table and regime summary.

\subsection{S3 Inference ablation and measurement error}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS1_relmeaserr_ablation_v8.pdf}
\caption{Supplementary measurement-error view of the no-DC projection ablation. Removing the projection increases measurement inconsistency.}
\label{fig:supp_relmeaserr_ablation}
\end{figure*}

\subsection{S4 Noise sweep}
\input{tables/tableS2_noise_sweep.tex}

\subsection{S5 CS-TV baseline}
\input{tables/tableS3_cstv_baseline.tex}

\subsection{S6 DC-row control}
\input{tables/tableS4_dc_row_control.tex}

\subsection{S7 Statistics, confidence intervals, and class-wise diagnostics}
\input{tables/tableS5_statistics_ci.tex}
\input{tables/tableS6_classwise.tex}

\subsection{S8 Runtime}
\input{tables/tableS7_runtime.tex}

\FloatBarrier
\section{Data and Code Availability}
The code, trained checkpoint manifests, exported Rademacher measurement operators, and detailed supplementary CSV tables will be made available upon publication or reasonable request. Rademacher reproduction requires the exported exact measurement operator and the cache-rebuilt evaluation path."""


CITATION_AUDIT = """# Citation Audit

This purified draft keeps the existing bibliography from the submission baseline.

Manual checks still required:
- Confirm every bibliographic record against the target journal style.
- Confirm author order, page numbers, issue numbers, and DOI availability.
- Confirm whether arXiv-only entries are acceptable for the target venue.
"""


INTERNAL_NOTE = """# Phase25/26 Exploration Not For Paper

The Phase25 and Phase26 exploration documents were inspected only to identify material that must remain outside the submission manuscript and supplement.

Excluded from the purified paper:
- PCA oracle / PCA prior / PCA subspace diagnostics.
- Sampling-ratio scaling fits and extrapolations.
- Architecture pilot, network replacement, NAFNet, and unrolled-ISTA diagnostics.
- Gate-decision or recommend-full planning language.

These items are internal planning diagnostics. They are not part of the current submission narrative, figures, tables, or supplement.
"""


ABSTRACT_MD = """# Purified Abstract

""" + ABSTRACT.replace("\\%", "%") + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def inspect_exploration_docs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in EXPLORATION_DOCS:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            records.append(
                {
                    "path": str(path),
                    "exists": True,
                    "chars": len(text),
                    "used_for": "exclusion-check-only",
                }
            )
        else:
            records.append({"path": str(path), "exists": False})
    return records


def main() -> None:
    base_project = find_base_project()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    INTERNAL_NOTES.mkdir(parents=True, exist_ok=True)
    exploration_records = inspect_exploration_docs()

    reset_dir(PROJECT_OUT)
    copy_tree_filtered(base_project, PROJECT_OUT)

    required_dirs = [
        PROJECT_OUT / "sections",
        PROJECT_OUT / "supplement",
        PROJECT_OUT / "figures",
        PROJECT_OUT / "tables",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    write_text(PROJECT_OUT / "sections" / "abstract.tex", ABSTRACT)
    write_text(PROJECT_OUT / "sections" / "introduction.tex", INTRODUCTION)
    write_text(PROJECT_OUT / "sections" / "experimental_protocol.tex", EXPERIMENTAL_PROTOCOL)
    write_text(PROJECT_OUT / "sections" / "validation_ablation.tex", VALIDATION)
    write_text(PROJECT_OUT / "sections" / "discussion.tex", DISCUSSION)
    write_text(PROJECT_OUT / "sections" / "limitations.tex", LIMITATIONS)
    write_text(PROJECT_OUT / "supplement" / "supplement.tex", SUPPLEMENT)
    write_text(PROJECT_OUT / "citation_audit.md", CITATION_AUDIT)
    write_text(OUTPUT_ROOT / "abstract_purified.md", ABSTRACT_MD)
    write_text(INTERNAL_NOTES / "PHASE25_26_EXPLORATION_NOT_FOR_PAPER.md", INTERNAL_NOTE)

    manifest = {
        "phase": 27,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_project": str(base_project),
        "output_project": str(PROJECT_OUT),
        "exploration_docs_inspected": exploration_records,
        "submission_policy": "Phase25/26 exploration excluded from main manuscript and supplement.",
    }
    write_text(OUTPUT_ROOT / "phase27_purification_manifest.json", json.dumps(manifest, indent=2))

    print(
        {
            "base_project": str(base_project),
            "output_project": str(PROJECT_OUT),
            "manifest": str(OUTPUT_ROOT / "phase27_purification_manifest.json"),
            "abstract": str(OUTPUT_ROOT / "abstract_purified.md"),
        }
    )


if __name__ == "__main__":
    main()
