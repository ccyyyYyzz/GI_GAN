from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
SOURCE_PROJECT = ROOT / "outputs_phase28_paragraph_polish" / "latex_project_v28"
OUT = ROOT / "outputs_phase29_final_submission_polish"
PROJECT = OUT / "latex_project_final"


COMPILED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".pdf",
    ".run.xml",
    ".synctex.gz",
    ".toc",
}


DATA_AVAILABILITY = r"""
\section{Data and Code Availability}
The code, trained-checkpoint manifests, exported Rademacher measurement operators, and detailed supplementary CSV tables will be made available upon publication or reasonable request. Reproducing Rademacher results requires the exported exact measurement operator and the cache-rebuilt evaluation path described in the manuscript.
"""


SUPPLEMENT = r"""
\section{Supplementary Material}
The supplement provides compact curated summaries for reproducibility and diagnostic interpretation. Complete CSV tables are described in the data and code availability statement.

\subsection{S1 Exact-operator reproducibility}
\input{tables/tableS1_exact_a.tex}

\subsection{S2 RelMeasErr ablation}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS1_relmeaserr_ablation_final.pdf}
\caption{Supplementary measurement-error view of the no measurement-consistency projection ablation. Removing the projection increases measurement inconsistency.}
\label{fig:supp_relmeaserr_ablation}
\end{figure*}

\subsection{S3 Finite-noise sweep summary}
\input{tables/tableS2_noise_sweep.tex}

\subsection{S4 CS-TV compressed-sensing baseline}
\input{tables/tableS3_cstv_baseline.tex}

\subsection{S5 Low-frequency Hadamard DC-row control}
\input{tables/tableS4_dc_row_control.tex}

\subsection{S6 Bootstrap CI and class-wise diagnostic}
\input{tables/tableS5_statistics_ci.tex}
\input{tables/tableS6_classwise.tex}

\subsection{S7 Runtime and complexity}
\input{tables/tableS7_runtime.tex}
"""


METHOD_FINAL = r"""
\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{Physical data solution}
We first compute a regularized data solution
\begin{equation}
x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This solution is a physical initialization, not a learned hallucination. It is determined by the forward operator and the bucket measurements. \(x_{\rm data}\) should be interpreted as a measured-component representative, not as a visually complete reconstruction. Its quality depends on the measurement family.

\subsection{Approximate null-space residual}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space projection. The neural reconstructor predicts \(r_\theta=G_\theta(x_{\rm data},z)\), and the intermediate reconstruction is \(\tilde{x}=x_{\rm data}+P_N(r_\theta)\). This step encourages the network to complete information not directly determined by the measurements. The projection removes the component of the proposed residual that would be visible to the measurement operator. Thus the network is not asked to overwrite the measured component; it proposes structure in directions that are weakly observed or unobserved by \(A\).

\subsection{Measurement-consistency projection}
To restore agreement with the bucket measurements, we apply
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
Although \(P_N\) restricts the neural residual, it is only an approximate null-space projection when \(\lambda>0\). In addition, the data solution, the refiner, numerical factorization, and intensity clipping can all introduce measurement inconsistency. The final projection \(\Pi_y\) therefore acts as an audit step on the complete image rather than only on the neural residual. The role of the neural network is therefore restricted by the measurement operator: it refines the missing component but is followed by an explicit projection back to the measured affine set. \Cref{fig:mechanism} visualizes the same logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism_final.pdf}
\caption{\textbf{Mechanism.} Low-sampling GI is treated as measurement-constrained completion: measured row-space information is preserved, missing structure is completed through a neural residual, and the output is projected back to the measurement-consistent set.}
\label{fig:mechanism}
\end{figure*}

\subsection{Two-stage refiner}
The implemented high-quality reconstructor uses a two-stage structure. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner then predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
The first stage enforces the main measurement-consistent null-space structure, while the refiner improves image fidelity. The final projection is retained after refinement so that the refiner cannot permanently move the output away from the measured signal. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

\subsection{Exact operator handling}
Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore requires reloading the exported exact operator. After replacing \(A\), all cached quantities derived from \(A\), including \(K=AA^T+\lambda I\) and its Cholesky factorization, must be rebuilt. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""


RESULTS_FINAL = r"""
\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
We first ask whether natural images can be reconstructed at the 5\% sampling level under a leakage-free protocol. \Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the predefined operational STL-10 5\% high-quality threshold. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_final.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines denote predefined operational thresholds used only to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The visual comparison is meant to show what is recovered beyond the physical initialization: Rademacher backprojections are noise-like, scrambled Hadamard backprojections contain more structure, and the neural reconstruction restores object-level content. Images are enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the leakage-free evaluation metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_final.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples were reselected for clearer object structure. Error maps use a shared high-percentile scale. The examples are qualitative visualizations; all quantitative conclusions are based on Table 1.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.

\subsection{Measurement-family attribution}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned refinement. Final PSNR alone is insufficient to explain the role of the measurement family. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_final.pdf}
\caption{\textbf{Measurement attribution.} Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Low-frequency Hadamard points are diagnostic rather than primary STL-10 high-quality claims.}
\label{fig:measurement_attribution}
\end{figure*}
"""


VALIDATION_FINAL = r"""
\section{Validation and Ablation}
\subsection{Exact-A reproducibility}
Rademacher measurements require exact-operator evaluation. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence. This audit is important because random sensing results cannot be reproduced by regenerating a nominally identical random matrix; the exact exported operator must be used.

\subsection{Inference-time ablation}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This no measurement-consistency projection condition shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_final.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. \(-\mathrm{MC}\) removes the final measurement-consistency projection. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This is a negative-control test: the model should fail when the measurement vector is corrupted. The result indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by PGD \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda \operatorname{TV}(x).
\end{equation}
We refer to this baseline as CS-TV. This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. It is a lightweight small-subset traditional baseline, and under the tested settings the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_final.pdf}
\caption{\textbf{Robustness and baselines.} These diagnostics support finite-noise stability and measurement dependence within the tested conditions; they do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. The DC-row result explains why low-frequency Hadamard backprojection behaves differently from Rademacher and scrambled Hadamard. It is a diagnostic of one measurement family, not a general explanation of all reconstructions.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if Path(name).suffix in COMPILED_SUFFIXES:
            ignored.add(name)
    return ignored


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(SOURCE_PROJECT, PROJECT, ignore=_ignore)


def update_main() -> None:
    main = PROJECT / "main.tex"
    text = main.read_text(encoding="utf-8")
    if r"\input{sections/data_availability.tex}" not in text:
        text = text.replace(
            r"\input{sections/limitations.tex}\FloatBarrier" + "\n" + r"\input{sections/conclusion.tex}\FloatBarrier",
            r"\input{sections/limitations.tex}\FloatBarrier" + "\n" + r"\input{sections/data_availability.tex}\FloatBarrier" + "\n" + r"\input{sections/conclusion.tex}\FloatBarrier",
        )
    main.write_text(text, encoding="utf-8")


def update_sources() -> None:
    write_text(PROJECT / "sections" / "method.tex", METHOD_FINAL)
    write_text(PROJECT / "sections" / "results.tex", RESULTS_FINAL)
    write_text(PROJECT / "sections" / "validation_ablation.tex", VALIDATION_FINAL)
    write_text(PROJECT / "sections" / "data_availability.tex", DATA_AVAILABILITY)
    write_text(PROJECT / "supplement" / "supplement.tex", SUPPLEMENT)


def citation_keys_from_sources() -> set[str]:
    keys: set[str] = set()
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT / "sections").glob("*.tex"))
    source_text += "\n" + (PROJECT / "main.tex").read_text(encoding="utf-8")
    for match in re.finditer(r"\\cite(?:[tp])?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}", source_text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def parse_bib_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = match.group(1)
        start = match.start()
        next_match = re.search(r"\n@\w+\s*\{", text[match.end() :])
        end = match.end() + next_match.start() + 1 if next_match else len(text)
        entries[key] = text[start:end]
    return entries


def field(entry: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*[\{{\"]([^}}\"]+)", entry, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def write_citation_audit() -> None:
    bib_path = PROJECT / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8")
    entries = parse_bib_entries(bib_text)
    all_bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)
    cited = citation_keys_from_sources()
    missing = sorted(cited - set(entries))
    uncited = sorted(set(entries) - cited)
    duplicate_keys = sorted(key for key, count in Counter(all_bib_keys).items() if count > 1)
    title_counter = Counter(re.sub(r"[^a-z0-9]+", " ", field(entry, "title").lower()).strip() for entry in entries.values())
    duplicate_titles = sorted(title for title, count in title_counter.items() if title and count > 1)

    incomplete: list[str] = []
    for key, entry in entries.items():
        title = field(entry, "title")
        year = field(entry, "year")
        venue = field(entry, "journal") or field(entry, "booktitle") or field(entry, "publisher") or field(entry, "howpublished")
        if not title or not year or not venue:
            incomplete.append(key)

    malformed_cites: list[str] = []
    source_paths = list((PROJECT / "sections").glob("*.tex")) + [PROJECT / "main.tex"]
    for path in source_paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "\\cite" in line and not re.search(r"\\cite(?:[tp])?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^{}]+\}", line):
                malformed_cites.append(f"{path.name}:{line_no}")

    todo_hits = bool(re.search(r"TODO|Reference Placeholders", bib_text, flags=re.IGNORECASE))
    lines = [
        "# Citation Audit",
        "",
        f"- Cited keys: {len(cited)}",
        f"- Bibliography entries: {len(entries)}",
        f"- Missing cited keys: {', '.join(missing) if missing else 'none'}",
        f"- Uncited bibliography entries: {', '.join(uncited) if uncited else 'none'}",
        f"- Duplicate keys: {', '.join(duplicate_keys) if duplicate_keys else 'none'}",
        f"- Duplicate obvious titles: {', '.join(duplicate_titles) if duplicate_titles else 'none'}",
        f"- Entries missing obvious title/year/venue fields: {', '.join(sorted(incomplete)) if incomplete else 'none'}",
        f"- Malformed citation commands: {', '.join(malformed_cites) if malformed_cites else 'none'}",
        f"- TODO or reference-placeholder text in references.bib: {'yes' if todo_hits else 'no'}",
        "",
        "Missing DOI fields are acceptable for this audit and were not treated as failures.",
    ]
    audit = "\n".join(lines)
    write_text(PROJECT / "citation_audit.md", audit)
    write_text(OUT / "citation_audit.md", audit)


def write_submission_checklist() -> None:
    checklist = """# Submission Checklist

- Final LaTeX project built from the previous polished manuscript.
- Main result numbers preserved.
- No new training or experiments were introduced.
- Figure 1 mechanism graphic regenerated.
- Figure 2 primary metrics graphic regenerated without in-panel threshold labels.
- Figure 3, Figure 4, Figure 5, and Figure 6 exported with final filenames.
- Table/Figure terminology uses measurement-consistency wording and `-MC`.
- Data and code availability section added.
- Supplement compressed to exact-operator reproducibility, RelMeasErr ablation, noise, CS-TV, DC-row, bootstrap/class-wise, and runtime summaries.
- Citation audit generated.

Manual checks still required: author information, affiliation, target journal template, anonymity requirements, final public repository or data-hosting details, and reference factual accuracy.
"""
    write_text(PROJECT / "submission_checklist.md", checklist)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("main_final_polished.pdf", "supplement_final_polished.pdf", "main_final_polished.txt", "supplement_final_polished.txt"):
        path = OUT / stale
        if path.exists():
            path.unlink()
    copy_project()
    update_main()
    update_sources()
    write_citation_audit()
    write_submission_checklist()
    print(
        {
            "output_dir": str(OUT),
            "latex_project": str(PROJECT),
            "citation_audit": str(PROJECT / "citation_audit.md"),
            "submission_checklist": str(PROJECT / "submission_checklist.md"),
        }
    )


if __name__ == "__main__":
    main()
