"""Assemble the two English mentor-contact proposals from the verified workflow output.

Reads the workflow JSON result and writes two standalone, pdflatex-compilable .tex files
with a shared English preamble.
"""
import json
import os

OUT_JSON = r"D:\tmp\claude\E--ns-mc-gan-gi-code-fcc-phase1\649423c0-024c-43d5-9296-469610640f92\tasks\wof8tkr6b.output"
DEST = r"E:\ns_mc_gan_gi_code_fcc_phase1"

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1.6cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage[font=small,labelfont=bf,skip=3pt]{caption}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue]{hyperref}

\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{5pt}{2pt}
\titlespacing*{\paragraph}{0pt}{3pt}{0.7em}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.22em}

\title{\vspace{-1.4cm}\large\textbf{%(title)s}}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-2.2cm}

%(body)s

\end{document}
"""

# Mechanism figure injected between "Background and Motivation" and
# "Proposed Approach" for each proposal.
FIGS = {
    "proposal_scheme1_baseline.tex": r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.82\textwidth]{mfig_scheme1_mechanism.pdf}
\caption{Working principle of the binary-amplitude Fresnel zone plate.
\textbf{(a)} Focusing principle: a collimated wave is transmitted only by the
open half-wave zones, whose contributions arrive in phase at a common focus a
distance $f$ away; the finest outer zone sets the manufacturing limit
$\Delta r_N\approx\lambda f/(2R)\approx20~\mu$m. \textbf{(b)} Top-view zone
structure generated from the half-wave index $q_f(\rho)$, with zone radii
$r_n=\sqrt{n\lambda f}$ and outer radius $R=2$~mm.}
\label{fig:s1-mechanism}
\end{figure}
""",
    "proposal_scheme2B_radial_binary.tex": r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.78\textwidth]{mfig_scheme2b_mechanism.pdf}
\caption{Radial binary optimization as a target-encoding mechanism.
\textbf{(a)} A target axial intensity (here a dual focus at $135$ and
$165$~mm) and the response of an optimized pattern. \textbf{(b)} The aperture
is encoded as a vector of binary ring states $a_m\in\{0,1\}$, which
\textbf{(c)} generates a concentric binary mask. \textbf{(d)} The ring states
are optimized against the target by binary local search / simulated annealing,
minimizing the objective $J(a)$.}
\label{fig:s2b-mechanism}
\end{figure}
""",
}

FILES = [
    "proposal_scheme1_baseline.tex",
    "proposal_scheme2B_radial_binary.tex",
]

with open(OUT_JSON, "r", encoding="utf-8") as fh:
    data = json.load(fh)

results = data["result"]
assert len(results) == 2, f"expected 2 proposals, got {len(results)}"

for fname, res in zip(FILES, results):
    title = res["final_title"]
    body = res["final_latex_body"]
    # Inject the mechanism figure just before "Proposed Approach".
    fig = FIGS.get(fname, "")
    anchor = r"\section{Proposed Approach}"
    if fig and anchor in body:
        body = body.replace(anchor, fig + "\n" + anchor, 1)
    elif fig:
        raise SystemExit(f"anchor not found in {fname}; cannot place figure")
    # Compact timeline tables (keep them on the second page).
    for tab in (r"\begin{tabular}{@{}cl@{}}", r"\begin{tabular}{ll}"):
        body = body.replace(tab, r"\small" + tab, 1)
    tex = PREAMBLE % {"title": title, "body": body}
    path = os.path.join(DEST, fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"wrote {path}  ({len(tex)} chars, page_est={res['page_estimate']})")
    print(f"   title: {title}")
