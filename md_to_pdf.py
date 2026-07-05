"""Render the paper draft markdown to PDF via xelatex (no pandoc needed).

Targeted markdown->LaTeX for PAPER_DRAFT.md: headings, bold/italic, inline code,
lists, one pipe table, $...$ / $$...$$ math (preserved), a small unicode map.
Appends the figures (PDF) at the end so the review PDF is self-contained.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

XELATEX = r"D:/Program Files/texlive/2024/bin/windows/xelatex.exe"
PAPER = Path("outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper")

UNICODE_RAW = {"…": r"\ldots{}", "↑": r"$\uparrow$", "↓": r"$\downarrow$"}


def esc(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def inline(text: str) -> str:
    math, code, raw = [], [], []

    def take(store, m):
        store.append(m.group(0) if store is not raw else m.group(1))
        return f"@@@{'MCR'[[math, code, raw].index(store)]}{len(store) - 1}@@@"

    text = re.sub(r"\$\$.+?\$\$", lambda m: take(math, m), text, flags=re.S)
    text = re.sub(r"\$.+?\$", lambda m: take(math, m), text)
    text = re.sub(r"`([^`]+)`", lambda m: (code.append(m.group(1)) or f"@@@C{len(code) - 1}@@@"), text)
    text = text.replace("—", "---").replace("–", "--")
    for u, r in UNICODE_RAW.items():
        while u in text:
            raw.append(r)
            text = text.replace(u, f"@@@R{len(raw) - 1}@@@", 1)
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    for i, c in enumerate(code):
        text = text.replace(f"@@@C{i}@@@", r"\texttt{" + esc(c) + "}")
    for i, r in enumerate(raw):
        text = text.replace(f"@@@R{i}@@@", r)
    for i, m in enumerate(math):
        text = text.replace(f"@@@M{i}@@@", m)
    return text


def convert(md: str) -> str:
    lines = md.split("\n")
    title = "Paper"
    out, i, n = [], 0, len(lines)
    list_mode = None  # 'itemize' | 'enumerate'

    def close_list():
        nonlocal list_mode
        if list_mode:
            out.append(f"\\end{{{list_mode}}}")
            list_mode = None

    while i < n:
        ln = lines[i]
        s = ln.strip()
        # inline figure directive: [FIG: file.pdf | width | Caption text]
        mfig = re.match(r"^\[FIG:\s*([^|\]]+?)\s*\|\s*([\d.]+)\s*\|\s*(.*)\]$", s)
        if mfig:
            close_list()
            fname, width, cap = mfig.group(1).strip(), mfig.group(2).strip(), mfig.group(3).strip()
            out.append(r"\begin{figure}[H]\centering")
            out.append(rf"\includegraphics[width={width}\linewidth]{{{fname}}}")
            out.append(r"\caption{" + inline(cap) + "}")
            out.append(r"\end{figure}")
            i += 1
            continue
        # table block
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_list()
            header = [c.strip() for c in s.strip("|").split("|")]
            ncol = len(header)
            out.append(r"\begin{center}\begin{tabular}{l" + "c" * (ncol - 1) + "}")
            out.append(r"\toprule")
            out.append(" & ".join(inline(c) for c in header) + r" \\")
            out.append(r"\midrule")
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append(" & ".join(inline(c) for c in cells) + r" \\")
                i += 1
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}\end{center}")
            continue
        # blockquote box (scope/protocol boxes): consecutive '>' lines -> framed parbox
        if s.startswith(">"):
            close_list()
            qlines = []
            while i < n and lines[i].strip().startswith(">"):
                qlines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            body = r"\par\smallskip ".join(inline(q) for q in qlines if q)
            out.append(r"\medskip\noindent\fbox{\parbox{0.96\linewidth}{" + body + r"}}\medskip")
            continue
        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            close_list()
            level, txt = len(m.group(1)), inline(m.group(2))
            if level == 1:
                if title == "Paper":
                    title = m.group(2)          # only the FIRST h1 is the title
                else:
                    out.append(r"\clearpage\section*{" + txt + "}")
                    if "appendices" in m.group(2).lower():
                        out.append(r"\setcounter{figure}{0}\renewcommand{\thefigure}{S\arabic{figure}}")
            elif level == 2:
                out.append(r"\section*{" + txt + "}")
            elif level == 3:
                out.append(r"\subsection*{" + txt + "}")
            else:
                out.append(r"\subsubsection*{" + txt + "}")
            i += 1
            continue
        # horizontal rule
        if re.match(r"^---+$", s):
            close_list()
            out.append(r"\medskip\hrule\medskip")
            i += 1
            continue
        # lists
        mb = re.match(r"^[-*]\s+(.*)$", s)
        mo = re.match(r"^\d+\.\s+(.*)$", s)
        if mb or mo:
            want = "itemize" if mb else "enumerate"
            if list_mode != want:
                close_list()
                out.append(f"\\begin{{{want}}}")
                list_mode = want
            out.append(r"\item " + inline((mb or mo).group(1)))
            i += 1
            continue
        # blank
        if s == "":
            close_list()
            out.append("")
            i += 1
            continue
        # paragraph
        close_list()
        out.append(inline(s))
        i += 1
    close_list()

    preamble = "\n".join([
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{array}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{parskip}",
        r"\title{" + inline(title) + "}",
        r"\author{}\date{}",
        r"\begin{document}",
        r"\maketitle",
    ])
    figs = []
    # order matches the prose figure references (method=Fig1, Pareto=Fig2, qualitative=Fig3, cross-rate=Fig4)
    fig_specs = [("METHOD_DIAGRAM_3D.pdf", "Pseudo-3D mechanism of measurement-consistent VQGAN detail fusion. Every measurement-consistent image lies on the affine plane $\\{x:Ax=y\\}=x_0+\\mathcal{N}(A)$ (drawn in oblique 3D); the bucket fixes only the orthogonal row-space (normal) direction, so leaving the plane changes $y$. From the measurement-audited LMMSE anchor $x_0$, the fusion weight $B$ slides the reconstruction along the in-plane dial from the VQAE structure point $x_A$ ($B{=}0$) to the VQGAN detail point $x_G$ ($B{=}1$), with balanced the validation-selected operating point. Because $AP_0=0$, every $B$ stays on the plane and $A\\hat{x}_B=y$ holds exactly. The true scene $x^*$ also lies on the plane, but its null-space location is unknowable from $y$."),
                 ("PARETO_FIGURE.pdf", "Perception--distortion operating points (locked filled, development hollow)."),
                 ("QUALITATIVE_GRID.pdf", "Locked-split reconstructions on fixed samples."),
                 ("rate_generalization_figure.pdf", "Cross-sampling-rate generalization (development-level, 3 seeds/rate): the balanced-fusion LPIPS advantage over VQAE holds at 2\\%, 5\\% (locked), and 10\\% sampling."),
                 ("B_CURVE.pdf", "Fine-grained perception--distortion frontier: a dense 21-point sweep of the fusion weight $B$ (development split), smooth and monotone from VQAE ($B{=}0$) to full VQGAN ($B{=}1$).")]
    if "[FIG:" in md:
        fig_specs = []          # inline directives supersede the appended review-figure block
    if any((PAPER / f).exists() for f, _ in fig_specs):
        figs.append(r"\clearpage\section*{Figures}")
        for f, cap in fig_specs:
            if (PAPER / f).exists():
                figs.append(r"\begin{figure}[h!]\centering")
                figs.append(r"\includegraphics[width=\linewidth]{" + f + "}")
                figs.append(r"\caption{" + inline(cap) + "}")
                figs.append(r"\end{figure}\par\medskip")
    return preamble + "\n" + "\n".join(out) + "\n" + "\n".join(figs) + "\n\\end{document}\n"


def main():
    src = PAPER / "PAPER_DRAFT.md"
    tex = convert(src.read_text(encoding="utf-8"))
    texpath = PAPER / "PAPER_DRAFT.tex"
    texpath.write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run([XELATEX, "-interaction=nonstopmode", "-halt-on-error", "PAPER_DRAFT.tex"],
                           cwd=str(PAPER), capture_output=True, text=True)
    pdf = PAPER / "PAPER_DRAFT.pdf"
    ok = pdf.exists()
    print("PDF:", pdf if ok else "FAILED")
    if not ok:
        print(r.stdout[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
