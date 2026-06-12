from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase38_professional_figure"
FIG_DIR = OUT / "figures"
EDIT_PACK = OUT / "figure1_edit_pack"

SVG = FIG_DIR / "fig1_professional_mechanism_v38.svg"
SOURCE_SVG = FIG_DIR / "fig1_professional_mechanism_v38_source.svg"
PDF = FIG_DIR / "fig1_professional_mechanism_v38.pdf"
PNG = FIG_DIR / "fig1_professional_mechanism_v38_600dpi.png"
TIKZ = FIG_DIR / "fig1_professional_mechanism_v38.tikz.tex"
INFO = OUT / "INKSCAPE_INFO.json"

WIDTH = 1800
HEIGHT = 950

FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"
BG = "#ffffff"
TEXT = "#1f2328"
MUTED = "#5f6b7a"
BASELINE = "#6f8fae"
BASE_FILL = "#eef5fb"
BLUE = "#1f77b4"
BLUE_FILL = "#eff7ff"
ORANGE = "#d97904"
ORANGE_FILL = "#fff6e8"
PURPLE = "#7b4ab8"
PURPLE_FILL = "#f6f0ff"
GREEN = "#238b45"
GREEN_FILL = "#effaf2"
STROKE = "#c9d1dc"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(
    x: float,
    y: float,
    content: str | list[str],
    size: int = 18,
    fill: str = TEXT,
    weight: str = "400",
    anchor: str = "middle",
    line_gap: float = 1.22,
) -> str:
    lines = [content] if isinstance(content, str) else content
    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{size * line_gap:.1f}"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str, r: int = 24, sw: float = 3) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, sw: float = 3, arrow: bool = True) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"{marker}/>'
    )


def small_label(x: float, y: float, content: str, fill: str = MUTED) -> str:
    return text(x, y, content, size=16, fill=fill, anchor="middle")


def checkerboard(x: float, y: float, size: int = 54, label: str = "a_i") -> str:
    cell = size / 4
    pieces = [f'<g aria-label="checkerboard pattern {esc(label)}">']
    pieces.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#ffffff" stroke="{BLUE}" stroke-width="2"/>')
    for row in range(4):
        for col in range(4):
            color = "#202124" if (row + col) % 2 == 0 else "#f7faff"
            pieces.append(
                f'<rect x="{x + col * cell:.1f}" y="{y + row * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="{color}" stroke="#d0d7de" stroke-width="0.5"/>'
            )
    pieces.append(text(x + size / 2, y + size + 23, label, size=16, fill=BLUE))
    pieces.append("</g>")
    return "\n".join(pieces)


def vector_icon(x: float, y: float, values: list[float], color: str, label: str) -> str:
    pieces = [f'<g aria-label="{esc(label)} vector icon">']
    pieces.append(f'<rect x="{x}" y="{y}" width="118" height="78" rx="14" fill="#ffffff" stroke="{color}" stroke-width="2.4"/>')
    for i, value in enumerate(values):
        h = 46 * value
        bx = x + 18 + i * 22
        by = y + 58 - h
        pieces.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="13" height="{h:.1f}" rx="3" fill="{color}" opacity="0.82"/>')
    pieces.append(text(x + 59, y + 104, label, size=17, fill=color))
    pieces.append("</g>")
    return "\n".join(pieces)


def image_box(x: float, y: float, w: float, h: float, label: str, note: str, color: str, fill: str) -> str:
    pieces = [f'<g aria-label="{esc(label)} schematic image">']
    pieces.append(rect(x, y, w, h, fill, color, r=20, sw=2.6))
    pieces.append(f'<rect x="{x + 28}" y="{y + 26}" width="{w - 56}" height="{h - 88}" rx="16" fill="#f7f9fb" stroke="{color}" stroke-width="2"/>')
    # Simple vector-only schematic texture.
    cx = x + w / 2
    cy = y + 68
    for i, radius in enumerate([48, 34, 21, 10]):
        opacity = 0.10 + i * 0.07
        pieces.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" opacity="{opacity:.2f}"/>')
    pieces.append(text(x + w / 2, y + h - 42, label, size=18, fill=color, weight="700"))
    pieces.append(text(x + w / 2, y + h - 18, note, size=15, fill=MUTED))
    pieces.append("</g>")
    return "\n".join(pieces)


def filter_icon(x: float, y: float, show_label: bool = True) -> str:
    pieces = [
        '<g aria-label="P_N residual filter icon">',
        f'<polygon points="{x},{y} {x + 130},{y} {x + 84},{y + 70} {x + 72},{y + 118} {x + 58},{y + 118} {x + 46},{y + 70}" '
        f'fill="{PURPLE_FILL}" stroke="{PURPLE}" stroke-width="3" stroke-linejoin="round"/>',
        f'<line x1="{x + 35}" y1="{y + 32}" x2="{x + 95}" y2="{y + 32}" stroke="{PURPLE}" stroke-width="2.5"/>',
        f'<line x1="{x + 52}" y1="{y + 63}" x2="{x + 78}" y2="{y + 63}" stroke="{PURPLE}" stroke-width="2.5"/>',
    ]
    if show_label:
        pieces.append(text(x + 65, y + 147, "P_N filter", size=17, fill=PURPLE, weight="700"))
    pieces.append("</g>")
    return "\n".join(pieces)


def audit_icon(x: float, y: float, show_label: bool = True) -> str:
    pieces = [
        '<g aria-label="Pi_y bucket audit icon">',
        f'<rect x="{x}" y="{y}" width="126" height="118" rx="22" fill="{GREEN_FILL}" stroke="{GREEN}" stroke-width="3"/>',
        f'<path d="M{x + 30},{y + 63} L{x + 54},{y + 86} L{x + 96},{y + 38}" fill="none" stroke="{GREEN}" stroke-width="9" '
        'stroke-linecap="round" stroke-linejoin="round"/>',
        f'<line x1="{x + 28}" y1="{y + 26}" x2="{x + 98}" y2="{y + 26}" stroke="{GREEN}" stroke-width="3"/>',
    ]
    if show_label:
        pieces.append(text(x + 63, y + 147, "Pi_y audit", size=17, fill=GREEN, weight="700"))
    pieces.append("</g>")
    return "\n".join(pieces)


def block(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str | list[str],
    lines: list[str],
    note: str | list[str],
    fill: str,
    stroke: str,
    formula_size: int = 24,
    note_size: int = 17,
) -> str:
    pieces = [rect(x, y, w, h, fill, stroke, r=24, sw=3)]
    title_lines = [title] if isinstance(title, str) else title
    pieces.append(text(x + w / 2, y + 40, title_lines, size=22, fill=TEXT, weight="700", line_gap=1.05))
    start_y = y + 84 + max(0, len(title_lines) - 1) * 14
    for i, formula in enumerate(lines):
        pieces.append(text(x + w / 2, start_y + i * 32, formula, size=formula_size, fill=TEXT))
    note_lines = [note] if isinstance(note, str) else note
    if any(note_lines):
        pieces.append(text(x + w / 2, y + h - 44, note_lines, size=note_size, fill=stroke, line_gap=1.1))
    return "\n".join(pieces)


def build_svg() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{MUTED}"/>',
        "</marker>",
        "</defs>",
        "<metadata>"
        "A^T y; sum_i y_i a_i; q = (AA^T + lambda I)^-1 y; x_data = A^T q; "
        "r_theta = G_theta(x_data); P_N filter; Pi_y audit."
        "</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        text(900, 54, "From GI correlation to measurement-audited neural completion", size=34, weight="700"),
    ]

    # Shared input.
    parts.append(rect(54, 128, 338, 604, "#f8fbff", STROKE, r=28, sw=3))
    parts.append(text(223, 174, "Known-pattern", size=24, weight="700"))
    parts.append(text(223, 203, "bucket measurements", size=24, weight="700"))
    parts.append(checkerboard(94, 242, 54, "a_1"))
    parts.append(checkerboard(176, 242, 54, "a_i"))
    parts.append(checkerboard(258, 242, 54, "a_m"))
    parts.append(vector_icon(134, 356, [0.35, 0.82, 0.52, 0.72], BLUE, "bucket vector y"))
    parts.append(text(223, 503, "y_i = <a_i, x> + ε_i", size=21, fill=TEXT))
    parts.append(text(223, 548, "y = Ax + ε,   m << n", size=22, fill=TEXT, weight="700"))
    parts.append(text(223, 636, ["same measurements", "enter both paths"], size=17, fill=MUTED))

    # Split arrows.
    parts.append(line(392, 430, 480, 250, color=MUTED, sw=3))
    parts.append(line(392, 430, 480, 565, color=MUTED, sw=3))

    # Conventional upper path.
    parts.append(block(480, 132, 420, 235, "Conventional GI / raw BP", ["x̂_GI = Aᵀy = ∑ᵢ yᵢaᵢ"], "raw bucket weights y_i", BASE_FILL, BASELINE))
    parts.append(line(900, 250, 960, 250, color=BASELINE, sw=3))
    parts.append(image_box(960, 150, 210, 205, "GI/BP image", "physical but incomplete", BASELINE, "#f6f9fc"))

    # Lower path.
    parts.append(block(480, 466, 330, 246, "Regularized data solution", ["q = (AAᵀ + λI)⁻¹y", "x_data = Aᵀq = ∑ᵢ qᵢaᵢ"], "", BLUE_FILL, BLUE))
    parts.append(vector_icon(515, 594, [0.35, 0.82, 0.52, 0.72], BASELINE, "raw y"))
    parts.append(line(640, 634, 675, 634, color=BLUE, sw=2.6))
    parts.append(vector_icon(682, 594, [0.50, 0.66, 0.58, 0.61], BLUE, "decorrelated q"))

    parts.append(line(810, 588, 850, 588, color=MUTED, sw=3))
    parts.append(block(
        850,
        486,
        250,
        210,
        "Candidate residual",
        ["r_θ = G_θ", "(x_data)"],
        ["network proposes", "missing structure", "not a final image"],
        ORANGE_FILL,
        ORANGE,
        formula_size=21,
        note_size=14,
    ))

    parts.append(line(1100, 588, 1140, 588, color=MUTED, sw=3))
    parts.append(rect(1140, 486, 220, 210, PURPLE_FILL, PURPLE, r=24, sw=3))
    parts.append(text(1245, 528, "Residual filter", size=22, weight="700"))
    parts.append(text(1245, 574, "r_N = P_N(r_θ)", size=21))
    parts.append(filter_icon(1180, 596, show_label=False))
    parts.append(text(1250, 718, ["P_N filter", "removes A-visible residual"], size=14, fill=PURPLE, weight="700", line_gap=1.08))

    parts.append(line(1360, 588, 1390, 588, color=MUTED, sw=3))
    parts.append(block(
        1390,
        486,
        220,
        210,
        ["Measured +", "missing"],
        ["x̃ = x_data", "+ r_N"],
        ["anchor + learned", "completion"],
        "#f8fbff",
        BLUE,
        formula_size=20,
        note_size=15,
    ))

    parts.append(line(1610, 588, 1640, 588, color=MUTED, sw=3))
    parts.append(rect(1640, 486, 130, 210, GREEN_FILL, GREEN, r=24, sw=3))
    parts.append(text(1705, 528, "Bucket", size=19, fill=TEXT, weight="700"))
    parts.append(text(1705, 552, "audit", size=19, fill=TEXT, weight="700"))
    parts.append(audit_icon(1642, 560, show_label=False))
    parts.append(text(1705, 680, "Pi_y", size=18, fill=GREEN, weight="700"))

    parts.append(image_box(1518, 740, 252, 140, "measurement-audited", "reconstruction tied to y", GREEN, "#f7fcf8"))
    parts.append(line(1705, 696, 1705, 740, color=GREEN, sw=2.8))

    # Bottom core transition.
    parts.append(rect(342, 792, 1116, 112, "#ffffff", STROKE, r=26, sw=2.8))
    parts.append(text(900, 828, "core transition", size=18, fill=MUTED, weight="700"))
    parts.append(text(900, 867, "Aᵀy  ->  Aᵀ(AAᵀ + λI)⁻¹y  ->  Π_y[x_data + P_N(G_θ)]", size=25, fill=TEXT))
    parts.append(text(900, 894, "raw GI correlation -> regularized measured anchor -> learned completion with final audit", size=16, fill=MUTED))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def find_inkscape() -> tuple[bool, str | None, str | None]:
    candidates = []
    which = shutil.which("inkscape")
    if which:
        candidates.append(which)
    fallback = Path("C:/Program Files/Inkscape/bin/inkscape.com")
    if fallback.exists():
        candidates.append(str(fallback))
    for candidate in candidates:
        try:
            proc = subprocess.run([candidate, "--version"], check=False, capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        if proc.returncode == 0:
            return True, candidate, (proc.stdout or proc.stderr).strip()
    return False, candidates[0] if candidates else None, None


def export_with_inkscape(command: str | None) -> None:
    if not command:
        return
    subprocess.run([command, str(SVG), "--export-filename", str(PDF)], check=True)
    subprocess.run([command, str(SVG), "--export-filename", str(PNG), "--export-dpi", "600"], check=True)


def write_tikz() -> None:
    TIKZ.write_text(
        r"""
% Editable TikZ fallback for Figure 1. The publication source is the SVG.
\begin{tikzpicture}[font=\sffamily,>=latex]
\node[draw,rounded corners,align=center,minimum width=3.0cm,minimum height=1.4cm] (input) {Known-pattern\\bucket measurements\\$y=Ax+\epsilon,\ m\ll n$};
\node[draw,rounded corners,align=center,right=1.2cm of input,minimum width=3.8cm,minimum height=1.2cm] (raw) {Conventional GI / raw BP\\$\hat{x}_{GI}=A^Ty=\sum_i y_i a_i$};
\node[draw,rounded corners,align=center,below=1.0cm of raw,minimum width=3.8cm,minimum height=1.5cm] (data) {Regularized data solution\\$q=(AA^T+\lambda I)^{-1}y$\\$x_{data}=A^Tq$};
\node[draw,rounded corners,align=center,right=0.8cm of data] (res) {Candidate residual\\$r_\theta=G_\theta(x_{data})$};
\node[draw,rounded corners,align=center,right=0.8cm of res] (filter) {Residual filter\\$r_N=P_N(r_\theta)$};
\node[draw,rounded corners,align=center,right=0.8cm of filter] (audit) {Bucket audit\\$\hat{x}=\Pi_y(x_{data}+r_N)$};
\draw[->] (input) -- (raw);
\draw[->] (input) -- (data);
\draw[->] (data) -- (res);
\draw[->] (res) -- (filter);
\draw[->] (filter) -- (audit);
\node[below=1.0cm of filter,align=center] {$A^Ty \rightarrow A^T(AA^T+\lambda I)^{-1}y \rightarrow \Pi_y[x_{data}+P_N(G_\theta)]$};
\end{tikzpicture}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_guides() -> None:
    (EDIT_PACK / "FIGURE1_EDITING_GUIDE.md").write_text(
        """# Figure 1 Editing Guide

Open `fig1_professional_mechanism_v38.svg` in Inkscape, Illustrator, or Affinity Designer.

Manual adjustment priority:
1. Enlarge formulas if the target journal reduces the figure strongly.
2. Align modules and arrows after any text edits.
3. Reduce explanatory text before reducing formula size.
4. Keep the \(P_N\) filter and \(\Pi_y\) audit visually distinct.
5. Export PDF and 600 dpi PNG after manual edits.

The SVG text is intentionally editable; avoid rasterizing text during manual editing.
""",
        encoding="utf-8",
    )
    (EDIT_PACK / "FIGURE1_STYLE_GUIDE.md").write_text(
        """# Figure 1 Style Guide

- Canvas: 1800 x 950 px, white background.
- Fonts: Arial / Helvetica / DejaVu Sans.
- Minimum text size: 15 px.
- Conventional GI path: muted blue-gray.
- Regularized measured anchor: blue.
- Neural residual: orange.
- Residual filter: purple.
- Measurement audit: green.
- Do not add laser, DMD, lens, CCD, camera, or other hardware optical-path drawings.
- Keep the figure as a conceptual reconstruction mechanism diagram.
""",
        encoding="utf-8",
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EDIT_PACK.mkdir(parents=True, exist_ok=True)
    svg_text = build_svg()
    SVG.write_text(svg_text, encoding="utf-8")
    SOURCE_SVG.write_text(svg_text, encoding="utf-8")
    write_tikz()
    found, command, version = find_inkscape()
    if found:
        export_with_inkscape(command)
    info = {"found": found, "command_path": command, "version": version}
    INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_guides()
    for path in [SVG, SOURCE_SVG, PDF, PNG, TIKZ]:
        if path.exists():
            shutil.copy2(path, EDIT_PACK / path.name)
    print({"figure_svg": str(SVG), "inkscape": info, "pdf_exists": PDF.exists(), "png_exists": PNG.exists()})


if __name__ == "__main__":
    main()
