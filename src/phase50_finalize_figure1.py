"""Phase 50 Figure 1 finalization.

This script uses the user-provided Figure 1 draft as the source artifact and
generates the final editable SVG/PDF/PNG outputs for the manuscript package.
It does not train models or change any numerical result.
"""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path


OUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase50_final_figure1")
FIG_DIR = OUT_DIR / "figures"
LATEX_DIR = OUT_DIR / "latex_project_v50"
LATEX_FIG_DIR = LATEX_DIR / "figures"
USER_SVG = Path("C:/Users/CYZ的computer/Downloads/fig1_draft.svg")
PROJECT_SVG = Path(__file__).resolve().parents[1] / "fig1_draft.svg"


BLUE = "#2b5ba8"
BLUE_LIGHT = "#dce8f8"
ORANGE = "#e07b39"
ORANGE_LIGHT = "#fbe8d8"
PURPLE = "#6f4bb7"
PURPLE_LIGHT = "#eee7fb"
GREEN = "#1f8a70"
GREEN_LIGHT = "#ddf2ec"
GRAY = "#777777"
GRAY_LIGHT = "#f1f1f1"
DARK = "#1a1a1a"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def text(x: float, y: float, s: str, size: int = 16, fill: str = DARK, weight: str = "400",
         anchor: str = "start", style: str = "") -> str:
    extra = f";{style}" if style else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'style="letter-spacing:0{extra}">{esc(s)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = DARK, width: float = 2.2,
         arrow: bool = True, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}" fill="none"{dash_attr}{marker}/>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str = "white", stroke: str = DARK,
         width: float = 2.0, rx: float = 8, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
    )


def circle(cx: float, cy: float, r: float, fill: str = "white", stroke: str = DARK,
           width: float = 2.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def multiline(x: float, y: float, lines: list[str], size: int = 14, fill: str = DARK,
              weight: str = "400", leading: float = 18, anchor: str = "start") -> str:
    return "\n".join(text(x, y + i * leading, item, size, fill, weight, anchor) for i, item in enumerate(lines))


def ghost(cx: float, cy: float, scale: float = 1.0) -> str:
    w, h = 54 * scale, 64 * scale
    x, y = cx - w / 2, cy - h / 2
    body = (
        f'<path d="M{x:.1f},{y+h:.1f} L{x:.1f},{y+24*scale:.1f} '
        f'C{x:.1f},{y+7*scale:.1f} {cx-16*scale:.1f},{y:.1f} {cx:.1f},{y:.1f} '
        f'C{cx+16*scale:.1f},{y:.1f} {x+w:.1f},{y+7*scale:.1f} {x+w:.1f},{y+24*scale:.1f} '
        f'L{x+w:.1f},{y+h:.1f} '
        f'L{x+w-9*scale:.1f},{y+h-8*scale:.1f} L{x+w-18*scale:.1f},{y+h:.1f} '
        f'L{x+w-27*scale:.1f},{y+h-8*scale:.1f} L{x+w-36*scale:.1f},{y+h:.1f} '
        f'L{x+w-45*scale:.1f},{y+h-8*scale:.1f} Z" fill="#f7f7f7" stroke="#333333" stroke-width="{2.2*scale:.1f}"/>'
    )
    eyes = (
        f'<circle cx="{cx-10*scale:.1f}" cy="{cy-7*scale:.1f}" r="{3.5*scale:.1f}" fill="#333333"/>'
        f'<circle cx="{cx+10*scale:.1f}" cy="{cy-7*scale:.1f}" r="{3.5*scale:.1f}" fill="#333333"/>'
    )
    return body + eyes


def pattern_stack(x: float, y: float) -> str:
    parts = []
    for i, dx in enumerate([26, 13, 0]):
        parts.append(rect(x + dx, y + dx * 0.35, 60, 60, "#ffffff", "#555555", 1.5, 0))
        step = 7
        for row in range(8):
            for col in range(8):
                if (row * 3 + col * 5 + i) % 4 in (0, 1):
                    parts.append(
                        f'<rect x="{x+dx+col*step+2:.1f}" y="{y+dx*0.35+row*step+2:.1f}" '
                        f'width="4.5" height="4.5" fill="#1a1a1a"/>'
                    )
    return "\n".join(parts)


def bucket_vector(x: float, y: float, w: float, h: float) -> str:
    parts = [rect(x, y, w, h, "#eef4ff", BLUE, 2.0, 0)]
    for i in range(46):
        xx = x + 3 + i * (w - 6) / 46
        shade = 50 + (i * 37) % 170
        parts.append(f'<rect x="{xx:.1f}" y="{y+3:.1f}" width="3.1" height="{h-6:.1f}" fill="rgb({shade},{shade},{shade})"/>')
    return "\n".join(parts)


def final_svg() -> str:
    svg: list[str] = []
    svg.append("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="640" viewBox="0 0 1500 640" version="1.1">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,3 L0,6 Z" fill="context-stroke"/>
  </marker>
  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="1" stdDeviation="1" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
</defs>
<rect width="1200" height="640" fill="#ffffff"/>
""")

    # Panel labels and panel A.
    svg.append(text(18, 42, "a", 24, DARK, "700"))
    svg.append(text(305, 42, "b", 24, DARK, "700"))
    svg.append(text(1190, 42, "c", 24, DARK, "700"))

    svg.append(ghost(72, 132, 1.0))
    svg.append(text(72, 203, "object x", 16, DARK, "400", "middle"))
    svg.append(circle(130, 128, 10, "#ffffff", DARK, 2.4))
    svg.append(text(130, 134, "×", 18, DARK, "700", "middle"))
    svg.append(pattern_stack(148, 80))
    svg.append(text(196, 203, "patterns aᵢ", 16, DARK, "400", "middle"))
    svg.append(text(196, 224, "known rows of A", 13, GRAY, "400", "middle"))
    svg.append(line(240, 128, 282, 128, DARK, 2.2, True))
    svg.append('<path d="M288 104 C318 112 318 144 288 152 Z" fill="#666666"/>')
    svg.append(text(306, 124, "bucket", 14, GRAY, "400", "middle"))
    svg.append(text(306, 143, "detector", 14, GRAY, "400", "middle"))
    svg.append(line(306, 156, 306, 292, DARK, 2.4, True))
    svg.append(bucket_vector(28, 292, 246, 27))
    svg.append(line(274, 306, 330, 306, BLUE, 2.6, True))
    svg.append(multiline(92, 348, ["y = Ax + ε ∈ Rᵐ", "m ≪ n"], 18, DARK, "400", 25))
    svg.append(text(58, 382, "computational forward model", 14, GRAY))

    # Panel B backgrounds.
    svg.append(rect(330, 45, 760, 49, "#ffffff", BLUE, 3.0, 8))
    svg.append(text(710, 73, "Bλₒₚ = Aᵀ(AAᵀ + λₒₚ I)⁻¹", 21, DARK, "400", "middle"))
    svg.append(text(710, 91, "fixed physical solver", 13, GRAY, "400", "middle"))

    svg.append(rect(330, 112, 760, 108, BLUE_LIGHT, "none", 0, 8))
    svg.append(rect(330, 252, 760, 138, ORANGE_LIGHT, "none", 0, 8))
    svg.append(rect(330, 420, 760, 155, GREEN_LIGHT, "none", 0, 8))
    svg.append(text(320, 180, "ANCHOR", 16, BLUE, "700", "middle", "writing-mode:vertical-rl"))
    svg.append(text(320, 324, "GATE", 16, ORANGE, "700", "middle", "writing-mode:vertical-rl"))
    svg.append(text(320, 501, "AUDIT", 16, GREEN, "700", "middle", "writing-mode:vertical-rl"))

    # Anchor row.
    svg.append(text(360, 137, "configured physical anchor", 17, BLUE, "700"))
    svg.append(rect(392, 152, 125, 42, "#ffffff", BLUE, 3.0, 8))
    svg.append(text(454, 179, "x_data = D(y)", 18, DARK, "400", "middle"))
    svg.append(line(517, 173, 555, 173, BLUE, 2.5, True))
    svg.append(rect(556, 144, 86, 66, "#f7f7f7", "#999999", 1.4, 0))
    svg.append(ghost(599, 176, 0.54))
    svg.append(text(599, 224, "measured anchor", 13, GRAY, "400", "middle"))
    svg.append(line(597, 210, 597, 255, BLUE, 2.0, True))

    # Gate row.
    svg.append(text(360, 276, "proposal -> admitted residual", 17, ORANGE, "700"))
    boxes = [
        (360, 304, 87, 48, "x_data", "#ffffff", BLUE),
        (474, 304, 86, 48, "Gθ", "#ffffff", ORANGE),
        (587, 304, 75, 48, "rθ", "#ffffff", ORANGE),
        (680, 292, 180, 72, "P_N^λ = I - Bλₒₚ A", "#ffffff", PURPLE),
        (887, 304, 64, 48, "rN", "#ffffff", PURPLE),
    ]
    for x, y, w, h, label, fill, stroke in boxes:
        svg.append(rect(x, y, w, h, fill, stroke, 2.8, 8))
        svg.append(text(x + w / 2, y + h / 2 + 6, label, 18 if w > 80 else 19, stroke if label in {"Gθ", "rN"} else DARK, "400", "middle"))
    for x1, x2 in [(447, 474), (560, 587), (662, 680), (860, 887)]:
        svg.append(line(x1, 328, x2, 328, PURPLE if x1 >= 662 else ORANGE, 2.4, True))
    svg.append(circle(983, 328, 16, "#ffffff", GREEN, 2.8))
    svg.append(text(983, 335, "+", 24, GREEN, "700", "middle"))
    svg.append(line(951, 328, 967, 328, PURPLE, 2.4, True))
    svg.append(line(983, 344, 983, 420, GREEN, 2.4, True))
    svg.append(line(455, 192, 455, 304, BLUE, 1.8, True))
    svg.append(line(599, 210, 599, 304, BLUE, 1.8, True))
    svg.append(text(760, 382, "defined by Bλₒₚ", 13, GRAY, "400", "middle"))
    svg.append(line(760, 292, 760, 255, PURPLE, 1.8, True, "5 5"))
    svg.append(rect(635, 226, 370, 24, GRAY_LIGHT, "#cccccc", 1.0, 12))
    svg.append(text(820, 243, "loss gradients update Gθ/Rφ only;  A, Bλₒₚ, P_N, Πy fixed", 13, GRAY, "400", "middle"))

    # Audit row.
    svg.append(text(360, 445, "remeasure -> compare -> correct", 17, GREEN, "700"))
    svg.append(rect(368, 468, 118, 48, "#ffffff", GREEN, 2.8, 8))
    svg.append(text(427, 498, "x̃ = x_data + rN", 16, DARK, "400", "middle"))
    svg.append(line(486, 492, 528, 492, GREEN, 2.4, True))
    svg.append(rect(528, 468, 66, 48, "#ffffff", GREEN, 2.8, 8))
    svg.append(text(561, 498, "A", 20, GREEN, "700", "middle"))
    svg.append(line(594, 492, 626, 492, GREEN, 2.4, True))
    svg.append(rect(626, 468, 125, 48, "#ffffff", GREEN, 2.8, 8))
    svg.append(text(688, 498, "ey = A x̃ - y", 16, DARK, "400", "middle"))
    svg.append(line(751, 492, 782, 492, GREEN, 2.4, True))
    svg.append(rect(782, 468, 103, 48, "#ffffff", BLUE, 2.8, 8))
    svg.append(text(834, 498, "δ = Bλₒₚ ey", 16, DARK, "400", "middle"))
    svg.append(line(885, 492, 927, 492, GREEN, 2.4, True))
    svg.append(circle(946, 492, 16, "#ffffff", "#c0392b", 2.8))
    svg.append(text(946, 499, "−", 24, "#c0392b", "700", "middle"))
    svg.append(line(962, 492, 1004, 492, GREEN, 2.4, True))
    svg.append(rect(1004, 468, 84, 48, "#ffffff", GREEN, 2.8, 8))
    svg.append(text(1046, 498, "x̂ final", 17, DARK, "400", "middle"))
    svg.append(text(706, 542, "bucket audit", 14, GRAY, "400", "middle"))

    # Panel C.
    svg.append(text(1228, 70, "Idealized geometry", 21, DARK, "700"))
    svg.append(text(1288, 95, "λₒₚ > 0: soft gate/audit", 13, GRAY, "400", "middle"))
    svg.append(line(1215, 540, 1432, 540, DARK, 2.2, True))
    svg.append(line(1225, 548, 1225, 124, DARK, 2.2, True))
    svg.append(text(1324, 570, "measured component / Range(Aᵀ)", 14, DARK, "400", "middle"))
    svg.append(text(1206, 334, "unmeasured freedom / Null(A)", 14, DARK, "400", "middle", "writing-mode:vertical-rl"))
    svg.append('<path d="M1278 256 C1278 183 1368 146 1384 194 C1394 226 1445 218 1433 282 C1425 327 1363 319 1321 324 C1289 327 1267 301 1278 256 Z" fill="#ebebeb" stroke="#999999" stroke-width="2"/>')
    svg.append(text(1324, 236, "natural-image prior M", 13, GRAY, "400", "middle"))
    svg.append(line(1352, 132, 1352, 535, BLUE, 3.0, False))
    svg.append(text(1372, 146, "Cy: Ax = y", 14, BLUE, "400"))
    svg.append(circle(1352, 540, 5, BLUE, BLUE, 1.0))
    svg.append(text(1318, 528, "x_data", 14, BLUE, "400", "end"))
    svg.append(text(1352, 280, "★", 28, DARK, "700", "middle"))
    svg.append(text(1367, 278, "x*", 15, DARK, "400"))
    svg.append(line(1352, 500, 1373, 438, ORANGE, 3.0, True))
    svg.append(text(1340, 484, "gate", 14, ORANGE, "400", "end", "writing-mode:vertical-rl"))
    svg.append(line(1352, 418, 1377, 450, GREEN, 3.0, True))
    svg.append(text(1392, 446, "audit", 14, GREEN, "400"))

    svg.append("</svg>\n")
    return "\n".join(svg)


def inkscape_path() -> str:
    preferred = Path("C:/Program Files/Inkscape/bin/inkscape.com")
    if preferred.exists():
        return str(preferred)
    found = shutil.which("inkscape") or shutil.which("inkscape.com")
    if not found:
        raise RuntimeError("Inkscape was not found on PATH.")
    return found


def run_export(svg_path: Path, pdf_path: Path, png_path: Path) -> str:
    ink = inkscape_path()
    subprocess.run(
        [ink, str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"],
        check=True,
    )
    subprocess.run(
        [ink, str(svg_path), "--export-type=png", "--export-dpi=600", f"--export-filename={png_path}"],
        check=True,
    )
    return ink


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    source_svg = USER_SVG if USER_SVG.exists() else PROJECT_SVG
    if not source_svg.exists():
        raise FileNotFoundError("Could not locate fig1_draft.svg from user Downloads or project root.")
    shutil.copy2(source_svg, FIG_DIR / "fig1_draft_source.svg")

    svg_path = FIG_DIR / "fig1_operator_circuit_final.svg"
    pdf_path = FIG_DIR / "fig1_operator_circuit_final.pdf"
    png_path = FIG_DIR / "fig1_operator_circuit_final_600dpi.png"
    svg_path.write_text(final_svg(), encoding="utf-8")
    ink = run_export(svg_path, pdf_path, png_path)

    shutil.copy2(svg_path, LATEX_FIG_DIR / svg_path.name)
    shutil.copy2(pdf_path, LATEX_FIG_DIR / pdf_path.name)
    shutil.copy2(png_path, LATEX_FIG_DIR / png_path.name)
    (OUT_DIR / "phase50_inkscape_path.txt").write_text(ink + "\n", encoding="utf-8")

    print(f"wrote {svg_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"inkscape={ink}")


if __name__ == "__main__":
    main()
