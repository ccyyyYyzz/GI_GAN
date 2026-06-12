from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase39_anchor_proposal_audit"
FIG_DIR = OUT / "figures"
COMP_IMG = OUT / "mechanism_components" / "component_images"

SVG = FIG_DIR / "fig1_anchor_proposal_audit_v39.svg"
PDF = FIG_DIR / "fig1_anchor_proposal_audit_v39.pdf"
PNG = FIG_DIR / "fig1_anchor_proposal_audit_v39_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"

WIDTH = 2300
HEIGHT = 1550
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"
TEXT = "#1f2328"
MUTED = "#5f6b7a"
STROKE = "#c9d1dc"
BLUE = "#1f77b4"
BLUE_FILL = "#eff7ff"
ORANGE = "#d97904"
ORANGE_FILL = "#fff7ed"
GREEN = "#238b45"
GREEN_FILL = "#effaf2"
SOFT = "#f8fafc"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(
    x: float,
    y: float,
    content: str | list[str],
    size: int = 20,
    fill: str = TEXT,
    weight: str = "400",
    anchor: str = "middle",
    gap: float = 1.18,
) -> str:
    lines = [content] if isinstance(content, str) else content
    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{size * gap:.1f}"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = STROKE, r: int = 24, sw: float = 2.5) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, sw: float = 3.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" marker-end="url(#arrow)"/>'


def image_href(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def image_tile(x: float, y: float, size: float, path: Path, label: str | None = None, border: str = STROKE) -> str:
    parts = [rect(x, y, size, size, "#ffffff", border, r=16, sw=2.0)]
    parts.append(
        f'<image x="{x + 8:.1f}" y="{y + 8:.1f}" width="{size - 16:.1f}" height="{size - 16:.1f}" '
        f'preserveAspectRatio="xMidYMid slice" href="{image_href(path)}"/>'
    )
    if label:
        parts.append(text(x + size / 2, y + size + 24, label, size=17, fill=MUTED, weight="700"))
    return "\n".join(parts)


def flow_block(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    formula: str | list[str],
    note: str | list[str],
    fill: str,
    stroke: str,
    image: Path | None = None,
    icon: str | None = None,
) -> str:
    parts = [rect(x, y, w, h, fill, stroke, r=28, sw=3)]
    parts.append(text(x + w / 2, y + 42, title, size=25, fill=TEXT, weight="700"))
    if image is not None:
        parts.append(image_tile(x + w / 2 - 70, y + 70, 140, image, border=stroke))
    elif icon == "set":
        parts.append(text(x + w / 2, y + 110, ["y = Ax + eps", "m << n"], size=20, fill=TEXT, weight="700", gap=1.2))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="44" fill="{}" stroke="{}" stroke-width="2" opacity="0.85"/>'.format(x + 130, y + 198, BLUE_FILL, BLUE))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="44" fill="{}" stroke="{}" stroke-width="2" opacity="0.80"/>'.format(x + 190, y + 198, ORANGE_FILL, ORANGE))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="44" fill="{}" stroke="{}" stroke-width="2" opacity="0.75"/>'.format(x + 160, y + 154, GREEN_FILL, GREEN))
        parts.append(text(x + w / 2, y + 268, "C_y = {x : Ax = y}", size=19, fill=TEXT, weight="700"))
    elif icon == "gate":
        parts.append(f'<path d="M{x + 78},{y + 92} H{x + 190} L{x + 150},{y + 165} V{x * 0 + y + 218} H{x + 118} V{x * 0 + y + 165} Z" fill="#ffffff" stroke="{GREEN}" stroke-width="4" stroke-linejoin="round"/>')
        parts.append(f'<line x1="{x + 98:.1f}" y1="{y + 126:.1f}" x2="{x + 170:.1f}" y2="{y + 126:.1f}" stroke="{GREEN}" stroke-width="3"/>')
        parts.append(f'<line x1="{x + 115:.1f}" y1="{y + 156:.1f}" x2="{x + 153:.1f}" y2="{y + 156:.1f}" stroke="{GREEN}" stroke-width="3"/>')
    elif icon == "audit":
        parts.append(f'<rect x="{x + 72:.1f}" y="{y + 86:.1f}" width="148" height="130" rx="24" fill="#ffffff" stroke="{GREEN}" stroke-width="4"/>')
        parts.append(f'<path d="M{x + 105},{y + 154} L{x + 134},{y + 184} L{x + 188},{y + 124}" fill="none" stroke="{GREEN}" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append(f'<line x1="{x + 104:.1f}" y1="{y + 112:.1f}" x2="{x + 188:.1f}" y2="{y + 112:.1f}" stroke="{GREEN}" stroke-width="3"/>')
    if image is not None:
        parts.append(text(x + w / 2, y + 242, formula, size=16, fill=TEXT, weight="700", gap=1.1))
        parts.append(text(x + w / 2, y + h - 58, note, size=16, fill=stroke, weight="700", gap=1.1))
    else:
        if icon == "set":
            parts.append(text(x + w / 2, y + h - 44, note, size=17, fill=stroke, weight="700", gap=1.1))
            return "\n".join(parts)
        parts.append(text(x + w / 2, y + h - 82, formula, size=18, fill=TEXT, weight="700", gap=1.15))
        parts.append(text(x + w / 2, y + h - 42, note, size=17, fill=stroke, weight="700", gap=1.1))
    return "\n".join(parts)


def bottom_header(x: float, y: float, w: float, label: str) -> str:
    return text(x + w / 2, y, label, size=18, fill=MUTED, weight="700")


def evidence_row(y: float, method: str, label: str) -> str:
    x0 = 125
    sizes = [146, 146, 146, 146, 146, 260]
    gap = 28
    paths = {
        "gt": COMP_IMG / f"{method}_gt.png",
        "x_data": COMP_IMG / f"{method}_x_data.png",
        "pre": COMP_IMG / f"{method}_pre_audit_no_mc.png",
        "final": COMP_IMG / f"{method}_final.png",
        "err": COMP_IMG / f"{method}_abs_error.png",
        "bar": COMP_IMG / f"{method}_measurement_residual_bar.png",
    }
    parts = [text(38, y + 82, label, size=22, fill=TEXT, weight="700", anchor="start")]
    x = x0
    for key, size in zip(["gt", "x_data", "pre", "final", "err"], sizes[:5]):
        parts.append(image_tile(x, y, size, paths[key]))
        x += size + gap
    parts.append(rect(x, y + 8, sizes[-1], 130, "#ffffff", STROKE, r=16, sw=2))
    parts.append(f'<image x="{x + 8:.1f}" y="{y + 17:.1f}" width="{sizes[-1] - 16:.1f}" height="112" preserveAspectRatio="xMidYMid meet" href="{image_href(paths["bar"])}"/>')
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{MUTED}"/>',
        "</marker>",
        "</defs>",
        "<metadata>real STL-10 examples; Anchor -> Proposal -> Legality Filter -> Audit; no conventional GI comparison main path</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(WIDTH / 2, 58, "Measurement-audited neural completion: the network proposes, the measurements audit", size=34, fill=TEXT, weight="700"),
    ]

    bx, by, bw, bh, gap = 52, 110, 400, 350, 38
    blocks = [
        ("Low-sampling evidence", ["y = Ax + eps,  m << n", "C_y = {x : Ax = y}"], "many feasible images", SOFT, STROKE, None, "set"),
        ("Measured anchor", ["x_data = A^T", "(AA^T + lambda I)^-1 y"], ["tied to buckets", "incomplete"], BLUE_FILL, BLUE, COMP_IMG / "rad5_x_data.png", None),
        ("Neural proposal", ["r_theta = G_theta", "(x_data)"], ["proposes missing structure", "not final image"], ORANGE_FILL, ORANGE, COMP_IMG / "rad5_pre_audit_no_mc.png", None),
        ("Legality filter", "r_N = P_N(r_theta)", ["only measurement-silent", "content may enter"], GREEN_FILL, GREEN, None, "gate"),
        ("Bucket audit", ["x_hat = Pi_y", "[x_data + r_N]"], "checks A x_hat approx y", GREEN_FILL, GREEN, COMP_IMG / "rad5_final.png", None),
    ]
    for i, args in enumerate(blocks):
        x = bx + i * (bw + gap)
        parts.append(flow_block(x, by, bw, bh, *args))
        if i < len(blocks) - 1:
            parts.append(line(x + bw + 6, by + bh / 2, x + bw + gap - 6, by + bh / 2))

    parts.append(rect(52, 505, 2196, 150, "#ffffff", STROKE, r=26, sw=2.5))
    parts.append(text(104, 550, "Result badge", size=18, fill=MUTED, weight="700", anchor="start"))
    parts.append(text(430, 550, "STL-10 5% final quality", size=23, fill=TEXT, weight="700"))
    parts.append(text(790, 550, "Rad: 22.316 dB / 0.635 SSIM", size=22, fill=GREEN, weight="700"))
    parts.append(text(1190, 550, "Scr: 22.271 dB / 0.632 SSIM", size=22, fill=GREEN, weight="700"))
    parts.append(text(1120, 610, "Removing final audit is strongest failure mode", size=22, fill=ORANGE, weight="700"))
    parts.append(text(1650, 610, "Rad-5: 22.202 -> 19.399;   Scr-5: 22.155 -> 6.352", size=20, fill=ORANGE, weight="700"))

    parts.append(text(1100, 735, "Real STL-10 evidence strip", size=26, fill=TEXT, weight="700"))
    headers = [
        ("GT", 125, 146),
        ("x_data / BP", 299, 146),
        ("without final audit", 473, 146),
        ("final audited output", 647, 146),
        ("absolute error", 821, 146),
        ("measurement residual", 995, 260),
    ]
    for label, x, w in headers:
        parts.append(bottom_header(x, 780, w, label))
    parts.append(evidence_row(810, "rad5", "Rad-5"))
    parts.append(evidence_row(1005, "scr5", "Scr-5"))

    parts.append(rect(1320, 820, 880, 330, "#ffffff", STROKE, r=24, sw=2.5))
    parts.append(text(1760, 865, "Interpretation", size=24, fill=TEXT, weight="700"))
    parts.append(text(1760, 915, ["The anchor keeps measured evidence visible.", "The network proposes missing structure.", "The legality filter limits what can enter.", "The final audit checks the bucket signal."], size=21, fill=TEXT, gap=1.35))
    parts.append(text(1760, 1112, "The no-audit column is an inference ablation, not a separately trained unconstrained network.", size=17, fill=MUTED))

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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SVG.write_text(build_svg(), encoding="utf-8")
    found, command, version = find_inkscape()
    if found:
        export_with_inkscape(command)
    INFO.write_text(json.dumps({"found": found, "command_path": command, "version": version}, indent=2), encoding="utf-8")
    print({"figure_svg": str(SVG), "pdf_exists": PDF.exists(), "png_exists": PNG.exists(), "inkscape": found})


if __name__ == "__main__":
    main()
