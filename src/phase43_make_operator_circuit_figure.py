from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE = ROOT / "outputs_phase41_inkscape_signal_trace"
COMP = BASE / "components" / "rad5"
OUT = ROOT / "outputs_phase43_operator_circuit"
FIG_DIR = OUT / "figures"

SVG = FIG_DIR / "fig1_operator_circuit_v43.svg"
PDF = FIG_DIR / "fig1_operator_circuit_v43.pdf"
PNG = FIG_DIR / "fig1_operator_circuit_v43_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"
SAMPLE_REPORT = OUT / "FIGURE1_SAMPLE_REPORT.json"

WIDTH = 2200
HEIGHT = 1250
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"
TEXT = "#1f2328"
MUTED = "#5f6b7a"
STROKE = "#c9d1dc"

BLUE = "#1f77b4"
BLUE_FILL = "#eff7ff"
ORANGE = "#d97904"
ORANGE_FILL = "#fff7ed"
PURPLE = "#7b61b8"
PURPLE_FILL = "#f4f0ff"
GREEN = "#238b45"
GREEN_FILL = "#effaf2"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def href(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def text(
    x: float,
    y: float,
    value: str | list[str],
    size: int = 20,
    fill: str = TEXT,
    weight: str = "400",
    anchor: str = "middle",
    gap: float = 1.14,
) -> str:
    lines = [value] if isinstance(value, str) else value
    spans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{size * gap:.1f}"
        spans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        + "".join(spans)
        + "</text>"
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = STROKE,
    r: int = 18,
    sw: float = 2.3,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
    )


def marker_defs() -> str:
    markers = []
    for name, color in [
        ("blue", BLUE),
        ("orange", ORANGE),
        ("purple", PURPLE),
        ("green", GREEN),
        ("muted", MUTED),
    ]:
        markers.append(
            f'<marker id="arrow-{name}" markerWidth="14" markerHeight="14" refX="11" refY="7" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 14 7 L 0 14 z" fill="{color}"/></marker>'
        )
    return "\n".join(markers)


def marker_name(color: str) -> str:
    return {
        BLUE: "blue",
        ORANGE: "orange",
        PURPLE: "purple",
        GREEN: "green",
        MUTED: "muted",
    }.get(color, "muted")


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = MUTED,
    sw: float = 3.0,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    marker = marker_name(color)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" '
        f'marker-end="url(#arrow-{marker})"{dash}/>'
    )


def curve(path_d: str, color: str = MUTED, sw: float = 3.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    marker = marker_name(color)
    return (
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow-{marker})"{dash}/>'
    )


def image_box(
    x: float,
    y: float,
    w: float,
    h: float,
    path: Path,
    label: str | None = None,
    stroke: str = STROKE,
) -> str:
    parts = [rect(x, y, w, h, "#ffffff", stroke, r=12, sw=1.7)]
    if path.exists():
        parts.append(
            f'<image x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" '
            f'preserveAspectRatio="xMidYMid meet" href="{href(path)}"/>'
        )
    else:
        parts.append(text(x + w / 2, y + h / 2 + 6, "not exported", size=22, fill=MUTED, weight="700"))
    if label:
        parts.append(text(x + w / 2, y + h + 23, label, size=18, fill=MUTED, weight="700"))
    return "\n".join(parts)


def panel(x: float, y: float, w: float, h: float, title: str, color: str, fill: str) -> str:
    return "\n".join(
        [
            rect(x, y, w, h, fill, color, r=28, sw=3.0),
            text(x + 28, y + 38, title, size=28, fill=color, weight="700", anchor="start"),
        ]
    )


def lock_icon(x: float, y: float, label: str, color: str = PURPLE) -> str:
    return "\n".join(
        [
            f'<rect x="{x:.1f}" y="{y + 16:.1f}" width="54" height="38" rx="8" fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            f'<path d="M{x + 12:.1f},{y + 18:.1f} v-9 c0,-23 30,-23 30,0 v9" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>',
            text(x + 27, y + 78, label, size=18, fill=color, weight="700"),
        ]
    )


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        marker_defs(),
        "</defs>",
        "<metadata>Operator-centered circuit figure. One representative eval-only sample reused from exported components.</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(
            WIDTH / 2,
            48,
            "One calibrated operator anchors, gates, and audits neural completion",
            size=42,
            fill=TEXT,
            weight="700",
        ),
        text(
            WIDTH / 2,
            84,
            "The network proposes missing structure; the known measurement operator decides what can remain.",
            size=23,
            fill=MUTED,
            weight="700",
        ),
    ]

    # Acquisition strip: intentionally small and secondary.
    strip_x, strip_y, strip_w, strip_h = 135, 115, 1930, 145
    parts.append(rect(strip_x, strip_y, strip_w, strip_h, "#ffffff", BLUE, r=24, sw=2.4))
    parts.append(text(strip_x + 26, strip_y + 37, "Acquisition strip", size=24, fill=BLUE, weight="700", anchor="start"))
    parts.append(image_box(strip_x + 330, strip_y + 22, 190, 88, COMP / "pattern_preview.png", "known patterns A", BLUE))
    parts.append(image_box(strip_x + 565, strip_y + 22, 270, 88, COMP / "bucket_vector.png", "bucket vector y", BLUE))
    parts.append(text(strip_x + 1055, strip_y + 63, "y = A x + eps", size=30, fill=TEXT, weight="700"))
    parts.append(text(strip_x + 1535, strip_y + 52, "Known A, measured y; test-time inputs are A,y.", size=24, fill=TEXT, weight="700"))

    # Central operator core.
    core_x, core_y, core_w, core_h = 675, 315, 850, 190
    parts.append(rect(core_x, core_y, core_w, core_h, BLUE_FILL, BLUE, r=38, sw=4.0))
    parts.append(text(core_x + core_w / 2, core_y + 50, "calibrated physical inverse core", size=31, fill=BLUE, weight="700"))
    parts.append(text(core_x + core_w / 2, core_y + 98, "B_lambda = A^T(AA^T + lambda I)^-1", size=33, fill=TEXT, weight="700"))
    parts.append(text(core_x + core_w / 2, core_y + 142, "fixed, not learned", size=24, fill=BLUE, weight="700"))
    parts.append(
        text(
            core_x + core_w / 2,
            core_y + 176,
            "network proposes; calibrated operator anchors, gates, audits",
            size=20,
            fill=MUTED,
            weight="700",
        )
    )

    # Three operator branches.
    anchor_x, panel_y, anchor_w, panel_h = 95, 585, 480, 360
    gate_x, gate_w = 615, 735
    audit_x, audit_w = 1395, 710
    parts.append(panel(anchor_x, panel_y, anchor_w, panel_h, "1. Anchor", BLUE, BLUE_FILL))
    parts.append(text(anchor_x + 240, panel_y + 78, "x_data = B_lambda y", size=25, fill=TEXT, weight="700"))
    parts.append(image_box(anchor_x + 145, panel_y + 104, 190, 168, COMP / "x_data.png", "measured anchor", BLUE))
    parts.append(text(anchor_x + 240, panel_y + 313, "from buckets, incomplete", size=21, fill=BLUE, weight="700"))

    parts.append(panel(gate_x, panel_y, gate_w, panel_h, "2. Gate", PURPLE, PURPLE_FILL))
    parts.append(text(gate_x + 220, panel_y + 78, "r_theta = G_theta(x_data)", size=23, fill=ORANGE, weight="700"))
    parts.append(image_box(gate_x + 48, panel_y + 116, 160, 135, COMP / "x_data.png", "x_data", BLUE))
    parts.append(image_box(gate_x + 268, panel_y + 116, 160, 135, COMP / "raw_residual.png", "raw residual", ORANGE))
    parts.append(arrow(gate_x + 212, panel_y + 184, gate_x + 260, panel_y + 184, ORANGE, 3.2))
    parts.append(text(gate_x + 348, panel_y + 289, "G_theta proposes only", size=18, fill=ORANGE, weight="700"))
    parts.append(rect(gate_x + 475, panel_y + 111, 205, 58, "#ffffff", PURPLE, r=18, sw=2.5))
    parts.append(text(gate_x + 577, panel_y + 146, "P_N = I - B_lambda A", size=18, fill=PURPLE, weight="700"))
    parts.append(image_box(gate_x + 497, panel_y + 199, 160, 92, COMP / "filtered_residual.png", "filtered residual", PURPLE))
    parts.append(arrow(gate_x + 431, panel_y + 184, gate_x + 468, panel_y + 184, PURPLE, 3.2))
    parts.append(text(gate_x + 367, panel_y + 326, "proposal -> admissible residual", size=21, fill=TEXT, weight="700"))
    parts.append(text(gate_x + 367, panel_y + 352, "P_N removes the A-visible part", size=18, fill=PURPLE, weight="700"))

    parts.append(panel(audit_x, panel_y, audit_w, panel_h, "3. Audit", GREEN, GREEN_FILL))
    parts.append(text(audit_x + 190, panel_y + 78, "x_tilde = x_data + r_N", size=23, fill=TEXT, weight="700"))
    parts.append(image_box(audit_x + 42, panel_y + 112, 150, 132, COMP / "pre_audit.png", "pre-audit", ORANGE))
    parts.append(rect(audit_x + 232, panel_y + 96, 158, 48, "#ffffff", GREEN, r=15, sw=2.4))
    parts.append(text(audit_x + 311, panel_y + 126, "y_tilde = A x_tilde", size=16, fill=GREEN, weight="700"))
    parts.append(rect(audit_x + 232, panel_y + 175, 158, 48, "#ffffff", GREEN, r=15, sw=2.4))
    parts.append(text(audit_x + 311, panel_y + 205, "e_y = y_tilde - y", size=17, fill=GREEN, weight="700"))
    parts.append(rect(audit_x + 232, panel_y + 254, 158, 48, "#ffffff", BLUE, r=15, sw=2.4))
    parts.append(text(audit_x + 311, panel_y + 284, "B_lambda e_y", size=18, fill=BLUE, weight="700"))
    parts.append(image_box(audit_x + 438, panel_y + 112, 132, 118, COMP / "final_audited.png", "final", GREEN))
    parts.append(image_box(audit_x + 592, panel_y + 112, 75, 75, COMP / "abs_error_final.png", "error", GREEN))
    parts.append(image_box(audit_x + 430, panel_y + 258, 230, 64, COMP / "relmeaserr_bar.png", None, GREEN))
    parts.append(arrow(audit_x + 196, panel_y + 178, audit_x + 225, panel_y + 120, GREEN, 3.0))
    parts.append(arrow(audit_x + 311, panel_y + 147, audit_x + 311, panel_y + 169, GREEN, 3.0))
    parts.append(arrow(audit_x + 311, panel_y + 226, audit_x + 311, panel_y + 248, BLUE, 3.0))
    parts.append(arrow(audit_x + 394, panel_y + 278, audit_x + 430, panel_y + 168, GREEN, 3.0))
    parts.append(text(audit_x + 545, panel_y + 249, "RelMeasErr before vs after", size=17, fill=GREEN, weight="700"))
    parts.append(text(audit_x + 545, panel_y + 326, "remeasure -> compare -> correct", size=20, fill=GREEN, weight="700"))
    parts.append(text(audit_x + 545, panel_y + 350, "final image must pass the bucket audit", size=17, fill=GREEN, weight="700"))

    # Core branches. They stop at panels to avoid visual clutter inside the branch details.
    parts.append(curve(f"M {core_x + 125:.1f},{core_y + core_h:.1f} C 500,542 350,540 {anchor_x + 240:.1f},{panel_y - 12:.1f}", BLUE, 4.0))
    parts.append(curve(f"M {core_x + core_w / 2:.1f},{core_y + core_h:.1f} C 1100,528 995,540 {gate_x + gate_w / 2:.1f},{panel_y - 12:.1f}", PURPLE, 4.0))
    parts.append(curve(f"M {core_x + core_w - 125:.1f},{core_y + core_h:.1f} C 1660,537 1755,540 {audit_x + 355:.1f},{panel_y - 12:.1f}", GREEN, 4.0))

    # Compact training inset.
    inset_x, inset_y, inset_w, inset_h = 310, 990, 1580, 170
    parts.append(rect(inset_x, inset_y, inset_w, inset_h, "#ffffff", STROKE, r=28, sw=2.5))
    parts.append(text(inset_x + 32, inset_y + 42, "Training only", size=27, fill=TEXT, weight="700", anchor="start"))
    parts.append(
        text(
            inset_x + 520,
            inset_y + 67,
            "L = L_img(x_hat, x) + alpha ||A x_hat - y||",
            size=26,
            fill=TEXT,
            weight="700",
        )
    )
    parts.append(rect(inset_x + 975, inset_y + 32, 260, 78, ORANGE_FILL, ORANGE, r=18, sw=2.6))
    parts.append(text(inset_x + 1105, inset_y + 66, "G_theta / R_phi", size=23, fill=ORANGE, weight="700"))
    parts.append(text(inset_x + 1105, inset_y + 94, "trainable", size=19, fill=ORANGE, weight="700"))
    parts.append(arrow(inset_x + 780, inset_y + 68, inset_x + 965, inset_y + 68, ORANGE, 3.0, dashed=True))
    parts.append(text(inset_x + 880, inset_y + 40, "loss -> updates neural modules only", size=18, fill=ORANGE, weight="700"))
    lock_y = inset_y + 52
    parts.append(lock_icon(inset_x + 1320, lock_y, "A", PURPLE))
    parts.append(lock_icon(inset_x + 1402, lock_y, "B_lambda", PURPLE))
    parts.append(lock_icon(inset_x + 1495, lock_y, "P_N", PURPLE))
    parts.append(lock_icon(inset_x + 1577, lock_y, "Pi_y", PURPLE))
    parts.append(text(inset_x + 1477, inset_y + 36, "fixed", size=20, fill=PURPLE, weight="700"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def find_inkscape() -> tuple[bool, str | None, str | None]:
    candidates = []
    preferred = Path("C:/Program Files/Inkscape/bin/inkscape.com")
    if preferred.exists():
        candidates.append(str(preferred))
    which = shutil.which("inkscape")
    if which:
        candidates.append(which)
    fallback = Path("D:/bin/inkscape.COM")
    if fallback.exists():
        candidates.append(str(fallback))
    for candidate in dict.fromkeys(candidates):
        try:
            proc = subprocess.run([candidate, "--version"], check=False, capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        if proc.returncode == 0:
            return True, candidate, (proc.stdout or proc.stderr).strip()
    return False, candidates[0] if candidates else None, None


def export_with_inkscape(command: str) -> None:
    subprocess.run([command, str(SVG), "--export-type=pdf", "--export-filename", str(PDF)], check=True)
    subprocess.run([command, str(SVG), "--export-type=png", "--export-dpi=600", "--export-filename", str(PNG)], check=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SVG.write_text(build_svg(), encoding="utf-8")
    missing = sorted(
        path.name
        for path in [
            COMP / "x_data.png",
            COMP / "raw_residual.png",
            COMP / "filtered_residual.png",
            COMP / "pre_audit.png",
            COMP / "final_audited.png",
            COMP / "abs_error_final.png",
            COMP / "bucket_vector.png",
            COMP / "relmeaserr_bar.png",
        ]
        if not path.exists()
    )
    SAMPLE_REPORT.write_text(
        json.dumps({"component_dir": str(COMP), "sample": "rad5", "missing_components": missing}, indent=2),
        encoding="utf-8",
    )
    found, command, version = find_inkscape()
    if found and command:
        export_with_inkscape(command)
    INFO.write_text(
        json.dumps({"found": found, "command_path": command, "version": version}, indent=2),
        encoding="utf-8",
    )
    print({"svg": str(SVG), "pdf": PDF.exists(), "png": PNG.exists(), "inkscape_found": found, "missing": missing})


if __name__ == "__main__":
    main()
