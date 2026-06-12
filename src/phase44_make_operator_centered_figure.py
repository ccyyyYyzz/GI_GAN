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
OUT = ROOT / "outputs_phase44_operator_centered"
FIG_DIR = OUT / "figures"

SVG = FIG_DIR / "fig1_operator_centered_v44.svg"
PDF = FIG_DIR / "fig1_operator_centered_v44.pdf"
PNG = FIG_DIR / "fig1_operator_centered_v44_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"
SAMPLE_REPORT = OUT / "FIGURE1_SAMPLE_REPORT.json"

WIDTH = 1800
HEIGHT = 1000
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
    *,
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


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = STROKE, r: int = 16, sw: float = 2) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
    )


def marker_defs() -> str:
    markers = []
    for name, color in [("blue", BLUE), ("orange", ORANGE), ("purple", PURPLE), ("green", GREEN), ("muted", MUTED)]:
        markers.append(
            f'<marker id="arrow-{name}" markerWidth="13" markerHeight="13" refX="10" refY="6.5" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 13 6.5 L 0 13 z" fill="{color}"/></marker>'
        )
    return "\n".join(markers)


def marker_name(color: str) -> str:
    return {BLUE: "blue", ORANGE: "orange", PURPLE: "purple", GREEN: "green", MUTED: "muted"}.get(color, "muted")


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, sw: float = 3.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" '
        f'marker-end="url(#arrow-{marker_name(color)})"{dash}/>'
    )


def curve(path_d: str, color: str = MUTED, sw: float = 3.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow-{marker_name(color)})"{dash}/>'
    )


def image_box(x: float, y: float, w: float, h: float, path: Path, label: str | None, stroke: str) -> str:
    parts = [rect(x, y, w, h, "#ffffff", stroke, r=10, sw=1.6)]
    if path.exists():
        parts.append(
            f'<image x="{x + 6:.1f}" y="{y + 6:.1f}" width="{w - 12:.1f}" height="{h - 12:.1f}" '
            f'preserveAspectRatio="xMidYMid meet" href="{href(path)}"/>'
        )
    else:
        parts.append(text(x + w / 2, y + h / 2 + 5, "missing", size=18, fill=MUTED, weight="700"))
    if label:
        parts.append(text(x + w / 2, y + h + 22, label, size=18, fill=MUTED, weight="700"))
    return "\n".join(parts)


def role_panel(x: float, y: float, w: float, h: float, title: str, color: str, fill: str) -> str:
    return "\n".join(
        [
            rect(x, y, w, h, fill, color, r=22, sw=2.8),
            text(x + 24, y + 36, title, size=24, fill=color, weight="700", anchor="start"),
        ]
    )


def lock_icon(x: float, y: float, label: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x:.1f}" y="{y + 14:.1f}" width="48" height="34" rx="7" fill="#ffffff" stroke="{PURPLE}" stroke-width="2.8"/>',
            f'<path d="M{x + 11:.1f},{y + 16:.1f} v-8 c0,-20 26,-20 26,0 v8" fill="none" stroke="{PURPLE}" stroke-width="3.5" stroke-linecap="round"/>',
            text(x + 24, y + 70, label, size=18, fill=PURPLE, weight="700"),
        ]
    )


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        marker_defs(),
        "</defs>",
        "<metadata>Phase 44 operator-centered Figure 1. One representative sample; no metric badges or measurement-family comparison.</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(WIDTH / 2, 42, "Physics-gated and audited neural completion circuit", size=36, fill=TEXT, weight="700"),
        text(
            WIDTH / 2,
            76,
            "One calibrated operator anchors the data, gates the proposal, and audits the output.",
            size=22,
            fill=MUTED,
            weight="700",
        ),
    ]

    # Compact input strip.
    strip_x, strip_y, strip_w, strip_h = 92, 105, 1616, 118
    parts.append(rect(strip_x, strip_y, strip_w, strip_h, "#ffffff", BLUE, r=20, sw=2.2))
    parts.append(text(strip_x + 28, strip_y + 36, "known sensing and bucket vector", size=22, fill=BLUE, weight="700", anchor="start"))
    parts.append(image_box(strip_x + 390, strip_y + 20, 160, 64, COMP / "pattern_preview.png", "patterns A", BLUE))
    parts.append(image_box(strip_x + 594, strip_y + 20, 225, 64, COMP / "bucket_vector.png", "bucket y", BLUE))
    parts.append(text(strip_x + 1010, strip_y + 62, "y = A x + eps", size=27, fill=TEXT, weight="700"))
    parts.append(text(strip_x + 1374, strip_y + 54, ["test-time evidence:", "the same A and y"], size=20, fill=TEXT, weight="700"))

    # Central inverse core.
    core_x, core_y, core_w, core_h = 510, 270, 780, 154
    parts.append(rect(core_x, core_y, core_w, core_h, BLUE_FILL, BLUE, r=30, sw=3.6))
    parts.append(text(core_x + core_w / 2, core_y + 42, "calibrated inverse core", size=28, fill=BLUE, weight="700"))
    parts.append(text(core_x + core_w / 2, core_y + 88, "B_lambda = A^T(AA^T + lambda I)^-1", size=30, fill=TEXT, weight="700"))
    parts.append(text(core_x + core_w / 2, core_y + 126, "one frozen physical operator reused in three roles", size=21, fill=BLUE, weight="700"))

    panel_y, panel_h = 510, 280
    anchor_x, anchor_w = 72, 405
    gate_x, gate_w = 515, 620
    audit_x, audit_w = 1173, 555
    parts.append(role_panel(anchor_x, panel_y, anchor_w, panel_h, "1. Anchor", BLUE, BLUE_FILL))
    parts.append(text(anchor_x + anchor_w / 2, panel_y + 76, "x_data = B_lambda y", size=23, fill=TEXT, weight="700"))
    parts.append(image_box(anchor_x + 128, panel_y + 96, 150, 126, COMP / "x_data.png", None, BLUE))
    parts.append(text(anchor_x + anchor_w / 2, panel_y + 246, "measured component", size=18, fill=MUTED, weight="700"))
    parts.append(text(anchor_x + anchor_w / 2, panel_y + 268, "bucket-tied but incomplete", size=18, fill=BLUE, weight="700"))

    parts.append(role_panel(gate_x, panel_y, gate_w, panel_h, "2. Gate", PURPLE, PURPLE_FILL))
    parts.append(text(gate_x + 153, panel_y + 76, "r_theta = G_theta(x_data,z)", size=21, fill=ORANGE, weight="700"))
    parts.append(image_box(gate_x + 40, panel_y + 103, 122, 103, COMP / "x_data.png", None, BLUE))
    parts.append(image_box(gate_x + 212, panel_y + 103, 122, 103, COMP / "raw_residual.png", None, ORANGE))
    parts.append(arrow(gate_x + 166, panel_y + 157, gate_x + 204, panel_y + 157, ORANGE, 3))
    parts.append(rect(gate_x + 386, panel_y + 102, 178, 52, "#ffffff", PURPLE, r=14, sw=2.5))
    parts.append(text(gate_x + 475, panel_y + 134, "P_N = I - B_lambda A", size=18, fill=PURPLE, weight="700"))
    parts.append(image_box(gate_x + 414, panel_y + 174, 122, 66, COMP / "filtered_residual.png", None, PURPLE))
    parts.append(arrow(gate_x + 337, panel_y + 157, gate_x + 379, panel_y + 157, PURPLE, 3))
    parts.append(text(gate_x + 101, panel_y + 230, "anchor", size=18, fill=MUTED, weight="700"))
    parts.append(text(gate_x + 273, panel_y + 230, "proposal", size=18, fill=ORANGE, weight="700"))
    parts.append(text(gate_x + 475, panel_y + 257, "admitted by the gate", size=18, fill=PURPLE, weight="700"))

    parts.append(role_panel(audit_x, panel_y, audit_w, panel_h, "3. Audit", GREEN, GREEN_FILL))
    parts.append(text(audit_x + 160, panel_y + 76, "x_tilde = x_data + r_N", size=21, fill=TEXT, weight="700"))
    parts.append(image_box(audit_x + 36, panel_y + 101, 116, 103, COMP / "pre_audit.png", "candidate", ORANGE))
    parts.append(rect(audit_x + 184, panel_y + 96, 146, 44, "#ffffff", GREEN, r=12, sw=2.2))
    parts.append(text(audit_x + 257, panel_y + 124, "A x_tilde", size=18, fill=GREEN, weight="700"))
    parts.append(rect(audit_x + 184, panel_y + 159, 146, 44, "#ffffff", GREEN, r=12, sw=2.2))
    parts.append(text(audit_x + 257, panel_y + 187, "A x_tilde - y", size=18, fill=GREEN, weight="700"))
    parts.append(rect(audit_x + 184, panel_y + 222, 146, 44, "#ffffff", BLUE, r=12, sw=2.2))
    parts.append(text(audit_x + 257, panel_y + 250, "B_lambda error", size=18, fill=BLUE, weight="700"))
    parts.append(image_box(audit_x + 381, panel_y + 101, 110, 98, COMP / "final_audited.png", "final", GREEN))
    parts.append(image_box(audit_x + 498, panel_y + 101, 44, 44, COMP / "abs_error_final.png", "error", GREEN))
    parts.append(arrow(audit_x + 154, panel_y + 152, audit_x + 178, panel_y + 118, GREEN, 2.8))
    parts.append(arrow(audit_x + 257, panel_y + 142, audit_x + 257, panel_y + 154, GREEN, 2.8))
    parts.append(arrow(audit_x + 257, panel_y + 204, audit_x + 257, panel_y + 217, BLUE, 2.8))
    parts.append(arrow(audit_x + 333, panel_y + 242, audit_x + 376, panel_y + 150, GREEN, 2.8))
    parts.append(text(audit_x + audit_w / 2, panel_y + 264, "corrected by bucket audit", size=18, fill=GREEN, weight="700"))

    parts.append(curve(f"M {core_x + 140},{core_y + core_h} C 420,468 314,475 {anchor_x + anchor_w / 2},{panel_y - 10}", BLUE, 3.8))
    parts.append(curve(f"M {core_x + core_w / 2},{core_y + core_h} C 875,466 820,476 {gate_x + gate_w / 2},{panel_y - 10}", PURPLE, 3.8))
    parts.append(curve(f"M {core_x + core_w - 140},{core_y + core_h} C 1385,468 1430,476 {audit_x + audit_w / 2},{panel_y - 10}", GREEN, 3.8))

    # Training inset.
    inset_x, inset_y, inset_w, inset_h = 215, 840, 1370, 116
    parts.append(rect(inset_x, inset_y, inset_w, inset_h, "#ffffff", STROKE, r=22, sw=2.2))
    parts.append(text(inset_x + 26, inset_y + 34, "training", size=23, fill=TEXT, weight="700", anchor="start"))
    parts.append(text(inset_x + 470, inset_y + 78, "losses pass through frozen physics", size=21, fill=TEXT, weight="700"))
    parts.append(rect(inset_x + 765, inset_y + 27, 210, 58, ORANGE_FILL, ORANGE, r=14, sw=2.4))
    parts.append(text(inset_x + 870, inset_y + 52, "G_theta / R_phi", size=20, fill=ORANGE, weight="700"))
    parts.append(text(inset_x + 870, inset_y + 76, "trainable", size=18, fill=ORANGE, weight="700"))
    parts.append(arrow(inset_x + 625, inset_y + 78, inset_x + 755, inset_y + 78, ORANGE, 2.8, dashed=True))
    for i, label in enumerate(["A", "B_lambda", "P_N", "Pi_y"]):
        parts.append(lock_icon(inset_x + 1036 + i * 72, inset_y + 32, label))
    parts.append(text(inset_x + 1152, inset_y + 31, "fixed", size=18, fill=PURPLE, weight="700"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def find_inkscape() -> tuple[bool, str | None, str | None]:
    candidates = []
    preferred = Path("C:/Program Files/Inkscape/bin/inkscape.com")
    if preferred.exists():
        candidates.append(str(preferred))
    found = shutil.which("inkscape")
    if found:
        candidates.append(found)
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
    required = [
        COMP / "pattern_preview.png",
        COMP / "bucket_vector.png",
        COMP / "x_data.png",
        COMP / "raw_residual.png",
        COMP / "filtered_residual.png",
        COMP / "pre_audit.png",
        COMP / "final_audited.png",
        COMP / "abs_error_final.png",
    ]
    missing = sorted(path.name for path in required if not path.exists())
    SAMPLE_REPORT.write_text(
        json.dumps(
            {
                "component_dir": str(COMP),
                "representative_sample": "one exported STL-10 evaluation example",
                "missing_components": missing,
                "figure_policy": "operator-centered; no Rad/Scr comparison, no PSNR/SSIM badges, no ablation bars",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    found, command, version = find_inkscape()
    if found and command:
        export_with_inkscape(command)
    INFO.write_text(json.dumps({"found": found, "command_path": command, "version": version}, indent=2), encoding="utf-8")
    print({"svg": str(SVG), "pdf": PDF.exists(), "png": PNG.exists(), "inkscape_found": found, "missing": missing})


if __name__ == "__main__":
    main()
