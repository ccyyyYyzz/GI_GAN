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
OUT = ROOT / "outputs_phase42_closed_loop_figure"
FIG_DIR = OUT / "figures"

SVG = FIG_DIR / "fig1_closed_loop_v42.svg"
PDF = FIG_DIR / "fig1_closed_loop_v42.pdf"
PNG = FIG_DIR / "fig1_closed_loop_v42_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"

WIDTH = 2200
HEIGHT = 1200
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"
TEXT = "#1f2328"
MUTED = "#5f6b7a"
STROKE = "#c9d1dc"
TEAL = "#0f766e"
TEAL_FILL = "#ecfeff"
BLUE = "#1f77b4"
BLUE_FILL = "#eff7ff"
ORANGE = "#d97904"
ORANGE_FILL = "#fff7ed"
PURPLE = "#7b61b8"
PURPLE_FILL = "#f4f0ff"
GREEN = "#238b45"
GREEN_FILL = "#effaf2"
RED = "#a23b3b"
RED_FILL = "#fff5f5"


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
    gap: float = 1.16,
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


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = STROKE, r: int = 18, sw: float = 2.4) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = MUTED,
    sw: float = 3.0,
    dashed: bool = False,
    marker: str = "arrow",
) -> str:
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" marker-end="url(#{marker})"{dash}/>'


def curve(path_d: str, color: str = MUTED, sw: float = 3.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    return f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'


def image_box(x: float, y: float, w: float, h: float, path: Path, label: str | None = None, stroke: str = STROKE) -> str:
    parts = [rect(x, y, w, h, "#ffffff", stroke, r=12, sw=1.8)]
    parts.append(
        f'<image x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" '
        f'preserveAspectRatio="xMidYMid meet" href="{href(path)}"/>'
    )
    if label:
        parts.append(text(x + w / 2, y + h + 22, label, size=18, fill=MUTED, weight="700"))
    return "\n".join(parts)


def state_box(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    image: Path,
    formula: str,
    note: str,
    stroke: str,
    fill: str,
    extra_image: Path | None = None,
) -> str:
    parts = [rect(x, y, w, h, fill, stroke, r=18, sw=2.6)]
    parts.append(text(x + w / 2, y + 34, title, size=22, fill=TEXT, weight="700"))
    if extra_image is None:
        parts.append(image_box(x + w / 2 - 72, y + 58, 144, 126, image, stroke=stroke))
    else:
        parts.append(image_box(x + 25, y + 58, 132, 126, image, "final", stroke=stroke))
        parts.append(image_box(x + 166, y + 58, 82, 82, extra_image, "error", stroke=stroke))
    parts.append(text(x + w / 2, y + h - 62, formula, size=18, fill=TEXT, weight="700"))
    parts.append(text(x + w / 2, y + h - 28, note, size=18, fill=stroke, weight="700"))
    return "\n".join(parts)


def lock_icon(x: float, y: float, label: str, color: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x:.1f}" y="{y + 16:.1f}" width="52" height="38" rx="8" fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            f'<path d="M{x + 12:.1f},{y + 18:.1f} v-8 c0,-22 28,-22 28,0 v8" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>',
            text(x + 26, y + 78, label, size=18, fill=color, weight="700"),
        ]
    )


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{MUTED}"/>',
        "</marker>",
        "</defs>",
        "<metadata>Closed-loop measurement-rail mechanism figure. One representative eval-only sample reused from exported components.</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(WIDTH / 2, 48, "Measurement-audited neural completion", size=42, fill=TEXT, weight="700"),
        text(WIDTH / 2, 82, "The bucket signal anchors, filters, and audits the neural reconstruction.", size=22, fill=MUTED, weight="700"),
    ]

    rail_x, rail_y, rail_w, rail_h = 80, 112, 2040, 185
    parts.append(rect(rail_x, rail_y, rail_w, rail_h, TEAL_FILL, TEAL, r=28, sw=3.2))
    parts.append(text(rail_x + 210, rail_y + 39, "Measurement evidence rail", size=28, fill=TEAL, weight="700"))
    parts.append(text(rail_x + 210, rail_y + 75, "Known A, bucket vector y", size=21, fill=TEXT, weight="700"))
    parts.append(image_box(rail_x + 460, rail_y + 26, 230, 120, COMP / "pattern_preview.png", "known patterns A", stroke=TEAL))
    parts.append(image_box(rail_x + 750, rail_y + 26, 320, 120, COMP / "bucket_vector.png", "bucket vector y", stroke=TEAL))
    parts.append(text(rail_x + 1295, rail_y + 78, "y = A x + eps", size=31, fill=TEXT, weight="700"))
    parts.append(text(rail_x + 1705, rail_y + 58, ["same measured signal", "used three times"], size=24, fill=TEAL, weight="700"))

    eng_x, eng_y, eng_w, eng_h = 110, 365, 1980, 385
    parts.append(rect(eng_x, eng_y, eng_w, eng_h, "#ffffff", STROKE, r=34, sw=3.0))
    parts.append(text(eng_x + 430, eng_y + 42, "Image completion engine", size=29, fill=TEXT, weight="700"))
    state_y = eng_y + 78
    state_w, state_h, gap = 330, 260, 37
    sx = [eng_x + 44 + i * (state_w + gap) for i in range(5)]
    states = [
        ("Measured anchor", COMP / "x_data.png", "x_data", "measurement-derived, incomplete", BLUE, BLUE_FILL, None),
        ("Neural proposal", COMP / "raw_residual.png", "r_theta = G_theta(x_data)", "raw residual", ORANGE, ORANGE_FILL, None),
        ("Residual filter", COMP / "filtered_residual.png", "r_N = P_N(r_theta)", "filtered residual", PURPLE, PURPLE_FILL, None),
        ("Pre-audit image", COMP / "pre_audit.png", "x_tilde = x_data + r_N", "completed, not audited", ORANGE, ORANGE_FILL, None),
        ("Final audited image", COMP / "final_audited.png", "x_hat = Pi_y(x_tilde)", "audited output", GREEN, GREEN_FILL, COMP / "abs_error_final.png"),
    ]
    for i, (title, img, formula, note, stroke, fill, extra) in enumerate(states):
        parts.append(state_box(sx[i], state_y, state_w, state_h, title, img, formula, note, stroke, fill, extra))
        if i < 4:
            parts.append(arrow(sx[i] + state_w + 8, state_y + 130, sx[i + 1] - 8, state_y + 130, MUTED, 3.3))

    # Pi_y as the audit operator on the final transition, not as a detached box.
    piy_x, piy_y = sx[3] + state_w + 14, state_y + 70
    parts.append(rect(piy_x, piy_y, 94, 78, GREEN_FILL, GREEN, r=18, sw=3))
    parts.append(text(piy_x + 47, piy_y + 34, "Pi_y", size=28, fill=GREEN, weight="700"))
    parts.append(text(piy_x + 47, piy_y + 61, "audit", size=18, fill=GREEN, weight="700"))

    # Measurement rail arrows: y used for anchor, audit, and measurement loss.
    parts.append(arrow(sx[0] + 165, rail_y + rail_h + 4, sx[0] + 165, state_y - 12, TEAL, 4.0))
    parts.append(text(sx[0] + 165, 338, "forms x_data", size=20, fill=TEAL, weight="700"))
    parts.append(arrow(piy_x + 48, rail_y + rail_h + 4, piy_x + 48, piy_y - 12, TEAL, 4.0))
    parts.append(text(piy_x + 48, 338, "audits x_hat", size=20, fill=TEAL, weight="700"))

    inset_x, inset_y = 1810, 760
    parts.append(rect(inset_x, inset_y, 350, 145, "#ffffff", GREEN, r=22, sw=2.5))
    parts.append(text(inset_x + 175, inset_y + 33, "RelMeasErr before vs after", size=21, fill=TEXT, weight="700"))
    parts.append(image_box(inset_x + 44, inset_y + 48, 260, 78, COMP / "relmeaserr_bar.png", stroke=GREEN))
    parts.append(arrow(inset_x + 175, rail_y + rail_h + 4, inset_x + 175, inset_y - 10, TEAL, 4.0))
    parts.append(text(inset_x + 175, 338, "checks A x_hat - y", size=20, fill=TEAL, weight="700"))

    loop_y = 905
    parts.append(text(560, loop_y - 28, "Training feedback loop", size=28, fill=TEXT, weight="700"))
    parts.append(rect(130, loop_y, 260, 118, "#ffffff", BLUE, r=22, sw=2.5))
    parts.append(text(260, loop_y + 37, "ground truth x", size=21, fill=TEXT, weight="700"))
    parts.append(text(260, loop_y + 66, "training only", size=19, fill=BLUE, weight="700"))
    parts.append(image_box(172, loop_y + 76, 78, 42, COMP / "ground_truth.png", stroke=BLUE))
    parts.append(image_box(272, loop_y + 76, 78, 42, COMP / "final_audited.png", stroke=GREEN))

    loss_x = 520
    parts.append(rect(loss_x, loop_y, 520, 140, RED_FILL, RED, r=24, sw=2.7))
    parts.append(text(loss_x + 260, loop_y + 40, "loss", size=25, fill=TEXT, weight="700"))
    parts.append(text(loss_x + 260, loop_y + 76, "L_img(x_hat, x) + L_meas(A x_hat, y)", size=22, fill=TEXT, weight="700"))
    parts.append(text(loss_x + 260, loop_y + 112, "image target + remeasured bucket signal", size=19, fill=RED, weight="700"))

    parts.append(rect(1145, loop_y, 315, 118, "#ffffff", PURPLE, r=22, sw=2.6))
    parts.append(text(1302, loop_y + 38, "fixed physics layers", size=22, fill=TEXT, weight="700"))
    parts.append(text(1302, loop_y + 76, "A, P_N, Pi_y", size=21, fill=PURPLE, weight="700"))
    parts.append(text(1302, loop_y + 104, "not learned", size=18, fill=PURPLE, weight="700"))

    parts.append(rect(1118, loop_y + 150, 372, 100, "#ffffff", PURPLE, r=22, sw=2.3))
    parts.append(lock_icon(1168, loop_y + 166, "A", PURPLE))
    parts.append(lock_icon(1278, loop_y + 166, "P_N", PURPLE))
    parts.append(lock_icon(1400, loop_y + 166, "Pi_y", PURPLE))

    train_x = 1545
    parts.append(rect(train_x, loop_y, 360, 118, "#ffffff", ORANGE, r=22, sw=2.7))
    parts.append(text(train_x + 180, loop_y + 40, "trainable modules", size=22, fill=TEXT, weight="700"))
    parts.append(text(train_x + 180, loop_y + 78, "G_theta / R_phi", size=23, fill=ORANGE, weight="700"))
    parts.append(text(train_x + 180, loop_y + 106, "updated by losses", size=18, fill=ORANGE, weight="700"))

    parts.append(arrow(390, loop_y + 59, loss_x - 12, loop_y + 59, RED, 3.2, dashed=True))
    parts.append(arrow(loss_x + 520 + 10, loop_y + 70, 1145 - 10, loop_y + 70, PURPLE, 3.2, dashed=True))
    parts.append(arrow(1460 + 10, loop_y + 70, train_x - 10, loop_y + 70, ORANGE, 3.2, dashed=True))
    parts.append(text(1512, loop_y - 18, "updates neural proposal", size=20, fill=ORANGE, weight="700"))

    parts.append(text(WIDTH / 2, 1182, "Loss backpropagates through fixed physics layers and updates only G_theta/R_phi; the measurement operator and projections stay fixed.", size=18, fill=TEXT, weight="700"))
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
    found, command, version = find_inkscape()
    if found and command:
        export_with_inkscape(command)
    INFO.write_text(
        json.dumps({"found": found, "command_path": command, "version": version}, indent=2),
        encoding="utf-8",
    )
    print({"svg": str(SVG), "pdf": PDF.exists(), "png": PNG.exists(), "inkscape_found": found})


if __name__ == "__main__":
    main()
