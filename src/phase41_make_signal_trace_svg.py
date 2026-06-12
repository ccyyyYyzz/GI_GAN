from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase41_inkscape_signal_trace"
COMP = OUT / "components"
FIG_DIR = OUT / "figures"
EDIT_PACK = OUT / "figure1_edit_pack"

SVG = FIG_DIR / "fig1_signal_trace_v41.svg"
PDF = FIG_DIR / "fig1_signal_trace_v41.pdf"
PNG = FIG_DIR / "fig1_signal_trace_v41_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"

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
SOFT = "#f8fafc"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def image_href(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def text(
    x: float,
    y: float,
    content: str | list[str],
    size: int = 20,
    fill: str = TEXT,
    weight: str = "400",
    anchor: str = "middle",
    gap: float = 1.18,
    style: str = "",
) -> str:
    lines = [content] if isinstance(content, str) else content
    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{size * gap:.1f}"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" {style}>'
        + "".join(tspans)
        + "</text>"
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = STROKE, r: int = 18, sw: float = 2.5) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, sw: float = 3.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round" marker-end="url(#arrow)"{dash}/>'


def image_box(x: float, y: float, w: float, h: float, path: Path, label: str | None = None, stroke: str = STROKE) -> str:
    parts = [rect(x, y, w, h, "#ffffff", stroke, r=12, sw=1.8)]
    parts.append(
        f'<image x="{x + 6:.1f}" y="{y + 6:.1f}" width="{w - 12:.1f}" height="{h - 12:.1f}" '
        f'preserveAspectRatio="xMidYMid meet" href="{image_href(path)}"/>'
    )
    if label:
        parts.append(text(x + w / 2, y + h + 20, label, size=16, fill=MUTED, weight="700"))
    return "\n".join(parts)


def node(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    formula: str | list[str],
    note: str | list[str],
    fill: str,
    stroke: str,
    images: list[tuple[Path, float, float, float, float, str | None]] | None = None,
) -> str:
    parts = [rect(x, y, w, h, fill, stroke, r=22, sw=3)]
    parts.append(text(x + w / 2, y + 36, title, size=24, fill=TEXT, weight="700"))
    if images:
        for p, ix, iy, iw, ih, label in images:
            parts.append(image_box(x + ix, y + iy, iw, ih, p, label=label, stroke=stroke))
    parts.append(text(x + w / 2, y + h - 70, formula, size=17, fill=TEXT, weight="700", gap=1.08))
    parts.append(text(x + w / 2, y + h - 30, note, size=16, fill=stroke, weight="700", gap=1.12))
    return "\n".join(parts)


def build_svg() -> str:
    r = COMP / "rad5"
    s = COMP / "scr5"
    node_y = 115
    node_w = 286
    node_h = 375
    gap = 22
    x0 = 42
    xs = [x0 + i * (node_w + gap) for i in range(7)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{MUTED}"/>',
        "</marker>",
        "</defs>",
        "<metadata>Phase 41 signal-trace mechanism figure; real eval-only intermediate images; SVG text editable in Inkscape.</metadata>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(WIDTH / 2, 56, "Signal trace of measurement-audited neural completion", size=38, fill=TEXT, weight="700"),
        text(WIDTH / 2, 88, "Forward signal path", size=20, fill=MUTED, weight="700"),
    ]

    parts.append(
        node(
            xs[0],
            node_y,
            node_w,
            node_h,
            "A. Measurement",
            "y = A x + eps",
            ["known patterns", "scalar buckets"],
            BLUE_FILL,
            BLUE,
            [
                (r / "pattern_preview.png", 18, 62, 146, 116, "patterns"),
                (r / "ground_truth.png", 180, 62, 78, 78, "x"),
                (r / "bucket_vector.png", 18, 205, 240, 92, "bucket y"),
            ],
        )
    )
    parts.append(
        node(
            xs[1],
            node_y,
            node_w,
            node_h,
            "B. Data anchor",
            ["x_data = A^T", "(AA^T + lambda I)^-1 y"],
            ["measurement-derived", "incomplete"],
            BLUE_FILL,
            BLUE,
            [(r / "x_data.png", 63, 72, 160, 160, "x_data")],
        )
    )
    parts.append(
        node(
            xs[2],
            node_y,
            node_w,
            node_h,
            "C. Residual proposal",
            "r_theta = G_theta(x_data)",
            ["proposal", "not final image"],
            ORANGE_FILL,
            ORANGE,
            [(r / "raw_residual.png", 63, 72, 160, 160, "raw residual")],
        )
    )
    parts.append(
        node(
            xs[3],
            node_y,
            node_w,
            node_h,
            "D. Residual filter",
            "r_N = P_N(r_theta)",
            ["remove A-visible", "residual"],
            PURPLE_FILL,
            PURPLE,
            [(r / "filtered_residual.png", 63, 72, 160, 160, "filtered residual")],
        )
    )
    parts.append(
        node(
            xs[4],
            node_y,
            node_w,
            node_h,
            "E. Pre-audit image",
            "x_tilde = x_data + r_N",
            ["completed", "not audited"],
            ORANGE_FILL,
            ORANGE,
            [(r / "pre_audit.png", 63, 72, 160, 160, "x_tilde")],
        )
    )
    parts.append(
        node(
            xs[5],
            node_y,
            node_w,
            node_h,
            "F. Bucket audit",
            "Pi_y:  A x_hat approx y",
            ["remeasure", "correct discrepancy"],
            GREEN_FILL,
            GREEN,
            [
                (r / "measurement_residual_pre.png", 20, 64, 116, 82, "pre"),
                (r / "measurement_residual_post.png", 150, 64, 116, 82, "post"),
                (r / "relmeaserr_bar.png", 28, 174, 230, 96, "RelMeasErr"),
            ],
        )
    )
    parts.append(
        node(
            xs[6],
            node_y,
            node_w,
            node_h,
            "G. Final output",
            "x_hat = Pi_y(x_tilde)",
            ["measurement-audited", "reconstruction"],
            GREEN_FILL,
            GREEN,
            [
                (r / "final_audited.png", 24, 70, 135, 135, "final"),
                (r / "abs_error_final.png", 166, 70, 95, 95, "error"),
                (r / "psnr_ssim_bar.png", 28, 232, 230, 78, "sample metrics"),
            ],
        )
    )

    for i in range(6):
        parts.append(arrow(xs[i] + node_w + 3, node_y + node_h / 2, xs[i + 1] - 3, node_y + node_h / 2, MUTED, 3.2))

    # Real evidence mini-strip for both measurement families.
    strip_y = 530
    parts.append(text(WIDTH / 2, strip_y - 16, "Real intermediate images used in the signal trace", size=24, fill=TEXT, weight="700"))
    headers = ["GT", "x_data", "raw residual", "filtered residual", "pre-audit", "final", "RelMeasErr"]
    col_x = [210, 390, 570, 750, 930, 1110, 1290]
    for hx, label in zip(col_x, headers):
        parts.append(text(hx + 62, strip_y + 22, label, size=17, fill=MUTED, weight="700"))
    for row_idx, (folder, label) in enumerate([(r, "Rad-5"), (s, "Scr-5")]):
        y = strip_y + 45 + row_idx * 130
        parts.append(text(70, y + 68, label, size=21, fill=TEXT, weight="700", anchor="start"))
        img_names = [
            "ground_truth.png",
            "x_data.png",
            "raw_residual.png",
            "filtered_residual.png",
            "pre_audit.png",
            "final_audited.png",
        ]
        for x, name in zip(col_x, img_names):
            parts.append(image_box(x, y, 124, 100, folder / name))
        parts.append(image_box(col_x[-1], y, 210, 100, folder / "relmeaserr_bar.png"))

    # Badges and training feedback path.
    parts.append(rect(1540, 562, 610, 92, "#ffffff", STROKE, r=20, sw=2.3))
    parts.append(text(1845, 594, "STL-10 5% final quality", size=21, fill=TEXT, weight="700"))
    parts.append(text(1845, 630, "Rad 22.316 / 0.635   |   Scr 22.271 / 0.632", size=20, fill=GREEN, weight="700"))

    parts.append(rect(1540, 685, 610, 104, "#ffffff", STROKE, r=20, sw=2.3))
    parts.append(text(1845, 718, "Audit ablation, inference only", size=21, fill=TEXT, weight="700"))
    parts.append(text(1845, 756, "Rad-5 Full 22.202 vs -MC 19.399", size=18, fill=ORANGE, weight="700"))
    parts.append(text(1845, 782, "Scr-5 Full 22.155 vs -MC 6.352", size=18, fill=ORANGE, weight="700"))

    feedback_y = 900
    parts.append(text(WIDTH / 2, feedback_y - 25, "Training feedback path", size=24, fill=TEXT, weight="700"))
    parts.append(rect(210, feedback_y, 260, 115, "#ffffff", BLUE, r=20, sw=2.4))
    parts.append(text(340, feedback_y + 42, "ground truth x", size=21, fill=TEXT, weight="700"))
    parts.append(image_box(242, feedback_y + 56, 66, 48, r / "ground_truth.png"))
    parts.append(image_box(318, feedback_y + 56, 66, 48, r / "final_audited.png"))

    parts.append(rect(580, feedback_y, 270, 115, "#ffffff", GREEN, r=20, sw=2.4))
    parts.append(text(715, feedback_y + 45, "image loss", size=22, fill=TEXT, weight="700"))
    parts.append(text(715, feedback_y + 80, "L_img(x_hat, x)", size=18, fill=MUTED, weight="700"))

    parts.append(rect(960, feedback_y, 305, 115, "#ffffff", GREEN, r=20, sw=2.4))
    parts.append(text(1112, feedback_y + 45, "measurement loss", size=22, fill=TEXT, weight="700"))
    parts.append(text(1112, feedback_y + 80, "L_meas = ||A x_hat - y||", size=18, fill=MUTED, weight="700"))
    parts.append(image_box(1186, feedback_y + 62, 64, 38, r / "bucket_vector.png"))

    parts.append(rect(1390, feedback_y, 360, 115, "#ffffff", PURPLE, r=20, sw=2.4))
    parts.append(text(1570, feedback_y + 38, "fixed differentiable physics", size=21, fill=TEXT, weight="700"))
    parts.append(text(1570, feedback_y + 75, "A, P_N, Pi_y do not update", size=18, fill=MUTED, weight="700"))

    parts.append(rect(1840, feedback_y, 260, 115, "#ffffff", ORANGE, r=20, sw=2.4))
    parts.append(text(1970, feedback_y + 42, "trainable modules", size=21, fill=TEXT, weight="700"))
    parts.append(text(1970, feedback_y + 80, "G_theta and R_phi", size=18, fill=MUTED, weight="700"))

    parts.append(arrow(470, feedback_y + 58, 580, feedback_y + 58, GREEN, 3, dashed=True))
    parts.append(arrow(850, feedback_y + 58, 960, feedback_y + 58, GREEN, 3, dashed=True))
    parts.append(arrow(1265, feedback_y + 58, 1390, feedback_y + 58, PURPLE, 3, dashed=True))
    parts.append(arrow(1750, feedback_y + 58, 1840, feedback_y + 58, ORANGE, 3, dashed=True))
    parts.append(text(WIDTH / 2, 1068, "Loss backpropagates through fixed physics layers; the next batch updates the residual proposal, not the measurement operator.", size=21, fill=TEXT, weight="700"))
    parts.append(text(WIDTH / 2, 1110, "Training: x is known to generate y and compute losses. Inference: only A and y are used.", size=19, fill=MUTED, weight="700"))

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
            proc = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=20, check=False)
        except Exception:
            continue
        if proc.returncode == 0:
            return True, candidate, (proc.stdout or proc.stderr).strip()
    return False, candidates[0] if candidates else None, None


def export_with_inkscape(command: str) -> None:
    subprocess.run([command, str(SVG), "--export-type=pdf", "--export-filename", str(PDF)], check=True)
    subprocess.run([command, str(SVG), "--export-type=png", "--export-dpi=600", "--export-filename", str(PNG)], check=True)


def copy_edit_pack() -> None:
    if EDIT_PACK.exists():
        shutil.rmtree(EDIT_PACK)
    EDIT_PACK.mkdir(parents=True, exist_ok=True)
    for path in [SVG, PDF, PNG]:
        if path.exists():
            shutil.copy2(path, EDIT_PACK / path.name)
    all_png = EDIT_PACK / "component_pngs"
    all_png.mkdir(parents=True, exist_ok=True)
    for src in COMP.rglob("*.png"):
        dst = all_png / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (EDIT_PACK / "FIGURE1_EDITING_GUIDE.md").write_text(
        "\n".join(
            [
                "# Figure 1 Editing Guide",
                "",
                "Open `fig1_signal_trace_v41.svg` in `C:\\Program Files\\Inkscape\\bin\\inkscape.exe`.",
                "",
                "Manual adjustment priorities:",
                "1. Enlarge real image blocks if a target journal compresses the figure.",
                "2. Shorten text labels before reducing image sizes.",
                "3. Keep the main arrow path visible from measurement to final output.",
                "4. Keep the training feedback path below the main path and dashed.",
                "5. Keep `P_N` visually distinct from `Pi_y`.",
                "6. Keep RelMeasErr pre/post visible.",
                "7. Avoid repeating the title in the caption.",
                "8. Export PDF and 600dpi PNG after manual edits.",
                "",
                "Commands:",
                '& "C:\\Program Files\\Inkscape\\bin\\inkscape.com" fig1_signal_trace_v41.svg --export-type=pdf --export-filename=fig1_signal_trace_v41.pdf',
                '& "C:\\Program Files\\Inkscape\\bin\\inkscape.com" fig1_signal_trace_v41.svg --export-type=png --export-dpi=600 --export-filename=fig1_signal_trace_v41_600dpi.png',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (EDIT_PACK / "FIGURE1_STYLE_GUIDE.md").write_text(
        "\n".join(
            [
                "# Figure 1 Style Guide",
                "",
                "- Canvas: 2200 x 1250 px, white background.",
                "- Fonts: Arial / Helvetica / DejaVu Sans.",
                "- Main colors: blue = measurement/anchor, orange = neural proposal, purple = residual filter, green = audit/final.",
                "- Keep text as SVG text whenever possible.",
                "- Use Inkscape-exported PDF in LaTeX; keep PNG only for review and external previews.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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
    copy_edit_pack()
    print({"svg": str(SVG), "pdf": PDF.exists(), "png": PNG.exists(), "inkscape_found": found})


if __name__ == "__main__":
    main()
