from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase39_anchor_proposal_audit"
COMP = OUT / "mechanism_components"
IMG_DIR = COMP / "component_images"

PHASE15 = ROOT / "outputs_phase15"
PHASE14 = ROOT / "outputs_phase14"
PHASE16 = ROOT / "outputs_phase16" / "supplementary_experiments"

GRID_SOURCES = {
    "rad5": PHASE15 / "imported_noleak" / "rademacher5_hq_noise001_colab" / "eval_samples" / "recon_grid.png",
    "scr5": PHASE15 / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "eval_samples" / "recon_grid.png",
}
METRIC_SOURCES = {
    "rad5": PHASE15 / "imported_noleak" / "rademacher5_hq_noise001_colab" / "eval_metrics.json",
    "scr5": PHASE15 / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "eval_metrics.json",
}
NO_MC_IMAGE_DIRS = {
    "rad5": PHASE14 / "ablation_pack" / "inference_eval" / "stl10_rademacher5_colab_full" / "no_dc_project_inference" / "eval_samples_individual",
    "scr5": PHASE14 / "ablation_pack" / "inference_eval" / "stl10_scrambled5_colab_full" / "no_dc_project_inference" / "eval_samples_individual",
}
ABLATION_CSV = PHASE16 / "inference_ablation" / "real_inference_ablation_results.csv"

COLS = [(29, 369), (396, 736), (764, 1104), (1131, 1471)]
ROWS = [(100, 440), (463, 803), (825, 1165), (1188, 1528), (1550, 1890), (1913, 2253), (2275, 2615), (2638, 2977)]
SAMPLE_INDEX = 2

METHOD_META = {
    "rad5": {
        "label": "Rad-5",
        "method_id": "rademacher5_hq_noise001_colab",
        "family": "Rademacher",
        "sampling": "5%",
        "exact_a": "safe exact-A cache rebuild path required and available",
        "main_psnr": 22.316,
        "main_ssim": 0.635,
        "table3_full": 22.202,
        "table3_no_mc": 19.399,
    },
    "scr5": {
        "label": "Scr-5",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "family": "Scrambled Hadamard",
        "sampling": "5%",
        "exact_a": "deterministic structured operator; exact-A file not required",
        "main_psnr": 22.271,
        "main_ssim": 0.632,
        "table3_full": 22.155,
        "table3_no_mc": 6.352,
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ablation_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with ABLATION_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["method_id"], row["ablation_mode"])] = row
    return rows


def crop_grid_cell(grid_path: Path, row_idx: int, col_idx: int, out_path: Path) -> None:
    im = Image.open(grid_path).convert("RGB")
    x0, x1 = COLS[col_idx]
    y0, y1 = ROWS[row_idx]
    crop = im.crop((x0, y0, x1, y1)).resize((256, 256), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)


def normalize_existing_png(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB").resize((256, 256), Image.Resampling.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)


def make_unavailable_tile(path: Path, title: str) -> None:
    im = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((14, 14, 242, 242), radius=18, outline=(205, 213, 224), width=4, fill=(248, 250, 252))
    draw.text((128, 96), title, anchor="mm", fill=(31, 35, 40))
    draw.text((128, 134), "not directly exported", anchor="mm", fill=(95, 107, 122))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def make_residual_bar(path: Path, pre: float, final: float, label: str) -> None:
    im = Image.new("RGB", (360, 132), "white")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((0, 0, 359, 131), radius=14, outline=(205, 213, 224), width=2)
    draw.text((18, 18), label, fill=(31, 35, 40))
    max_v = max(pre, final, 1e-6)
    entries = [("without audit", pre, (217, 121, 4)), ("final audit", final, (35, 139, 69))]
    for i, (name, val, color) in enumerate(entries):
        y = 48 + i * 38
        draw.text((18, y + 7), name, fill=(95, 107, 122))
        draw.rounded_rectangle((135, y, 330, y + 20), radius=5, fill=(241, 245, 249))
        w = int(195 * (val / max_v))
        draw.rounded_rectangle((135, y, 135 + max(2, w), y + 20), radius=5, fill=color)
        draw.text((336, y + 10), f"{val:.3g}", anchor="lm", fill=(31, 35, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_method(method: str, ablation_rows: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    meta = METHOD_META[method]
    metrics = load_json(METRIC_SOURCES[method])
    final_row = ablation_rows[(meta["method_id"], "full_model")]
    no_mc_row = ablation_rows[(meta["method_id"], "no_dc_project")]

    grid = GRID_SOURCES[method]
    names = ["gt", "x_data", "final", "abs_error"]
    for col, name in enumerate(names):
        crop_grid_cell(grid, SAMPLE_INDEX, col, IMG_DIR / f"{method}_{name}.png")

    no_mc_src = NO_MC_IMAGE_DIRS[method] / f"sample_{SAMPLE_INDEX:03d}_recon.png"
    if no_mc_src.exists():
        normalize_existing_png(no_mc_src, IMG_DIR / f"{method}_pre_audit_no_mc.png")
        pre_available = True
    else:
        make_unavailable_tile(IMG_DIR / f"{method}_pre_audit_no_mc.png", "pre-audit / no-MC")
        pre_available = False

    make_unavailable_tile(IMG_DIR / f"{method}_raw_residual_unavailable.png", "raw residual")
    make_unavailable_tile(IMG_DIR / f"{method}_filtered_residual_unavailable.png", "filtered residual")
    make_residual_bar(
        IMG_DIR / f"{method}_measurement_residual_bar.png",
        float(no_mc_row["rel_meas_err"]),
        float(final_row["rel_meas_err"]),
        f"{meta['label']} RelMeasErr",
    )

    component_rows = [
        {
            "method": method,
            "display_label": meta["label"],
            "component": "ground_truth",
            "image": str(IMG_DIR / f"{method}_gt.png"),
            "psnr": "",
            "ssim": "",
            "rel_meas_err": "",
            "source": str(grid),
            "notes": "cropped real STL-10 sample from final no-leak evaluation grid",
        },
        {
            "method": method,
            "display_label": meta["label"],
            "component": "x_data_backprojection",
            "image": str(IMG_DIR / f"{method}_x_data.png"),
            "psnr": metrics["backprojection"]["psnr"],
            "ssim": metrics["backprojection"]["ssim"],
            "rel_meas_err": metrics["backprojection"]["rel_meas_error"],
            "source": str(grid),
            "notes": "measured anchor / BP image",
        },
        {
            "method": method,
            "display_label": meta["label"],
            "component": "pre_audit_no_mc",
            "image": str(IMG_DIR / f"{method}_pre_audit_no_mc.png"),
            "psnr": no_mc_row["psnr"],
            "ssim": no_mc_row["ssim"],
            "rel_meas_err": no_mc_row["rel_meas_err"],
            "source": str(no_mc_src),
            "notes": "inference ablation without final audit; not separately trained",
        },
        {
            "method": method,
            "display_label": meta["label"],
            "component": "final_audited_output",
            "image": str(IMG_DIR / f"{method}_final.png"),
            "psnr": metrics["model"]["psnr"],
            "ssim": metrics["model"]["ssim"],
            "rel_meas_err": metrics["model"]["rel_meas_error"],
            "source": str(grid),
            "notes": "final audited reconstruction from no-leak evaluation",
        },
        {
            "method": method,
            "display_label": meta["label"],
            "component": "absolute_error",
            "image": str(IMG_DIR / f"{method}_abs_error.png"),
            "psnr": "",
            "ssim": "",
            "rel_meas_err": "",
            "source": str(grid),
            "notes": "absolute error crop from no-leak evaluation grid",
        },
        {
            "method": method,
            "display_label": meta["label"],
            "component": "measurement_residual_profile",
            "image": str(IMG_DIR / f"{method}_measurement_residual_bar.png"),
            "psnr": "",
            "ssim": "",
            "rel_meas_err": "",
            "source": str(ABLATION_CSV),
            "notes": "bar chart compares no-audit and final-audit RelMeasErr",
        },
    ]

    availability = [
        {
            "method": method,
            "x_data_available": True,
            "raw_candidate_residual_available": False,
            "filtered_residual_available": False,
            "pre_audit_or_no_mc_available": pre_available,
            "final_output_available": True,
            "rel_meas_err_available": True,
            "notes": "raw residual and filtered residual were not directly exported by existing checkpoints; unavailable placeholders are explicit and not used as evidence",
        }
    ]
    return component_rows, availability


def write_manifest(component_rows: list[dict[str, object]], availability: list[dict[str, object]]) -> None:
    lines = [
        "# Mechanism Component Manifest",
        "",
        "This folder is eval-only packaging of existing no-leak evaluation and inference-ablation artifacts. No training or new large experiment was run.",
        "",
        f"Representative sample index: `{SAMPLE_INDEX}` from the existing STL-10 evaluation grids.",
        "",
        "## Availability",
        "",
    ]
    for row in availability:
        lines.append(
            f"- {row['method']}: x_data={row['x_data_available']}, raw_residual={row['raw_candidate_residual_available']}, "
            f"filtered_residual={row['filtered_residual_available']}, pre_audit/no-MC={row['pre_audit_or_no_mc_available']}, final={row['final_output_available']}."
        )
    lines += [
        "",
        "## Measurement Audit Notes",
        "",
        "- Rademacher uses the imported exact-A artifact and the safe cache-rebuild evaluation path in the locked no-leak results.",
        "- The no-MC image/metrics are inference ablations, not separately trained unconstrained networks.",
        "- Raw residual and filtered residual hooks are not directly available in the saved artifacts, so they are marked unavailable rather than reconstructed by assumption.",
        "",
        "## Files",
        "",
    ]
    for row in component_rows:
        lines.append(f"- `{row['component']}` / `{row['method']}`: `{row['image']}`")
    (COMP / "component_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ablation_rows = read_ablation_rows()
    component_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    for method in ["rad5", "scr5"]:
        rows, availability = export_method(method, ablation_rows)
        component_rows.extend(rows)
        availability_rows.extend(availability)

    write_csv(
        COMP / "component_metrics.csv",
        component_rows,
        ["method", "display_label", "component", "image", "psnr", "ssim", "rel_meas_err", "source", "notes"],
    )
    write_csv(
        COMP / "available_components.csv",
        availability_rows,
        [
            "method",
            "x_data_available",
            "raw_candidate_residual_available",
            "filtered_residual_available",
            "pre_audit_or_no_mc_available",
            "final_output_available",
            "rel_meas_err_available",
            "notes",
        ],
    )
    write_manifest(component_rows, availability_rows)
    print({"output": str(COMP), "component_rows": len(component_rows), "availability_rows": len(availability_rows)})


if __name__ == "__main__":
    main()
