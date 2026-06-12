from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


E_ROOT = Path("E:/ns_mc_gan_gi")
OUT = E_ROOT / "outputs_phase18b_figfix"
PHASE15 = E_ROOT / "outputs_phase15"
PHASE16 = E_ROOT / "outputs_phase16" / "supplementary_experiments"
PHASE18 = E_ROOT / "outputs_phase18_rewrite"
REGISTRY = PHASE15 / "noleak_registry.csv"
IMPORTED = PHASE15 / "imported_noleak"

TABLES = {
    "exact_a": PHASE16 / "exactA_reeval" / "exactA_reeval_results.csv",
    "attribution": PHASE16 / "attribution" / "attribution_final.csv",
    "ablation": PHASE16 / "inference_ablation" / "real_inference_ablation_results.csv",
    "noise": PHASE16 / "noise_sweep" / "noise_sweep_results.csv",
    "baseline": PHASE16 / "traditional_baselines" / "tv_pgd_baseline_results.csv",
    "dc_row": PHASE16 / "dc_row_control" / "dc_row_final.csv",
    "statistics": PHASE16 / "statistics" / "statistics_ci.csv",
    "classwise": PHASE16 / "classwise" / "classwise_stl10_metrics.csv",
    "perturbation": PHASE16 / "measurement_perturbation" / "measurement_perturbation.csv",
    "runtime": PHASE16 / "runtime_complexity" / "runtime_complexity.csv",
}

TITLE = "High-Quality Low-Sampling Ghost Imaging via Measurement-Consistent Null-Space Neural Reconstruction"

METHOD_ORDER = [
    "rademacher5_hq_noise001_colab",
    "scrambled_hadamard5_hq_noise001_colab",
    "rademacher10_full_noise001_colab",
    "scrambled_hadamard10_full_noise001_colab",
    "mnist_hadamard5_full_colab",
    "fashion_hadamard5_full_colab",
]

STL_METHODS = METHOD_ORDER[:4]
SIMPLE_METHODS = METHOD_ORDER[4:]

METHOD_LABEL = {
    "rademacher5_hq_noise001_colab": "Rad-5",
    "scrambled_hadamard5_hq_noise001_colab": "Scr-5",
    "rademacher10_full_noise001_colab": "Rad-10",
    "scrambled_hadamard10_full_noise001_colab": "Scr-10",
    "mnist_hadamard5_full_colab": "MNIST",
    "fashion_hadamard5_full_colab": "Fashion",
}

LONG_LABEL = {
    "rademacher5_hq_noise001_colab": "STL-10 Rademacher 5%",
    "scrambled_hadamard5_hq_noise001_colab": "STL-10 scrambled Hadamard 5%",
    "rademacher10_full_noise001_colab": "STL-10 Rademacher 10%",
    "scrambled_hadamard10_full_noise001_colab": "STL-10 scrambled Hadamard 10%",
    "mnist_hadamard5_full_colab": "MNIST Hadamard 5%",
    "fashion_hadamard5_full_colab": "Fashion-MNIST Hadamard 5%",
}

BLUE = "#1F5F8B"
LIGHT_BLUE = "#DCECF5"
ORANGE = "#D7872F"
LIGHT_ORANGE = "#F8E3C7"
GREEN = "#3D8B5E"
LIGHT_GREEN = "#DDEFE4"
RED = "#B94A48"
GRAY = "#555555"
LIGHT_GRAY = "#E8EAED"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    val = as_float(value)
    if math.isfinite(val):
        return f"{val:.{digits}f}"
    return str(value) if value is not None else ""


def registry_rows() -> list[dict[str, str]]:
    by_id = {row.get("method_id", ""): row for row in read_csv(REGISTRY)}
    return [by_id[mid] for mid in METHOD_ORDER if mid in by_id]


def registry_by_id() -> dict[str, dict[str, str]]:
    return {row.get("method_id", ""): row for row in read_csv(REGISTRY)}


def table(name: str) -> list[dict[str, str]]:
    return read_csv(TABLES[name])


def metric(mid: str, key: str) -> str:
    return registry_by_id().get(mid, {}).get(key, "")


def method_summary(mid: str) -> str:
    row = registry_by_id()[mid]
    return f"{METHOD_LABEL[mid]}: PSNR {fmt(row['psnr'])}, SSIM {fmt(row['ssim'])}, RelMeasErr {fmt(row['rel_meas_err'])}"


def tex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def tex_table(rows: list[dict[str, Any]], fields: list[str], caption: str, label: str, *, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\linewidth"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{'l' * len(fields)}}}",
        r"\toprule",
        " & ".join(tex_escape(f) for f in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(f, "")) for f in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", rf"\end{{{env}}}"])
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(f, "")).replace("|", "/") for f in fields) + "|")
    return "\n".join(lines)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, *, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> list[Path]:
    out = ensure_dir(OUT / "figures")
    paths: list[Path] = []
    for ext in exts:
        path = out / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def image_grid_path(mid: str) -> Path:
    return IMPORTED / mid / "eval_samples" / "recon_grid.png"


def detect_runs(values: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    flags = values > threshold
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        runs.append((start, len(flags)))
    return runs


def grid_cell_bounds(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    im = Image.open(path).convert("L")
    arr = np.asarray(im)
    mask = arr < 245
    x_profile = mask[120:, :].mean(axis=0)
    x_runs = detect_runs(x_profile, 0.05, 120)
    y_profile = mask[:, 20 : arr.shape[1] - 20].mean(axis=1)
    y_runs = [r for r in detect_runs(y_profile, 0.05, 150) if r[0] > 70]
    if len(x_runs) < 4:
        w = arr.shape[1]
        x_runs = [(25, 375), (405, 755), (785, 1135), (1165, min(w - 10, 1510))]
    if len(y_runs) < 8:
        h = arr.shape[0]
        y_runs = [(105 + i * 360, min(445 + i * 360, h - 10)) for i in range(8)]
    return x_runs[:4], y_runs[:8]


def crop_grid_cell(mid: str, row_idx: int, col_idx: int) -> Image.Image:
    path = image_grid_path(mid)
    im = Image.open(path).convert("RGB")
    xs, ys = grid_cell_bounds(path)
    x0, x1 = xs[col_idx]
    y0, y1 = ys[row_idx]
    pad = 2
    return im.crop((max(0, x0 + pad), max(0, y0 + pad), min(im.width, x1 - pad), min(im.height, y1 - pad)))


def crop_sample(mid: str, row_idx: int) -> dict[str, Image.Image]:
    names = ["GT", "Backprojection", "Reconstruction", "Abs error"]
    return {name: crop_grid_cell(mid, row_idx, i) for i, name in enumerate(names)}


def copy_figures_to_latex(fig_dir: Path, latex_fig_dir: Path) -> None:
    ensure_dir(latex_fig_dir)
    for path in fig_dir.glob("*.*"):
        if path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, latex_fig_dir / path.name)


def source_manifest() -> dict[str, Any]:
    return {
        "registry": str(REGISTRY),
        "phase18_rewrite": str(PHASE18),
        "phase16_tables": {name: str(path) for name, path in TABLES.items()},
        "reconstruction_grids": {mid: str(image_grid_path(mid)) for mid in METHOD_ORDER},
        "output": str(OUT),
        "note": "Phase18B uses saved no-leak evaluation grids and Phase15/16 CSVs only; no training is launched.",
    }
