from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


E_ROOT = Path("E:/ns_mc_gan_gi")
OUT = E_ROOT / "outputs_phase18_rewrite"
PHASE15 = E_ROOT / "outputs_phase15"
PHASE16 = E_ROOT / "outputs_phase16" / "supplementary_experiments"
REGISTRY = PHASE15 / "noleak_registry.csv"

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
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def pct(value: Any) -> str:
    val = as_float(value)
    if math.isfinite(val):
        return f"{val * 100:.0f}%"
    return str(value)


def registry_rows() -> list[dict[str, str]]:
    by_id = {row.get("method_id", ""): row for row in read_csv(REGISTRY)}
    return [by_id[mid] for mid in METHOD_ORDER if mid in by_id]


def registry_by_id() -> dict[str, dict[str, str]]:
    return {row.get("method_id", ""): row for row in read_csv(REGISTRY)}


def table(name: str) -> list[dict[str, str]]:
    return read_csv(TABLES[name])


def main_results_rows() -> list[dict[str, str]]:
    rows = []
    for row in registry_rows():
        dataset = row.get("dataset", "")
        ratio = as_float(row.get("sampling_ratio"))
        psnr = as_float(row.get("psnr"))
        ssim = as_float(row.get("ssim"))
        if dataset == "STL-10" and abs(ratio - 0.05) < 1e-6:
            hq = psnr >= 20.0 and ssim >= 0.60
        elif dataset == "STL-10" and abs(ratio - 0.10) < 1e-6:
            hq = psnr >= 22.0 and ssim >= 0.65
        else:
            hq = psnr >= 25.0 and ssim >= 0.80
        rows.append(
            {
                "method_id": row.get("method_id", ""),
                "label": METHOD_LABEL.get(row.get("method_id", ""), row.get("method_id", "")),
                "long_label": LONG_LABEL.get(row.get("method_id", ""), row.get("display_name", "")),
                "dataset": dataset,
                "sampling": pct(row.get("sampling_ratio")),
                "measurement": row.get("measurement_family", "").replace("_", " "),
                "psnr": fmt(row.get("psnr")),
                "ssim": fmt(row.get("ssim")),
                "bp_psnr": fmt(row.get("backproj_psnr")),
                "delta_psnr": fmt(row.get("delta_psnr")),
                "hq": "yes" if hq else "no",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in shown:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/").replace("\n", " ") for field in fields) + "|")
    if limit is not None and len(rows) > limit:
        lines.append("|...|" + "|".join([""] * (len(fields) - 1)) + "|")
    return "\n".join(lines)


def tex_escape_text(text: Any) -> str:
    s = str(text)
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in s)


def tex_table(rows: list[dict[str, Any]], fields: list[str], caption: str, label: str, *, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    align = "l" * len(fields)
    width = r"\textwidth" if wide else r"\linewidth"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape_text(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape_text(row.get(field, "")) for field in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", rf"\end{{{env}}}"])
    return "\n".join(lines)


def prose_to_tex(text: str) -> str:
    # Preserve display math blocks and simple LaTeX commands; escape only plain percent signs.
    out: list[str] = []
    in_math = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "$$":
            out.append(r"\[" if not in_math else r"\]")
            in_math = not in_math
        elif in_math:
            out.append(line)
        elif not stripped:
            out.append("")
        elif stripped.startswith(r"\begin") or stripped.startswith(r"\end") or stripped.startswith(r"\toprule") or stripped.startswith(r"\midrule") or stripped.startswith(r"\bottomrule"):
            out.append(line)
        else:
            out.append(line.replace("%", r"\%"))
    return "\n".join(out)


def strip_internal_terms(text: str) -> str:
    return re.sub(r"Phase1[4567]", "the archived evaluation", text)


def source_manifest() -> dict[str, Any]:
    return {
        "registry": str(REGISTRY),
        "tables": {key: str(path) for key, path in TABLES.items()},
        "output": str(OUT),
        "note": "Main manuscript text avoids internal phase names; source paths are listed for audit only.",
    }
