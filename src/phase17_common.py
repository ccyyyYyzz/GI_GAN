from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E_ROOT = Path("E:/ns_mc_gan_gi")
PHASE15 = E_ROOT / "outputs_phase15"
PHASE16 = E_ROOT / "outputs_phase16" / "supplementary_experiments"
PHASE17 = E_ROOT / "outputs_phase17"

REGISTRY = PHASE15 / "noleak_registry.csv"
PHASE15R_REPORT = PHASE15 / "repro_debug" / "PHASE15R_RADEMACHER_REPRO_REPORT.md"
PHASE16_REPORT = PHASE16 / "_report" / "PHASE16_SUPPLEMENTARY_REPORT.md"
PHASE16_AGGREGATE = PHASE16 / "_aggregate" / "PHASE16_AGGREGATE_SUMMARY.md"
PHASE16_SUPPORTED = PHASE16 / "_report" / "PHASE16_SUPPORTED_CLAIMS.md"

PHASE16_TABLES = {
    "exact_a_reeval": PHASE16 / "exactA_reeval" / "exactA_reeval_results.csv",
    "attribution": PHASE16 / "attribution" / "attribution_final.csv",
    "ablation": PHASE16 / "inference_ablation" / "real_inference_ablation_results.csv",
    "noise": PHASE16 / "noise_sweep" / "noise_sweep_results.csv",
    "traditional_baselines": PHASE16 / "traditional_baselines" / "tv_pgd_baseline_results.csv",
    "dc_row": PHASE16 / "dc_row_control" / "dc_row_final.csv",
    "statistics": PHASE16 / "statistics" / "statistics_ci.csv",
    "classwise": PHASE16 / "classwise" / "classwise_stl10_metrics.csv",
    "perturbation": PHASE16 / "measurement_perturbation" / "measurement_perturbation.csv",
    "runtime": PHASE16 / "runtime_complexity" / "runtime_complexity.csv",
}

METHOD_ORDER = [
    "mnist_hadamard5_full_colab",
    "fashion_hadamard5_full_colab",
    "scrambled_hadamard5_hq_noise001_colab",
    "rademacher5_hq_noise001_colab",
    "scrambled_hadamard10_full_noise001_colab",
    "rademacher10_full_noise001_colab",
]

METHOD_LABELS = {
    "mnist_hadamard5_full_colab": "MNIST Hadamard 5%",
    "fashion_hadamard5_full_colab": "Fashion-MNIST Hadamard 5%",
    "scrambled_hadamard5_hq_noise001_colab": "STL-10 scrambled Hadamard 5%",
    "rademacher5_hq_noise001_colab": "STL-10 Rademacher 5%",
    "scrambled_hadamard10_full_noise001_colab": "STL-10 scrambled Hadamard 10%",
    "rademacher10_full_noise001_colab": "STL-10 Rademacher 10%",
}

TITLE = "High-Quality Low-Sampling Ghost Imaging via Measurement-Consistent Null-Space Neural Reconstruction"
CORE_CLAIM = (
    "Strict no-leak experiments show high-quality low-sampling ghost imaging / single-pixel imaging "
    "when a neural reconstructor is constrained by measurement consistency and null-space structure."
)
DO_NOT_CLAIM = [
    "strict state-of-the-art performance",
    "universal or adversarial robustness",
    "low-frequency Hadamard 5% is high-quality on STL-10",
    "binary learned illumination is the main contribution",
    "GAN is the final main mechanism",
    "TV-PGD is exhaustively optimized",
    "first deep-learning ghost imaging method",
    "first data-consistency or null-space neural inverse method",
]


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


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value) if value is not None else ""


def tex_escape(value: Any) -> str:
    text = str(value)
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
    return "".join(replacements.get(ch, ch) for ch in text)


def markdown_table(rows: list[dict[str, Any]], fields: list[str], labels: dict[str, str] | None = None, limit: int | None = None) -> str:
    labels = labels or {}
    shown = rows if limit is None else rows[:limit]
    lines = ["|" + "|".join(labels.get(field, field) for field in fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in shown:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/").replace("\n", " ") for field in fields) + "|")
    if limit is not None and len(rows) > limit:
        lines.append("|...|" + "|".join([""] * (len(fields) - 1)) + "|")
    return "\n".join(lines)


def latex_table(rows: list[dict[str, Any]], fields: list[str], caption: str, label: str) -> str:
    cols = "l" * len(fields)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        " & ".join(tex_escape(field) for field in fields) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(field, "")) for field in fields) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def registry_rows() -> list[dict[str, str]]:
    by_id = {row.get("method_id", ""): row for row in read_csv(REGISTRY)}
    return [by_id[mid] for mid in METHOD_ORDER if mid in by_id]


def table_rows(name: str) -> list[dict[str, str]]:
    return read_csv(PHASE16_TABLES[name])


def main_result_rows() -> list[dict[str, str]]:
    rows = []
    for row in registry_rows():
        rows.append(
            {
                "method": METHOD_LABELS.get(row.get("method_id", ""), row.get("method_id", "")),
                "dataset": row.get("dataset", ""),
                "sampling": f"{float(row.get('sampling_ratio', 0.0)) * 100:.0f}%",
                "family": row.get("measurement_family", ""),
                "psnr": fnum(row.get("psnr")),
                "ssim": fnum(row.get("ssim")),
                "bp_psnr": fnum(row.get("backproj_psnr")),
                "delta_psnr": fnum(row.get("delta_psnr")),
                "source": "Phase15 noleak_registry.csv",
            }
        )
    return rows


def stl_main_rows() -> list[dict[str, str]]:
    return [row for row in main_result_rows() if row["dataset"] == "STL-10"]


def simple_main_rows() -> list[dict[str, str]]:
    return [row for row in main_result_rows() if row["dataset"] != "STL-10"]


def cite_paths() -> dict[str, str]:
    return {
        "Phase15 no-leak registry": str(REGISTRY),
        "Phase15R Rademacher report": str(PHASE15R_REPORT),
        "Phase16 supplementary report": str(PHASE16_REPORT),
        "Phase16 aggregate summary": str(PHASE16_AGGREGATE),
        "Phase16 supported claims": str(PHASE16_SUPPORTED),
    }


def output_file_purpose(path: Path) -> str:
    parts = set(path.parts)
    name = path.name
    if "evidence_index" in parts:
        return "claim-to-evidence index"
    if "manuscript" in parts:
        return "English manuscript draft" if "manuscript" in name else "citation verification aid"
    if "chinese_report" in parts:
        return "Chinese technical report draft"
    if "supplement" in parts:
        return "supplementary material draft"
    if "figure_table_pack" in parts:
        return "figure/table planning and captions"
    if "submission_pack" in parts:
        return "journal submission planning material"
    if "defense" in parts:
        return "defense slides and Q&A preparation"
    if "reviewer_risk_register" in parts:
        return "reviewer risk register"
    if name == "FINAL_PAPER_CHECKLIST.md":
        return "final manuscript safety checklist"
    if name == "PHASE17_MANIFEST.md":
        return "Phase17 output manifest"
    return "Phase17 generated artifact"
