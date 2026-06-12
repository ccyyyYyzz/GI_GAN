from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


PHASE6_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase6")
PHASE7_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase7")

FIELDS = [
    "method",
    "physical_pattern_type",
    "pattern_mode",
    "effective_A_mode",
    "model_psnr",
    "model_ssim",
    "score",
    "hard_flip_fraction",
    "soft_flip_delta",
    "A_rel_fro_delta",
    "secant_rip_initial",
    "secant_rip_final",
    "secant_rip_delta",
    "mean_abs_offdiag_corr",
    "pattern_mean",
    "pattern_std",
    "row_std_mean",
    "continuous_contrast",
    "pattern_attribution_note",
    "g_only_reference_score",
    "pattern_trainable_minus_g_only",
    "pattern_only_gain",
    "swap_learnedG_initialA_score",
    "swap_learnedG_learnedA_score",
    "swap_fixedG_learnedA_score",
    "measurement_quality_secrip",
    "measurement_quality_condition",
    "measurement_quality_bucket_snr_proxy",
    "checkpoint",
    "sample_image",
    "pattern_image",
    "pattern_change_image",
    "status",
]


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(metrics: dict | None, weight: float = 10.0):
    if not metrics:
        return ""
    model = metrics.get("model", {})
    if "psnr" not in model or "ssim" not in model:
        return ""
    return float(model["psnr"]) + weight * float(model["ssim"])


def phase6_score(method: str) -> str:
    for row in read_csv(PHASE6_ROOT / "phase6_control_results.csv"):
        if row.get("method") == method and row.get("score") not in ("", None):
            return row["score"]
    return ""


def diagnostics_for(run_dir: Path) -> dict:
    return (
        read_json(run_dir / "eval_pattern_diagnostics" / "pattern_diagnostics.json")
        or read_json(run_dir / "pattern_diagnostics" / "pattern_diagnostics.json")
        or {}
    )


def specs():
    return [
        ("Flip-aware Alpha1", PHASE7_ROOT / "flipaware_alpha1_5pct", "binary", "Phase6 G-only"),
        ("Flip-aware Alpha0.5", PHASE7_ROOT / "flipaware_alpha0p5_5pct", "binary", "Phase6 G-only"),
        ("Flip-aware Aggressive", PHASE7_ROOT / "flipaware_aggressive_5pct", "binary", "Phase6 G-only"),
        ("Continuous G-only", PHASE7_ROOT / "continuous_g_only_5pct", "continuous", "self"),
        ("Continuous Physical", PHASE7_ROOT / "continuous_physical_5pct", "continuous", "continuous_g_only"),
        ("Continuous Pattern-only", PHASE7_ROOT / "continuous_pattern_only_5pct", "continuous", "continuous_g_only"),
        ("Continuous Long", PHASE7_ROOT / "continuous_long_5pct", "continuous", "continuous_g_only"),
    ]


def swap_scores(kind: str) -> dict:
    path = PHASE7_ROOT / "pattern_swap" / kind / "pattern_swap_metrics.csv"
    rows = read_csv(path)
    out = {}
    for row in rows:
        out[row.get("method", "")] = row.get("score", "")
    return out


def measurement_quality(kind: str) -> dict:
    return read_json(PHASE7_ROOT / "measurement_quality" / kind / "measurement_quality.json") or {}


def row_from_run(method: str, run_dir: Path, physical_type: str, ref_kind: str) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json")
    diagnostics = diagnostics_for(run_dir)
    pattern = (metrics or {}).get("pattern", {})
    row = {field: "" for field in FIELDS}
    row["method"] = method
    row["status"] = "ok" if metrics else "missing"
    row["physical_pattern_type"] = (
        (metrics or {}).get("pattern_physical_type")
        or pattern.get("pattern_physical_type")
        or physical_type
    )
    row["pattern_mode"] = pattern.get("pattern_mode", diagnostics.get("pattern_mode", ""))
    row["effective_A_mode"] = pattern.get("effective_A_mode", diagnostics.get("effective_A_mode", ""))
    if metrics:
        model = metrics.get("model", {})
        row["model_psnr"] = model.get("psnr", "")
        row["model_ssim"] = model.get("ssim", "")
        row["score"] = score(metrics)
    for key in [
        "hard_flip_fraction",
        "soft_flip_delta",
        "A_rel_fro_delta",
        "secant_rip_initial",
        "secant_rip_final",
        "secant_rip_delta",
        "pattern_attribution_note",
    ]:
        row[key] = diagnostics.get(key, (metrics or {}).get(f"pattern_{key}", ""))
    row["mean_abs_offdiag_corr"] = pattern.get("mean_abs_offdiag_corr", "")
    row["pattern_mean"] = pattern.get("mean", "")
    row["pattern_std"] = pattern.get("std", "")
    row["row_std_mean"] = pattern.get("row_std_mean", "")
    row["continuous_contrast"] = pattern.get("continuous_contrast", "")
    phase6_g = phase6_score("G-only Fine-tune")
    continuous_g = read_json(PHASE7_ROOT / "continuous_g_only_5pct" / "eval_metrics.json")
    continuous_g_score = score(continuous_g)
    if ref_kind == "continuous_g_only" and continuous_g_score != "":
        row["g_only_reference_score"] = continuous_g_score
    elif ref_kind == "self":
        row["g_only_reference_score"] = row["score"]
    else:
        row["g_only_reference_score"] = phase6_g
    try:
        row["pattern_trainable_minus_g_only"] = float(row["score"]) - float(row["g_only_reference_score"])
    except Exception:
        row["pattern_trainable_minus_g_only"] = ""
    if method == "Continuous Pattern-only":
        fixed = phase6_score("Fixed Rademacher")
        try:
            row["pattern_only_gain"] = float(row["score"]) - float(fixed)
        except Exception:
            row["pattern_only_gain"] = ""
    swap_kind = "continuous_physical_5pct" if "Continuous" in method else "flipaware_alpha1_5pct"
    swaps = swap_scores(swap_kind)
    row["swap_learnedG_initialA_score"] = swaps.get("Learned G + Initial A", "")
    row["swap_learnedG_learnedA_score"] = swaps.get("Learned G + Learned A", "")
    row["swap_fixedG_learnedA_score"] = swaps.get("Fixed G + Learned A", "")
    mq_kind = "continuous_physical_5pct" if "Continuous Physical" in method else ""
    if method == "Continuous G-only":
        mq_kind = "continuous_g_only_5pct"
    if method.startswith("Flip-aware"):
        mq_kind = "flipaware_alpha1_5pct"
    mq = measurement_quality(mq_kind) if mq_kind else {}
    row["measurement_quality_secrip"] = mq.get("secant_rip_loss", "")
    row["measurement_quality_condition"] = mq.get("gram_condition_number", "")
    row["measurement_quality_bucket_snr_proxy"] = mq.get("bucket_snr_proxy", "")
    row["checkpoint"] = str(run_dir / "best_score.pt") if (run_dir / "best_score.pt").exists() else ""
    row["sample_image"] = (
        str(run_dir / "eval_samples" / "recon_grid.png")
        if (run_dir / "eval_samples" / "recon_grid.png").exists()
        else ""
    )
    row["pattern_image"] = (
        str(run_dir / "eval_patterns" / "final_patterns.png")
        if (run_dir / "eval_patterns" / "final_patterns.png").exists()
        else ""
    )
    row["pattern_change_image"] = (
        str(run_dir / "eval_pattern_diagnostics" / "pattern_change_grid.png")
        if (run_dir / "eval_pattern_diagnostics" / "pattern_change_grid.png").exists()
        else ""
    )
    return row


def write_csv(rows: list[dict], path: Path, fields: list[str] = FIELDS) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value) -> str:
    if value in ("", None):
        return "missing"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict], path: Path, title: str, cols: list[str]) -> None:
    lines = [f"# {title}", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_fig(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_score(rows: list[dict], x_key: str, path: Path, xlabel: str) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("score") not in ("", None)]
    ok = [r for r in ok if r.get(x_key) not in ("", None, "missing")]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in ok:
        ax.scatter(float(row[x_key]), float(row["score"]), label=row["method"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    save_fig(fig, path)


def plot_continuous_vs_binary(rows: list[dict], path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("score") not in ("", None)]
    if not ok:
        return
    colors = ["#4c78a8" if r.get("physical_pattern_type") == "binary" else "#f58518" for r in ok]
    fig, ax = plt.subplots(figsize=(max(7, len(ok) * 1.1), 4.4))
    ax.bar(range(len(ok)), [float(r["score"]) for r in ok], color=colors)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([r["method"] for r in ok], rotation=35, ha="right")
    ax.set_ylabel("Score")
    save_fig(fig, path)


def write_latex(rows: list[dict], path: Path, cols: list[str]) -> None:
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    lines.append(" & ".join(cols).replace("_", "\\_") + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(col, "")).replace("_", "\\_") for col in cols) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase7 = ensure_dir(PHASE7_ROOT)
    rows = [row_from_run(*spec) for spec in specs()]
    write_csv(rows, phase7 / "phase7_results.csv")
    write_markdown(
        rows,
        phase7 / "phase7_results.md",
        "Phase 7 Results",
        [
            "method",
            "physical_pattern_type",
            "model_psnr",
            "model_ssim",
            "score",
            "hard_flip_fraction",
            "A_rel_fro_delta",
            "pattern_trainable_minus_g_only",
            "status",
        ],
    )
    swap_rows = []
    for path in sorted((phase7 / "pattern_swap").glob("*/pattern_swap_metrics.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["swap_experiment"] = path.parent.name
            swap_rows.append(row)
    write_csv(
        swap_rows,
        phase7 / "phase7_pattern_swap_results.csv",
        [
            "swap_experiment",
            "method",
            "generator_source",
            "pattern_source",
            "model_psnr",
            "model_ssim",
            "score",
            "pattern_hard_flip_fraction",
            "A_rel_fro_delta",
            "status",
        ],
    )
    mq_rows = []
    for path in sorted((phase7 / "measurement_quality").glob("*/measurement_quality.json")):
        row = read_json(path) or {}
        row["method"] = path.parent.name
        mq_rows.append(row)
    write_csv(
        mq_rows,
        phase7 / "phase7_measurement_quality.csv",
        [
            "method",
            "pattern_physical_type",
            "secant_rip_loss",
            "mean_abs_offdiag_corr",
            "gram_condition_number",
            "bucket_snr_proxy",
            "status",
        ],
    )
    plot_score(rows, "hard_flip_fraction", phase7 / "phase7_flip_vs_score.png", "Hard Flip Fraction")
    plot_score(rows, "A_rel_fro_delta", phase7 / "phase7_A_drift_vs_score.png", "A Rel Fro Delta")
    plot_continuous_vs_binary(rows, phase7 / "phase7_continuous_vs_binary.png")
    write_latex(
        rows,
        phase7 / "phase7_latex_table_main.tex",
        ["method", "model_psnr", "model_ssim", "score", "hard_flip_fraction", "A_rel_fro_delta"],
    )
    write_latex(
        swap_rows,
        phase7 / "phase7_latex_table_swap.tex",
        ["swap_experiment", "method", "model_psnr", "model_ssim", "score"],
    )
    write_latex(
        mq_rows,
        phase7 / "phase7_latex_table_measurement_quality.tex",
        ["method", "secant_rip_loss", "mean_abs_offdiag_corr", "gram_condition_number", "bucket_snr_proxy"],
    )
    print(f"Wrote Phase 7 aggregation to: {phase7}")


if __name__ == "__main__":
    main()
