from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


PHASE7_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase7")
ASSET_ROOT = PHASE7_ROOT / "paper_assets"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=None):
    try:
        if value in ("", None, "missing"):
            return default
        return float(value)
    except Exception:
        return default


def save_fig(fig, path_base: Path) -> None:
    ensure_dir(path_base.parent)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_flip(rows: list[dict], path_base: Path) -> None:
    ok = [r for r in rows if as_float(r.get("score")) is not None and as_float(r.get("hard_flip_fraction")) is not None]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for row in ok:
        ax.scatter(as_float(row["hard_flip_fraction"]), as_float(row["score"]), label=row["method"])
    ax.set_xlabel("Hard flip fraction")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    save_fig(fig, path_base)


def plot_continuous(rows: list[dict], path_base: Path) -> None:
    ok = [r for r in rows if as_float(r.get("score")) is not None]
    if not ok:
        return
    colors = ["#4c78a8" if r.get("physical_pattern_type") == "binary" else "#f58518" for r in ok]
    fig, ax = plt.subplots(figsize=(max(7, len(ok) * 1.05), 4.4))
    ax.bar(range(len(ok)), [as_float(r["score"]) for r in ok], color=colors)
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([r["method"] for r in ok], rotation=35, ha="right")
    save_fig(fig, path_base)


def plot_swap(rows: list[dict], path_base: Path) -> None:
    ok = [r for r in rows if as_float(r.get("score")) is not None]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(ok) * 0.9), 4.4))
    labels = [f"{r.get('swap_experiment', '')}\n{r.get('method', '')}" for r in ok]
    ax.bar(range(len(ok)), [as_float(r["score"]) for r in ok])
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    save_fig(fig, path_base)


def plot_measurement_quality(rows: list[dict], path_base: Path) -> None:
    ok = [r for r in rows if as_float(r.get("secant_rip_loss")) is not None]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(ok) * 1.0), 4.0))
    ax.bar(range(len(ok)), [as_float(r["secant_rip_loss"]) for r in ok])
    ax.set_ylabel("Secant-RIP loss")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([r["method"] for r in ok], rotation=35, ha="right")
    save_fig(fig, path_base)


def write_latex(rows: list[dict], path: Path, cols: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    lines.append(" & ".join(cols).replace("_", "\\_") + " \\\\")
    lines.append("\\hline")
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            try:
                value = f"{float(value):.4f}"
            except Exception:
                value = str(value or "missing")
            vals.append(value.replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claims(path: Path, rows: list[dict]) -> None:
    binary_ok = any(
        r.get("method", "").startswith("Flip-aware")
        and as_float(r.get("hard_flip_fraction"), 0.0) > 0.0
        and as_float(r.get("pattern_trainable_minus_g_only"), -1.0) > 0.02
        for r in rows
    )
    cont = next((r for r in rows if r.get("method") == "Continuous Physical"), None)
    cont_ok = bool(cont and as_float(cont.get("pattern_trainable_minus_g_only"), -1.0) > 0.02)
    claims = [
        ("Exact operator matching works", "yes", "Phase 5/6 fixed exact controls", "Not learned illumination"),
        ("Generator fine-tuning improves reconstruction", "yes", "Phase 6 G-only control", "Do not attribute to A"),
        (
            "Hard binary learned illumination improves reconstruction",
            "yes" if binary_ok else "no",
            "Phase 7 flip-aware hard flips and G-only delta",
            "Requires hard flip and swap support",
        ),
        (
            "Continuous physical illumination improves reconstruction",
            "yes" if cont_ok else "no",
            "Phase 7 continuous trainable vs continuous G-only",
            "Continuous is not binary",
        ),
        (
            "Secant-RIP proxy correlates with performance",
            "based on diagnostics",
            "measurement_quality outputs",
            "Proxy, not a proof",
        ),
    ]
    lines = ["\\begin{tabular}{llll}", "\\hline", "Claim & Supported? & Evidence & Caveat \\\\", "\\hline"]
    for claim in claims:
        lines.append(" & ".join(x.replace("_", "\\_") for x in claim) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_or_render_pattern_change(rows: list[dict], path_base: Path) -> None:
    src = ""
    for row in rows:
        if row.get("pattern_change_image"):
            src = row["pattern_change_image"]
            break
    if not src or not Path(src).exists():
        return
    ensure_dir(path_base.parent)
    shutil.copy2(src, path_base.with_suffix(".png"))
    img = plt.imread(src)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(img)
    ax.axis("off")
    save_fig(fig, path_base)


def main() -> None:
    assets = ensure_dir(ASSET_ROOT)
    rows = read_csv(PHASE7_ROOT / "phase7_results.csv")
    swap_rows = read_csv(PHASE7_ROOT / "phase7_pattern_swap_results.csv")
    mq_rows = read_csv(PHASE7_ROOT / "phase7_measurement_quality.csv")
    plot_flip(rows, assets / "fig_phase7_flip_vs_score")
    plot_continuous(rows, assets / "fig_phase7_continuous_vs_binary")
    plot_swap(swap_rows, assets / "fig_phase7_pattern_swap")
    plot_measurement_quality(mq_rows, assets / "fig_phase7_measurement_quality")
    copy_or_render_pattern_change(rows, assets / "fig_phase7_pattern_change_grid")
    write_latex(
        rows,
        assets / "table_phase7_main.tex",
        ["method", "model_psnr", "model_ssim", "score", "hard_flip_fraction", "A_rel_fro_delta"],
    )
    write_latex(
        swap_rows,
        assets / "table_phase7_swap.tex",
        ["swap_experiment", "method", "model_psnr", "model_ssim", "score"],
    )
    write_latex(
        mq_rows,
        assets / "table_phase7_measurement_quality.tex",
        ["method", "secant_rip_loss", "mean_abs_offdiag_corr", "gram_condition_number", "bucket_snr_proxy"],
    )
    write_claims(assets / "table_claims.tex", rows)
    print(f"Wrote Phase 7 paper assets to: {assets}")


if __name__ == "__main__":
    main()
