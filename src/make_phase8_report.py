from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8")
PSNR_THRESHOLD = 20.0
SSIM_THRESHOLD = 0.60


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value):
    try:
        return float(value)
    except Exception:
        return None


def _best(rows: list[dict], key: str):
    ok = [(r, _float(r.get(key))) for r in rows if _float(r.get(key)) is not None and r.get("status") == "ok"]
    if not ok:
        return None
    return max(ok, key=lambda item: item[1])[0]


def _row_named(rows: list[dict], contains: str):
    contains = contains.lower()
    return next((r for r in rows if contains in r.get("method", "").lower()), None)


def _summary(row) -> str:
    if not row or row.get("status") != "ok":
        return "missing"
    return (
        f"{row['method']} | PSNR={row.get('model_psnr', 'missing')} | "
        f"SSIM={row.get('model_ssim', 'missing')} | score={row.get('score', 'missing')} | "
        f"checkpoint={row.get('checkpoint', 'missing')}"
    )


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rows = _read_rows(OUTPUT_DIR / "phase8_results.csv")
    quality_path = OUTPUT_DIR / "quality_audit" / "QUALITY_AUDIT.md"
    related_path = OUTPUT_DIR / "related_work_table.md"
    examples_dir = OUTPUT_DIR / "quality_audit" / "examples"
    best_score = _best(rows, "score")
    best_psnr = _best(rows, "model_psnr")
    best_ssim = _best(rows, "model_ssim")
    strong_cont = _row_named(rows, "continuous physical wide")
    strong_fixed = _row_named(rows, "fixed wide 5")
    cont_g_only = _row_named(rows, "continuous g-only wide")
    direct_y = _row_named(rows, "direct y")
    mnist_fixed = _row_named(rows, "mnist fixed")
    mnist_cont = _row_named(rows, "mnist continuous")
    reaches = False
    if best_score:
        psnr, ssim = _float(best_score.get("model_psnr")), _float(best_score.get("model_ssim"))
        reaches = psnr is not None and ssim is not None and psnr >= PSNR_THRESHOLD and ssim >= SSIM_THRESHOLD
    conclusion = (
        "The strong reconstruction pipeline reaches a visually stronger regime and can support a limited high-quality claim under STL-10 64x64 5% setting."
        if reaches
        else "The current system demonstrates physics-consistent learned illumination but does not yet support a high-quality reconstruction claim."
    )
    lines = [
        "# Phase 8 Report",
        "",
        f"- Experiment datetime: {datetime.now().isoformat(timespec='seconds')}",
        "- Dataset path: E:/ns_mc_gan_gi/data",
        f"- Output path: {OUTPUT_DIR}",
        "- Current problem: existing reconstructions are visually and metrically modest",
        "",
        "## Phase 7 Summary",
        "",
        "- Continuous physical learned illumination had positive attribution signal.",
        "- Hard binary learned illumination did not support a reconstruction-gain claim.",
        "- Best Phase 7 continuous physical 5%: PSNR 17.6758, SSIM 0.4386, score 22.0620.",
        "",
        "## Artifacts",
        "",
        f"- Related work table: {related_path}",
        f"- Quality audit report: {quality_path}",
        f"- best/median/worst examples: {examples_dir}",
        "",
        "## Strong Baselines",
        "",
        f"- Strong fixed baseline: {_summary(strong_fixed)}",
        f"- Strong continuous physical: {_summary(strong_cont)}",
        f"- Continuous physical vs continuous G-only: strong={_summary(strong_cont)}; g_only={_summary(cont_g_only)}",
        f"- Direct y baseline: {_summary(direct_y)}",
        f"- MNIST sanity: fixed={_summary(mnist_fixed)}; continuous={_summary(mnist_cont)}",
        "",
        "## High-Quality Threshold",
        "",
        f"- Internal threshold: PSNR >= {PSNR_THRESHOLD} and SSIM >= {SSIM_THRESHOLD}",
        f"- Best score method: {best_score['method'] if best_score else 'missing'}",
        f"- Best PSNR method: {best_psnr['method'] if best_psnr else 'missing'}",
        f"- Best SSIM method: {best_ssim['method'] if best_ssim else 'missing'}",
        f"- Reaches threshold: {reaches}",
        "",
        "## Current Conclusion",
        "",
        conclusion,
        "",
        "## Paper-Safe Conclusions",
        "",
        "- Learned continuous physical illumination can be discussed as evidence when controlled comparisons support it.",
        "- Fixed measurement and stronger reconstruction baselines should be reported separately from learned illumination.",
        "",
        "## Conclusions Not Supported Yet",
        "",
        "- Do not claim high-quality STL-10 64x64 5% reconstruction unless the threshold and examples support it.",
        "- Do not claim binary learned illumination improves reconstruction.",
        "",
        "## Next Step",
        "",
        "- If strong baselines remain below threshold, move the main claim toward learned physical illumination evidence and run HQ/domain sanity experiments.",
    ]
    path = OUTPUT_DIR / "PHASE8_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 8 report to {path}")


if __name__ == "__main__":
    main()
