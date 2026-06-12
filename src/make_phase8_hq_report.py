from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8_hq")


def _rows() -> list[dict]:
    path = OUTPUT_DIR / "phase8_hq_results.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value):
    try:
        return float(value)
    except Exception:
        return None


def _truth(value) -> bool:
    return str(value).lower() == "true"


def _best(rows: list[dict]):
    scored = [(row, _float(row.get("hq_score"))) for row in rows if row.get("status") == "ok" and _float(row.get("hq_score")) is not None]
    return max(scored, key=lambda item: item[1])[0] if scored else None


def _find(rows: list[dict], name: str):
    name = name.lower()
    return next((r for r in rows if r.get("method", "").lower() == name), None)


def _summary(row) -> str:
    if not row or row.get("status") != "ok":
        return "missing"
    return (
        f"{row['method']} | dataset={row.get('dataset_name')} | ratio={row.get('sampling_ratio')} | "
        f"pattern={row.get('pattern_type')} | PSNR={row.get('model_psnr')} | "
        f"SSIM={row.get('model_ssim')} | hq_score={row.get('hq_score')}"
    )


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rows = _rows()
    best = _best(rows)
    stl10_10 = [r for r in rows if _truth(r.get("reaches_stl10_10pct_hq_threshold"))]
    stl10_5 = [r for r in rows if _truth(r.get("reaches_stl10_5pct_hq_threshold"))]
    simple = [r for r in rows if _truth(r.get("reaches_simple_domain_hq_threshold"))]
    reaches_any = bool(stl10_10 or stl10_5 or simple)
    first = (stl10_5 or stl10_10 or simple or [None])[0]
    if stl10_5:
        conclusion = "在 STL-10 grayscale 64x64 5% sampling 下，当前系统达到内部 high-quality threshold。"
    elif stl10_10:
        conclusion = "高质量可以在 10% 低采样下实现，5% 仍然是挑战。"
    elif simple:
        conclusion = "系统在结构化简单目标上达到高质量，但自然图像 STL-10 仍未达到。"
    else:
        conclusion = "当前 pipeline 仍不能支持 high-quality claim，需要进一步提高模型容量、训练长度或调整任务设定。"
    had = _find(rows, "hadamard_hq_10pct")
    rad = _find(rows, "rademacher_hq_10pct")
    cont = _find(rows, "continuous_physical_hq_10pct")
    examples_dir = OUTPUT_DIR / "paper_examples"
    lines = [
        "# Phase 8-HQ Report",
        "",
        f"- Experiment datetime: {datetime.now().isoformat(timespec='seconds')}",
        "- Dataset path: E:/ns_mc_gan_gi/data",
        f"- Output path: {OUTPUT_DIR}",
        "",
        "## Phase 7/8 Low-Quality Problem",
        "",
        "- Previous STL-10 64x64 5% reconstructions were around PSNR 17-18 and SSIM 0.4-0.46.",
        "- That regime is useful for learned illumination evidence but not enough for high-quality reconstruction claims.",
        "",
        "## HQ Pipeline",
        "",
        "- Hadamard/orthogonal fixed measurements.",
        "- HQ residual U-Net and optional two-stage refiner.",
        "- PSNR/SSIM-oriented losses with small adversarial weight.",
        "- AMP, EMA, staged refiner/adversarial training support.",
        "",
        "## Thresholds",
        "",
        "|setting|threshold|reached|",
        "|---|---|---|",
        f"|STL-10 10%|PSNR >= 22 and SSIM >= 0.65|{bool(stl10_10)}|",
        f"|STL-10 5%|PSNR >= 20 and SSIM >= 0.60|{bool(stl10_5)}|",
        f"|MNIST/FashionMNIST 5%|PSNR >= 25 and SSIM >= 0.80|{bool(simple)}|",
        "",
        "## Key Comparisons",
        "",
        f"- Hadamard 10%: {_summary(had)}",
        f"- Rademacher 10%: {_summary(rad)}",
        f"- Continuous physical 10%: {_summary(cont)}",
        f"- First high-quality setting: {first['method'] if first else 'missing'}",
        f"- Best hq_score setting: {best['method'] if best else 'missing'}",
        f"- best/median/worst examples: {examples_dir}",
        "",
        "## High-Quality Answer",
        "",
        f"- Reaches any internal high-quality threshold: {reaches_any}",
        f"- Conclusion: {conclusion}",
        "",
        "## Paper-Safe Conclusions",
        "",
        "- Report Hadamard and HQ reconstructor effects separately from learned illumination.",
        "- If only simple domains pass, limit the high-quality claim to those domains.",
        "- If continuous learned illumination beats the matching fixed/G-only HQ controls, report it as physical illumination evidence.",
        "",
        "## Conclusions Not Supported",
        "",
        "- Do not claim STL-10 5% high-quality if the threshold is not met.",
        "- Do not claim learned binary illumination improves reconstruction.",
        "- Do not use related-work metrics as strict SOTA comparisons unless datasets and sampling protocols match.",
        "",
        "## Next Step",
        "",
        "- Continue the best HQ setting longer, then run the nearest fixed/continuous attribution control under identical measurement and training budget.",
    ]
    path = OUTPUT_DIR / "PHASE8_HQ_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 8-HQ report to {path}")


if __name__ == "__main__":
    main()
