from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import ensure_dir


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8/quality_audit")

EXPERIMENTS = [
    ("Phase 2 fixed 5%", Path("E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct")),
    ("Phase 4 best 5%", Path("E:/ns_mc_gan_gi/outputs_phase4/matched_binary_no_freeze_5pct")),
    ("Phase 7 continuous G-only 5%", Path("E:/ns_mc_gan_gi/outputs_phase7/continuous_g_only_5pct")),
    ("Phase 7 continuous physical 5%", Path("E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct")),
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(metrics: dict | None, section: str, key: str):
    if not metrics:
        return "missing"
    return metrics.get(section, {}).get(key, "missing")


def _score(row: dict) -> float | None:
    try:
        return float(row["model_psnr"]) + 10.0 * float(row["model_ssim"])
    except Exception:
        return None


def collect_rows() -> list[dict]:
    rows = []
    for method, output_dir in EXPERIMENTS:
        metrics = _read_json(output_dir / "eval_metrics.json")
        sample = output_dir / "eval_samples" / "recon_grid.png"
        status = "ok" if metrics else "missing"
        row = {
            "method": method,
            "output_dir": str(output_dir),
            "status": status,
            "model_psnr": _metric(metrics, "model", "psnr"),
            "model_ssim": _metric(metrics, "model", "ssim"),
            "model_mse": _metric(metrics, "model", "mse"),
            "model_rel_meas_err": _metric(metrics, "model", "rel_meas_error"),
            "backproj_psnr": _metric(metrics, "backprojection", "psnr"),
            "backproj_ssim": _metric(metrics, "backprojection", "ssim"),
            "backproj_mse": _metric(metrics, "backprojection", "mse"),
            "backproj_rel_meas_err": _metric(metrics, "backprojection", "rel_meas_error"),
            "score": "missing",
            "sample_image": str(sample) if sample.exists() else "missing",
            "high_quality": "missing",
        }
        score = _score(row)
        if score is not None:
            row["score"] = score
            row["high_quality"] = bool(float(row["model_psnr"]) >= 20.0 and float(row["model_ssim"]) >= 0.60)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = ["method", "model_psnr", "model_ssim", "model_mse", "model_rel_meas_err", "score", "status"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row[h]) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows: list[dict], path: Path) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]

    def best_by(key: str):
        values = []
        for row in ok_rows:
            try:
                values.append((float(row[key]), row))
            except Exception:
                pass
        return max(values, default=(None, None))[1]

    best_psnr = best_by("model_psnr")
    best_ssim = best_by("model_ssim")
    best_score = best_by("score")
    ranked = sorted(ok_rows, key=lambda r: float(r["score"]), reverse=True)
    not_hq = [r["method"] for r in ok_rows if not bool(r["high_quality"])]
    lines = [
        "# Phase 8 Quality Audit",
        "",
        "## Aggregate Metrics",
        "",
    ]
    lines.extend((OUTPUT_DIR / "quality_audit_summary.md").read_text(encoding="utf-8").splitlines())
    lines.extend(
        [
            "",
            "## Quality Ranking",
            "",
        ]
    )
    for idx, row in enumerate(ranked, start=1):
        lines.append(f"{idx}. {row['method']}: score={_fmt(row['score'])}, PSNR={_fmt(row['model_psnr'])}, SSIM={_fmt(row['model_ssim'])}")
    lines.extend(
        [
            "",
            "## Best Methods",
            "",
            f"- Best PSNR: {best_psnr['method'] if best_psnr else 'missing'}",
            f"- Best SSIM: {best_ssim['method'] if best_ssim else 'missing'}",
            f"- Best score: {best_score['method'] if best_score else 'missing'}",
            "",
            "## Not High-Quality Under Internal Threshold",
            "",
        ]
    )
    if not_hq:
        lines.extend(f"- {name}" for name in not_hq)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Manual Visual Failure Checklist",
            "",
            "- [ ] 是否边缘模糊",
            "- [ ] 是否纹理丢失",
            "- [ ] 是否类别结构能辨认",
            "- [ ] 是否有 GAN hallucination",
            "- [ ] 是否过平滑",
            "- [ ] 是否有 checkerboard artifacts",
            "- [ ] 是否与 bucket measurement 保持一致",
            "",
            "## Sample Images",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['method']}: {row['sample_image']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rows = collect_rows()
    write_csv(rows, OUTPUT_DIR / "quality_audit_summary.csv")
    write_markdown(rows, OUTPUT_DIR / "quality_audit_summary.md")
    write_report(rows, OUTPUT_DIR / "QUALITY_AUDIT.md")
    print(f"Wrote quality audit to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
