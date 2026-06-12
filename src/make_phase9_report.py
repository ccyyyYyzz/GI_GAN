from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import ensure_dir


ROOT = Path("E:/ns_mc_gan_gi/outputs_phase9")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict], method: str) -> dict:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def metric_pair(row: dict) -> str:
    if not row or row.get("status") == "missing":
        return "missing"
    return f"PSNR={row.get('model_psnr', '')}, SSIM={row.get('model_ssim', '')}"


def lookup_backproj(rows: list[dict], ratio: str, label: str, mode: str) -> dict:
    for row in rows:
        if (
            str(row.get("sampling_ratio")) == ratio
            and row.get("pattern_label") == label
            and row.get("backprojection_mode") == mode
        ):
            return row
    return {}


def truthy(value) -> bool:
    return str(value).lower() in {"true", "1", "yes", "passed", "completed"}


def main() -> None:
    ensure_dir(ROOT)
    sanity = read_json(ROOT / "sanity_hadamard" / "hadamard_sanity.json")
    results = read_csv(ROOT / "phase9_results.csv")
    backproj = read_csv(ROOT / "phase9_backprojection_quality.csv")

    sanity_pass = bool(sanity.get("full_sampling_exact_passed", False))
    overfit = find_row(results, "overfit_hadamard_10pct")
    try:
        overfit_pass = (
            float(overfit.get("model_psnr", "nan")) >= 30.0
            and float(overfit.get("model_ssim", "nan")) >= 0.90
        )
    except Exception:
        overfit_pass = False
    h10 = find_row(results, "hadamard10_probe_nonoise")
    h10_noise = find_row(results, "hadamard10_probe_noise001")
    r10 = find_row(results, "rademacher10_probe_noise001")
    h5 = find_row(results, "hadamard5_probe_noise001")
    mnist = find_row(results, "mnist_hadamard5_hq")
    fashion = find_row(results, "fashion_hadamard5_hq")

    stl10_10_pass = truthy(h10.get("reaches_stl10_10pct_hq")) or truthy(h10_noise.get("reaches_stl10_10pct_hq"))
    stl10_5_pass = truthy(h5.get("reaches_stl10_5pct_hq"))
    simple_pass = truthy(mnist.get("reaches_simple_domain_hq")) or truthy(fashion.get("reaches_simple_domain_hq"))

    if not sanity_pass:
        conclusion = "Hadamard operator implementation is not yet reliable; fix measurement before training."
    elif not overfit_pass:
        conclusion = "Measurement is correct, but HQ model/training cannot overfit small data; focus on optimization/model bug."
    elif not stl10_10_pass:
        conclusion = "Pipeline can fit, but STL-10 10% generalization remains difficult; need more data/epochs/domain restriction."
    elif stl10_10_pass and not stl10_5_pass:
        conclusion = "High-quality is supported at 10% sampling, while 5% remains challenging."
    elif stl10_5_pass:
        conclusion = "High-quality STL-10 5% reconstruction is supported under the current internal threshold."
    elif simple_pass:
        conclusion = "High-quality claim should be restricted to simple structured targets."
    else:
        conclusion = "Current evidence is incomplete; missing runs should stay marked as missing."

    include_row = lookup_backproj(backproj, "0.1", "lowfreq_include_dc", "hadamard_zero_filled")
    skip_row = lookup_backproj(backproj, "0.1", "lowfreq_skip_dc", "hadamard_zero_filled")
    rad_row = lookup_backproj(backproj, "0.1", "rademacher", "ridge_pinv")
    lowfreq_row = include_row

    lines = [
        "# Phase 9 Report",
        "",
        "## Required Answers",
        "",
        f"1. Hadamard operator mathematically correct: {sanity_pass}.",
        f"2. Full sampling zero-filled exact reconstruction: {sanity_pass}, rel_error={sanity.get('full_sampling_rel_error', 'missing')}.",
        f"3. lowfreq_hadamard include DC vs skip DC: include={include_row or 'missing'}, skip={skip_row or 'missing'}.",
        f"4. zero-filled vs ridge/adjoint: see `{ROOT / 'phase9_backprojection_quality.csv'}`; rademacher ridge 10%={rad_row or 'missing'}, lowfreq zero-filled 10%={lowfreq_row or 'missing'}.",
        f"5. small-set overfit success: {overfit_pass}.",
        f"6. If overfit failed, do not continue large training because the model/training path cannot yet fit 32 real STL-10 samples.",
        f"7. hadamard10 no-noise high quality: {metric_pair(h10)}, reaches={h10.get('reaches_stl10_10pct_hq', 'missing')}.",
        f"8. hadamard10 noise=0.01 high quality: {metric_pair(h10_noise)}, reaches={h10_noise.get('reaches_stl10_10pct_hq', 'missing')}.",
        f"9. rademacher10 control: {metric_pair(r10)}.",
        f"10. 5% worth continuing: {h5.get('status', 'missing')} / {metric_pair(h5)}.",
        f"11. MNIST/Fashion sanity: MNIST {metric_pair(mnist)}, Fashion {metric_pair(fashion)}.",
        f"12. Current high-quality reconstruction claim supported: {stl10_10_pass or stl10_5_pass or simple_pass}.",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Key Paths",
        "",
        f"- phase9_results.csv: {ROOT / 'phase9_results.csv'}",
        f"- phase9_backprojection_quality.csv: {ROOT / 'phase9_backprojection_quality.csv'}",
        f"- sanity report: {ROOT / 'sanity_hadamard' / 'hadamard_sanity.md'}",
        f"- overfit report: {ROOT / 'overfit_hadamard_10pct' / 'overfit_metrics.md'}",
    ]
    (ROOT / "PHASE9_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Phase 9 report written to: {ROOT / 'PHASE9_REPORT.md'}")


if __name__ == "__main__":
    main()
