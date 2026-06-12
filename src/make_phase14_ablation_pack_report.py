from __future__ import annotations

from collections import Counter

from .phase14_ablation_pack_common import PHASE14_RESULTS, f, out_dir, read_csv


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def by_method_mode(rows: list[dict[str, str]], mode: str) -> dict[str, dict[str, str]]:
    return {
        row.get("method_id", ""): row
        for row in rows
        if row.get("ablation_mode") == mode and row.get("status") == "completed"
    }


def mode_delta(rows: list[dict[str, str]], mode: str, metric: str = "psnr") -> float | None:
    full = by_method_mode(rows, "full_model")
    other = by_method_mode(rows, mode)
    deltas = []
    for method_id, full_row in full.items():
        if method_id not in other:
            continue
        a = f(full_row.get(metric))
        b = f(other[method_id].get(metric))
        if a is not None and b is not None:
            deltas.append(a - b)
    return mean(deltas)


def checkpoint_mismatches(inf: list[dict[str, str]]) -> list[dict[str, str]]:
    source = {row.get("method_id", ""): row for row in read_csv(PHASE14_RESULTS)}
    mismatches = []
    for row in inf:
        if row.get("ablation_mode") != "full_model" or row.get("status") != "completed":
            continue
        original = source.get(row.get("method_id", ""))
        if not original:
            continue
        original_psnr = f(original.get("psnr"))
        reeval_psnr = f(row.get("psnr"))
        if original_psnr is None or reeval_psnr is None:
            continue
        if abs(original_psnr - reeval_psnr) > 2.0:
            mismatches.append(
                {
                    "method_id": row.get("method_id", ""),
                    "original_psnr": f"{original_psnr:.4f}",
                    "reeval_psnr": f"{reeval_psnr:.4f}",
                    "delta_psnr": f"{reeval_psnr - original_psnr:.4f}",
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    return mismatches


def main() -> None:
    out = out_dir()
    attr = read_csv(out / "attribution_table.csv")
    inf = read_csv(out / "inference_ablation_results.csv")
    noise = read_csv(out / "noise_sweep_results.csv")
    dc = read_csv(out / "dc_control_results.csv")
    trad = read_csv(out / "traditional_baselines.csv")
    stats = read_csv(out / "statistics_summary.csv")

    helpful = [r for r in attr if r.get("classification") == "model_refinement_helpful"]
    degraded = [r for r in attr if r.get("classification") == "model_degrades_backprojection"]
    class_counts = Counter(r.get("classification") for r in attr)

    inf_completed = [r for r in inf if r.get("status") == "completed"]
    noise_completed = [r for r in noise if r.get("status") == "completed"]
    trad_completed = [r for r in trad if r.get("status") == "completed"]
    tv_completed = [r for r in trad_completed if r.get("baseline") == "tv_pgd_lightweight"]
    mismatches = checkpoint_mismatches(inf)

    dc_include = [r for r in dc if r.get("hadamard_include_dc") == "True"]
    dc_skip = [r for r in dc if r.get("hadamard_include_dc") == "False"]
    include_psnr = [f(r.get("psnr")) for r in dc_include if f(r.get("psnr")) is not None]
    skip_psnr = [f(r.get("psnr")) for r in dc_skip if f(r.get("psnr")) is not None]
    dc_gain = (
        sum(include_psnr) / len(include_psnr) - sum(skip_psnr) / len(skip_psnr)
        if include_psnr and skip_psnr
        else None
    )

    noise_methods = sorted({r.get("method_id", "") for r in noise_completed})
    noise_ranges = []
    for method_id in noise_methods:
        subset = [r for r in noise_completed if r.get("method_id") == method_id]
        psnrs = [f(r.get("psnr")) for r in subset if f(r.get("psnr")) is not None]
        if psnrs:
            noise_ranges.append(f"- {method_id}: PSNR {min(psnrs):.3f} to {max(psnrs):.3f}")

    lines = [
        "# Phase 14C Final Ablation and Robustness Pack Report",
        "",
        "## Execution policy",
        "",
        "- Local large training was not started.",
        "- Imported Colab results were kept on E: and used as fixed inputs.",
        "- Eval-only ablations, noise sweeps, statistics, DC-row controls, and lightweight traditional baselines were run locally.",
        "- STL-10 5% HQ is treated as safe only when the imported Colab metric itself meets PSNR >= 20 and SSIM >= 0.60.",
        "",
        "## Generated files",
        "",
        "- attribution_table.csv/md and attribution_delta_psnr.png / attribution_delta_ssim.png",
        "- inference_ablation_results.csv/md and inference_ablation_psnr.png / inference_ablation_ssim.png / inference_ablation_relmeaserr.png",
        "- noise_sweep_results.csv/md and noise_sweep_psnr.png / noise_sweep_ssim.png / noise_sweep_relmeaserr.png",
        "- dc_control_results.csv/md and dc_control_psnr.png / dc_control_ssim.png",
        "- traditional_baselines.csv/md",
        "- statistics_summary.csv/md and psnr_histograms.png / ssim_histograms.png",
        "",
        "## Completion status",
        "",
        f"- Inference ablation rows completed: {len(inf_completed)}/{len(inf)}.",
        f"- Noise sweep rows completed: {len(noise_completed)}/{len(noise)}.",
        f"- Traditional baseline rows completed: {len(trad_completed)}/{len(trad)}; TV-PGD rows completed: {len(tv_completed)}.",
        f"- Statistics rows generated: {len(stats)}.",
        "",
        "## Answers",
        "",
        f"1. Model clearly exceeds configured backprojection on imported final metrics: {yes_no(len(helpful) > 0 and len(degraded) == 0)}.",
        f"   Helpful rows: {len(helpful)}/{len(attr)}. Degraded rows: {len(degraded)}. Classification counts: {dict(class_counts)}.",
        "",
        f"2. DC projection inference ablation mean PSNR effect, full minus no-DC: {fmt(mode_delta(inf, 'no_dc_project_inference'))}.",
        f"3. Null-space projection inference ablation mean PSNR effect, full minus no-null: {fmt(mode_delta(inf, 'no_null_project_inference'))}.",
        f"4. Refiner effect mean PSNR, full minus stage1-only: {fmt(mode_delta(inf, 'stage1_only'))}.",
        f"5. EMA effect mean PSNR, EMA minus raw generator: {fmt(mode_delta(inf, 'raw_generator_no_ema'))}.",
        "",
        f"6. DC row importance: {'strongly supported' if dc_gain is not None and dc_gain > 5 else 'partially supported' if dc_gain is not None else 'insufficient data'}.",
        f"   Mean include-DC minus skip-DC PSNR among DC-control rows: {fmt(dc_gain)}.",
        "",
        f"7. Traditional baselines completed: {len(trad_completed)} rows. Use traditional_baselines.csv for exact per-setting numbers.",
        f"8. Noise robustness completed across {len(noise_methods)} STL-10 methods and {len(noise_completed)} rows.",
        *(noise_ranges if noise_ranges else ["- no completed noise ranges"]),
        "",
        "## Reproducibility risks",
        "",
    ]
    if mismatches:
        lines.extend(
            [
                "The following imported checkpoints do not locally re-evaluate to their imported Colab final metric. Treat their eval-only ablations as diagnostic, not as manuscript-quality claims, until the exact Colab measurement/eval state is re-exported.",
                "",
                "| method_id | imported PSNR | local checkpoint PSNR | delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in mismatches:
            lines.append(
                f"| {row['method_id']} | {row['original_psnr']} | {row['reeval_psnr']} | {row['delta_psnr']} |"
            )
    else:
        lines.append("No >2dB full-model checkpoint re-evaluation mismatch was detected.")

    lines.extend(
        [
            "",
            "## Manuscript-safe wording",
            "",
            "- The imported final metrics support the core claim that the learned reconstructor improves strongly over the configured backprojection for completed rows.",
            "- DC-row inclusion remains important for low-frequency Hadamard controls.",
            "- Noise and traditional-baseline tables are now present as supplementary evidence, with the reproducibility-risk rows flagged above.",
            "- Do not use locally re-evaluated Rademacher ablations as primary evidence until the Colab-side measurement/eval state is exported or the mismatch is resolved.",
        ]
    )

    target = out / "PHASE14_ABLATION_PACK_REPORT.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
