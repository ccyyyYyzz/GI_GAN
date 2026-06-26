from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase82_lineA_closure"
FIG = OUT / "figures"

P79_ROOT = ROOT / "outputs_phase79_posterior_anti_collapse"
P79_BASE = P79_ROOT / "baseline_rad5_collapse_eval"
P79_DIAG = P79_ROOT / "rad5_rowspace_diversity_diagnostic"
P80 = ROOT / "outputs_phase80_posterior_calibration" / "rad5_centered_diversity_anchor"
P81 = ROOT / "outputs_phase81_diversity_weight_scan"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def cov_at(summary: dict[str, Any], space: str, level: float) -> float:
    for row in summary["coverage_curve"]:
        if row["space"] == space and abs(float(row["level"]) - level) < 1e-9:
            return float(row["empirical_coverage"])
    raise KeyError((space, level))


def phase_metrics() -> dict[str, Any]:
    baseline = load_json(P79_BASE / "criteria_summary.json")
    p79_criteria = load_json(P79_DIAG / "final_criteria_eval" / "criteria_summary.json")
    p79_spec = load_json(P79_DIAG / "final_criteria_eval" / "p0_spectrum_summary.json")
    p79_cov = load_json(P79_DIAG / "calibration_validation" / "session_01_coverage" / "coverage_summary.json")
    p79_kappa = load_json(P79_DIAG / "calibration_validation" / "session_02_kappa" / "kappa_summary.json")
    p79_mean = load_json(P79_DIAG / "calibration_validation" / "session_03_mean_shift" / "mean_shift_summary.json")

    p80_train = load_json(P80 / "calibration_repair_summary.json")
    p80_criteria = load_json(P80 / "criteria_eval" / "criteria_summary.json")
    p80_spec = load_json(P80 / "criteria_eval" / "p0_spectrum_summary.json")
    p80_cov = load_json(P80 / "calibration_validation" / "session_01_coverage" / "coverage_summary.json")
    p80_kappa = load_json(P80 / "calibration_validation" / "session_02_kappa" / "kappa_summary.json")
    p80_mean = load_json(P80 / "calibration_validation" / "session_03_mean_shift" / "mean_shift_summary.json")

    p81_scan = load_json(P81 / "scan_summary.json")

    return {
        "baseline": baseline,
        "p79_criteria": p79_criteria,
        "p79_spec": p79_spec,
        "p79_cov": p79_cov,
        "p79_kappa": p79_kappa,
        "p79_mean": p79_mean,
        "p80_train": p80_train,
        "p80_criteria": p80_criteria,
        "p80_spec": p80_spec,
        "p80_cov": p80_cov,
        "p80_kappa": p80_kappa,
        "p80_mean": p80_mean,
        "p81_scan": p81_scan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_gate_rows(m: dict[str, Any]) -> list[dict[str, Any]]:
    p81_best = m["p81_scan"]["rows"][0]
    return [
        {
            "gate": 1,
            "name": "Deterministic Rad-5 collapse baseline",
            "claim": "Stochastic z is ignored by the deterministic checkpoint.",
            "std": m["baseline"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
            "p0_variance": m["baseline"]["overall"]["p0_variance_mean_over_y"],
            "pr_variance": m["baseline"]["overall"]["pr_variance_mean_over_y"],
            "relmeas_max": m["baseline"]["overall"]["relmeaserr_max_over_all"],
            "status": "FAIL anti-collapse; PASS measurement consistency",
        },
        {
            "gate": 2,
            "name": "Patched null/range criterion",
            "claim": "Exploding P0/PR ratio was a false pass unless absolute P0 variance is checked.",
            "std": m["baseline"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
            "p0_variance": m["baseline"]["overall"]["p0_variance_mean_over_y"],
            "pr_variance": m["baseline"]["overall"]["pr_variance_mean_over_y"],
            "status": "Baseline cleanly classified as collapse",
        },
        {
            "gate": 3,
            "name": "Row-space-only reconstruction diagnostic",
            "claim": "Full reconstruction loss was suppressing null-space diversity.",
            "std": m["p79_criteria"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
            "p0_variance": m["p79_criteria"]["overall"]["p0_variance_mean_over_y"],
            "pr_variance": m["p79_criteria"]["overall"]["pr_variance_mean_over_y"],
            "relmeas_max": m["p79_criteria"]["overall"]["relmeaserr_max_over_all"],
            "status": "PASS anti-collapse criteria",
        },
        {
            "gate": 4,
            "name": "Structured, non-white null-space variation",
            "claim": "Recovered diversity is low-frequency image structure, not white noise.",
            "p0_variation_slope": m["p79_spec"]["p0_variation"]["loglog_spectral_slope"],
            "high_to_low": m["p79_spec"]["p0_variation"]["high_to_low_power_ratio"],
            "status": "PASS spectral sanity",
        },
        {
            "gate": 5,
            "name": "Calibration audit of Phase79",
            "claim": "Diversity exists but posterior is not calibrated; failure shape is center drift.",
            "pixel_cov_90": cov_at(m["p79_cov"], "pixel", 0.9),
            "p0_cov_90": cov_at(m["p79_cov"], "p0_random_direction", 0.9),
            "kappa_det": m["p79_kappa"]["kappa_vs_deterministic"],
            "p0_mean_offset": m["p79_mean"]["means"]["mean_vs_det_p0_rmse"],
            "status": "FAIL calibration; kappa admissible",
        },
        {
            "gate": 6,
            "name": "P0 mean anchor + centered diversity",
            "claim": "Mean anchor fixes most center drift but over-constrains spread.",
            "std": m["p80_criteria"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
            "p0_variance": m["p80_criteria"]["overall"]["p0_variance_mean_over_y"],
            "pixel_cov_90": cov_at(m["p80_cov"], "pixel", 0.9),
            "p0_cov_90": cov_at(m["p80_cov"], "p0_random_direction", 0.9),
            "p0_mean_offset": m["p80_mean"]["means"]["mean_vs_det_p0_rmse"],
            "status": "Center improved; spread too narrow",
        },
        {
            "gate": 7,
            "name": "Anchor/diversity scan",
            "claim": "Spread increase improves coverage but plateaus around 45-48%.",
            "pixel_cov_90": p81_best["pixel_cov_90"],
            "p0_cov_90": p81_best["p0_cov_90"],
            "kappa_det": p81_best["kappa_vs_deterministic"],
            "p0_mean_offset": p81_best["mean_vs_det_p0_rmse"],
            "p0_variation_slope": p81_best["p0_variation_slope"],
            "status": "Coverage ceiling, not white-noise false success",
        },
        {
            "gate": 8,
            "name": "Base null-space accuracy bottleneck",
            "claim": "Calibration ceiling is set by base P0 center error, not by sampler mechanics.",
            "base_p0_det_to_gt_rmse": m["p79_mean"]["means"]["p0_det_to_gt_rmse"],
            "best_p0_cov_90": p81_best["p0_cov_90"],
            "best_pixel_cov_90": p81_best["pixel_cov_90"],
            "status": "Stop line A expansion; base retraining is a new problem",
        },
    ]


def plot_std_p0var(m: dict[str, Any]) -> Path:
    labels = ["Collapse", "Phase79", "Phase80", "Div2", "Div4", "Div8"]
    stds = [
        m["baseline"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
        m["p79_criteria"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
        m["p80_criteria"]["overall"]["mean_pixel_std_unclipped_mean_over_y"],
        *[row["fixed_y_std"] for row in m["p81_scan"]["rows"]],
    ]
    p0vars = [
        m["baseline"]["overall"]["p0_variance_mean_over_y"],
        m["p79_criteria"]["overall"]["p0_variance_mean_over_y"],
        m["p80_criteria"]["overall"]["p0_variance_mean_over_y"],
        *[row["fixed_y_p0_variance"] for row in m["p81_scan"]["rows"]],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, stds, color=["#777777", "#3684a5", "#cc8a33", "#58a65c", "#58a65c", "#58a65c"])
    axes[0].axhline(0.01, color="crimson", linestyle="--", linewidth=1, label="std gate")
    axes[0].set_yscale("log")
    axes[0].set_title("Mean pixel std")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, p0vars, color=["#777777", "#3684a5", "#cc8a33", "#58a65c", "#58a65c", "#58a65c"])
    axes[1].axhline(1e-4, color="crimson", linestyle="--", linewidth=1, label="P0 var gate")
    axes[1].set_yscale("log")
    axes[1].set_title("P0 variance")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig1_collapse_to_recovery_std_p0var.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def read_spectrum_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    bins: list[int] = []
    var: list[float] = []
    gt: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bins.append(int(row["bin"]))
            var.append(float(row["p0_variation_power"]))
            gt.append(float(row["p0_ground_truth_power"]))
    return bins, var, gt


def plot_spectrum() -> Path:
    spectra = [
        ("Phase79 variation", P79_DIAG / "final_criteria_eval" / "p0_radial_power_spectrum.csv"),
        ("Phase80 variation", P80 / "criteria_eval" / "p0_radial_power_spectrum.csv"),
        ("Phase81 div2 variation", P81 / "rad5_centered_anchor2_div2" / "criteria_eval" / "p0_radial_power_spectrum.csv"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, path in spectra:
        bins, var, gt = read_spectrum_csv(path)
        ax.plot(bins, var, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.plot(bins, gt, color="black", linestyle="--", linewidth=1.2, label="P0 ground truth reference")
    ax.set_yscale("log")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("normalized power")
    ax.set_title("P0 variation spectra remain low-frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig2_p0_radial_power_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_coverage(m: dict[str, Any]) -> Path:
    labels = ["Phase79", "Phase80", "Div2", "Div4", "Div8"]
    pixel = [
        cov_at(m["p79_cov"], "pixel", 0.9),
        cov_at(m["p80_cov"], "pixel", 0.9),
        *[row["pixel_cov_90"] for row in m["p81_scan"]["rows"]],
    ]
    p0 = [
        cov_at(m["p79_cov"], "p0_random_direction", 0.9),
        cov_at(m["p80_cov"], "p0_random_direction", 0.9),
        *[row["p0_cov_90"] for row in m["p81_scan"]["rows"]],
    ]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.18 for i in x], pixel, width=0.36, label="pixel 90%")
    ax.bar([i + 0.18 for i in x], p0, width=0.36, label="P0 dir 90%")
    ax.axhline(0.9, color="crimson", linestyle="--", linewidth=1, label="nominal 90%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage evolution and final plateau")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig3_coverage_evolution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ceiling(m: dict[str, Any]) -> Path:
    labels = ["Phase79", "Phase80", "Div2", "Div4", "Div8"]
    p0_cov = [
        cov_at(m["p79_cov"], "p0_random_direction", 0.9),
        cov_at(m["p80_cov"], "p0_random_direction", 0.9),
        *[row["p0_cov_90"] for row in m["p81_scan"]["rows"]],
    ]
    mean_to_gt = [
        m["p79_mean"]["means"]["p0_mean_to_gt_rmse"],
        m["p80_mean"]["means"]["p0_mean_to_gt_rmse"],
        *[row["p0_mean_to_gt_rmse"] for row in m["p81_scan"]["rows"]],
    ]
    offset = [
        m["p79_mean"]["means"]["mean_vs_det_p0_rmse"],
        m["p80_mean"]["means"]["mean_vs_det_p0_rmse"],
        *[row["mean_vs_det_p0_rmse"] for row in m["p81_scan"]["rows"]],
    ]
    base = m["p79_mean"]["means"]["p0_det_to_gt_rmse"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(mean_to_gt, p0_cov, s=60)
    for x, y, label in zip(mean_to_gt, p0_cov, labels):
        axes[0].annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].axvline(base, color="crimson", linestyle="--", linewidth=1, label=f"base P0 RMSE={base:.3f}")
    axes[0].axhline(0.9, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("P0 sample mean to GT RMSE")
    axes[0].set_ylabel("P0 90% empirical coverage")
    axes[0].set_title("Coverage ceiling near biased base center")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, offset, color="#7b9acc")
    axes[1].axhline(m["p79_mean"]["means"]["mean_vs_det_p0_rmse"], color="crimson", linestyle="--", linewidth=1, label="Phase79 center drift")
    axes[1].set_ylabel("sample mean vs deterministic P0 RMSE")
    axes[1].set_title("Center drift after repairs")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig4_coverage_ceiling_vs_base_p0_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def collect_audit(m: dict[str, Any], fig_paths: list[Path], gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    script_paths = [
        Path("E:/ns_mc_gan_gi_code/src/phase79_rad5_rowspace_diversity_diagnostic.py"),
        Path("E:/ns_mc_gan_gi_code/src/phase80_rad5_centered_diversity_calibration.py"),
        Path("E:/ns_mc_gan_gi_code/scripts/eval_posterior_sampling_criteria.py"),
        Path("E:/ns_mc_gan_gi_code/scripts/eval_posterior_calibration.py"),
        Path("E:/ns_mc_gan_gi_code/scripts/aggregate_posterior_calibration_shards.py"),
        Path("E:/ns_mc_gan_gi_code/scripts/assemble_lineA_closure.py"),
    ]
    small_hash_paths = [
        P79_BASE / "criteria_summary.json",
        P79_BASE / "per_sample_outputs.npz",
        P79_DIAG / "diagnostic_protocol.json",
        P79_DIAG / "split_manifest.json",
        P79_DIAG / "final_criteria_eval" / "criteria_summary.json",
        P79_DIAG / "final_criteria_eval" / "per_sample_outputs.npz",
        P79_DIAG / "calibration_validation" / "sample_bank_manifest.json",
        P79_DIAG / "calibration_validation" / "calibration_artifact_manifest.json",
        P80 / "calibration_repair_protocol.json",
        P80 / "split_manifest.json",
        P80 / "criteria_eval" / "criteria_summary.json",
        P80 / "criteria_eval" / "per_sample_outputs.npz",
        P80 / "calibration_validation" / "sample_bank_manifest.json",
        P80 / "calibration_validation" / "calibration_artifact_manifest.json",
        P81 / "scan_summary.json",
        P81 / "scan_summary.csv",
    ]
    for row in m["p81_scan"]["rows"]:
        root = Path(row["checkpoint"]).parents[1]
        small_hash_paths.extend(
            [
                root / "calibration_repair_protocol.json",
                root / "split_manifest.json",
                root / "criteria_eval" / "criteria_summary.json",
                root / "criteria_eval" / "per_sample_outputs.npz",
                root / "calibration_shards_aggregate" / "aggregate_calibration_summary.json",
            ]
        )
    records = []
    for path in script_paths + small_hash_paths + fig_paths:
        records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
                "kind": "script" if path in script_paths else ("figure" if path in fig_paths else "artifact"),
            }
        )
    checkpoint_records = [
        {
            "name": "baseline_rad5",
            "path": m["baseline"]["checkpoint"],
            "sha256": m["baseline"]["checkpoint_sha256"],
        },
        {
            "name": "phase79_final",
            "path": m["p79_criteria"]["checkpoint"],
            "sha256": m["p79_criteria"]["checkpoint_sha256"],
        },
        {
            "name": "phase80_final",
            "path": m["p80_criteria"]["checkpoint"],
            "sha256": m["p80_criteria"]["checkpoint_sha256"],
        },
        *[
            {"name": row["name"], "path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
            for row in m["p81_scan"]["rows"]
        ],
    ]
    sample_bank_records = [
        {
            "name": "phase79_calibration_bank",
            "path": m["p79_cov"]["sample_bank"]["samples_path"],
            "sha256": m["p79_cov"]["sample_bank"]["samples_sha256"],
            "N": m["p79_cov"]["sample_bank"]["N"],
            "K": m["p79_cov"]["sample_bank"]["K"],
        },
        {
            "name": "phase80_calibration_bank",
            "path": m["p80_cov"]["sample_bank"]["samples_path"],
            "sha256": m["p80_cov"]["sample_bank"]["samples_sha256"],
            "N": m["p80_cov"]["sample_bank"]["N"],
            "K": m["p80_cov"]["sample_bank"]["K"],
        },
    ]
    for row in m["p81_scan"]["rows"]:
        for i, sha in enumerate(row["shard_sample_sha256"].split(";")):
            sample_bank_records.append(
                {
                    "name": f"{row['name']}_shard_{i}",
                    "path": None,
                    "sha256": sha,
                    "N": 500,
                    "K": 50,
                }
            )
    return {
        "status": "complete",
        "output_dir": str(OUT),
        "gate_rows": gate_rows,
        "checkpoint_records": checkpoint_records,
        "sample_bank_records": sample_bank_records,
        "file_records": records,
        "notes": [
            "Large sample-bank hashes are taken from the already-written sample_bank_manifest or scan_summary records.",
            "Training split hashes are preserved in split_manifest.json and criteria summaries; evaluation used frozen main_rad5 cache.",
            "No new posterior training was launched by this closure script.",
        ],
    }


def write_markdown(m: dict[str, Any], gate_rows: list[dict[str, Any]], fig_paths: list[Path], audit: dict[str, Any]) -> Path:
    best = m["p81_scan"]["rows"][0]
    lines = [
        "# Line A Mechanistic Closure: Measurement-Consistent GAN Posterior Sampling",
        "",
        "## Core Claim",
        "",
        "Line A should stop here. The experiment did not produce a calibrated posterior sampler, but it did isolate a publishable mechanism: in this Rad-5 ghost-imaging setup, measurement consistency and anti-collapse can be achieved, yet posterior calibration is capped by the null-space accuracy of the deterministic base reconstruction. Once the sample cloud is anchored around a biased base P0 center, diversity tuning improves coverage only up to a ceiling of about 48%.",
        "",
        "## Fixed Experimental Invariants",
        "",
        "- Setup: Rad-5 Rademacher ghost-imaging configuration; dataset/sampling rate unchanged.",
        "- Projection audit: each sample is evaluated after measurement-consistency projection; RelMeasErr remained below 1e-2 in all reported gates.",
        "- Sampling audit: fixed y with K=50 for anti-collapse gates; full frozen main_rad5 test cache with N=2000, K=50 for calibration gates.",
        "- Split discipline: training phases use STL10 train+unlabeled train partition; calibration evaluation uses frozen main_rad5 cache.",
        "- Pre-registered criteria were not relaxed after seeing results; the final scan uses coverage as the spread target, not the std lower bound.",
        "",
        "## Eight Gates",
        "",
        "| Gate | Result | Key Evidence | Status |",
        "|---:|---|---|---|",
    ]
    for row in gate_rows:
        evidence_parts = []
        for key in [
            "std",
            "p0_variance",
            "pixel_cov_90",
            "p0_cov_90",
            "p0_mean_offset",
            "p0_variation_slope",
            "base_p0_det_to_gt_rmse",
            "best_p0_cov_90",
        ]:
            if key in row:
                evidence_parts.append(f"{key}={row[key]:.4g}" if isinstance(row[key], float) else f"{key}={row[key]}")
        lines.append(f"| {row['gate']} | {row['name']} | {'; '.join(evidence_parts)} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            "1. The original deterministic checkpoint was measurement-consistent but collapsed under stochastic z. Its mean pixel std was only 9.83e-4, and absolute P0 variance was 1.09e-6. The initially huge P0/PR ratio was therefore a false signal caused by an almost-zero denominator.",
            "2. Removing reconstruction loss from the null space confirmed the suspected disease: full image reconstruction loss was flattening P0 diversity because each y has only one supervised x. Row-space-only reconstruction plus adversarial/diversity terms lifted fixed-y std to 0.0335 and P0 variance to 1.28e-3 while preserving RelMeasErr.",
            "3. The restored variation was not white noise. Phase79 P0 variation had a log-log spectral slope of -2.62, with low-frequency power strongly dominating high-frequency power.",
            "4. Calibration then failed in a different way. Phase79 90% intervals covered only 38.1% of pixels and 40.7% of random P0 directions. kappa stayed admissible, but the sample mean shifted by 0.0549 in P0 and was farther from GT than the deterministic point estimate.",
            "5. A P0 sample-mean anchor plus centered diversity fixed most of the location failure: P0 mean offset dropped to 0.018. But this also crushed spread, reducing 90% coverage to about 10%.",
            "6. Rebalancing anchor/diversity recovered spread and lifted 90% coverage to about 45-48%. Increasing lambda_diversity from 2 to 4 to 8 did not continue improving coverage, and spectra stayed low-frequency rather than turning white.",
            f"7. The best scan point was lambda_anchor=2, lambda_diversity=2: pixel 90% coverage={best['pixel_cov_90']:.3f}, P0 90% coverage={best['p0_cov_90']:.3f}, kappa_det={best['kappa_vs_deterministic']:.3f}, P0 mean offset={best['mean_vs_det_p0_rmse']:.3f}.",
            f"8. The base deterministic P0-to-GT RMSE is {m['p79_mean']['means']['p0_det_to_gt_rmse']:.3f}. This biased center is the observed calibration bottleneck: widening samples around it cannot make a calibrated posterior without solving the harder base null-space reconstruction problem.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in fig_paths:
        lines.append(f"- {path.name}: `{path}`")
    lines.extend(
        [
            "",
            "## Reproducibility Ledger",
            "",
            f"- Audit JSON: `{OUT / 'lineA_artifact_audit.json'}`",
            f"- Gate table CSV: `{OUT / 'lineA_gate_table.csv'}`",
            f"- Top-level scan summary: `{P81 / 'scan_summary.json'}`",
            "- All checkpoint, sample-bank, per-sample-output, split, and protocol paths with hashes are recorded in the audit JSON.",
            "",
            "## Stop Decision",
            "",
            "Line A has reached its useful stopping point. Continuing inside posterior mechanics is unlikely to remove the coverage ceiling; the next real lever would be retraining a base reconstructor with substantially better P0 accuracy, which is a separate long-horizon problem. Compute should move back to Line B.",
            "",
        ]
    )
    path = OUT / "lineA_mechanistic_narrative.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown_zh(m: dict[str, Any], gate_rows: list[dict[str, Any]], fig_paths: list[Path]) -> Path:
    best = m["p81_scan"]["rows"][0]
    lines = [
        "# 线A机理收尾: measurement-consistent GAN posterior sampling",
        "",
        "## 核心结论",
        "",
        "线A在这里停止扩展。它没有得到校准后验采样器,但给出了一个可发表的机理性结果:在 Rad-5 鬼成像 setup 中,测量一致性和 anti-collapse 可以同时做到,但后验校准的上限由 deterministic base 重建器的零空间精度决定,不是由 posterior sampling trick 本身决定。当样本云围绕一个有偏的 P0 中心展开时,调 diversity 只能把 90% 覆盖率推到约 48%,再加宽不会继续接近 90%。",
        "",
        "## 不变量",
        "",
        "- 数据和采样率固定为 Rad-5,没有换 dataset 或 measurement rate。",
        "- 训练只使用 train split; calibration 评估使用 frozen main_rad5 test cache。",
        "- anti-collapse gate 使用固定 y、K=50; calibration gate 使用 N=2000、K=50。",
        "- 所有样本都有 measurement consistency 审计; 本线所有成功/失败判断中的 RelMeasErr 都远低于 1e-2。",
        "- 判据没有事后放宽:最终扫描明确用 coverage 调 spread,而不是用 std 下界替代校准。",
        "",
        "## 八个 gate",
        "",
        "| Gate | 机理节点 | 关键数值 | 结论 |",
        "|---:|---|---|---|",
    ]
    for row in gate_rows:
        evidence = []
        for key in [
            "std",
            "p0_variance",
            "pixel_cov_90",
            "p0_cov_90",
            "p0_mean_offset",
            "p0_variation_slope",
            "base_p0_det_to_gt_rmse",
            "best_p0_cov_90",
        ]:
            if key in row:
                value = row[key]
                evidence.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
        lines.append(f"| {row['gate']} | {row['name']} | {'; '.join(evidence)} | {row['status']} |")
    lines.extend(
        [
            "",
            "## 机理链",
            "",
            "1. baseline deterministic checkpoint 在测量上一致,但 z 基本被忽略: mean pixel std=9.83e-4, P0 variance=1.09e-6。原来的 P0/PR ratio 爆炸只是 PR 分母接近浮点零造成的假阳性。",
            "2. 给判据2加绝对 P0 variance 后,baseline 被干净判为 collapse: P0 和 PR 两个子空间的多样性都近似为零。",
            "3. 把 reconstruction loss 改成只惩罚 row-space 后,collapse 被打开: Phase79 fixed-y std=0.0335, P0 variance=1.28e-3, PR variance 仍约为 5.7e-12, RelMeasErr 仍受控。这确认 full recon loss 在零空间压平多样性。",
            "4. Phase79 的 P0 variation 谱 slope=-2.62,低频主导,不是白噪声。这个结果说明恢复的是结构化图像补全自由度,不是垃圾噪声。",
            "5. 但 Phase79 不是校准后验: 90% nominal coverage 只有 38.1% pixel / 40.7% P0 direction。kappa 在 [1,2] 内,说明主要坏在 location/中心,不是 covariance 量级完全错。",
            "6. P0 mean anchor + centered diversity 把 P0 center offset 从 0.0549 降到 0.018,说明中心漂移机制判断正确;但锚太强导致 spread 被压窄,90% coverage 只剩约 10%。",
            "7. 重新平衡 anchor/diversity 后,coverage 回升但卡住: 最佳 lambda_anchor=2, lambda_diversity=2, pixel 90% coverage=0.453, P0 90% coverage=0.478, kappa_det=1.327, P0 variation slope=-2.181。lambda_diversity 到 4/8 没继续提高 coverage。",
            f"8. 根本瓶颈是 base P0 中心误差: deterministic-to-GT P0 RMSE={m['p79_mean']['means']['p0_det_to_gt_rmse']:.3f}。围绕这个偏移中心采样,覆盖率存在约 48% 的天花板;突破它需要重训更准的 base reconstructor,这是新方向,不是线A继续调 posterior 的 gate。",
            "",
            "## 图表",
            "",
        ]
    )
    for path in fig_paths:
        lines.append(f"- {path.name}: `{path}`")
    lines.extend(
        [
            "",
            "## 可复现性",
            "",
            f"- artifact/hash 审计: `{OUT / 'lineA_artifact_audit.json'}`",
            f"- 八 gate 表: `{OUT / 'lineA_gate_table.csv'}`",
            f"- Phase81 扫描总表: `{P81 / 'scan_summary.json'}`",
            "- audit JSON 记录 checkpoint、per-sample output、sample bank shard、split manifest、protocol/criteria summary 和脚本 hash。",
            "",
            "## 停止决定",
            "",
            "线A停止。继续 posterior 机制调参不能解决 base P0 center error; 下一步应该把注意力和算力转回线B。",
            "",
        ]
    )
    path = OUT / "lineA_mechanistic_narrative_zh.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ensure_dirs()
    metrics = phase_metrics()
    gate_rows = build_gate_rows(metrics)
    write_csv(OUT / "lineA_gate_table.csv", gate_rows)
    fig_paths = [
        plot_std_p0var(metrics),
        plot_spectrum(),
        plot_coverage(metrics),
        plot_ceiling(metrics),
    ]
    audit = collect_audit(metrics, fig_paths, gate_rows)
    narrative_path = write_markdown(metrics, gate_rows, fig_paths, audit)
    narrative_zh_path = write_markdown_zh(metrics, gate_rows, fig_paths)
    audit["file_records"].append(
        {
            "path": str(narrative_path),
            "exists": True,
            "bytes": narrative_path.stat().st_size,
            "sha256": sha256_file(narrative_path),
            "kind": "narrative",
        }
    )
    audit["file_records"].append(
        {
            "path": str(narrative_zh_path),
            "exists": True,
            "bytes": narrative_zh_path.stat().st_size,
            "sha256": sha256_file(narrative_zh_path),
            "kind": "narrative_zh",
        }
    )
    audit["file_records"].append(
        {
            "path": str(OUT / "lineA_gate_table.csv"),
            "exists": True,
            "bytes": (OUT / "lineA_gate_table.csv").stat().st_size,
            "sha256": sha256_file(OUT / "lineA_gate_table.csv"),
            "kind": "gate_table",
        }
    )
    audit_path = OUT / "lineA_artifact_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    file_rows = audit["file_records"]
    write_csv(OUT / "lineA_artifact_audit.csv", file_rows)
    summary = {
        "status": "complete",
        "output_dir": str(OUT),
        "narrative": str(narrative_path),
        "narrative_sha256": sha256_file(narrative_path),
        "narrative_zh": str(narrative_zh_path),
        "narrative_zh_sha256": sha256_file(narrative_zh_path),
        "gate_table": str(OUT / "lineA_gate_table.csv"),
        "gate_table_sha256": sha256_file(OUT / "lineA_gate_table.csv"),
        "audit_json": str(audit_path),
        "audit_json_sha256": sha256_file(audit_path),
        "figures": [{"path": str(path), "sha256": sha256_file(path)} for path in fig_paths],
        "best_observed": {
            "lambda_anchor": 2.0,
            "lambda_diversity": 2.0,
            "pixel_cov_90": metrics["p81_scan"]["rows"][0]["pixel_cov_90"],
            "p0_cov_90": metrics["p81_scan"]["rows"][0]["p0_cov_90"],
            "base_p0_det_to_gt_rmse": metrics["p79_mean"]["means"]["p0_det_to_gt_rmse"],
        },
    }
    summary_path = OUT / "closure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
