from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


PH76 = Path("E:/ns_mc_gan_gi/outputs_phase76_high_upside_auditable_gan_exploration")
PH75 = Path("E:/ns_mc_gan_gi/outputs_phase75_final_high_tier_validation")
PH73 = Path("E:/ns_mc_gan_gi/outputs_phase73_overnight_gauge_gan_expansion")
PH71 = Path("E:/ns_mc_gan_gi/outputs_phase71_gauge_cgan_paired_seeds")
PH72 = Path("E:/ns_mc_gan_gi/outputs_phase72_scr10_gauge_cgan_regime_validation")
OUT = Path("E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly")


REQUIRED_PHASE76_REPORTS = [
    "PHASE76_HIGH_UPSIDE_FINAL_REPORT.md",
    "UNMEASURED_CONTENT_MAP_REPORT.md",
    "ALPHA_IDENTITY_CHECK_REPORT.md",
    "FAILURE_DETECTOR_AUC_REPORT.md",
    "Z_VARIATION_DIAGNOSTIC_REPORT.md",
    "SECOND_INVERSE_PROBLEM_TOY_FEASIBILITY.md",
    "AUDITABLE_GAN_DRAFT_OUTLINE.md",
    "FIGURE_PLAN_PHASE76.md",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def latex_escape(text: str) -> str:
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


def source_report_path(name: str) -> Path | None:
    candidates = [PH76 / name, PH76 / "reports" / name]
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_required_sources() -> tuple[dict[str, str], list[dict[str, object]]]:
    reports: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for name in REQUIRED_PHASE76_REPORTS:
        path = source_report_path(name)
        if path is None:
            rows.append(
                {
                    "required_name": name,
                    "status": "missing",
                    "resolved_path": "",
                    "sha256": "",
                    "bytes": 0,
                }
            )
            reports[name] = ""
            continue
        reports[name] = read_text(path)
        rows.append(
            {
                "required_name": name,
                "status": "read",
                "resolved_path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return reports, rows


def make_evidence_summary() -> dict[str, object]:
    alpha = read_csv_rows(PH76 / "tables" / "alpha_identity_check.csv")
    failure = read_csv_rows(PH76 / "tables" / "failure_detector_auc.csv")
    zvar = read_csv_rows(PH76 / "tables" / "z_variation_metrics.csv")
    second = read_csv_rows(PH76 / "tables" / "second_task_diagnostic.csv")
    unmeasured = read_csv_rows(PH76 / "tables" / "unmeasured_content_summary.csv")
    regime_map = read_csv_rows(PH75 / "regime_map_final.csv")
    scr10 = read_csv_rows(PH72 / "scr10_gauge_signal_auc.csv")
    rad5 = read_csv_rows(PH73 / "rad5_gauge_signal_auc.csv")
    shortcut = read_csv_rows(PH75 / "shortcut_stress_summary.csv")
    scr5_seed = read_csv_rows(PH71 / "scr5_seed_delta_metrics.csv")

    max_alpha_resid = max((fnum(r.get("max_Av_alpha_minus_Av0")) for r in alpha), default=0.0)
    rel_ranges: dict[str, float] = {}
    for regime in sorted({r.get("regime", "") for r in alpha}):
        vals = [fnum(r.get("mean_post_relmeaserr")) for r in alpha if r.get("regime") == regime]
        if vals:
            rel_ranges[regime] = max(vals) - min(vals)

    best_failure = max(failure, key=lambda r: fnum(r.get("auc")), default={})
    z_mean = mean([fnum(r.get("pixel_std_mean")) for r in zvar]) if zvar else 0.0
    second_auc = {r.get("model", ""): fnum(r.get("auc")) for r in second}

    shortcut_row = [
        r
        for r in shortcut
        if r.get("model") == "gauge_D_score"
        and r.get("base_kind") == "fake_mean"
        and r.get("perturb_type") == "row"
        and r.get("alpha") == "0.1"
    ]
    standard_row = [
        r
        for r in shortcut
        if r.get("model") == "standard_D_score"
        and r.get("base_kind") == "fake_mean"
        and r.get("perturb_type") == "row"
        and r.get("alpha") == "0.1"
    ]

    scr5_positive = 0
    scr5_count = 0
    for row in scr5_seed:
        metric = row.get("metric", "")
        if metric in {"lpips", "rapsd_distance"}:
            scr5_count += 1
            if fnum(row.get("improvement_positive_means_C_better")) > 0:
                scr5_positive += 1

    return {
        "alpha_rows": alpha,
        "alpha_max_Av_alpha_minus_Av0": max_alpha_resid,
        "alpha_relmeaserr_range_by_regime": rel_ranges,
        "failure_rows": failure,
        "best_failure_feature": best_failure.get("feature", ""),
        "best_failure_auc": fnum(best_failure.get("auc")),
        "best_failure_ci": f"{best_failure.get('auc_ci_low', '')}-{best_failure.get('auc_ci_high', '')}",
        "z_pixel_std_mean": z_mean,
        "second_task_auc_by_model": second_auc,
        "unmeasured_rows": unmeasured,
        "regime_map_rows": regime_map,
        "scr10_rows": scr10,
        "rad5_rows": rad5,
        "shortcut_gauge_row_delta_alpha_0p1": fnum(shortcut_row[0].get("mean_abs_delta_vs_alpha0")) if shortcut_row else 0.0,
        "shortcut_standard_row_delta_alpha_0p1": fnum(standard_row[0].get("mean_abs_delta_vs_alpha0")) if standard_row else 0.0,
        "scr5_lpips_rapsd_positive_count": scr5_positive,
        "scr5_lpips_rapsd_total_count": scr5_count,
    }


def claim_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rel_ranges = summary["alpha_relmeaserr_range_by_regime"]
    assert isinstance(rel_ranges, dict)
    regime_rows = summary["regime_map_rows"]
    assert isinstance(regime_rows, list)
    regime_summary = "; ".join(
        f"{r.get('regime')}: AUC {r.get('gauge_auc')} ({r.get('outcome')})" for r in regime_rows
    )
    return [
        {
            "id": "C1",
            "claim": "The GAN branch should be framed as a prior/detail engine whose outputs remain subordinate to a measurement certificate.",
            "support": "Phase76 final report and Phase75/73 readiness reports.",
            "allowed_strength": "Main positioning claim.",
            "caveat": "Do not present GAN scores as the certificate.",
        },
        {
            "id": "C2",
            "claim": "Pi_y^lambda is the explicit measurement certificate used to restore and audit bucket consistency.",
            "support": "Phase71 and Phase75 reports confirm RelMeasErr is evaluated after the projection/audit path.",
            "allowed_strength": "Method claim.",
            "caveat": "Paper 1 supplies the premise; do not restate Paper 1 performance as GAN evidence.",
        },
        {
            "id": "C3",
            "claim": "P0 xhat can be visualized as an unmeasured-content or prior-supplied content map.",
            "support": "UNMEASURED_CONTENT_MAP_REPORT.md and fig_unmeasured_content_maps.",
            "allowed_strength": "Accountability visualization claim.",
            "caveat": "It identifies measurement-unconstrained content, not truth/falsity by itself.",
        },
        {
            "id": "C4",
            "claim": "The alpha trust knob changes P0 content while leaving measured consistency effectively invariant.",
            "support": f"alpha_identity_check.csv: max Av-alpha difference {summary['alpha_max_Av_alpha_minus_Av0']:.3e}; RelMeasErr ranges {rel_ranges}.",
            "allowed_strength": "Core technical claim.",
            "caveat": "Quality changes are prior-axis changes; measured consistency belongs to the certificate path.",
        },
        {
            "id": "C5",
            "claim": "Gauge equalization removes a residual shortcut available to a naive discriminator.",
            "support": f"Phase75 shortcut stress: standard row delta {summary['shortcut_standard_row_delta_alpha_0p1']:.6f}, gauge row delta {summary['shortcut_gauge_row_delta_alpha_0p1']:.6f} at alpha 0.1.",
            "allowed_strength": "Mechanism/safety claim.",
            "caveat": "This is evidence against a known shortcut, not a universal guarantee.",
        },
        {
            "id": "C6",
            "claim": "The diagnostic gate explains regime dependence: Scr-5 and Rad-5 are positive, while Scr-10 and Rad-10 are weak-risk regimes.",
            "support": regime_summary,
            "allowed_strength": "Evidence-scope claim.",
            "caveat": "Do not claim broad validity beyond tested regimes.",
        },
        {
            "id": "C7",
            "claim": "Failure detection is a preliminary stress-label signal only.",
            "support": f"FAILURE_DETECTOR_AUC_REPORT.md: best feature {summary['best_failure_feature']} AUC {summary['best_failure_auc']:.3f}, CI {summary['best_failure_ci']}.",
            "allowed_strength": "Limitation / future-work claim.",
            "caveat": "Not deployable; not a validated distribution-shift detector.",
        },
        {
            "id": "C8",
            "claim": "Latent z sampling collapsed in this branch and should not be used as uncertainty evidence.",
            "support": f"Z_VARIATION_DIAGNOSTIC_REPORT.md: mean pixel std {summary['z_pixel_std_mean']:.6f}.",
            "allowed_strength": "Negative result.",
            "caveat": "Do not make a sampling-calibration claim.",
        },
        {
            "id": "C9",
            "claim": "The second inverse-problem result is a toy feasibility check for the gauge diagnostic idea.",
            "support": f"SECOND_INVERSE_PROBLEM_TOY_FEASIBILITY.md: {summary['second_task_auc_by_model']}.",
            "allowed_strength": "Supplementary feasibility note.",
            "caveat": "It is not a trained second reconstruction method.",
        },
    ]


def unsupported_rows() -> list[dict[str, object]]:
    return [
        {
            "claim": "The method has broad benchmark dominance over all inverse solvers.",
            "reason": "No vetted external diffusion/PnP or full competitive benchmark has been completed.",
            "replacement": "Report a measurement-accountable GAN prior branch under tested regimes.",
        },
        {
            "claim": "The method is empirically above diffusion inverse solvers.",
            "reason": "Phase75/76 explicitly leave diffusion/PnP comparison as future evidence.",
            "replacement": "Position against diffusion solvers conceptually via certificate and prior-axis control.",
        },
        {
            "claim": "The GAN improves measured consistency.",
            "reason": "Measured consistency is imposed by the audit/projection certificate.",
            "replacement": "The GAN changes prior-supplied detail while the certificate preserves RelMeasErr.",
        },
        {
            "claim": "P0 xhat establishes truth or falsity of a structure.",
            "reason": "P0 maps identify unmeasured components; ground truth is unavailable in deployment.",
            "replacement": "Use unmeasured-content map and, where GT exists, null-error as an evaluation-only diagnostic.",
        },
        {
            "claim": "The failure/OOD detector is reliable.",
            "reason": "Best signal is weak-to-moderate on artificial stress labels.",
            "replacement": "Call it a preliminary failure-signal exploration.",
        },
        {
            "claim": "z sampling gives calibrated sampling spread.",
            "reason": "Phase76 found the z path collapsed.",
            "replacement": "Report collapse and exclude sampling-based uncertainty claims.",
        },
        {
            "claim": "Paper 1's main reconstruction gains are GAN results.",
            "reason": "Paper 1 is the measurement-accountability premise, not this GAN branch.",
            "replacement": "Cite Paper 1 only for separability of quality and certificate auditability.",
        },
    ]


def figure_inventory_rows() -> list[dict[str, object]]:
    roots = [
        ("Phase76", PH76 / "figs"),
        ("Phase75", PH75),
        ("Phase73", PH73),
        ("Phase71", PH71),
        ("Phase72", PH72),
    ]
    roles = {
        "fig_unmeasured_content_maps": "Main: unmeasured-content maps.",
        "fig_alpha_metric_curves": "Main: alpha trust/quality curves.",
        "fig_alpha_trust_sharpness_tradeoff": "Main or supplement: prior-content tradeoff.",
        "fig_alpha_relmeaserr_invariance": "Main: certificate invariance under alpha.",
        "score_vs_relmeaserr_standard_vs_gauge": "Main: shortcut safety.",
        "regime_map_auc_and_outcome": "Main: diagnostic gate/regime dependence.",
        "fig_failure_signal_boxplots": "Supplement: preliminary failure signals.",
        "fig_failure_detector_roc": "Supplement: failure signal ROC.",
        "fig_z_samples": "Supplement: z collapse.",
        "fig_z_uncertainty_map": "Supplement: z collapse visualization.",
        "scr5_seed_visual_grid": "Supplement: Scr-5 paired seed visuals.",
        "rad5_visual_grid": "Supplement: Rad-5 paired seed visuals.",
        "scr10_gauge_score_histograms": "Supplement: Scr-10 weak gate.",
    }
    rows: list[dict[str, object]] = []
    for phase, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".png", ".pdf"}:
                continue
            stem = path.stem
            role = "Archive/supporting asset."
            for key, value in roles.items():
                if key in stem:
                    role = value
                    break
            rows.append(
                {
                    "source_phase": phase,
                    "asset_name": path.name,
                    "asset_path": str(path),
                    "format": path.suffix.lower().lstrip("."),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "paper_role": role,
                }
            )
    return rows


def build_claim_map_md(rows: list[dict[str, object]], source_rows: list[dict[str, object]]) -> str:
    return f"""
# Final Claim Map: Auditable GAN Paper

Generated: `{now_iso()}`

Scope lock: this map is assembled from Phase76 plus Phase75/73/71/72 evidence already on disk. It records paper-allowed claims only. No generator, reconstruction network, GAN fine-tune, or first-paper result was trained or modified in Phase77.

## Source Read Log

{markdown_table(source_rows, ["required_name", "status", "resolved_path", "sha256", "bytes"])}

## Allowed Claims

{markdown_table(rows, ["id", "claim", "support", "allowed_strength", "caveat"])}

## Paper-1 Boundary

Paper 1 may be used only as the premise that measurement accountability is separable from perceptual/detail priors and that an audit certificate exists. Its main empirical results are not GAN results.
"""


def build_unsupported_md(rows: list[dict[str, object]]) -> str:
    return f"""
# Unsupported Claims: Auditable GAN Paper

Generated: `{now_iso()}`

These claims are explicitly excluded from the Phase77 draft. They are listed to prevent accidental overreach during later editing.

{markdown_table(rows, ["claim", "reason", "replacement"])}
"""


def build_related_work_md() -> str:
    return r"""
# Related Work Positioning: Auditable GAN Reconstruction

## DDNM And Range-Null Methods

DDNM-style and range-null inverse solvers explicitly separate measured/range consistency from null-space synthesis. This paper adopts the same accountability instinct but uses it for a GAN reconstruction branch in ghost imaging: the measured part is certified by \(\Pi_y^\lambda\), while \(P_0\hat{x}\) is exposed as prior-supplied content. The contribution is not a new general solver; it is a way to audit what an adversarial prior changes when the bucket measurements are held fixed.

## Diffusion Inverse Solvers

Diffusion inverse methods provide powerful priors for inverse problems and can be combined with data-consistency updates. This draft does not claim empirical superiority over diffusion solvers. The positioning is complementary: diffusion methods often emphasize sampling and restoration quality, while this paper emphasizes an explicit measurement certificate, a visible unmeasured-content map, and a diagnostic gate for deciding whether an adversarial prior has usable signal in a given measurement regime.

## Adversarial Regularizers

Adversarial losses have long been used to improve perceptual detail, often at the risk of hiding measurement errors or favoring plausible texture. Here the adversarial component is constrained by gauge equalization and post-hoc auditing. The discriminator is not treated as evidence of physical correctness; it is a prior/detail pressure whose output must pass the certificate.

## Null-Space Learning

Null-space learning asks a model to place plausible content where the measurement operator is silent. The key risk is unaccountable hallucination. This paper turns the same decomposition into a reporting object: \(P_0\hat{x}\) is not hidden in a final image, but presented as the portion of the reconstruction supplied by the prior.

## Data Consistency And Projection

Projection and data-consistency layers enforce agreement with the forward model. The role of \(\Pi_y^\lambda\) here is stronger than a training penalty: it is the final measurement certificate used after the GAN prior has supplied detail. The draft therefore attributes measured consistency to the certificate path, while assigning visual/detail changes to the prior path.

## GAN Inverse Imaging

GAN priors have been used for compressed sensing and inverse imaging through latent optimization, conditional generation, and adversarial fine-tuning. The present branch is narrower and more auditable: it reports when the prior supplies detail, uses gauge-equalized diagnostics to avoid a residual shortcut, and includes regime gates that can stop the GAN branch when signal is weak.

## Paper 1 Premise

The first measurement-accountability paper is used only as a premise: reconstruction quality and measurement certificate auditing can be separated. This paper starts from that separation and studies a GAN prior branch under the same accountability principle.
"""


def build_abstracts_md() -> str:
    return r"""
# Abstract Options: Auditable GAN Paper

## Option A: Measurement-Certificate Focus

Generative priors can add plausible detail to severely underdetermined ghost-imaging reconstructions, but their most useful content lies exactly where the measurements are silent. We present an auditable GAN reconstruction framework that separates prior-supplied detail from measured consistency. A projection certificate \(\Pi_y^\lambda\) enforces bucket consistency after the GAN prior is applied, while \(P_0\hat{x}\) is reported as an unmeasured-content map. This enables a measurement-invariant trust knob: scaling the prior component changes image detail but leaves the certificate-controlled measurement error effectively unchanged. We further use gauge-equalized diagnostics to remove a residual shortcut from adversarial evaluation and to gate regimes where the adversarial prior has usable signal. The resulting evidence supports Scr-5/Rad-5 operation and identifies Scr-10/Rad-10 as weak-signal regimes. The method is positioned as accountable prior control, not as a new measurement certificate or sampling-confidence estimator.

## Option B: Trust-Control Focus

We study how to use a GAN prior in ghost imaging without allowing perceptual detail to masquerade as measured evidence. The proposed auditable pipeline decomposes the reconstruction into a certified range component and a prior-supplied null component. The range component is controlled by \(\Pi_y^\lambda\); the null component, \(P_0\hat{x}\), is exposed as an unmeasured-content map. This decomposition gives a simple alpha trust control that changes visual detail while preserving the measured error under the same certificate. Across the available evidence, gauge-equalized diagnostics explain why adversarial signal is useful in Scr-5 and Rad-5 but weak in Scr-10 and Rad-10. Failure detection remains preliminary, and latent sampling collapsed in this branch, so no sampling-based uncertainty claim is made.

## Option C: Short-Paper Focus

This paper presents an accountability-first view of GAN reconstruction for ghost imaging. Rather than treating adversarial detail as physical evidence, we certify the measured component with \(\Pi_y^\lambda\) and expose \(P_0\hat{x}\) as the prior-supplied content map. A measurement-invariant alpha knob varies the GAN-supplied detail without changing the audited measurement error, and gauge-equalized diagnostics check whether the discriminator sees useful null-space signal rather than a residual shortcut. The evidence supports a cautious GAN branch in Scr-5/Rad-5 and a stop decision in weak Scr-10/Rad-10 regimes. The contribution is not a broad benchmark claim; it is a controlled way to inspect, trust-adjust, and gate an adversarial prior.
"""


def build_figure_captions_md(inventory: list[dict[str, object]]) -> str:
    def asset(name: str) -> str:
        matches = [r["asset_path"] for r in inventory if name in str(r["asset_name"]) or name in str(r["asset_path"])]
        return str(matches[0]) if matches else "TO_DRAW_OR_SELECT"

    rows = [
        {
            "figure": "Figure 1",
            "asset": "TO_DRAW",
            "caption": "Auditable GAN reconstruction pipeline. A conditional GAN/refiner supplies prior detail, the projection certificate \\(\\Pi_y^\\lambda\\) restores bucket consistency, and the null component \\(P_0\\hat{x}\\) is reported as unmeasured/prior-supplied content rather than hidden inside the final image.",
        },
        {
            "figure": "Figure 2",
            "asset": asset("fig_unmeasured_content_maps.png"),
            "caption": "Unmeasured-content maps for representative Scr-5 and Rad-5 reconstructions. The maps visualize where the final image depends on the prior because the measurement operator is silent. They are accountability maps, not standalone proof that a visible structure is true or false.",
        },
        {
            "figure": "Figure 3",
            "asset": asset("fig_alpha_metric_curves.png"),
            "caption": "Measurement-invariant alpha trust control. Scaling the prior component changes perceptual/detail metrics while the post-audit measurement error remains controlled by the same certificate.",
        },
        {
            "figure": "Figure 4",
            "asset": asset("fig_alpha_relmeaserr_invariance.png"),
            "caption": "Identity check for alpha trust control. Across tested alpha values, the measured component is unchanged up to numerical tolerance after projection, demonstrating that alpha modifies prior-supplied content rather than measured consistency.",
        },
        {
            "figure": "Figure 5",
            "asset": asset("score_vs_relmeaserr_standard_vs_gauge.png"),
            "caption": "Shortcut stress test. A standard discriminator can respond to row-space residual changes, while the gauge-equalized diagnostic removes the residual shortcut and tests for signal in canonicalized images.",
        },
        {
            "figure": "Figure 6",
            "asset": asset("regime_map_auc_and_outcome.png"),
            "caption": "Diagnostic gate across regimes. Scr-5 and Rad-5 show usable adversarial signal and paired-seed evidence, while Scr-10 and Rad-10 remain weak-signal regimes where the branch should stop without explicit acceptance.",
        },
        {
            "figure": "Supplement Figure S1",
            "asset": asset("fig_failure_signal_boxplots.png"),
            "caption": "Preliminary failure-signal features under artificial stress labels. The best AUC is weak-to-moderate, so these plots motivate future failure analysis rather than a deployable detector.",
        },
        {
            "figure": "Supplement Figure S2",
            "asset": asset("fig_z_samples.png"),
            "caption": "Latent-sampling diagnostic. Samples vary too little to support uncertainty reporting in this branch; the result is recorded as a negative finding.",
        },
    ]
    return f"""
# Final Figure Captions

Generated: `{now_iso()}`

{markdown_table(rows, ["figure", "asset", "caption"])}

See `FIGURE_ASSET_INVENTORY.csv` for hashes and source locations.
"""


def build_paper_md(claims: list[dict[str, object]], summary: dict[str, object]) -> str:
    rel_ranges = summary["alpha_relmeaserr_range_by_regime"]
    return f"""
# Auditable GAN Reconstruction for Ghost Imaging: Unmeasured-Content Maps and Measurement-Invariant Trust Control

## Abstract

Generative priors can add plausible detail to severely underdetermined ghost-imaging reconstructions, but their most useful content lies where the measurements are silent. We present an auditable GAN reconstruction framework that separates prior-supplied detail from measured consistency. A projection certificate \\(\\Pi_y^\\lambda\\) enforces bucket consistency after the GAN prior is applied, while \\(P_0\\hat{{x}}\\) is reported as an unmeasured-content map. This gives a measurement-invariant trust knob: scaling the prior component changes visual detail but leaves the certificate-controlled measurement error effectively unchanged. Gauge-equalized diagnostics remove a residual shortcut from adversarial evaluation and explain why the branch is positive in Scr-5/Rad-5 but weak in Scr-10/Rad-10. Failure detection is preliminary, and latent sampling collapsed in this branch, so uncertainty from z sampling is not claimed.

## 1. Introduction

Ghost imaging is an inverse problem with a large unmeasured subspace. In such settings, a visually appealing reconstruction can contain content that the bucket measurements did not determine. This is not automatically a defect: a prior is useful precisely because it fills unmeasured structure. The problem is accountability. A reconstruction method should make clear which part is certified by the measurements and which part is supplied by the prior.

This paper assembles the GAN branch under that accountability principle. The GAN is not presented as a measurement certificate. It is a prior/detail engine. The certificate is \\(\\Pi_y^\\lambda\\), and the visible audit object is \\(P_0\\hat{{x}}\\), the unmeasured/prior-supplied content map.

The first measurement-accountability paper is used only as a premise: quality and certificate auditing can be separated. This draft studies what happens when an adversarial prior is inserted into that separated pipeline.

## 2. Contributions

1. We define an auditable GAN reconstruction pipeline for ghost imaging in which the GAN supplies detail and \\(\\Pi_y^\\lambda\\) supplies the measurement certificate.
2. We report \\(P_0\\hat{{x}}\\) as an unmeasured-content map, making prior-supplied content visible.
3. We introduce an alpha trust knob that scales prior content while preserving the measured component under the certificate.
4. We use gauge-equalized diagnostics to avoid a residual shortcut in discriminator evaluation.
5. We use the diagnostic gate to scope the branch: Scr-5 and Rad-5 are positive; Scr-10 and Rad-10 are weak-signal regimes.

## 3. Method

Let \\(A\\) be the ghost-imaging measurement operator and \\(y\\) the bucket measurement. The audited reconstruction is written as a combination of a certified range component and a prior-supplied null component. In the notation used throughout the evidence pack,

\\[
\\hat{{x}} = \\Pi_y^\\lambda(v), \\qquad P_0\\hat{{x}} = P_0 v,
\\]

where \\(v\\) is the candidate from the GAN/refiner path. The projection certificate controls the measured consistency, while the null component records what the prior supplied in directions not fixed by the measurements.

### 3.1 Unmeasured-Content Maps

The map \\(P_0\\hat{{x}}\\) should be read as prior-supplied content. It is a useful accountability layer because it marks where the image depends on the learned prior. It does not, by itself, prove that any structure is false. When ground truth exists in a benchmark, null-error maps can be used for evaluation, but deployment only has the accountability map.

### 3.2 Measurement-Invariant Alpha Trust Control

For a trust value \\(\\alpha\\), the prior component can be scaled while preserving the measured part through the same projection certificate. Phase76 verifies that \\(A v_\\alpha - A v_0\\) remains at numerical tolerance, with maximum recorded difference `{summary['alpha_max_Av_alpha_minus_Av0']:.3e}` and RelMeasErr ranges `{rel_ranges}`. Thus alpha changes prior content and visual/detail behavior, not the certified measurement residual.

### 3.3 Gauge-Equalized Diagnostic

A discriminator can cheat if it sees residual features or row-space inconsistencies. The gauge-equalized diagnostic canonicalizes real/fake images so the discriminator must rely on prior-content differences rather than direct measurement residuals. The Phase75 shortcut stress test shows the intended behavior: standard row-space perturbation score delta `{summary['shortcut_standard_row_delta_alpha_0p1']:.6f}` versus gauge delta `{summary['shortcut_gauge_row_delta_alpha_0p1']:.6f}` at the same stress level.

## 4. Evidence

### 4.1 Unmeasured Content And Trust Control

Phase76 identifies the strongest paper direction as auditable GAN reconstruction: unmeasured-content maps plus a measurement-invariant alpha knob. The unmeasured-content report finds stable prior-content summaries across Scr-5 and Rad-5 arms, and the alpha identity check confirms that the trust knob operates along the prior axis after certificate projection.

### 4.2 Paired-Seed And Regime Evidence

The Scr-5 paired seed validation found C better than B on LPIPS/RAPSD criteria across counted seeds, with PSNR changes inside the predefined budget. Phase73 adds Rad-5 robustness, with gauge diagnostic AUC around `0.877` and paired-seed evidence. In contrast, Scr-10 and Rad-10 are weak-gate regimes. The final regime map is:

{markdown_table(summary['regime_map_rows'], ['regime', 'gauge_auc', 'auc_ci', 'outcome', 'decision'])}

The branch should therefore be scoped as positive in Scr-5/Rad-5 and explicitly cautious in Scr-10/Rad-10.

### 4.3 Failure Signals And Negative Findings

The failure-signal study is preliminary: the best Phase76 feature is `{summary['best_failure_feature']}` with AUC `{summary['best_failure_auc']:.3f}` and CI `{summary['best_failure_ci']}` under artificial stress labels. This is useful for future triage but not enough for a reliable detector claim.

The z-variation diagnostic is a negative result. Mean pixel standard deviation is `{summary['z_pixel_std_mean']:.6f}`, so the branch does not support sampling-based uncertainty.

### 4.4 Toy Second Inverse Problem

The random-mask inpainting toy supports the feasibility of applying a gauge diagnostic idea outside GI, but only as a toy diagnostic. It is not a trained second inverse solver.

## 5. Related Work Positioning

This paper relates to range-null methods, diffusion inverse solvers, adversarial regularization, null-space learning, projection/data consistency, and GAN inverse imaging. The central difference is not broader benchmark dominance; it is accountable separation of prior detail from measured consistency.

## 6. Limitations

The current branch does not include real human 2AFC responses. A vetted diffusion/PnP empirical comparator remains future evidence. Failure detection is preliminary. The z path collapsed. The toy second inverse-problem result is not a generalization claim. Paper 1's main results are not reused as GAN results.

## 7. Conclusion

Auditable GAN reconstruction is best framed as measurement-invariant trust control over prior-supplied content. The GAN can improve the prior/detail axis in selected regimes, but the certificate remains \\(\\Pi_y^\\lambda\\), and the accountability object remains \\(P_0\\hat{{x}}\\). This makes the branch useful when the diagnostic gate shows real prior signal, and easy to stop when the gate is weak.
"""


def build_supplement_md(summary: dict[str, object], source_rows: list[dict[str, object]]) -> str:
    return f"""
# Supplement: Auditable GAN Reconstruction

## S1. Source Provenance

Phase77 is a paper-assembly pass only. It reads existing Phase76/75/73/71/72 outputs and writes only to `E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly`.

{markdown_table(source_rows, ["required_name", "status", "resolved_path", "sha256", "bytes"])}

## S2. Alpha Identity Check

The maximum recorded measured-component difference across alpha settings is `{summary['alpha_max_Av_alpha_minus_Av0']:.3e}`. RelMeasErr ranges by regime are `{summary['alpha_relmeaserr_range_by_regime']}`.

{markdown_table(summary['alpha_rows'], ["regime", "alpha", "max_Av_alpha_minus_Av0", "mean_post_relmeaserr", "max_P0_xhat_minus_alpha_P0v_rel"])}

## S3. Failure Signal Table

These are artificial stress-label diagnostics only.

{markdown_table(summary['failure_rows'], ["feature", "orientation", "auc", "auc_ci_low", "auc_ci_high"])}

## S4. Regime Map

{markdown_table(summary['regime_map_rows'], ["regime", "gauge_auc", "auc_ci", "outcome", "decision"])}

## S5. Second Inverse-Problem Toy

The random-mask toy supports diagnostic feasibility only; it is not a new trained reconstruction method.

{summary['second_task_auc_by_model']}

## S6. Negative z Result

Mean pixel standard deviation under z variation was `{summary['z_pixel_std_mean']:.6f}`, so sampling-based uncertainty is excluded from the paper claims.
"""


def markdown_to_simple_tex(title: str, markdown: str) -> str:
    lines = markdown.strip().splitlines()
    body: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            body.append(r"\section*{" + latex_escape(line[2:].strip()) + "}")
        elif line.startswith("## "):
            body.append(r"\section{" + latex_escape(line[3:].strip()) + "}")
        elif line.startswith("### "):
            body.append(r"\subsection{" + latex_escape(line[4:].strip()) + "}")
        elif line.startswith("|"):
            if not in_table:
                body.append(r"\begin{verbatim}")
                in_table = True
            body.append(line)
        else:
            if in_table:
                body.append(r"\end{verbatim}")
                in_table = False
            if line.strip() == "":
                body.append("")
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. ") or line.startswith("5. "):
                body.append(latex_escape(line))
            else:
                body.append(line)
    if in_table:
        body.append(r"\end{verbatim}")
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage[strings]{underscore}",
            r"\usepackage{hyperref}",
            r"\title{" + latex_escape(title) + "}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            *body,
            r"\end{document}",
        ]
    )


def build_attack_bank_md() -> str:
    rows = [
        {
            "attack": "Is this just adding hallucinations?",
            "answer": "The paper exposes prior-supplied content as P0 xhat and keeps the measured component under Pi_y^lambda. It does not claim the unmeasured content is automatically true.",
            "needed_evidence": "Figure 2, alpha identity table, limitations.",
        },
        {
            "attack": "Could the discriminator be exploiting measurement residuals?",
            "answer": "Gauge-equalized diagnostics remove the residual shortcut; Phase75 stress tests show standard D is row-sensitive while gauge D is not.",
            "needed_evidence": "Figure 5 and shortcut stress report.",
        },
        {
            "attack": "Why not use diffusion?",
            "answer": "The paper is not a broad solver comparison. It studies accountable GAN prior control with explicit certificate and null-content maps.",
            "needed_evidence": "Related work and limitation section.",
        },
        {
            "attack": "Does the GAN improve measurement error?",
            "answer": "No. Measurement error is controlled by the certificate. The GAN affects the prior/detail component.",
            "needed_evidence": "Alpha identity check and method section.",
        },
        {
            "attack": "Does this generalize to all measurement regimes?",
            "answer": "No. The diagnostic gate explicitly stops weak Scr-10/Rad-10 regimes and supports Scr-5/Rad-5 only.",
            "needed_evidence": "Regime map.",
        },
        {
            "attack": "Is the failure detector reliable?",
            "answer": "No. It is preliminary and based on artificial stress labels; the draft says so.",
            "needed_evidence": "Failure detector AUC report.",
        },
        {
            "attack": "Can z sampling be interpreted as uncertainty?",
            "answer": "No. The z diagnostic collapsed, so the paper excludes that claim.",
            "needed_evidence": "Z variation diagnostic.",
        },
        {
            "attack": "Are Paper 1 results being recycled?",
            "answer": "Paper 1 is only a premise for measurement accountability and audit certificates. The GAN branch uses its own Phase69-76 evidence.",
            "needed_evidence": "Claim map boundary note.",
        },
    ]
    return f"""
# Reviewer Attack Bank: Auditable GAN Paper

Generated: `{now_iso()}`

{markdown_table(rows, ["attack", "answer", "needed_evidence"])}
"""


def build_ready_report_md(source_rows: list[dict[str, object]], claims: list[dict[str, object]], unsupported: list[dict[str, object]]) -> str:
    missing = [r for r in source_rows if r["status"] != "read"]
    status = "READY_FOR_CAUTIONARY_DRAFT" if not missing else "READY_WITH_SOURCE_WARNINGS"
    return f"""
# Phase77 Auditable GAN Ready Report

Generated: `{now_iso()}`

Readiness status: `{status}`

## Deliverables

- `CLAIMS_AUDITABLE_GAN_FINAL.md`
- `UNSUPPORTED_CLAIMS_AUDITABLE_GAN_FINAL.md`
- `auditable_gan_paper_v1.md`
- `auditable_gan_paper_v1.tex`
- `supplement_auditable_gan_v1.md`
- `supplement_auditable_gan_v1.tex`
- `FIGURE_CAPTIONS_FINAL.md`
- `FIGURE_ASSET_INVENTORY.csv`
- `ABSTRACT_OPTIONS_FINAL.md`
- `REVIEWER_ATTACK_BANK_AUDITABLE_GAN.md`
- `RELATED_WORK_AUDITABLE_GAN.md`

## Safety Confirmation

- No training was run.
- No new experiment was run.
- No checkpoint was modified.
- No first-paper main result was modified.
- Phase77 wrote only inside `E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly`.

## Claim Readiness

Allowed claims recorded: `{len(claims)}`.

Excluded claims recorded: `{len(unsupported)}`.

Missing required Phase76 reports: `{len(missing)}`.

## Publication Readiness

The draft is ready as a cautious assembly manuscript centered on auditable GAN reconstruction, unmeasured-content maps, and measurement-invariant trust control. It is not ready for a broad empirical dominance story because human 2AFC responses and a vetted external diffusion/PnP comparator remain absent.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    reports, source_rows = collect_required_sources()
    summary = make_evidence_summary()
    claims = claim_rows(summary)
    unsupported = unsupported_rows()
    inventory = figure_inventory_rows()

    write_csv(
        OUT / "phase77_source_read_log.csv",
        source_rows,
        ["required_name", "status", "resolved_path", "sha256", "bytes"],
    )
    write_csv(
        OUT / "FIGURE_ASSET_INVENTORY.csv",
        inventory,
        ["source_phase", "asset_name", "asset_path", "format", "bytes", "sha256", "paper_role"],
    )

    write_text(OUT / "CLAIMS_AUDITABLE_GAN_FINAL.md", build_claim_map_md(claims, source_rows))
    write_text(OUT / "UNSUPPORTED_CLAIMS_AUDITABLE_GAN_FINAL.md", build_unsupported_md(unsupported))
    write_text(OUT / "RELATED_WORK_AUDITABLE_GAN.md", build_related_work_md())
    write_text(OUT / "ABSTRACT_OPTIONS_FINAL.md", build_abstracts_md())
    write_text(OUT / "FIGURE_CAPTIONS_FINAL.md", build_figure_captions_md(inventory))
    write_text(OUT / "REVIEWER_ATTACK_BANK_AUDITABLE_GAN.md", build_attack_bank_md())

    paper_md = build_paper_md(claims, summary)
    supplement_md = build_supplement_md(summary, source_rows)
    write_text(OUT / "auditable_gan_paper_v1.md", paper_md)
    write_text(
        OUT / "auditable_gan_paper_v1.tex",
        markdown_to_simple_tex("Auditable GAN Reconstruction for Ghost Imaging", paper_md),
    )
    write_text(OUT / "supplement_auditable_gan_v1.md", supplement_md)
    write_text(
        OUT / "supplement_auditable_gan_v1.tex",
        markdown_to_simple_tex("Supplement: Auditable GAN Reconstruction", supplement_md),
    )
    write_text(OUT / "PHASE77_AUDITABLE_GAN_READY_REPORT.md", build_ready_report_md(source_rows, claims, unsupported))

    manifest = {
        "generated_utc": now_iso(),
        "output_dir": str(OUT),
        "input_dirs": {
            "phase76": str(PH76),
            "phase75": str(PH75),
            "phase73": str(PH73),
            "phase71": str(PH71),
            "phase72": str(PH72),
        },
        "phase77_actions": [
            "read existing reports/tables/figures",
            "assembled paper and supplement drafts",
            "assembled claim map, unsupported claim map, figures, related work, reviewer bank",
        ],
        "no_training": True,
        "no_new_experiments": True,
        "no_checkpoint_modification": True,
        "no_first_paper_modification": True,
        "required_reports": source_rows,
    }
    write_text(OUT / "PHASE77_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    write_text(
        OUT / "PHASE77_RUNLOG.md",
        f"""# Phase77 Runlog

Generated: `{manifest['generated_utc']}`

Ran paper assembly only. No training, no new experiment, no checkpoint edit, and no first-paper edit.
""",
    )

    print(f"Phase77 assembly complete: {OUT}")


if __name__ == "__main__":
    main()
