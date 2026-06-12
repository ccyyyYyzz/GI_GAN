from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROMPT = Path("C:/Users/CYZ的computer/Downloads/prompt_codex.md")
MEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/scrambled_hadamard5_hq_noise001_colab")
PILOT_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling")
PHASE59_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase59_gan_sampling_mode_g1")
PHASE60_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase60_gan_sampling_mode_g2")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def csv_metric(rows: list[dict[str, str]], metric: str, column: str) -> float:
    for row in rows:
        if row.get("metric") == metric:
            return to_float(row.get(column))
    return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    x = to_float(value)
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def looks_like_transfer_manifest(path: Path) -> bool:
    text = read_text(path).lower()
    return '"parts"' in text or '"zip"' in text or "chunk_mb" in text


def locate_split_candidates(root: Path) -> tuple[list[Path], list[Path]]:
    candidates: list[Path] = []
    ignored: list[Path] = []
    if not root.exists():
        return candidates, ignored
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not any(token in name for token in ["split", "index", "indices", "ids"]):
            continue
        if path.suffix.lower() not in [".json", ".csv", ".npy", ".npz", ".pt", ".yaml", ".yml"]:
            continue
        if "split_manifest" in name and looks_like_transfer_manifest(path):
            ignored.append(path)
        else:
            candidates.append(path)
    return sorted(candidates), sorted(ignored)


def make_placeholder_pdf(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    image = Image.new("RGB", (1300, 780), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 1276, 756), outline=(40, 40, 40), width=2)
    draw.text((60, 70), title, fill=(0, 0, 0))
    y = 130
    for line in lines:
        draw.text((60, y), line, fill=(20, 20, 20))
        y += 34
    image.save(png)
    image.save(path, "PDF", resolution=120.0)


def write_yaml_like(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "figs").mkdir(exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, ROOT / "prompt_codex_input.md")

    identity = read_json(ROOT / "checks" / "identity_gate_results.json")
    mean_metrics = read_json(MEAN_ROOT / "eval_metrics.json").get("model", {})
    phase60_prov = read_json(PHASE60_ROOT / "phase60_provenance_status.json")
    phase60_safety = read_json(PHASE60_ROOT / "g2_safety_status.json")
    g1_rows = read_csv(PHASE59_ROOT / "g1_key_metric_table.csv")
    optional_rows = read_csv(PILOT_ROOT / "optional_gan_results.csv")
    posterior_rows = read_csv(PILOT_ROOT / "posterior_sampling_metrics.csv")
    command_log = read_text(PILOT_ROOT / "command_log.txt").strip()
    source_code = read_text(REPO / "src" / "phase53C_optional_gan_posterior.py")
    has_explicit_z = (" z" in source_code or "z_" in source_code) and "torch.randn" in source_code
    repeated_deterministic_reconstruct = "for _ in range(K)" in source_code and "reconstruct_from_measurements" in source_code and not has_explicit_z

    mean_psnr = to_float(mean_metrics.get("psnr"))
    mean_ssim = to_float(mean_metrics.get("ssim"))
    mean_rel = to_float(mean_metrics.get("rel_meas_error"))
    g1_psnr = csv_metric(g1_rows, "PSNR", "sampling_mode_scr5")
    g1_ssim = csv_metric(g1_rows, "SSIM", "sampling_mode_scr5")
    g1_rel = csv_metric(g1_rows, "RelMeasErr", "sampling_mode_scr5")
    g1_std = csv_metric(g1_rows, "mean_pixel_std", "sampling_mode_scr5")
    g1_null_ratio = csv_metric(g1_rows, "null_variance_ratio", "sampling_mode_scr5")
    delta_psnr = g1_psnr - mean_psnr
    kappa = 10 ** (-delta_psnr / 10.0) if not math.isnan(delta_psnr) else float("nan")
    rel_ratio = abs(g1_rel - mean_rel) / max(abs(mean_rel), 1e-12)

    main_split_candidates, main_split_ignored = locate_split_candidates(MEAN_ROOT)
    pilot_split_candidates, pilot_split_ignored = locate_split_candidates(PILOT_ROOT)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mean_mode_scr5": {
            "root": str(MEAN_ROOT),
            "checkpoint": str(MEAN_ROOT / "last.pt"),
            "checkpoint_sha256": sha256(MEAN_ROOT / "last.pt"),
            "config": str(MEAN_ROOT / "resolved_config.yaml"),
            "metrics": mean_metrics,
            "split_candidates": [str(p) for p in main_split_candidates],
            "ignored_transfer_manifests": [str(p) for p in main_split_ignored],
            "split_hash_status": "not_locatable" if not main_split_candidates else "candidate_files_need_human_parse",
        },
        "g1_pilot": {
            "root": str(PILOT_ROOT),
            "source_checkpoint": str(PILOT_ROOT / "scr5" / "source_checkpoint.pt"),
            "source_checkpoint_sha256": sha256(PILOT_ROOT / "scr5" / "source_checkpoint.pt"),
            "optional_gan_results": str(PILOT_ROOT / "optional_gan_results.csv"),
            "posterior_sampling_metrics": str(PILOT_ROOT / "posterior_sampling_metrics.csv"),
            "split_candidates": [str(p) for p in pilot_split_candidates],
            "ignored_transfer_manifests": [str(p) for p in pilot_split_ignored],
            "split_hash_status": "not_locatable" if not pilot_split_candidates else "candidate_files_need_human_parse",
        },
        "phase60_gate": {
            "provenance_status": str(PHASE60_ROOT / "phase60_provenance_status.json"),
            "safety_status": str(PHASE60_ROOT / "g2_safety_status.json"),
            "unsafe_to_run": phase60_safety.get("unsafe_to_run", True),
            "reasons": phase60_safety.get("reasons", []),
        },
    }
    write_json(ROOT / "PROVENANCE_SAMPLING.json", provenance)

    anomaly_classification = "mixed_protocol_drift_and_z_disabled_or_collapsed_budget_confound_possible_leakage_undeterminable"
    g1_lines = [
        "# G1 Postmortem",
        "",
        "## Classification",
        "",
        f"`{anomaly_classification}`",
        "",
        "The G1 pilot is not valid sampling-mode evidence. The checkpoint provenance matches the published Scr-5 mean-mode checkpoint at initialization, but the pilot then ran a gated optional generator update and did not save a final per-sample stochastic artifact set. The evaluation loop repeatedly calls deterministic reconstruction without an explicit z input. Split hashes for the main no-leak and pilot eval are not locatable, so leakage/protocol drift cannot be ruled out.",
        "",
        "## Evidence Chain",
        "",
        f"- Published mean Scr-5 metrics: PSNR {fmt(mean_psnr)}, SSIM {fmt(mean_ssim)}, RelMeasErr {fmt(mean_rel, 6)} from `{MEAN_ROOT / 'eval_metrics.json'}`.",
        f"- G1 pilot metrics: PSNR {fmt(g1_psnr)}, SSIM {fmt(g1_ssim)}, RelMeasErr {fmt(g1_rel, 6)} from `{PHASE59_ROOT / 'g1_key_metric_table.csv'}`.",
        f"- G1 PSNR advantage: {fmt(delta_psnr)} dB; kappa proxy {fmt(kappa, 4)} < 1, outside [1, 2].",
        f"- Mean pixel std proxy: {fmt(g1_std, 8)}; null variance ratio proxy: {fmt(g1_null_ratio, 6)}.",
        f"- Certificate invariance/reportable residual: G1 RelMeasErr {fmt(g1_rel, 6)} vs mean {fmt(mean_rel, 6)}; relative difference {fmt(rel_ratio, 6)}.",
        f"- Initialization checkpoint SHA match according to Phase60: `{phase60_prov.get('checkpoint_sha_match', False)}`.",
        f"- Main split candidates: {len(main_split_candidates)}; pilot split candidates: {len(pilot_split_candidates)}.",
        f"- Stochastic z active in old eval code: `{has_explicit_z}`. Repeated deterministic reconstruct pattern found: `{repeated_deterministic_reconstruct}`.",
        f"- Pilot command: `{command_log}`",
        "",
        "## Training Budget vs Mean Mode",
        "",
        "- Mean-mode config reports 80 epochs in `resolved_config.yaml`.",
        "- G1 optional pilot command reports `--critic_epochs 8 --max_steps 180 --num_samples_per_y 8`.",
        "- Because the final fine-tuned checkpoint and optimizer state were not saved as individual evidence, the budget confound cannot be fully separated from protocol drift.",
        "",
        "## Leakage Question",
        "",
        "No main no-leak train/val/test split hash and no pilot eval split hash were found. The file named `scrambled5_noleak_split_manifest.json` is a transfer chunk manifest, not a data split provenance file.",
    ]
    (ROOT / "G1_POSTMORTEM.md").write_text("\n".join(g1_lines) + "\n", encoding="utf-8")

    g2_config = {
        "name": "phase_sampling_mode_g2_preflight_scr5_null_gauge",
        "status": "prepared_but_not_launch_ready",
        "dataset": "STL-10",
        "task": "Scr-5 only",
        "initialize_generator_from": str(MEAN_ROOT / "last.pt"),
        "mean_mode_reference": {
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "rel_meas_error": mean_rel,
        },
        "measurement": {
            "sampling_ratio": 0.05,
            "m": 205,
            "n": 4096,
            "operator": "scrambled Hadamard, orthonormal rows",
            "audit": "Pi_y_lambda stays ON in deliverable path",
            "exact_projector": "P0 = I - Q Q^T, Q = orth(A^T)",
        },
        "discriminator": {
            "type": "small projection-conditioned conv net",
            "inputs_allowed": ["P0_xhat", "x_data"],
            "inputs_forbidden": ["full_xhat", "A_xhat_minus_y", "RelMeasErr", "audit_delta", "Pi_y_xhat_minus_xhat"],
            "loss": "hinge",
        },
        "generator_loss": {
            "data_term": "existing reconstruction loss on sample mean over P=2 stochastic z samples",
            "adversarial_term": "beta * -D(P0_xhat, x_data)",
            "diversity_term": "rcGAN std reward; fallback maximize E||xhat(z1)-xhat(z2)||_1 if rcGAN unavailable",
            "beta_sweep": ["0.3*beta0", "beta0", "3*beta0"],
        },
        "smoke_test_plan": {
            "max_iterations": 200,
            "max_training_images": 256,
            "fixed_seed": 6002,
            "pass_criteria": [
                "finite moving losses",
                "D not saturated; real/fake margins reported",
                "per-sample saving works for K=32",
                "pixel std increases from init",
                "certificate invariant at init and after smoke",
                "kappa computable end-to-end",
            ],
        },
        "safety_blockers": phase60_safety.get("reasons", []) + [
            "Old pilot eval path has no explicit stochastic z; G2 stochastic branch needs implementation review before smoke.",
            "No individual G1 stochastic samples exist for cross-checking metrics.",
        ],
    }
    write_yaml_like(ROOT / "G2_CONFIG.yaml", g2_config)

    ready_blockers = [
        "No saved main no-leak train/val/test split hashes are available.",
        "Pilot split/eval index hashes are not available.",
        "Old G1 code path appears deterministic with no explicit stochastic z.",
        "Controlled G2 smoke was not run because provenance is unsafe and stochastic branch implementation has not been reviewed.",
    ]
    ready_lines = [
        "# G2 Ready Dossier",
        "",
        "## Exact Launch Command",
        "",
        "Do not launch yet. After blockers are resolved, the intended launch command should use the prepared `G2_CONFIG.yaml` and write all outputs to a new timestamped directory, with audit enabled and K=32 per-sample saving.",
        "",
        "## Estimated Wall-Clock And Memory",
        "",
        "Not measured because smoke was not run. Any estimate would be fabricated.",
        "",
        "## Pre-Registered Acceptance Band",
        "",
        "- kappa >= 1.15.",
        "- Observed delta PSNR consistent with -10 log10(kappa) within +/-0.3 dB.",
        "- RelMeasErr unchanged to 1e-6 relative with audit on.",
        "- Visible diversity in an 8x4 sample grid.",
        "",
        "## Pre-Registered No-Go Reading",
        "",
        "If kappa ~= 1 despite diversity pressure, report: adversarial fine-tuning reduces to the mean mode at this information budget. This is an acceptable publishable negative outcome.",
        "",
        "## Artifacts A Future Run Must Emit",
        "",
        "- train/val/test split indices and SHA256 hashes.",
        "- all K=32 individual stochastic samples per test image plus z seeds.",
        "- per-sample PSNR/SSIM/RelMeasErr and sample-mean metrics.",
        "- pixelwise std maps, null variance ratio, kappa proxy.",
        "- LPIPS/FID/KID only if packages and local weights are available.",
        "- smoke loss curves and D real/fake margins.",
        "",
        f"READY TO LAUNCH: no - blockers: {ready_blockers}",
    ]
    (ROOT / "G2_READY.md").write_text("\n".join(ready_lines) + "\n", encoding="utf-8")

    make_placeholder_pdf(
        ROOT / "figs" / "smoke_std_curve.pdf",
        "Smoke std curve not produced",
        [
            "Smoke test was intentionally skipped.",
            "Reason: main no-leak split hashes are missing.",
            "Reason: G2 stochastic branch needs implementation review.",
            "No training or fine-tuning was launched.",
        ],
    )

    runlog = [
        "# Runlog",
        "",
        f"- UTC created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Prompt copied from: `{PROMPT}`",
        "- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/checks/gan_identity_gate.py --output results/sampling_mode_20260612_151210Z/checks/identity_gate_results.json`",
        f"- Identity gate pass: `{identity.get('pass', False)}`.",
        "- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m compileall results/sampling_mode_20260612_151210Z`",
        "- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/build_sampling_dossier.py`",
        "- Split-hash command not run because no main or pilot data split index file was locatable; this is recorded as a blocker in PROVENANCE_SAMPLING.json.",
        "- Full G2 training launched: `false`.",
        "- G2 smoke training launched: `false`; blocked by unsafe provenance.",
    ]
    (ROOT / "RUNLOG.md").write_text("\n".join(runlog) + "\n", encoding="utf-8")

    perceptual = {}
    try:
        sys.path.insert(0, str(ROOT))
        from sampling_metrics import optional_perceptual_availability

        perceptual = optional_perceptual_availability()
    except Exception as exc:
        perceptual = {"error": str(exc)}

    report_rows = [
        {
            "G1 anomaly cause": anomaly_classification,
            "certificate-invariance result": f"RelMeasErr {fmt(g1_rel, 6)} vs {fmt(mean_rel, 6)}; relative diff {fmt(rel_ratio, 6)}",
            "infra status": "utilities generated; split hashes not locatable; per-sample saving module ready but untested on G2",
            "smoke-test verdict": "skipped_unsafe_provenance",
            "READY-TO-LAUNCH flag": "no",
        }
    ]
    report = [
        "# Sampling-Mode GAN Track Dossier",
        "",
        "This dossier prepares the exploratory GAN sampling-mode track up to the safety gate. It does not modify the main pipeline, does not overwrite results, and does not launch full G2 training.",
        "",
        "## Status By Task",
        "",
        f"- S-1 identity gate: {'PASS' if identity.get('pass') else 'FAIL'}; results in `checks/identity_gate_results.json`.",
        "- S0 G1 forensic post-mortem: completed; see `G1_POSTMORTEM.md`.",
        "- S1 infrastructure repair: utility modules generated (`eval_sampling.py`, `sampling_metrics.py`, `tools/split_hash.py`); no main code edited.",
        "- S2 G2 preflight: config prepared; smoke not run because provenance is unsafe and stochastic branch needs review.",
        "- S3 launch dossier: completed with `READY TO LAUNCH: no`.",
        "",
        "## Perceptual Metrics Availability",
        "",
        f"`{json.dumps(perceptual, sort_keys=True)}`",
        "",
        "## Evidence Files",
        "",
        f"- Phase60 safety gate: `{PHASE60_ROOT / 'g2_safety_status.json'}`.",
        f"- Phase60 provenance: `{PHASE60_ROOT / 'phase60_provenance_status.json'}`.",
        f"- G1 key table: `{PHASE59_ROOT / 'g1_key_metric_table.csv'}`.",
        f"- Pilot command log: `{PILOT_ROOT / 'command_log.txt'}`.",
        f"- Old pilot source: `{REPO / 'src' / 'phase53C_optional_gan_posterior.py'}`.",
        "",
        "## What I could not determine and why",
        "",
        "- I could not confirm that the pilot used the same test indices as the main no-leak split because no saved data split hash/index files were found.",
        "- I could not separate leakage from protocol drift because both main and pilot split hashes are missing.",
        "- I could not prove active stochastic sampling in G1 because the old eval loop has no explicit z path and individual stochastic samples were not saved.",
        "- I could not measure smoke wall-clock/memory because running G2 smoke would proceed past an unsafe provenance gate.",
        "",
        "|G1 anomaly cause|certificate-invariance result|infra status|smoke-test verdict|READY-TO-LAUNCH flag|",
        "|---|---|---|---|---|",
    ]
    for row in report_rows:
        report.append(
            "|"
            + "|".join(
                [
                    row["G1 anomaly cause"],
                    row["certificate-invariance result"],
                    row["infra status"],
                    row["smoke-test verdict"],
                    row["READY-TO-LAUNCH flag"],
                ]
            )
            + "|"
        )
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "root": str(ROOT),
        "files": sorted(set(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()) | {"DELIVERABLES_MANIFEST.json"}),
        "ready_to_launch": False,
        "full_training_launched": False,
        "smoke_training_launched": False,
    }
    write_json(ROOT / "DELIVERABLES_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
