from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
OUT = ROOT / "outputs" / "compatibility" / "phase1_3_final_locked_eval"
GENERATOR_CKPT = Path("E:/ns_mc_gan_gi/outputs_phase79_posterior_anti_collapse/rad5_rowspace_diversity_diagnostic/checkpoints/final.pt")
A_RAD5 = Path("E:/ns_mc_gan_gi/results/cert_package_20260612/cache/A_rad5.npy")
PARENT_FINAL_INDICES = ROOT / "outputs" / "compatibility" / "phase1_1_corrected_rad5" / "reports" / "final_locked_test_indices.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.3 preregistered freeze/audit for final locked evaluation.")
    parser.add_argument("--output_dir", default=str(OUT))
    parser.add_argument("--phase12_dir", default=str(PHASE12))
    parser.add_argument("--allow_final", action="store_true", help="Only has effect if all frozen selector artifacts exist.")
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}\n"


def hash_paths(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        key = str(path)
        if path.exists() and path.is_file():
            out[key] = {"exists": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            out[key] = {"exists": False, "sha256": None, "bytes": None}
    return out


def write_preregistration(out: Path, phase12: Path, missing: list[str]) -> None:
    selector_report = read_json(phase12 / "reports" / "gate_report_e2b_64_selector.json")
    coverage_report = read_json(phase12 / "reports" / "gate_report_e2a_64_coverage.json")
    text = f"""# Phase 1.3 Preregistration

## Status

This preregistration is written before any Phase 1.3 final locked-test scoring.
The final test is **not** evaluated unless all frozen artifacts exist and pass
hash verification.

Current freeze status:

`BLOCKED_MISSING_SELECTOR_ARTIFACTS`

Missing frozen selector artifacts:

{chr(10).join(f"- {m}" for m in missing) if missing else "- none"}

## Frozen Facts From Phase 1.2

- Operator: Rad-5 / 64x64 / m=205.
- Coverage classification: `{coverage_report.get("classification")}`.
- Selector validation classification: `{selector_report.get("classification")}`.
- Validation-selected primary deployment selector: `dm_fcc_seed3`.
- Validation selection rule: `best_dual_key` in Phase 1.2 selector gate.
- K: `16`.
- Primary endpoint: exact canonicalized P0 RMSE.

## Primary Hypothesis H1

DM-FCC seed3 selected candidate versus random candidate expectation.
Delta is `RMSE_DM_FCC - RMSE_random_expected`; negative is better.

H1 requires:

1. mean delta < 0;
2. paired bootstrap 95% CI upper bound < 0;
3. relative P0-RMSE improvement >= 1%;
4. oracle gain fraction >= 0.20;
5. no RelMeasErr/candidate-order/row residual shortcut.

## Secondary Hypotheses

H2 compares DM-FCC seed3 against validation-selected scalar and sum-image
selectors.  H3 compares three frozen DM-FCC seeds against three frozen scratch
dual seeds.  H4 audits measurement residuals, candidate index, seed id, and
shared exact row anchor.

## Blocker

Phase 1.2 did not persist loadable selector checkpoints/scalers for the
validation-selected models.  Recreating them would require retraining or
rerunning model selection, which Phase 1.3 explicitly forbids.  Therefore
Stage A and Stage B are blocked in this run.
"""
    (out / "freeze_bundle" / "phase1_3_preregistration.md").write_text(text, encoding="utf-8")


def expected_selector_artifacts(phase12: Path) -> dict[str, Path]:
    return {
        "primary_dm_fcc_seed3_checkpoint": phase12 / "selector_checkpoints" / "dm_fcc_seed3.pt",
        "scratch_seed1_checkpoint": phase12 / "selector_checkpoints" / "scratch_seed1.pt",
        "scratch_seed2_checkpoint": phase12 / "selector_checkpoints" / "scratch_seed2.pt",
        "scratch_seed3_checkpoint": phase12 / "selector_checkpoints" / "scratch_seed3.pt",
        "raw_fcc_seed1_checkpoint": phase12 / "selector_checkpoints" / "raw_fcc_seed1.pt",
        "raw_fcc_seed2_checkpoint": phase12 / "selector_checkpoints" / "raw_fcc_seed2.pt",
        "raw_fcc_seed3_checkpoint": phase12 / "selector_checkpoints" / "raw_fcc_seed3.pt",
        "dm_fcc_seed1_checkpoint": phase12 / "selector_checkpoints" / "dm_fcc_seed1.pt",
        "dm_fcc_seed2_checkpoint": phase12 / "selector_checkpoints" / "dm_fcc_seed2.pt",
        "structural_dm_fcc_seed1_checkpoint": phase12 / "selector_checkpoints" / "structural_dm_fcc_seed1.pt",
        "structural_dm_fcc_seed2_checkpoint": phase12 / "selector_checkpoints" / "structural_dm_fcc_seed2.pt",
        "structural_dm_fcc_seed3_checkpoint": phase12 / "selector_checkpoints" / "structural_dm_fcc_seed3.pt",
        "scalar_pair_selector_model": phase12 / "selector_checkpoints" / "scalar_pair_selector.joblib",
        "sum_image_selector_model": phase12 / "selector_checkpoints" / "sum_image_selector.joblib",
    }


def final_integrity_audit(out: Path, phase12: Path) -> dict[str, Any]:
    final_manifest = phase12 / "manifests" / "final_locked_test_64_manifest.json"
    manifest = read_json(final_manifest) if final_manifest.exists() else {}
    final_indices = np.load(PARENT_FINAL_INDICES).astype(np.int64) if PARENT_FINAL_INDICES.exists() else np.array([], dtype=np.int64)
    overlaps: dict[str, int] = {}
    for cache_name in ["train_64_selector_k16.pt", "val_64_selector_k16.pt"]:
        path = phase12 / "candidate_cache" / cache_name
        if path.exists():
            obj = torch.load(path, map_location="cpu", weights_only=False)
            overlaps[cache_name] = int(len(set(map(int, obj.get("indices", []))) & set(map(int, final_indices))))
    dev_manifest = phase12 / "manifests" / "candidate_pool_dev_64.json"
    if dev_manifest.exists():
        dev = read_json(dev_manifest)
        dev_idx = [int(x["source_index"]) for x in dev.get("images", [])]
        overlaps["candidate_pool_dev_64"] = int(len(set(dev_idx) & set(map(int, final_indices))))
    existing_final_outputs = [
        str(p)
        for p in (out / "reports").glob("final_locked_test_*")
        if p.name not in {"final_test_integrity_audit.json"}
    ]
    if any(v > 0 for v in overlaps.values()):
        status = "INVALID_SPLIT_OVERLAP"
    elif bool(manifest.get("final_test_evaluated", False)) or existing_final_outputs:
        status = "POSSIBLY_SEEN_OR_CONTAMINATED"
    else:
        status = "CLEAN_UNSEEN_FINAL_TEST"
    audit = {
        "status": status,
        "final_manifest": str(final_manifest),
        "final_manifest_exists": final_manifest.exists(),
        "final_test_evaluated_flag": manifest.get("final_test_evaluated"),
        "final_source_indices_count": int(final_indices.size),
        "final_source_indices_sha256": hashlib.sha256(np.ascontiguousarray(final_indices).tobytes()).hexdigest() if final_indices.size else None,
        "overlap_counts": overlaps,
        "existing_final_result_files": existing_final_outputs,
        "ground_truth_metric_files_found_before_stage_b": existing_final_outputs,
    }
    save_json(out / "reports" / "final_test_integrity_audit.json", audit)
    return audit


def create_source_snapshot(out: Path, files: list[Path]) -> Path:
    zip_path = out / "freeze_bundle" / "source_snapshot.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists() and path.is_file():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
    return zip_path


def build_freeze_bundle(out: Path, phase12: Path, missing: list[str], integrity: dict[str, Any]) -> dict[str, Any]:
    freeze = ensure(out / "freeze_bundle")
    source_files = [
        ROOT / "phase1_2_rad5_64_pipeline.py",
        ROOT / "phase1_3_freeze_and_audit.py",
        ROOT / "src" / "projections.py",
        ROOT / "src" / "compatibility_model.py",
        ROOT / "src" / "phase1_1_controls.py",
        ROOT / "tests" / "test_phase1_2_rad5_64.py",
    ]
    config_files = [
        phase12 / "reports" / "gate_report_e2a_64_coverage.json",
        phase12 / "reports" / "gate_report_e2b_64_selector.json",
        phase12 / "reports" / "selector_validation_ablation.json",
        phase12 / "manifests" / "final_locked_test_64_manifest.json",
    ]
    checkpoints = [GENERATOR_CKPT, A_RAD5, *expected_selector_artifacts(phase12).values()]
    save_json(freeze / "source_file_hashes.json", hash_paths(source_files))
    save_json(freeze / "config_hashes.json", hash_paths(config_files))
    save_json(freeze / "checkpoint_hashes.json", hash_paths(checkpoints))
    dependency_versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import sklearn

        dependency_versions["sklearn"] = sklearn.__version__
    except Exception as exc:
        dependency_versions["sklearn"] = f"UNAVAILABLE: {exc!r}"
    save_json(freeze / "dependency_versions.json", dependency_versions)
    (freeze / "git_status_before_final.txt").write_text(git_text(["status", "--short"]), encoding="utf-8")
    (freeze / "git_commit_or_snapshot.txt").write_text(git_text(["rev-parse", "HEAD"]), encoding="utf-8")
    if (phase12 / "manifests" / "final_locked_test_64_manifest.json").exists():
        shutil.copy2(phase12 / "manifests" / "final_locked_test_64_manifest.json", freeze / "final_locked_test_64_manifest.json")
    seed_policy = {
        "K": 16,
        "global_candidate_seed": "not_consumed_in_phase1_3_because_selector_artifacts_missing",
        "policy": "stable sha256(source_index,candidate_index,global_seed), preregistered but not used",
    }
    save_json(freeze / "candidate_seed_policy.json", seed_policy)
    metric_defs = {
        "primary": "exact canonicalized P0 RMSE before clipping",
        "random_baseline": "mean over all K candidates in the fixed pool",
        "oracle_gain_fraction": "(L_random - L_selected)/(L_random - L_oracle), unclipped",
    }
    save_json(freeze / "metric_definitions.json", metric_defs)
    selector_registry = {
        "primary": {
            "name": "dm_fcc_seed3",
            "validation_selected": True,
            "artifact_status": "missing",
            "expected_path": str(expected_selector_artifacts(phase12)["primary_dm_fcc_seed3_checkpoint"]),
        },
        "baselines": {
            name: {"expected_path": str(path), "artifact_status": "exists" if path.exists() else "missing"}
            for name, path in expected_selector_artifacts(phase12).items()
        },
    }
    save_json(freeze / "selector_registry.json", selector_registry)
    snapshot = create_source_snapshot(out, source_files)
    freeze_manifest = {
        "phase": "Phase1.3",
        "freeze_status": "BLOCKED_MISSING_SELECTOR_ARTIFACTS" if missing else "READY",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_primary_model": "dm_fcc_seed3",
        "missing_artifacts": missing,
        "final_integrity_status": integrity["status"],
        "source_snapshot_zip": str(snapshot),
        "source_snapshot_sha256": sha256_file(snapshot),
        "generator_checkpoint": str(GENERATOR_CKPT),
        "generator_checkpoint_sha256": sha256_file(GENERATOR_CKPT) if GENERATOR_CKPT.exists() else None,
        "A_path": str(A_RAD5),
        "A_file_sha256": sha256_file(A_RAD5) if A_RAD5.exists() else None,
        "K": 16,
        "primary_endpoint": "exact canonicalized P0 RMSE",
        "hypothesis_thresholds": {
            "H1_relative_improvement_min": 0.01,
            "H1_oracle_gain_fraction_min": 0.20,
            "H1_bootstrap_ci_upper_lt_zero": True,
        },
    }
    save_json(freeze / "phase1_3_freeze_manifest.json", freeze_manifest)
    write_preregistration(out, phase12, missing)
    # Because the primary selector is absent, do not emit FINAL_EVAL_FROZEN.json.
    blocked = {
        "status": "FINAL_EVAL_BLOCKED",
        "reason": "Missing frozen selector artifacts from Phase 1.2; retraining is forbidden in Phase 1.3.",
        "freeze_manifest": str(freeze / "phase1_3_freeze_manifest.json"),
        "missing_artifacts": missing,
    }
    save_json(freeze / "FINAL_EVAL_BLOCKED.json", blocked)
    return freeze_manifest


def write_blocked_stage_reports(out: Path, missing: list[str], integrity: dict[str, Any]) -> None:
    ensure(out / "blind_inference")
    blocked = {
        "status": "not_run",
        "reason": "Stage A blind inference requires FINAL_EVAL_FROZEN.json, which was not created because selector artifacts are missing.",
        "missing_artifacts": missing,
    }
    save_json(out / "blind_inference" / "BLIND_INFERENCE_BLOCKED.json", blocked)
    reports = {
        "final_locked_test_primary_results.json": {"status": "not_run", "reason": "final evaluation blocked before Stage A"},
        "final_locked_test_all_models.json": {"status": "not_run", "reason": "frozen model artifacts unavailable"},
        "final_method_seed_analysis.json": {"status": "not_run", "reason": "frozen seed checkpoints unavailable"},
        "final_paired_bootstrap.json": {"status": "not_run", "reason": "no final metrics computed"},
        "final_hypothesis_decisions.json": {
            "classification": "FINAL_EVALUATION_INVALID",
            "reason": "Cannot run preregistered final evaluation without frozen primary selector artifact.",
        },
        "final_shortcut_audit.json": {"status": "not_run", "reason": "no selector inference performed"},
    }
    for name, payload in reports.items():
        save_json(out / "reports" / name, payload)
    (out / "reports" / "final_scientific_conclusion.md").write_text(
        "# Phase 1.3 Final Scientific Conclusion\n\n"
        "Classification: `FINAL_EVALUATION_INVALID` for this attempted run.\n\n"
        "The final locked test was not consumed.  Phase 1.2 did not persist the "
        "validation-selected selector checkpoints/scalers, and Phase 1.3 forbids "
        "retraining or reconstructing them.  Therefore Stage A and Stage B were "
        "blocked before any final-test metrics were computed.\n",
        encoding="utf-8",
    )
    (out / "reports" / "final_reproducibility_report.md").write_text(
        "# Final Reproducibility Report\n\n"
        "- Freeze bundle created.\n"
        "- FINAL_EVAL_FROZEN.json not created.\n"
        "- Final test integrity audit status: `" + str(integrity["status"]) + "`.\n"
        "- Final scoring not run.\n",
        encoding="utf-8",
    )
    (out / "reports" / "final_test_incident_log.md").write_text(
        "# Final Test Incident Log\n\nNo final-test incident occurred because final scoring was not started.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    start = time.time()
    out = ensure(Path(args.output_dir))
    ensure(out / "reports")
    ensure(out / "freeze_bundle")
    phase12 = Path(args.phase12_dir)
    (out / "reports" / "command_log.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    artifacts = expected_selector_artifacts(phase12)
    missing = [f"{name}: {path}" for name, path in artifacts.items() if not path.exists()]
    integrity = final_integrity_audit(out, phase12)
    freeze_manifest = build_freeze_bundle(out, phase12, missing, integrity)
    if missing or integrity["status"] != "CLEAN_UNSEEN_FINAL_TEST":
        write_blocked_stage_reports(out, missing, integrity)
        status = {
            "phase": "Phase1.3",
            "status": "blocked_before_final",
            "classification": "FINAL_EVALUATION_INVALID",
            "final_test_consumed": False,
            "final_manifest_marked_evaluated": False,
            "missing_artifact_count": len(missing),
            "final_integrity_status": integrity["status"],
            "runtime_seconds": time.time() - start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        save_json(out / "reports" / "runtime_and_memory.json", status)
        save_json(out / "reports" / "implementation_status_phase1_3.json", status)
        (out / "reports" / "BLOCKERS.md").write_text(
            "# BLOCKERS\n\n"
            "Phase 1.3 final evaluation was blocked before Stage A.\n\n"
            "## Missing frozen selector artifacts\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nRetraining or reconstructing these models is forbidden by the Phase 1.3 protocol.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "blocked_before_final", "missing_artifact_count": len(missing), "output_dir": str(out)}, indent=2))
        return 0
    # This branch is intentionally unavailable in the current artifact state.
    raise RuntimeError("Unexpected READY state: Stage A/B implementation should be invoked by a separate frozen runner.")


if __name__ == "__main__":
    raise SystemExit(main())
