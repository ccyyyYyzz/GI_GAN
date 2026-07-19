"""One-shot Round59 replay using the already materialized held-out caches.

This runner is intentionally separate from the Round56 evaluator.  It reads
only ``/content/gan_r56_heldout_recovery/lane{N}/rate{05,10}/cache`` and
writes a new, fail-closed result tree.  It never downloads STL-10, prepares a
cache, trains a model, or mutates the Round56 tree.

The diagnostic is expected to expose ``--final-target raw_y`` and to write the
same value to ``summary.json``.  Keeping that contract here makes a missing or
misnamed core option an immediate failure rather than a silent fallback to the
previous projected target.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


RATES = ("05", "10")
FINAL_TARGET = "raw_y"
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260719


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_FILE:{path}")
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return decoded


def lane_from_archive(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"RATE_ARCHIVE_MISSING:{path}")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    lane = int(manifest["seed"])
    if lane not in (0, 1, 2):
        raise RuntimeError(f"INVALID_LANE_INDEX:{lane}")
    return lane


def verify_hash_map(entries: dict[str, str], *, label: str) -> dict[str, str]:
    verified: dict[str, str] = {}
    for raw_path, expected in sorted(entries.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label}_MISSING:{path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label}_HASH_MISMATCH:{path}:{actual}:{expected}")
        verified[str(path)] = actual
    return verified


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def current_code_hashes(repo_root: Path) -> dict[str, str]:
    """Hash all code which participates in the Round59 invocation.

    These hashes are descriptive rather than compared with the old freeze:
    adding raw-y support necessarily changes the diagnostic implementation.
    The frozen weights/operator/cache are compared with their prior receipts.
    """
    relpaths = (
        "diagnose_fiber_orthogonal_highpass_innovation.py",
        "diagnose_afrb_proposal_headroom.py",
        "diagnose_fiber_residual_frequency_fusion.py",
        "diagnose_vqgan_causal_disagreement_controls.py",
        "gan_high_quality_gi.py",
        "train_fiber_residual_phase_gan.py",
        "train_vqae_centered_residual_adapter.py",
        "src/fiber_orthogonal_innovation.py",
        "src/gauge_geometry.py",
        "src/metrics.py",
        "src/projections.py",
        "run_frozen_fohi_raw_fiber_once.py",
    )
    hashes: dict[str, str] = {}
    for relpath in relpaths:
        path = repo_root / relpath
        if not path.is_file():
            raise FileNotFoundError(f"ROUND59_CODE_MISSING:{path}")
        hashes[relpath] = sha256(path)
    return hashes


def cache_receipt(
    *, source_lane: Path, rate: str, expected_images: int
) -> dict[str, Any]:
    cache_dir = source_lane / f"rate{rate}" / "cache"
    cache = cache_dir / "test_cache.pt"
    manifest_path = cache_dir / "test_cache_manifest.json"
    if not cache.is_file():
        raise FileNotFoundError(f"ROUND56_CACHE_MISSING:{cache}")
    manifest = read_json(manifest_path)
    if int(manifest.get("test_images", -1)) != int(expected_images):
        raise RuntimeError(f"ROUND56_CACHE_IMAGE_COUNT_MISMATCH:{rate}")
    if int(manifest.get("included_development_raw_hash_overlap", -1)) != 0:
        raise RuntimeError(f"ROUND56_CACHE_DEVELOPMENT_OVERLAP:{rate}")
    return {
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256(manifest_path),
        "test_images": int(manifest["test_images"]),
        "included_development_raw_hash_overlap": int(
            manifest["included_development_raw_hash_overlap"]
        ),
    }


def preflight(
    *, freeze_manifest: Path, repo_root: Path, source_lane: Path, lane_index: int
) -> dict[str, Any]:
    manifest = read_json(freeze_manifest)
    if manifest.get("status") != "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN":
        raise RuntimeError("FREEZE_MANIFEST_STATUS_MISMATCH")
    lane = manifest.get("lanes", {}).get(str(lane_index))
    if not isinstance(lane, dict):
        raise RuntimeError(f"UNFROZEN_LANE:{lane_index}")
    prior_complete = read_json(source_lane / "HELDOUT_ONCE_COMPLETE.json")
    if prior_complete.get("status") != "VQGAN_GUIDED_FOHI_HELDOUT_LANE_COMPLETE":
        raise RuntimeError("ROUND56_COMPLETE_RECEIPT_STATUS_MISMATCH")
    if int(prior_complete.get("lane_index", -1)) != lane_index:
        raise RuntimeError("ROUND56_COMPLETE_RECEIPT_LANE_MISMATCH")

    artifact_hashes = verify_hash_map(
        lane["artifact_sha256"], label="FROZEN_ARTIFACT"
    )
    caches = {
        rate: cache_receipt(
            source_lane=source_lane,
            rate=rate,
            expected_images=int(manifest["expected_test_images"]),
        )
        for rate in RATES
    }
    for rate, cache in caches.items():
        old_rate = prior_complete.get("rates", {}).get(rate, {})
        if old_rate.get("test_cache_manifest_sha256") != cache["cache_manifest_sha256"]:
            raise RuntimeError(f"ROUND56_CACHE_RECEIPT_HASH_MISMATCH:{rate}")

    return {
        "status": "ROUND59_RAW_FIBER_PREFLIGHT_VERIFIED",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lane_index": lane_index,
        "git_head": git_head(repo_root),
        "freeze_manifest": str(freeze_manifest),
        "freeze_manifest_sha256": sha256(freeze_manifest),
        "source_round56_lane": str(source_lane),
        "source_round56_complete_receipt_sha256": sha256(
            source_lane / "HELDOUT_ONCE_COMPLETE.json"
        ),
        "frozen_weight_and_config_sha256": artifact_hashes,
        "current_code_sha256": current_code_hashes(repo_root),
        "reused_caches": caches,
        "method_parameters": {
            **dict(manifest["method_parameters"]),
            "final_target": FINAL_TARGET,
            "input_cache_policy": "reuse_only_round56_test_cache",
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }


def run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def diagnostic_command(
    *, repo_root: Path, cache: Path, frozen_rate: dict[str, Any], lane_index: int, rate: str, output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(repo_root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
        "--primary-val",
        str(cache),
        "--control-val",
        str(cache),
        "--config",
        str(frozen_rate["config"]),
        "--control-checkpoint",
        str(frozen_rate["structural_checkpoint"]),
        "--proposal-checkpoint",
        str(frozen_rate["proposal_checkpoint"]),
        "--filter-mode",
        "highpass",
        "--cutoff",
        "0.12",
        "--transition",
        "0.03",
        "--alpha",
        "0.5",
        "--batch-size",
        "32",
        "--exact-iterations",
        "4096",
        "--bootstrap-reps",
        str(BOOTSTRAP_REPS),
        "--seed",
        str(BOOTSTRAP_SEED + 100 * lane_index + int(rate)),
        "--evaluation-scope",
        "heldout",
        "--final-target",
        FINAL_TARGET,
        "--output-dir",
        str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/content/GI_GAN"))
    parser.add_argument("--rate-archive", type=Path, default=Path("/content/gan_rate_bundle.zip"))
    parser.add_argument("--source-root", type=Path, default=Path("/content/gan_r56_heldout_recovery"))
    parser.add_argument("--output-root", type=Path, default=Path("/content/gan_r59_raw_fiber"))
    args = parser.parse_args()

    lane_index = lane_from_archive(args.rate_archive)
    source_lane = args.source_root / f"lane{lane_index}"
    output_dir = args.output_root / f"lane{lane_index}"
    if output_dir.exists():
        raise RuntimeError(f"ROUND59_ONE_SHOT_OUTPUT_ALREADY_EXISTS:{output_dir}")

    preflight_receipt = preflight(
        freeze_manifest=args.freeze_manifest,
        repo_root=args.repo_root,
        source_lane=source_lane,
        lane_index=lane_index,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "preflight_receipt.json", preflight_receipt)
    write_json(
        output_dir / "ROUND59_STARTED.json",
        {
            "status": "ROUND59_RAW_FIBER_STARTED",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lane_index": lane_index,
            "final_target": FINAL_TARGET,
            "source_round56_lane": str(source_lane),
            "preflight_receipt_sha256": sha256(output_dir / "preflight_receipt.json"),
        },
    )
    started = time.time()
    try:
        manifest = read_json(args.freeze_manifest)
        frozen_lane = manifest["lanes"][str(lane_index)]
        outputs: dict[str, Any] = {}
        for rate in RATES:
            evaluation_dir = output_dir / f"rate{rate}" / "fohi"
            evaluation_dir.mkdir(parents=True, exist_ok=False)
            cache = source_lane / f"rate{rate}" / "cache" / "test_cache.pt"
            command = diagnostic_command(
                repo_root=args.repo_root,
                cache=cache,
                frozen_rate=frozen_lane["rates"][rate],
                lane_index=lane_index,
                rate=rate,
                output_dir=evaluation_dir,
            )
            run(command, cwd=args.repo_root, log_path=output_dir / f"rate{rate}" / "fohi.log")
            summary_path = evaluation_dir / "summary.json"
            vectors_path = evaluation_dir / "metric_vectors.npz"
            summary = read_json(summary_path)
            if summary.get("final_target") != FINAL_TARGET:
                raise RuntimeError(f"ROUND59_FINAL_TARGET_NOT_RECORDED:{rate}")
            if summary.get("evaluation_scope") != "heldout" or summary.get("test_split_opened") is not True:
                raise RuntimeError(f"ROUND59_HELDOUT_SCOPE_NOT_RECORDED:{rate}")
            if summary.get("operator_sha256") != frozen_lane["rates"][rate]["operator_sha256"]:
                raise RuntimeError(f"ROUND59_OPERATOR_IDENTITY_MISMATCH:{rate}")
            if not vectors_path.is_file():
                raise FileNotFoundError(f"ROUND59_METRIC_VECTORS_MISSING:{rate}")
            outputs[rate] = {
                "command": command,
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "metric_vectors": str(vectors_path),
                "metric_vectors_sha256": sha256(vectors_path),
                "reused_cache": preflight_receipt["reused_caches"][rate],
            }
        complete = {
            "status": "ROUND59_RAW_FIBER_LANE_COMPLETE",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lane_index": lane_index,
            "final_target": FINAL_TARGET,
            "evaluation_scope": "heldout",
            "method_parameters": preflight_receipt["method_parameters"],
            "code_sha256": preflight_receipt["current_code_sha256"],
            "frozen_weight_and_config_sha256": preflight_receipt["frozen_weight_and_config_sha256"],
            "reused_caches": preflight_receipt["reused_caches"],
            "rates": outputs,
            "runtime_seconds": time.time() - started,
        }
        write_json(output_dir / "ROUND59_COMPLETE.json", complete)
        print(json.dumps(complete, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        write_json(
            output_dir / "ROUND59_FAILED.json",
            {
                "status": "ROUND59_RAW_FIBER_FAILED",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "lane_index": lane_index,
                "final_target": FINAL_TARGET,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
