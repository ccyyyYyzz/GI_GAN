from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify_hash_map(entries: dict[str, str], *, label: str) -> dict[str, str]:
    verified = {}
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label}_MISSING:{path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label}_HASH_MISMATCH:{path}:{actual}:{expected}")
        verified[str(path)] = actual
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--bootstrap-reps", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if str(args.split) != "test":
        raise RuntimeError("HELDOUT_RUN_REQUIRES_STL10_TEST")
    if int(args.bootstrap_reps) != 20000 or int(args.bootstrap_seed) != 20260719:
        raise RuntimeError("HELDOUT_BOOTSTRAP_POLICY_MISMATCH")
    root = Path(__file__).resolve().parent
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN":
        raise RuntimeError("FREEZE_MANIFEST_STATUS_MISMATCH")
    if manifest.get("test_split_opened") is not False:
        raise RuntimeError("FREEZE_MANIFEST_ALREADY_OPENED")

    rate_archive = Path("/content/gan_rate_bundle.zip")
    if not rate_archive.is_file():
        raise FileNotFoundError(rate_archive)
    with zipfile.ZipFile(rate_archive) as archive:
        lane_index = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
    lane = manifest["lanes"].get(str(lane_index))
    if lane is None:
        raise RuntimeError(f"UNFROZEN_LANE:{lane_index}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_marker = args.output_dir / "HELDOUT_ONCE_STARTED.json"
    completed_marker = args.output_dir / "HELDOUT_ONCE_COMPLETE.json"
    if started_marker.exists() or completed_marker.exists():
        raise RuntimeError("ONE_SHOT_HELDOUT_ALREADY_INVOKED")

    code_hashes = verify_hash_map(manifest["code_sha256"], label="CODE")
    lane_hashes = verify_hash_map(lane["artifact_sha256"], label="ARTIFACT")
    preflight = {
        "status": "HELDOUT_PREFLIGHT_HASHES_VERIFIED",
        "lane_index": lane_index,
        "code_sha256": code_hashes,
        "artifact_sha256": lane_hashes,
        "test_split_opened": False,
        "unix_time": time.time(),
    }
    write_json(args.output_dir / "preflight_receipt.json", preflight)
    write_json(
        started_marker,
        {
            "status": "HELDOUT_ONCE_STARTED",
            "lane_index": lane_index,
            "test_split_opened": True,
            "unix_time": time.time(),
        },
    )
    started = time.time()
    rate_outputs = {}
    for rate in ("05", "10"):
        frozen = lane["rates"][rate]
        rate_dir = args.output_dir / f"rate{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = rate_dir / "cache"
        run(
            [
                sys.executable,
                "-u",
                str(root / "prepare_frozen_fohi_test_cache.py"),
                "--bundle-root",
                frozen["bundle_root"],
                "--dataset-root",
                str(args.dataset_root),
                "--rate",
                rate,
                "--lane-index",
                str(lane_index),
                "--output-dir",
                str(cache_dir),
            ],
            cwd=root,
            log_path=rate_dir / "cache.log",
        )
        cache = cache_dir / "test_cache.pt"
        evaluation_dir = rate_dir / "fohi"
        run(
            [
                sys.executable,
                "-u",
                str(root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
                "--primary-val",
                str(cache),
                "--control-val",
                str(cache),
                "--config",
                frozen["config"],
                "--control-checkpoint",
                frozen["structural_checkpoint"],
                "--proposal-checkpoint",
                frozen["proposal_checkpoint"],
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
                str(int(args.bootstrap_reps)),
                "--seed",
                str(int(args.bootstrap_seed) + 100 * lane_index + int(rate)),
                "--evaluation-scope",
                "heldout",
                "--output-dir",
                str(evaluation_dir),
            ],
            cwd=root,
            log_path=rate_dir / "fohi.log",
        )
        summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
        cache_manifest = json.loads(
            (cache_dir / "test_cache_manifest.json").read_text(encoding="utf-8")
        )
        if summary.get("test_split_opened") is not True or summary.get("validation_only") is not False:
            raise RuntimeError(f"HELDOUT_SCOPE_NOT_RECORDED:{rate}")
        expected_images = int(manifest["expected_test_images"])
        if cache_manifest.get("test_images") != expected_images:
            raise RuntimeError(f"EXPECTED_FROZEN_HASH_DISJOINT_TEST_IMAGES:{rate}")
        if cache_manifest.get("included_development_raw_hash_overlap") != 0:
            raise RuntimeError(f"INCLUDED_DEVELOPMENT_OVERLAP:{rate}")
        rate_outputs[rate] = {
            "summary": str(evaluation_dir / "summary.json"),
            "metric_vectors": str(evaluation_dir / "metric_vectors.npz"),
            "test_cache_manifest": str(cache_dir / "test_cache_manifest.json"),
            "summary_sha256": sha256(evaluation_dir / "summary.json"),
            "metric_vectors_sha256": sha256(evaluation_dir / "metric_vectors.npz"),
            "test_cache_manifest_sha256": sha256(cache_dir / "test_cache_manifest.json"),
        }

    payload = {
        "status": "VQGAN_GUIDED_FOHI_HELDOUT_LANE_COMPLETE",
        "lane_index": lane_index,
        "evaluation_scope": "heldout",
        "validation_only": False,
        "test_split_opened": True,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "method_parameters": manifest["method_parameters"],
        "rates": rate_outputs,
        "runtime_seconds": time.time() - started,
    }
    write_json(completed_marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
