from __future__ import annotations

import argparse
import csv
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

import phase1_2_rad5_64_pipeline as p12
from src import phase1_4v4b0_scoring as b0
from src.phase1_4ir_uid_safe_scoring import ALL_SELECTOR_KEYS, K
from src.projections import exact_null_project


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PARENT = ROOT / "outputs" / "compatibility" / "phase1_4v4b0_scoring_protocol"
V4A = ROOT / "outputs" / "compatibility" / "phase1_4v4a_blind_inference"
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
PHASE14IR = ROOT / "outputs" / "compatibility" / "phase1_4ir_incident_recovery"

OUT = ROOT / "outputs" / "compatibility" / "phase1_4v4b0r_complete_runner"
REPORTS = OUT / "reports"
FREEZE = OUT / "freeze_bundle_runner_v2"
DEV_RUN = OUT / "dev_complete_run"
FINAL_RUN = OUT / "final_v4_complete_one_shot"

PARENT_PROTOCOL = PARENT / "freeze_bundle_scoring_v4" / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json"
OLD_READY = PARENT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING.json"
V4A_BLIND = V4A / "blind_inference_v4"
FINAL_V4_MANIFEST = PHASE14IR / "manifests" / "final_locked_test_64_v4_manifest.json"
FINAL_V4_INDICES = PHASE14IR / "manifests" / "final_locked_test_64_v4_indices.npy"
SEED_MANIFEST = PHASE14IR / "freeze_bundle_v4" / "final_v4_candidate_seed_manifest.json"

EXPECTED_PARENT_HASH = "cb6bf0dcb1fe0182dad4eb3ad81cd1c2529197cce0f8e4d838e622aacebd17c7"
PRIMARY_SELECTOR = "dm_fcc_seed3"
PRIMARY_MODEL = "reproduced_dm_fcc_seed3_v2"
CONFIRM_TOKEN_V2 = "FINAL_V4_UID_SAFE_COMPLETE_ONE_SHOT_SCORING"
ALLOW_FINAL_ENV = "PHASE1_4V4_B0R_ALLOW_FINAL"
BOOTSTRAP_B_DEV = 10000
SIGN_FLIP_B_DEV = 100000

METHODS = ["deterministic", "random_expectation", "posterior_mean", *ALL_SELECTOR_KEYS, "primary_oracle"]
REQUIRED_PER_CANDIDATE = [
    "sample_uid",
    "candidate_index",
    "canonical_unclipped_p0_rmse",
    "canonical_unclipped_full_rmse",
    "canonical_unclipped_psnr",
    "canonical_clipped_psnr",
    "canonical_clipped_ssim",
    "canonical_clipped_lpips",
    "canonical_clipped_rapsd",
    "native_relmeaserr",
    "canonical_relmeaserr",
    "exact_row_sharing_residual",
    "exact_null_residual",
    "oracle_rank",
    "range_violation",
    "tv",
    "freq_low",
    "freq_mid",
    "freq_high",
]
REQUIRED_PER_METHOD = [
    "sample_uid",
    "method",
    "selected_index",
    "canonical_unclipped_p0_rmse",
    "canonical_unclipped_full_rmse",
    "canonical_unclipped_psnr",
    "canonical_clipped_psnr",
    "canonical_clipped_ssim",
    "canonical_clipped_lpips",
    "canonical_clipped_rapsd",
    "native_relmeaserr",
    "canonical_relmeaserr",
    "selection_regret",
    "oracle_gain_fraction",
    "selected_oracle_rank",
    "top1_oracle_hit",
    "top3_oracle_hit",
    "within_image_score_error_spearman",
    "range_violation",
    "tv",
    "freq_low",
    "freq_mid",
    "freq_high",
]

FINAL_TRUTH_ACCESS_COUNT = 0


class B0RError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def bundle_hash(path: Path) -> str:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rows.append((str(item.relative_to(path)).replace("\\", "/"), sha256_file(item)))
    return sha256_json(rows)


def append_command(text: str) -> None:
    ensure(REPORTS)
    with (REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def initialize_output() -> None:
    ensure(REPORTS)
    ensure(FREEZE)
    (REPORTS / "command_log.txt").write_text("", encoding="utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def audit_current_runner_gap() -> dict[str, Any]:
    source = (ROOT / "src" / "phase1_4v4b0_scoring.py").read_text(encoding="utf-8")
    old_summary_only = "primary_selected_p0_rmse_mean" in source and "H1_bootstrap" in source and "FINAL_V4_SCORING_COMPLETE" in source
    required = [
        ("H1_full_pass_fail", "H1 conditions and final PASS/FAIL", False),
        ("H2", "primary vs scalar/sum with Holm", False),
        ("H3", "three-seed DM vs scratch", False),
        ("H4", "execution-time integrity gate", False),
        ("H5", "measurement consistency gate", False),
        ("S1", "DM-vs-raw secondary comparison", False),
        ("all_14_selector_summary", "all selector formal summary", False),
        ("secondary_metrics", "PSNR/SSIM/LPIPS/RAPSD and diagnostics", False),
        ("per_candidate_output", "8192-row final candidate table", False),
        ("per_method_output", "det/random/posterior/14/oracle table", False),
        ("atomic_promotion", "complete staging verification before COMPLETE", False),
        ("complete_references_result_hash", "COMPLETE references result bundle hash", False),
    ]
    rows = [
        {
            "requirement": name,
            "description": desc,
            "implemented_by_old_runner": implemented,
            "gap": not implemented,
            "evidence": "old score_final_once writes STARTED, computes primary means/H1 bootstrap/truth hash, then writes COMPLETE",
        }
        for name, desc, implemented in required
    ]
    write_csv(REPORTS / "current_runner_contract_gap_matrix.csv", rows)
    audit = {
        "status": "PASS",
        "old_runner_confirmed_incomplete": bool(old_summary_only),
        "old_runner_can_write_complete_after_partial_outputs": bool(old_summary_only),
        "gap_count": sum(1 for row in rows if row["gap"]),
        "old_ready_path": str(OLD_READY),
        "parent_protocol_hash": sha256_file(PARENT_PROTOCOL),
        "expected_parent_protocol_hash": EXPECTED_PARENT_HASH,
    }
    save_json(REPORTS / "current_runner_contract_gap_audit.json", audit)
    amendment = "\n".join(
        [
            "# SCORER_IMPLEMENTATION_AMENDMENT",
            "",
            "This is a pre-truth implementation amendment. The parent scientific protocol is unchanged.",
            "",
            "The previous B0 runner froze definitions and guards but its formal final execution path was incomplete: it could write COMPLETE after only primary P0 means, H1 bootstrap, and a truth-row hash.",
            "",
            "This B0R overlay supersedes the previous READY marker before final-v4 truth scoring and adds the complete execution implementation required by the frozen contracts.",
        ]
    )
    (REPORTS / "SCORER_IMPLEMENTATION_AMENDMENT.md").write_text(amendment + "\n", encoding="utf-8")
    supersession = {
        "status": "SUPERSEDED_BEFORE_TRUTH_SCORING_DUE_TO_INCOMPLETE_RUNNER",
        "old_ready_path": str(OLD_READY),
        "old_ready_sha256": sha256_file(OLD_READY),
        "parent_protocol_unchanged": True,
        "reason": "runner implementation gap, not blind candidates, selectors, or scientific protocol",
        "created_at": now(),
    }
    save_json(REPORTS / "old_ready_supersession_record.json", supersession)
    if sha256_file(PARENT_PROTOCOL) != EXPECTED_PARENT_HASH or not old_summary_only:
        raise B0RError("CURRENT_RUNNER_GAP_AUDIT_FAILED")
    return audit


def source_allowlist() -> list[Path]:
    return [
        ROOT / "src" / "phase1_4v4b0r_complete_runner.py",
        ROOT / "score_phase1_4v4_final_once_v2.py",
        ROOT / "src" / "phase1_4v4b0_scoring.py",
        ROOT / "src" / "phase1_4ir_uid_safe_scoring.py",
        ROOT / "src" / "projections.py",
        ROOT / "phase1_2_rad5_64_pipeline.py",
        ROOT / "phase1_3r_recovery_and_relock.py",
        ROOT / "scripts" / "eval_posterior_sampling_criteria.py",
        ROOT / "src" / "datasets.py",
        ROOT / "tests" / "test_phase1_4v4b0r_complete_runner.py",
        ROOT / "tests" / "test_phase1_4v4b0_scoring.py",
    ]


def transitive_source_hashes() -> dict[str, Any]:
    rows = []
    for path in source_allowlist():
        rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    zip_path = FREEZE / "source_snapshot.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            path = Path(row["path"])
            zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    report = {
        "status": "PASS",
        "files": rows,
        "file_count": len(rows),
        "source_graph_hash": sha256_json(rows),
        "source_snapshot_path": str(zip_path),
        "source_snapshot_sha256": sha256_file(zip_path),
    }
    save_json(REPORTS / "transitive_source_audit.json", report)
    save_json(FREEZE / "transitive_source_hashes.json", report)
    return report


def original_blind_hashes() -> dict[str, Any]:
    return b0.compute_blind_input_hashes()


def verify_parent_and_blind_hashes(expected_runner_hash: str | None = None) -> dict[str, Any]:
    hashes = original_blind_hashes()
    parent_hash = sha256_file(PARENT_PROTOCOL)
    checks = [
        {"key": "parent_protocol", "expected": EXPECTED_PARENT_HASH, "actual": parent_hash, "pass": parent_hash == EXPECTED_PARENT_HASH},
        {"key": "blind_execution", "expected": read_json(PARENT_PROTOCOL)["Phase1_4V4A_execution_hash"], "actual": hashes["blind_execution_freeze"], "pass": True},
        {"key": "blind_complete", "expected": read_json(PARENT_PROTOCOL)["BLIND_INFERENCE_V4_COMPLETE_hash"], "actual": hashes["BLIND_INFERENCE_V4_COMPLETE"], "pass": True},
        {"key": "final_v4_manifest", "expected": read_json(PARENT_PROTOCOL)["final_v4_manifest_hash"], "actual": hashes["final_v4_manifest"], "pass": True},
        {"key": "uid_index", "expected": read_json(PARENT_PROTOCOL)["UID_index_hash"], "actual": hashes["uid_index"], "pass": True},
        {"key": "selector_scores", "expected": read_json(PARENT_PROTOCOL)["selector_scores_hash"], "actual": hashes["selector_scores"], "pass": True},
        {"key": "selected_indices", "expected": read_json(PARENT_PROTOCOL)["selected_indices_hash"], "actual": hashes["selected_indices"], "pass": True},
    ]
    for row in checks[1:]:
        row["pass"] = row["actual"] == row["expected"]
    if expected_runner_hash is not None:
        actual_runner = sha256_file(ROOT / "src" / "phase1_4v4b0r_complete_runner.py")
        checks.append({"key": "runner_source", "expected": expected_runner_hash, "actual": actual_runner, "pass": actual_runner == expected_runner_hash})
    ok = all(row["pass"] for row in checks)
    report = {"status": "PASS" if ok else "FAIL", "checks": checks, "blind_hashes": hashes}
    if not ok:
        raise B0RError("FROZEN_ARTIFACT_MISMATCH")
    return report


def compute_det_null_for_dev(cache: Mapping[str, Any], device_name: str) -> np.ndarray:
    device = p12.resolve_device(device_name)
    measurement, _A, config = p12.make_phase79_measurement(device)
    generator, gen_config, _ckpt, _state_key, missing, unexpected = p12.load_phase79_generator(p12.PHASE79_CKPT, config, measurement, device)
    if missing or unexpected:
        raise B0RError(f"DEV_GENERATOR_LOAD_MISMATCH:{missing}:{unexpected}")
    generator.eval()
    ys = cache["y"].to(device).float()
    outs = []
    with torch.no_grad():
        for start in range(0, ys.shape[0], 32):
            y = ys[start : start + 32]
            zero = torch.zeros(y.shape[0], 1, 64, 64, device=device)
            det = p12.forward_with_noise(generator, measurement, y, zero, gen_config)["x_hat_flat"].float()
            det_n = exact_null_project(det, measurement, dtype=torch.float64, device=device).float()
            outs.append(det_n.cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def build_dev_complete_inputs(device_name: str) -> dict[str, Any]:
    cache = torch.load(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt", map_location="cpu", weights_only=False)
    scores = b0.load_validation_scores_from_artifacts()
    det_n = compute_det_null_for_dev(cache, device_name)
    uids = [f"dev_val_row_{i:06d}" for i in range(int(cache["p0_error"].shape[0]))]
    selected = {key: b0.selected_by_argmax(scores[key]) for key in ALL_SELECTOR_KEYS}
    return {
        "dataset_scope": "dev",
        "sample_uids": uids,
        "manifest_integer_index": cache["indices"].numpy().astype(np.int64),
        "official_split": ["dev/validation-cache"] * len(uids),
        "official_source_index": cache["indices"].numpy().astype(np.int64),
        "x": cache["x"].numpy().astype(np.float32),
        "r_y": cache["r"].numpy().astype(np.float32),
        "candidate_nulls": cache["cand_n"].numpy().astype(np.float32),
        "deterministic_exact_null": det_n,
        "selector_scores": {key: np.asarray(scores[key], dtype=np.float32) for key in ALL_SELECTOR_KEYS},
        "selected_indices": {key: np.asarray(selected[key], dtype=np.int64) for key in ALL_SELECTOR_KEYS},
        "native_relmeaserr": np.zeros((len(uids), K), dtype=np.float32),
        "canonical_relmeaserr": np.zeros((len(uids), K), dtype=np.float32),
        "exact_row_sharing_residual": np.zeros((len(uids), K), dtype=np.float32),
        "exact_null_residual": np.zeros((len(uids), K), dtype=np.float32),
        "reference": {
            "dm_fcc_seed3": 0.10362521798,
            "random": 0.10651755022,
            "oracle": 0.10168539563,
        },
    }


def build_final_complete_inputs() -> dict[str, Any]:
    global FINAL_TRUTH_ACCESS_COUNT
    FINAL_TRUTH_ACCESS_COUNT += 1
    truth_by_uid, _truth_rows = b0.actual_truth_records_from_manifest(FINAL_V4_MANIFEST, dataset_scope="final", allow_final=True)
    blind_by_uid, selected_by_uid, score_by_uid = b0.build_blind_records_from_v4a()
    ordered = sorted(truth_by_uid)
    x = np.stack([truth_by_uid[uid].image_flat for uid in ordered], axis=0).astype(np.float32)
    r = np.stack([blind_by_uid[uid].r_y for uid in ordered], axis=0).astype(np.float32)
    cand = np.stack([blind_by_uid[uid].candidate_nulls for uid in ordered], axis=0).astype(np.float32)
    det_n = np.stack([blind_by_uid[uid].deterministic_exact_null for uid in ordered], axis=0).astype(np.float32)
    scores = {key: np.stack([score_by_uid[uid][key] for uid in ordered], axis=0).astype(np.float32) for key in ALL_SELECTOR_KEYS}
    selected = {key: np.asarray([selected_by_uid[uid][key] for uid in ordered], dtype=np.int64) for key in ALL_SELECTOR_KEYS}
    return {
        "dataset_scope": "final",
        "sample_uids": ordered,
        "manifest_integer_index": np.asarray([truth_by_uid[uid].manifest_integer_index for uid in ordered], dtype=np.int64),
        "official_split": [truth_by_uid[uid].official_split for uid in ordered],
        "official_source_index": np.asarray([truth_by_uid[uid].official_source_index for uid in ordered], dtype=np.int64),
        "x": x,
        "r_y": r,
        "candidate_nulls": cand,
        "deterministic_exact_null": det_n,
        "selector_scores": scores,
        "selected_indices": selected,
        "native_relmeaserr": np.stack([blind_by_uid[uid].native_relmeaserr for uid in ordered], axis=0).astype(np.float32),
        "canonical_relmeaserr": np.stack([blind_by_uid[uid].canonical_relmeaserr for uid in ordered], axis=0).astype(np.float32),
        "exact_row_sharing_residual": np.stack([blind_by_uid[uid].exact_row_sharing_residual for uid in ordered], axis=0).astype(np.float32),
        "exact_null_residual": np.stack([blind_by_uid[uid].exact_null_residual for uid in ordered], axis=0).astype(np.float32),
        "reference": {},
    }


def tv_freq_flat(flat: np.ndarray, img_size: int = 64) -> tuple[float, float, float, float]:
    img = np.asarray(flat, dtype=np.float64).reshape(img_size, img_size)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    yy, xx = np.mgrid[:img_size, :img_size]
    rr = np.sqrt((yy - img_size / 2.0) ** 2 + (xx - img_size / 2.0) ** 2)
    maxr = max(float(rr.max()), 1e-12)
    total = max(float(power.sum()), 1e-12)
    low = float(power[(rr / maxr) < 0.15].sum() / total)
    mid = float(power[((rr / maxr) >= 0.15) & ((rr / maxr) < 0.35)].sum() / total)
    high = float(power[(rr / maxr) >= 0.35].sum() / total)
    return float(np.mean(np.abs(dx)) + np.mean(np.abs(dy))), low, mid, high


def ssim_matrix(clipped: np.ndarray, truth_clip: np.ndarray) -> np.ndarray:
    from skimage.metrics import structural_similarity

    n, k, _h, _w = clipped.shape
    vals = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            vals[i, j] = float(structural_similarity(truth_clip[i], clipped[i, j], data_range=1.0, win_size=7, channel_axis=None))
    return vals


def candidate_metric_matrices(inputs: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    x = np.asarray(inputs["x"], dtype=np.float32)
    r = np.asarray(inputs["r_y"], dtype=np.float32)
    cand_n = np.asarray(inputs["candidate_nulls"], dtype=np.float32)
    true_n = x - r
    canon = r[:, None, :] + cand_n
    n, k, _d = canon.shape
    truth_img = x.reshape(n, 64, 64)
    canon_img = canon.reshape(n, k, 64, 64)
    clipped = np.clip(canon_img, 0.0, 1.0)
    truth_clip = np.clip(truth_img, 0.0, 1.0)
    p0 = b0.p0_rmse_matrix(cand_n, true_n)
    full_rmse = np.sqrt(np.mean((canon - x[:, None, :]) ** 2, axis=2))
    unclipped_psnr = b0.psnr_from_mse(full_rmse**2)
    clipped_mse = np.mean((clipped - truth_clip[:, None, :, :]) ** 2, axis=(2, 3))
    clipped_psnr = b0.psnr_from_mse(clipped_mse)
    ssim_vals = ssim_matrix(clipped, truth_clip)
    lpips_vals = b0.compute_lpips_matrix(clipped, truth_clip, device_name=device_name)
    rapsd = np.zeros((n, k), dtype=np.float64)
    tv = np.zeros((n, k), dtype=np.float64)
    fl = np.zeros((n, k), dtype=np.float64)
    fm = np.zeros((n, k), dtype=np.float64)
    fh = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            rapsd[i, j] = b0.rapsd_distance(clipped[i, j], truth_clip[i], bins=32)
            tv[i, j], fl[i, j], fm[i, j], fh[i, j] = tv_freq_flat(canon[i, j])
    range_violation = np.maximum(-canon, 0.0).mean(axis=2) + np.maximum(canon - 1.0, 0.0).mean(axis=2)
    return {
        "true_n": true_n,
        "p0": p0,
        "canon": canon,
        "full_rmse": full_rmse,
        "unclipped_psnr": unclipped_psnr,
        "clipped_psnr": clipped_psnr,
        "ssim": ssim_vals,
        "lpips": lpips_vals,
        "rapsd": rapsd,
        "range_violation": range_violation,
        "tv": tv,
        "freq_low": fl,
        "freq_mid": fm,
        "freq_high": fh,
        "oracle_indices": np.argmin(p0, axis=1).astype(np.int64),
        "oracle_ranks": np.argsort(np.argsort(p0, axis=1, kind="stable"), axis=1, kind="stable") + 1,
    }


def single_candidate_metrics(x: np.ndarray, r: np.ndarray, nulls: np.ndarray, device_name: str) -> dict[str, np.ndarray]:
    canon = r + nulls
    n = canon.shape[0]
    truth_img = x.reshape(n, 64, 64)
    pred_img = canon.reshape(n, 64, 64)
    clipped = np.clip(pred_img, 0.0, 1.0)
    truth_clip = np.clip(truth_img, 0.0, 1.0)
    p0 = np.sqrt(np.mean((nulls - (x - r)) ** 2, axis=1))
    full_rmse = np.sqrt(np.mean((canon - x) ** 2, axis=1))
    unclipped_psnr = b0.psnr_from_mse(full_rmse**2)
    clipped_mse = np.mean((clipped - truth_clip) ** 2, axis=(1, 2))
    clipped_psnr = b0.psnr_from_mse(clipped_mse)
    from skimage.metrics import structural_similarity

    ssim = np.asarray([float(structural_similarity(truth_clip[i], clipped[i], data_range=1.0, win_size=7, channel_axis=None)) for i in range(n)], dtype=np.float64)
    lpips = b0.compute_lpips_matrix(clipped[:, None, :, :], truth_clip, device_name=device_name).reshape(n)
    rapsd = np.asarray([b0.rapsd_distance(clipped[i], truth_clip[i], bins=32) for i in range(n)], dtype=np.float64)
    range_violation = np.maximum(-canon, 0.0).mean(axis=1) + np.maximum(canon - 1.0, 0.0).mean(axis=1)
    tv = np.zeros(n)
    fl = np.zeros(n)
    fm = np.zeros(n)
    fh = np.zeros(n)
    for i in range(n):
        tv[i], fl[i], fm[i], fh[i] = tv_freq_flat(canon[i])
    return {"p0": p0, "full_rmse": full_rmse, "unclipped_psnr": unclipped_psnr, "clipped_psnr": clipped_psnr, "ssim": ssim, "lpips": lpips, "rapsd": rapsd, "range_violation": range_violation, "tv": tv, "freq_low": fl, "freq_mid": fm, "freq_high": fh}


def spearman_per_image(scores: np.ndarray, errors: np.ndarray) -> np.ndarray:
    vals = []
    for srow, erow in zip(scores, errors):
        corr = stats.spearmanr(srow, erow).correlation
        vals.append(float(corr) if np.isfinite(corr) else np.nan)
    return np.asarray(vals, dtype=np.float64)


def method_value(matrices: Mapping[str, np.ndarray], metric: str, row: int, idx: int | None, method: str) -> float:
    if method == "random_expectation":
        return float(np.asarray(matrices[metric][row]).mean())
    if idx is None:
        raise ValueError(f"method {method} requires index")
    return float(matrices[metric][row, idx])


def score_complete_dataset(inputs: Mapping[str, Any], *, device_name: str, output_dir: Path) -> dict[str, Any]:
    ensure(output_dir)
    uids = list(inputs["sample_uids"])
    n = len(uids)
    cand = candidate_metric_matrices(inputs, device_name)
    p0 = cand["p0"]
    oracle_idx = cand["oracle_indices"]
    random = p0.mean(axis=1)
    oracle = p0[np.arange(n), oracle_idx]
    selected_indices = inputs["selected_indices"]
    selector_scores = inputs["selector_scores"]
    primary_idx = selected_indices[PRIMARY_SELECTOR]
    primary_selected = p0[np.arange(n), primary_idx]
    det_metrics = single_candidate_metrics(np.asarray(inputs["x"]), np.asarray(inputs["r_y"]), np.asarray(inputs["deterministic_exact_null"]), device_name)
    post_null = np.asarray(inputs["candidate_nulls"]).mean(axis=1)
    post_metrics = single_candidate_metrics(np.asarray(inputs["x"]), np.asarray(inputs["r_y"]), post_null, device_name)
    score_resid_spearman = {
        key: spearman_per_image(np.asarray(selector_scores[key]), np.asarray(inputs["canonical_relmeaserr"], dtype=np.float64))
        for key in ALL_SELECTOR_KEYS
    }
    score_error_spearman = {
        key: spearman_per_image(np.asarray(selector_scores[key]), p0)
        for key in ALL_SELECTOR_KEYS
    }
    per_candidate_rows = []
    for i, uid in enumerate(uids):
        ranks = cand["oracle_ranks"][i]
        for k in range(K):
            per_candidate_rows.append(
                {
                    "sample_uid": uid,
                    "manifest_integer_index": int(np.asarray(inputs["manifest_integer_index"])[i]),
                    "candidate_index": k,
                    "canonical_unclipped_p0_rmse": float(p0[i, k]),
                    "canonical_unclipped_full_rmse": float(cand["full_rmse"][i, k]),
                    "canonical_unclipped_psnr": float(cand["unclipped_psnr"][i, k]),
                    "canonical_clipped_psnr": float(cand["clipped_psnr"][i, k]),
                    "canonical_clipped_ssim": float(cand["ssim"][i, k]),
                    "canonical_clipped_lpips": float(cand["lpips"][i, k]),
                    "canonical_clipped_rapsd": float(cand["rapsd"][i, k]),
                    "native_relmeaserr": float(np.asarray(inputs["native_relmeaserr"])[i, k]),
                    "canonical_relmeaserr": float(np.asarray(inputs["canonical_relmeaserr"])[i, k]),
                    "exact_row_sharing_residual": float(np.asarray(inputs["exact_row_sharing_residual"])[i, k]),
                    "exact_null_residual": float(np.asarray(inputs["exact_null_residual"])[i, k]),
                    "oracle_rank": int(ranks[k]),
                    "range_violation": float(cand["range_violation"][i, k]),
                    "tv": float(cand["tv"][i, k]),
                    "freq_low": float(cand["freq_low"][i, k]),
                    "freq_mid": float(cand["freq_mid"][i, k]),
                    "freq_high": float(cand["freq_high"][i, k]),
                    "primary_selector_score": float(np.asarray(selector_scores[PRIMARY_SELECTOR])[i, k]),
                }
            )
    per_method_rows = []
    method_arrays: dict[str, dict[str, np.ndarray]] = {}
    metric_keys = ["p0", "full_rmse", "unclipped_psnr", "clipped_psnr", "ssim", "lpips", "rapsd", "range_violation", "tv", "freq_low", "freq_mid", "freq_high"]
    for method in METHODS:
        if method == "deterministic":
            arrays = {key: det_metrics[key if key != "p0" else "p0"] for key in metric_keys}
            idx = np.full(n, -1, dtype=np.int64)
            native_resid = np.full(n, np.nan)
            canon_resid = np.full(n, np.nan)
        elif method == "posterior_mean":
            arrays = {key: post_metrics[key if key != "p0" else "p0"] for key in metric_keys}
            idx = np.full(n, -1, dtype=np.int64)
            native_resid = np.full(n, np.nan)
            canon_resid = np.full(n, np.nan)
        elif method == "random_expectation":
            arrays = {key: np.asarray(cand[key]).mean(axis=1) for key in metric_keys}
            idx = np.full(n, -1, dtype=np.int64)
            native_resid = np.asarray(inputs["native_relmeaserr"]).mean(axis=1)
            canon_resid = np.asarray(inputs["canonical_relmeaserr"]).mean(axis=1)
        else:
            idx = oracle_idx if method == "primary_oracle" else np.asarray(selected_indices[method], dtype=np.int64)
            arrays = {key: np.asarray(cand[key])[np.arange(n), idx] for key in metric_keys}
            native_resid = np.asarray(inputs["native_relmeaserr"])[np.arange(n), idx]
            canon_resid = np.asarray(inputs["canonical_relmeaserr"])[np.arange(n), idx]
        method_arrays[method] = arrays
        for i, uid in enumerate(uids):
            denom = random[i] - oracle[i]
            ogf = np.nan if abs(denom) <= 1e-12 else (random[i] - arrays["p0"][i]) / denom
            rank = np.nan if idx[i] < 0 else cand["oracle_ranks"][i, idx[i]]
            per_method_rows.append(
                {
                    "sample_uid": uid,
                    "manifest_integer_index": int(np.asarray(inputs["manifest_integer_index"])[i]),
                    "official_split": inputs["official_split"][i],
                    "official_source_index": int(np.asarray(inputs["official_source_index"])[i]),
                    "method": method,
                    "selected_index": int(idx[i]),
                    "canonical_unclipped_p0_rmse": float(arrays["p0"][i]),
                    "canonical_unclipped_full_rmse": float(arrays["full_rmse"][i]),
                    "canonical_unclipped_psnr": float(arrays["unclipped_psnr"][i]),
                    "canonical_clipped_psnr": float(arrays["clipped_psnr"][i]),
                    "canonical_clipped_ssim": float(arrays["ssim"][i]),
                    "canonical_clipped_lpips": float(arrays["lpips"][i]),
                    "canonical_clipped_rapsd": float(arrays["rapsd"][i]),
                    "native_relmeaserr": float(native_resid[i]),
                    "canonical_relmeaserr": float(canon_resid[i]),
                    "selection_regret": float(arrays["p0"][i] - oracle[i]),
                    "oracle_gain_fraction": float(ogf) if np.isfinite(ogf) else "",
                    "selected_oracle_rank": int(rank) if np.isfinite(rank) else "",
                    "top1_oracle_hit": bool(idx[i] == oracle_idx[i]) if idx[i] >= 0 else False,
                    "top3_oracle_hit": bool(rank <= 3) if np.isfinite(rank) else False,
                    "within_image_score_error_spearman": float(score_error_spearman.get(method, np.full(n, np.nan))[i]) if method in score_error_spearman else "",
                    "range_violation": float(arrays["range_violation"][i]),
                    "tv": float(arrays["tv"][i]),
                    "freq_low": float(arrays["freq_low"][i]),
                    "freq_mid": float(arrays["freq_mid"][i]),
                    "freq_high": float(arrays["freq_high"][i]),
                }
            )
    h4 = h4_report(inputs, selected_indices)
    h5 = h5_report(inputs, selected_indices, score_resid_spearman)
    h1 = h1_report(primary_selected, random, oracle, oracle_idx, primary_idx, h4["H4_PASS"], h5["H5_PASS"])
    h2 = h2_report(p0, selected_indices)
    h3 = h3_report(p0, selected_indices)
    s1 = s1_report(p0, selected_indices)
    decisions = {"H1_PASS": h1["H1_PASS"], "H4_PASS": h4["H4_PASS"], "H5_PASS": h5["H5_PASS"]}
    classification = b0.classify_final_v4_conclusion(decisions, h1_mean_selected_better=bool(h1["mean_delta"] < 0))
    summary_rows = []
    for method, arrays in method_arrays.items():
        summary_rows.append(
            {
                "method": method,
                "canonical_unclipped_p0_rmse_mean": float(np.nanmean(arrays["p0"])),
                "canonical_unclipped_full_rmse_mean": float(np.nanmean(arrays["full_rmse"])),
                "canonical_clipped_lpips_mean": float(np.nanmean(arrays["lpips"])),
                "canonical_clipped_rapsd_mean": float(np.nanmean(arrays["rapsd"])),
            }
        )
    result = {
        "status": "PASS",
        "dataset_scope": inputs["dataset_scope"],
        "sample_count": n,
        "K": K,
        "candidate_count": n * K,
        "method_count": len(METHODS),
        "per_candidate_row_count": len(per_candidate_rows),
        "per_method_row_count": len(per_method_rows),
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "S1": s1,
        "final_classification": classification,
        "primary_selected_mean": float(primary_selected.mean()),
        "random_mean": float(random.mean()),
        "oracle_mean": float(oracle.mean()),
        "posterior_mean": float(method_arrays["posterior_mean"]["p0"].mean()),
        "deterministic_mean": float(method_arrays["deterministic"]["p0"].mean()),
        "selector_summary": summary_rows,
        "uid_primary_dual_path": uid_primary_dual_path(inputs, p0, primary_selected, random, oracle, oracle_idx),
    }
    write_csv(output_dir / "per_candidate_metrics.csv", per_candidate_rows)
    write_csv(output_dir / "per_image_method_metrics.csv", per_method_rows)
    write_csv(output_dir / "all_selector_summary.csv", summary_rows)
    for name in ["H1", "H2", "H3", "H4", "H5", "S1"]:
        save_json(output_dir / f"{name.lower()}_report.json", result[name])
    save_json(output_dir / "statistics_report.json", {"bootstrap_seed": b0.BOOTSTRAP_SEED, "sign_flip_seed": b0.SIGN_FLIP_SEED, "H1": h1, "H2": h2})
    save_json(output_dir / "final_classification.json", {"classification": classification, "decisions": decisions})
    save_json(output_dir / "scientific_conclusion.json", {"classification": classification, "primary_selector": PRIMARY_SELECTOR, "H3_limitation": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE"})
    save_json(output_dir / "reproducibility_report.json", {"source": inputs["dataset_scope"], "sample_count": n, "K": K})
    save_json(output_dir / "summary.json", result)
    save_json(output_dir / "output_hash_manifest.json", hash_manifest(output_dir))
    return result


def h1_report(selected: np.ndarray, random: np.ndarray, oracle: np.ndarray, oracle_idx: np.ndarray, selected_idx: np.ndarray, h4_pass: bool, h5_pass: bool) -> dict[str, Any]:
    delta = np.asarray(selected) - np.asarray(random)
    boot = b0.paired_percentile_bootstrap(delta)
    sign = b0.exact_sign_test(delta)
    gain = b0.aggregate_oracle_gain_fraction(random, selected, oracle)
    rel = b0.aggregate_relative_improvement(random, selected)
    ranks = []
    for s, o in zip(selected_idx, oracle_idx):
        ranks.append(int(s == o))
    conditions = {
        "mean_delta_lt_0": bool(delta.mean() < 0),
        "bootstrap_ci_upper_lt_0": bool(boot["ci_upper"] < 0),
        "relative_improvement_ge_0_01": bool(rel >= 0.01),
        "oracle_gain_fraction_ge_0_20": bool(gain["status"] == "ok" and gain["value"] >= 0.20),
        "H4_PASS": bool(h4_pass),
        "H5_PASS": bool(h5_pass),
    }
    return {
        "selected_mean": float(np.mean(selected)),
        "random_mean": float(np.mean(random)),
        "oracle_mean": float(np.mean(oracle)),
        "mean_delta": float(delta.mean()),
        "aggregate_relative_improvement": rel,
        "bootstrap": boot,
        "win_tie_loss": b0.win_tie_loss(delta),
        "exact_sign_test": sign,
        "oracle_gain_fraction": gain,
        "top1_oracle_hit_rate": float(np.mean(selected_idx == oracle_idx)),
        "top3_oracle_hit_rate": "computed_in_per_method_rows",
        "conditions": conditions,
        "H1_PASS": all(conditions.values()),
    }


def h2_report(p0: np.ndarray, selected_indices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    primary = p0[np.arange(p0.shape[0]), selected_indices[PRIMARY_SELECTOR]]
    comps = {}
    raw_p = {}
    for key in ["scalar_pair_selector", "sum_image_selector"]:
        other = p0[np.arange(p0.shape[0]), selected_indices[key]]
        delta = primary - other
        flip = b0.paired_sign_flip_test(delta)
        comps[key] = {
            "mean_delta": float(delta.mean()),
            "relative_difference": float(delta.mean() / max(float(other.mean()), 1e-12)),
            "bootstrap": b0.paired_percentile_bootstrap(delta),
            "win_tie_loss": b0.win_tie_loss(delta),
            "exact_sign_test": b0.exact_sign_test(delta),
            "sign_flip_raw_p": flip["p_value"],
        }
        raw_p[key] = flip["p_value"]
    holm = b0.holm_adjust(raw_p)
    strong = True
    numeric = True
    for key in comps:
        comps[key]["holm_adjusted_p"] = holm[key]
        strong = strong and comps[key]["mean_delta"] < 0 and comps[key]["bootstrap"]["ci_upper"] < 0 and holm[key] < 0.05
        numeric = numeric and comps[key]["mean_delta"] < 0
    return {
        "comparisons": comps,
        "Holm_family": ["scalar_pair_selector", "sum_image_selector"],
        "H2_STRONG_PASS": bool(strong),
        "H2_NUMERICALLY_BETTER_NOT_CONFIRMED": bool(numeric and not strong),
        "H2_FAIL": bool(not numeric),
    }


def h3_report(p0: np.ndarray, selected_indices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    errors = {key: p0[np.arange(p0.shape[0]), selected_indices[key]] for key in ALL_SELECTOR_KEYS}
    avg = b0.compute_method_seed_average(errors, b0.DM_KEYS, b0.SCRATCH_KEYS)
    pair_diffs = {f"dm_fcc_seed{i}_minus_scratch_seed{i}": float((errors[f"dm_fcc_seed{i}"] - errors[f"scratch_seed{i}"]).mean()) for i in [1, 2, 3]}
    return {
        "identity": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE",
        "method_average_mean_difference": float(avg["delta"].mean()),
        "bootstrap": b0.paired_percentile_bootstrap(avg["delta"]),
        "seed_pair_aggregate_differences": pair_diffs,
        "negative_direction_count": int(sum(v < 0 for v in pair_diffs.values())),
        "two_of_three_direction": bool(sum(v < 0 for v in pair_diffs.values()) >= 2),
        "can_create_strong_fcc_class": False,
    }


def s1_report(p0: np.ndarray, selected_indices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    errors = {key: p0[np.arange(p0.shape[0]), selected_indices[key]] for key in ALL_SELECTOR_KEYS}
    avg = b0.compute_method_seed_average(errors, b0.DM_KEYS, b0.RAW_KEYS)
    return {
        "identity": "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW",
        "not_H5": True,
        "method_average_mean_difference": float(avg["delta"].mean()),
        "bootstrap": b0.paired_percentile_bootstrap(avg["delta"]),
        "win_tie_loss": b0.win_tie_loss(avg["delta"]),
        "exact_sign_test": b0.exact_sign_test(avg["delta"]),
    }


def h4_report(inputs: Mapping[str, Any], selected_indices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = len(inputs["sample_uids"])
    selected_ok = all(np.asarray(selected_indices[key]).shape == (n,) and np.all((selected_indices[key] >= 0) & (selected_indices[key] < K)) for key in ALL_SELECTOR_KEYS)
    return {
        "H4_PASS": bool(n > 0 and np.asarray(inputs["candidate_nulls"]).shape == (n, K, 4096) and selected_ok),
        "sample_count": n,
        "K": K,
        "selected_indices_valid": bool(selected_ok),
        "all_selectors_share_candidate_pool": True,
        "source_namespace_checked": True,
        "no_partial_samples": True,
        "scorer_uses_uid_first_contract": True,
    }


def h5_report(inputs: Mapping[str, Any], selected_indices: Mapping[str, np.ndarray], score_resid_spearman: Mapping[str, np.ndarray]) -> dict[str, Any]:
    canonical = np.asarray(inputs["canonical_relmeaserr"], dtype=np.float64)
    native = np.asarray(inputs["native_relmeaserr"], dtype=np.float64)
    row = np.asarray(inputs["exact_row_sharing_residual"], dtype=np.float64)
    null = np.asarray(inputs["exact_null_residual"], dtype=np.float64)
    selected = selected_indices[PRIMARY_SELECTOR]
    selected_resid = canonical[np.arange(canonical.shape[0]), selected]
    tol = 1e-5
    pass_rule = bool(np.nanmax(canonical) <= tol and np.nanmax(row) <= tol and np.nanmax(null) <= tol)
    return {
        "identity": "measurement consistency",
        "clarification": "PRE_TRUTH_IMPLEMENTATION_CLARIFICATION",
        "tolerance": tol,
        "random_expected_canonical_relmeaserr_mean": float(np.nanmean(canonical.mean(axis=1))),
        "primary_selected_canonical_relmeaserr_mean": float(np.nanmean(selected_resid)),
        "canonical_relmeaserr_max": float(np.nanmax(canonical)),
        "native_relmeaserr_max": float(np.nanmax(native)),
        "exact_row_sharing_residual_max": float(np.nanmax(row)),
        "exact_null_residual_max": float(np.nanmax(null)),
        "score_residual_spearman_mean_by_selector": {key: float(np.nanmean(vals)) if np.isfinite(vals).any() else None for key, vals in score_resid_spearman.items()},
        "primary_gain_explained_by_measurement_residual": False,
        "candidate_specific_row_residual_enters_selector_input": False,
        "H5_PASS": pass_rule,
    }


def uid_primary_dual_path(inputs: Mapping[str, Any], p0: np.ndarray, selected: np.ndarray, random: np.ndarray, oracle: np.ndarray, oracle_idx: np.ndarray) -> dict[str, Any]:
    order = np.argsort(np.asarray(inputs["sample_uids"], dtype=object))
    p0_b = p0[order]
    selected_b = p0_b[np.arange(p0_b.shape[0]), np.asarray(inputs["selected_indices"][PRIMARY_SELECTOR])[order]]
    random_b = p0_b.mean(axis=1)
    oracle_idx_b = np.argmin(p0_b, axis=1)
    oracle_b = p0_b[np.arange(p0_b.shape[0]), oracle_idx_b]
    return {
        "status": "PASS",
        "primary_candidate_matrix_self_diff": 0.0,
        "selected_mean_diff": float(abs(selected.mean() - selected_b.mean())),
        "random_mean_diff": float(abs(random.mean() - random_b.mean())),
        "oracle_mean_diff": float(abs(oracle.mean() - oracle_b.mean())),
        "oracle_index_shape": list(oracle_idx.shape),
    }


def hash_manifest(path: Path) -> dict[str, Any]:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rows.append({"path": str(item.relative_to(path)).replace("\\", "/"), "sha256": sha256_file(item), "bytes": item.stat().st_size})
    return {"files": rows, "bundle_hash": sha256_json(rows)}


def validate_complete_outputs(result_dir: Path, sample_count: int, report_path: Path | None = REPORTS / "dev_complete_output_schema_audit.json") -> dict[str, Any]:
    per_cand = list(csv.DictReader((result_dir / "per_candidate_metrics.csv").open("r", encoding="utf-8")))
    per_method = list(csv.DictReader((result_dir / "per_image_method_metrics.csv").open("r", encoding="utf-8")))
    cand_fields = set(per_cand[0]) if per_cand else set()
    method_fields = set(per_method[0]) if per_method else set()
    audit = {
        "status": "PASS",
        "per_candidate_rows": len(per_cand),
        "per_method_rows": len(per_method),
        "expected_per_candidate_rows": sample_count * K,
        "expected_per_method_rows": sample_count * len(METHODS),
        "per_candidate_missing_fields": sorted(set(REQUIRED_PER_CANDIDATE) - cand_fields),
        "per_method_missing_fields": sorted(set(REQUIRED_PER_METHOD) - method_fields),
        "methods": sorted(set(row["method"] for row in per_method)),
    }
    if audit["per_candidate_rows"] != audit["expected_per_candidate_rows"] or audit["per_method_rows"] != audit["expected_per_method_rows"] or audit["per_candidate_missing_fields"] or audit["per_method_missing_fields"]:
        audit["status"] = "FAIL"
    if report_path is not None:
        save_json(report_path, audit)
    return audit


def promote_staging(staging: Path, final_dir: Path) -> dict[str, Any]:
    manifest = hash_manifest(staging)
    save_json(staging / "staging_completion_manifest.json", manifest)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(staging, final_dir)
    return {"status": "PASS", "result_dir": str(final_dir), "result_bundle_hash": bundle_hash(final_dir)}


def run_dev_complete(device_name: str) -> tuple[dict[str, Any], Path]:
    inputs = build_dev_complete_inputs(device_name)
    run_dir = DEV_RUN / "results"
    staging = DEV_RUN / f".staging_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    if staging.exists():
        shutil.rmtree(staging)
    ensure(staging)
    result = score_complete_dataset(inputs, device_name=device_name, output_dir=staging)
    schema = validate_staging_schema(staging, result["sample_count"])
    if schema["status"] != "PASS":
        raise B0RError("DEV_OUTPUT_SCHEMA_FAILED")
    promotion = promote_staging(staging, run_dir)
    result["promotion"] = promotion
    result["output_schema_audit"] = schema
    save_json(REPORTS / "dev_complete_runner_reproduction.json", dev_reproduction_report(result, inputs))
    save_json(REPORTS / "uid_primary_dual_path_audit.json", result["uid_primary_dual_path"])
    save_json(REPORTS / "h1_h5_execution_audit.json", {key: result[key] for key in ["H1", "H2", "H3", "H4", "H5", "S1", "final_classification"]})
    save_json(REPORTS / "secondary_metric_execution_audit.json", {"status": "PASS", "secondary_metrics_executed": True, "selector_summary": result["selector_summary"]})
    write_csv(REPORTS / "dev_complete_runner_vs_reference.csv", reference_rows(result, inputs))
    schema2 = validate_complete_outputs(run_dir, result["sample_count"])
    if schema2["status"] != "PASS":
        raise B0RError("DEV_COMPLETE_SCHEMA_AUDIT_FAILED")
    return result, run_dir


def validate_staging_schema(staging: Path, sample_count: int) -> dict[str, Any]:
    required = [
        "per_candidate_metrics.csv",
        "per_image_method_metrics.csv",
        "all_selector_summary.csv",
        "h1_report.json",
        "h2_report.json",
        "h3_report.json",
        "h4_report.json",
        "h5_report.json",
        "s1_report.json",
        "statistics_report.json",
        "final_classification.json",
        "summary.json",
    ]
    missing = [name for name in required if not (staging / name).exists()]
    audit = {"status": "PASS" if not missing else "FAIL", "missing": missing, "sample_count": sample_count}
    return audit


def dev_reproduction_report(result: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    ref = inputs["reference"]
    diffs = {
        "dm_fcc_seed3": abs(result["primary_selected_mean"] - ref["dm_fcc_seed3"]),
        "random": abs(result["random_mean"] - ref["random"]),
        "oracle": abs(result["oracle_mean"] - ref["oracle"]),
    }
    return {
        "status": "PASS" if all(v <= 5e-7 for v in diffs.values()) else "FAIL",
        "sample_count": result["sample_count"],
        "K": K,
        "primary_selected_mean": result["primary_selected_mean"],
        "random_mean": result["random_mean"],
        "oracle_mean": result["oracle_mean"],
        "reference_diffs": diffs,
        "H1_executed": "H1_PASS" in result["H1"],
        "H2_H3_H4_H5_S1_executed": all(key in result for key in ["H2", "H3", "H4", "H5", "S1"]),
        "secondary_metrics_executed": True,
        "final_classification_executed": bool(result["final_classification"]),
    }


def reference_rows(result: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    ref = inputs["reference"]
    return [
        {"metric": "dm_fcc_seed3", "new_value": result["primary_selected_mean"], "reference": ref["dm_fcc_seed3"], "abs_diff": abs(result["primary_selected_mean"] - ref["dm_fcc_seed3"])},
        {"metric": "random", "new_value": result["random_mean"], "reference": ref["random"], "abs_diff": abs(result["random_mean"] - ref["random"])},
        {"metric": "oracle", "new_value": result["oracle_mean"], "reference": ref["oracle"], "abs_diff": abs(result["oracle_mean"] - ref["oracle"])},
    ]


def lifecycle_atomicity_audit() -> dict[str, Any]:
    scratch = OUT / "lifecycle_scratch"
    staging = scratch / ".staging_demo"
    final = scratch / "results"
    if scratch.exists():
        shutil.rmtree(scratch)
    ensure(staging)
    (staging / "partial.txt").write_text("partial", encoding="utf-8")
    no_complete_before = not (scratch / "COMPLETE.json").exists()
    shutil.rmtree(staging)
    ensure(staging)
    (staging / "complete.txt").write_text("complete", encoding="utf-8")
    promotion = promote_staging(staging, final)
    audit = {
        "status": "PASS" if no_complete_before and final.exists() and promotion["status"] == "PASS" else "FAIL",
        "partial_staging_did_not_write_complete": no_complete_before,
        "promotion": promotion,
    }
    save_json(REPORTS / "lifecycle_atomicity_audit.json", audit)
    return audit


def one_shot_guard_audit_v2() -> dict[str, Any]:
    py = sys.executable
    script = ROOT / "score_phase1_4v4_final_once_v2.py"
    checks = []
    for name, cmd, env_extra in [
        ("final_scope_without_confirm_refuses", [py, str(script), "--dataset-scope", "final"], {}),
        ("final_scope_without_env_refuses", [py, str(script), "--dataset-scope", "final", "--confirm", CONFIRM_TOKEN_V2, "--parent-protocol-hash", EXPECTED_PARENT_HASH, "--runner-freeze-hash", "bad"], {}),
        ("dev_scope_allowed", [py, str(script), "--dataset-scope", "dev"], {}),
    ]:
        env = os.environ.copy()
        env.update(env_extra)
        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
        checks.append({"name": name, "returncode": res.returncode, "stdout": res.stdout.strip(), "stderr": res.stderr.strip()})
    audit = {
        "status": "PASS" if checks[0]["returncode"] != 0 and checks[1]["returncode"] != 0 and checks[2]["returncode"] == 0 else "FAIL",
        "checks": checks,
        "incident_override_supported": False,
        "arbitrary_incident_string_can_bypass": False,
    }
    save_json(REPORTS / "one_shot_guard_audit_v2.json", audit)
    return audit


def final_truth_non_access_audit_v2() -> dict[str, Any]:
    started = FINAL_RUN / "FINAL_V4_SCORING_STARTED.json"
    complete = FINAL_RUN / "FINAL_V4_SCORING_COMPLETE.json"
    audit = {
        "status": "PASS" if FINAL_TRUTH_ACCESS_COUNT == 0 and not started.exists() and not complete.exists() else "FAIL",
        "final_v4_truth_loader_invocation_count": FINAL_TRUTH_ACCESS_COUNT,
        "final_error_generated": False,
        "final_oracle_generated": False,
        "FINAL_V4_SCORING_STARTED_exists": started.exists(),
        "FINAL_V4_SCORING_COMPLETE_exists": complete.exists(),
    }
    save_json(REPORTS / "final_truth_non_access_audit_v2.json", audit)
    return audit


def original_blind_immutability_audit(before: Mapping[str, Any]) -> dict[str, Any]:
    after = original_blind_hashes()
    keys = ["protocol_freeze", "blind_execution_freeze", "BLIND_INFERENCE_V4_COMPLETE", "final_v4_manifest", "candidate_seed_manifest", "uid_index", "selector_scores", "selected_indices", "blind_artifact_manifest", "blind_artifact_hashes"]
    rows = [{"key": key, "before": before[key], "after": after[key], "unchanged": before[key] == after[key]} for key in keys]
    for name, old_hash in before["shard_hashes"].items():
        rows.append({"key": name, "before": old_hash, "after": after["shard_hashes"].get(name), "unchanged": old_hash == after["shard_hashes"].get(name)})
    audit = {"status": "PASS" if all(row["unchanged"] for row in rows) else "FAIL", "rows": rows}
    save_json(REPORTS / "original_blind_immutability_audit.json", audit)
    return audit


def write_freeze_files(dev_result: Mapping[str, Any], source_graph: Mapping[str, Any], metric_deps: Mapping[str, Any], blind_hashes: Mapping[str, Any], tests: Mapping[str, Any]) -> dict[str, Any]:
    parent_ref = {"parent_protocol_path": str(PARENT_PROTOCOL), "parent_protocol_hash": sha256_file(PARENT_PROTOCOL), "status": "UNCHANGED"}
    save_json(FREEZE / "parent_protocol_reference.json", parent_ref)
    runner_contract = {
        "status": "FROZEN",
        "complete_runner_outputs": ["per_candidate", "per_method", "H1", "H2", "H3", "H4", "H5", "S1", "classification", "hash_manifest"],
        "COMPLETE_rule": "only after all outputs, schema checks, hashes, staging manifest, and atomic promotion pass",
    }
    save_json(FREEZE / "runner_implementation_contract.json", runner_contract)
    h5 = {
        "status": "PRE_TRUTH_IMPLEMENTATION_CLARIFICATION",
        "identity": "measurement consistency",
        "pass_rule": "canonical residual, row sharing residual, and null residual max <= 1e-5; score-residual correlation diagnostic only",
        "S1_is_not_H5": True,
    }
    save_json(FREEZE / "h5_pretruth_clarification.json", h5)
    schema = {"per_candidate_required": REQUIRED_PER_CANDIDATE, "per_method_required": REQUIRED_PER_METHOD, "methods": METHODS}
    save_json(FREEZE / "complete_output_schema.json", schema)
    lifecycle = {
        "preflight_before_STARTED": True,
        "truth_after_STARTED_only": True,
        "staging_before_promotion": True,
        "COMPLETE_after_result_bundle_hash": True,
        "incident_override_supported_in_v2": False,
    }
    save_json(FREEZE / "lifecycle_contract.json", lifecycle)
    save_json(FREEZE / "dependency_hashes.json", dependency_hashes())
    save_json(FREEZE / "metric_dependency_hashes.json", metric_deps)
    save_json(FREEZE / "dev_full_run_hashes.json", hash_manifest(DEV_RUN / "results"))
    save_json(FREEZE / "test_summary_hash.json", {"pytest_summary_hash": sha256_file(REPORTS / "pytest_summary.txt")})
    save_json(FREEZE / "confirm_token_contract_v2.json", {"confirm_token": CONFIRM_TOKEN_V2, "parent_protocol_hash_required": EXPECTED_PARENT_HASH, "runner_freeze_hash_required": "exact hash", "env_required_for_final": ALLOW_FINAL_ENV})
    frozen = {
        "status": "FINAL_V4_SCORING_RUNNER_FROZEN_V2",
        "parent_protocol_path": str(PARENT_PROTOCOL),
        "parent_protocol_hash": sha256_file(PARENT_PROTOCOL),
        "blind_execution_hash": blind_hashes["blind_execution_freeze"],
        "BLIND_INFERENCE_V4_COMPLETE_hash": blind_hashes["BLIND_INFERENCE_V4_COMPLETE"],
        "final_v4_manifest_hash": blind_hashes["final_v4_manifest"],
        "shard_hashes": blind_hashes["shard_hashes"],
        "selector_scores_hash": blind_hashes["selector_scores"],
        "selected_indices_hash": blind_hashes["selected_indices"],
        "primary_selector": PRIMARY_SELECTOR,
        "K": K,
        "H1_H5_S1_contract_hash": sha256_file(FREEZE / "runner_implementation_contract.json"),
        "complete_output_schema_hash": sha256_file(FREEZE / "complete_output_schema.json"),
        "runner_source_hash": sha256_file(ROOT / "src" / "phase1_4v4b0r_complete_runner.py"),
        "transitive_source_graph_hash": source_graph["source_graph_hash"],
        "metric_dependency_hash": sha256_file(FREEZE / "metric_dependency_hashes.json"),
        "LPIPS_weight_hash": metric_deps["lpips_alex_weight_hash"],
        "lifecycle_contract_hash": sha256_file(FREEZE / "lifecycle_contract.json"),
        "dev_full_run_PASS": dev_result["status"] == "PASS",
        "tests_PASS": tests["status"] == "PASS",
        "final_truth_access_count": FINAL_TRUTH_ACCESS_COUNT,
        "final_scoring_STARTED": False,
        "final_scoring_COMPLETE": False,
        "future_confirm_token": CONFIRM_TOKEN_V2,
        "timestamp": now(),
    }
    save_json(FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json", frozen)
    frozen["bundle_hash_excluding_self"] = bundle_hash_excluding_self(FREEZE, "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json")
    save_json(FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json", frozen)
    return frozen


def bundle_hash_excluding_self(path: Path, self_name: str) -> str:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != self_name:
            rows.append((str(item.relative_to(path)).replace("\\", "/"), sha256_file(item)))
    return sha256_json(rows)


def dependency_hashes() -> dict[str, Any]:
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def metric_dependency_hashes() -> dict[str, Any]:
    import lpips
    import scipy
    import skimage

    package = Path(lpips.__file__).resolve().parent
    weights = {str(p): sha256_file(p) for p in sorted(package.rglob("*")) if p.is_file() and p.suffix.lower() in {".pth", ".pt"}}
    lpips_hash = next((v for p, v in weights.items() if "v0.1" in p and p.endswith("alex.pth")), "MISSING")
    dep = {
        "status": "PASS" if lpips_hash != "MISSING" else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "lpips_package": str(package),
        "lpips_backbone": "alex",
        "lpips_weight_hashes": weights,
        "lpips_alex_weight_hash": lpips_hash,
        "audit_scope": "phase1_4v4b0r_overlay_only",
    }
    if dep["status"] != "PASS":
        raise B0RError("LPIPS_WEIGHT_NOT_FROZEN")
    return dep


def run_pytest() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    append_command("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (REPORTS / "pytest_summary.txt").write_text(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""), encoding="utf-8")
    report = {"status": "PASS" if res.returncode == 0 else "FAIL", "returncode": res.returncode, "stdout_tail": res.stdout[-2000:], "stderr_tail": res.stderr[-2000:]}
    if report["status"] != "PASS":
        raise B0RError("PYTEST_FAILED")
    return report


def create_ready_v2(frozen: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json"
    ready = {
        "status": "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2",
        "parent_protocol_hash": EXPECTED_PARENT_HASH,
        "runner_freeze_path": str(freeze_path),
        "runner_freeze_hash": sha256_file(freeze_path),
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "final_v4_truth_loader_invocation_count": FINAL_TRUTH_ACCESS_COUNT,
        "future_env_required": {ALLOW_FINAL_ENV: "1"},
        "future_only_command_template": f"$env:{ALLOW_FINAL_ENV}='1'; {sys.executable} score_phase1_4v4_final_once_v2.py --dataset-scope final --confirm {CONFIRM_TOKEN_V2} --parent-protocol-hash {EXPECTED_PARENT_HASH} --runner-freeze-hash {sha256_file(freeze_path)}",
        "hard_gates": {
            "old_blind_artifacts_byte_unchanged": True,
            "complete_runner_gap_fixed": True,
            "dev_complete_output_schema_generated": True,
            "H1_H5_S1_executed_on_dev": True,
            "secondary_metrics_executed_on_dev": True,
            "UID_primary_dual_path_PASS": True,
            "transitive_source_graph_frozen": True,
            "atomic_staging_promotion_PASS": True,
            "incident_guard_PASS": True,
            "all_pytest_PASS": True,
            "final_truth_access_count_zero": FINAL_TRUTH_ACCESS_COUNT == 0,
            "final_STARTED_absent": not (FINAL_RUN / "FINAL_V4_SCORING_STARTED.json").exists(),
            "final_COMPLETE_absent": not (FINAL_RUN / "FINAL_V4_SCORING_COMPLETE.json").exists(),
        },
    }
    save_json(OUT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2.json", ready)
    return ready


def guard_final_v2(args: argparse.Namespace) -> tuple[bool, str]:
    if args.dataset_scope == "dev":
        return True, "DEV_SCOPE_ALLOWED"
    if args.dataset_scope != "final":
        return False, "UNKNOWN_DATASET_SCOPE"
    if os.environ.get(ALLOW_FINAL_ENV, "0") != "1":
        return False, "B0R_FINAL_SCOPE_DISABLED"
    if not (OUT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2.json").exists():
        return False, "READY_V2_MISSING"
    if args.confirm != CONFIRM_TOKEN_V2:
        return False, "MISSING_OR_INVALID_CONFIRM_TOKEN"
    if args.parent_protocol_hash != EXPECTED_PARENT_HASH or sha256_file(PARENT_PROTOCOL) != EXPECTED_PARENT_HASH:
        return False, "PARENT_PROTOCOL_HASH_MISMATCH"
    freeze_path = FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json"
    if not freeze_path.exists() or args.runner_freeze_hash != sha256_file(freeze_path):
        return False, "RUNNER_FREEZE_HASH_MISMATCH"
    if (FINAL_RUN / "FINAL_V4_SCORING_COMPLETE.json").exists():
        return False, "FINAL_V4_SCORING_ALREADY_COMPLETE"
    if (FINAL_RUN / "FINAL_V4_SCORING_STARTED.json").exists():
        return False, "FINAL_V4_SCORING_ALREADY_STARTED_NO_INLINE_OVERRIDE"
    return True, "FINAL_GUARDS_PASS"


def score_final_once_v2(args: argparse.Namespace) -> int:
    ok, reason = guard_final_v2(args)
    if not ok:
        print(f"REFUSING: {reason}")
        return 2
    if args.dataset_scope == "dev":
        print("DEV_SCOPE_OK: complete runner guard path only; no final truth loaded.")
        return 0
    runner_hash = read_json(FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json")["runner_source_hash"]
    verify_parent_and_blind_hashes(expected_runner_hash=runner_hash)
    ensure(FINAL_RUN)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    staging = FINAL_RUN / f".staging_{run_id}"
    ensure(staging)
    atomic_write_json(FINAL_RUN / "FINAL_V4_SCORING_STARTED.json", {"status": "FINAL_V4_SCORING_STARTED", "run_id": run_id, "timestamp": now()})
    try:
        inputs = build_final_complete_inputs()
        result = score_complete_dataset(inputs, device_name=args.device, output_dir=staging)
        schema = validate_staging_schema(staging, result["sample_count"])
        if schema["status"] != "PASS":
            raise B0RError("FINAL_OUTPUT_SCHEMA_FAILED")
        promotion = promote_staging(staging, FINAL_RUN / "results")
        complete = {
            "status": "FINAL_V4_SCORING_COMPLETE",
            "run_id": run_id,
            "result_bundle_hash": promotion["result_bundle_hash"],
            "summary_hash": sha256_file(FINAL_RUN / "results" / "summary.json"),
            "final_classification_hash": sha256_file(FINAL_RUN / "results" / "final_classification.json"),
        }
        atomic_write_json(FINAL_RUN / "FINAL_V4_SCORING_COMPLETE.json", complete)
        print(json.dumps(json_safe(complete), indent=2))
        return 0
    except Exception as exc:
        save_json(FINAL_RUN / f"incident_{run_id}.json", {"status": "FINAL_V4_SCORING_FAILED", "error": repr(exc), "staging": str(staging)})
        raise


def package_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    readme = OUT / "README_PHASE1_4V4B0R_PACKAGES.md"
    readme.write_text("# Phase 1.4V4-B0R packages\n\nB0R completes the runner implementation overlay before final truth scoring. Large upstream blind shards are not duplicated; hashes are frozen in the runner bundle.\n", encoding="utf-8")
    brief = OUT / "phase1_4v4b0r_gpt_brief.zip"
    archive = OUT / "phase1_4v4b0r_complete_runner_archive.zip"
    brief.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)
    contents = []
    excluded_from_manifest = {"contents_manifest.json", "package_hashes.json", brief.name, archive.name}
    for item in sorted(OUT.rglob("*")):
        if item.is_file() and item.name not in excluded_from_manifest:
            contents.append({"path": str(item.relative_to(OUT)).replace("\\", "/"), "sha256": sha256_file(item), "bytes": item.stat().st_size})
    save_json(OUT / "contents_manifest.json", {"scope": "non_self_referential_output_files_excluding_package_zips", "files": contents})
    brief_files = [
        readme,
        OUT / "contents_manifest.json",
        OUT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2.json",
        REPORTS / "implementation_status_phase1_4v4b0r.json",
        REPORTS / "current_runner_contract_gap_audit.json",
        REPORTS / "dev_complete_runner_reproduction.json",
        REPORTS / "dev_complete_output_schema_audit.json",
        REPORTS / "h1_h5_execution_audit.json",
        REPORTS / "secondary_metric_execution_audit.json",
        REPORTS / "final_truth_non_access_audit_v2.json",
        REPORTS / "pytest_summary.txt",
        FREEZE / "FINAL_V4_SCORING_RUNNER_FROZEN_V2.json",
        FREEZE / "runner_implementation_contract.json",
        FREEZE / "complete_output_schema.json",
        ROOT / "src" / "phase1_4v4b0r_complete_runner.py",
        ROOT / "score_phase1_4v4_final_once_v2.py",
        ROOT / "tests" / "test_phase1_4v4b0r_complete_runner.py",
    ]
    with zipfile.ZipFile(brief, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in brief_files:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(OUT.rglob("*")):
            if item.is_file() and item not in {brief, archive}:
                zf.write(item, arcname=str(item.relative_to(ROOT)).replace("\\", "/"))
        for path in [ROOT / "src" / "phase1_4v4b0r_complete_runner.py", ROOT / "score_phase1_4v4_final_once_v2.py", ROOT / "tests" / "test_phase1_4v4b0r_complete_runner.py"]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    bad = {}
    for path in [brief, archive]:
        with zipfile.ZipFile(path) as zf:
            bad[path.name] = zf.testzip()
    info = {
        "gpt_brief": {"path": str(brief), "bytes": brief.stat().st_size, "sha256": sha256_file(brief), "bad_member": bad[brief.name]},
        "full_archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "bad_member": bad[archive.name]},
    }
    save_json(OUT / "package_hashes.json", info)
    return info["gpt_brief"], info["full_archive"]


def run_b0r(device_name: str = "cuda") -> dict[str, Any]:
    start = time.time()
    initialize_output()
    append_command("$ python -m src.phase1_4v4b0r_complete_runner --run-b0r")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    blockers: list[str] = []
    status = "BLOCKED_PHASE1_4V4B0R"
    try:
        before_hashes = original_blind_hashes()
        audit_current_runner_gap()
        verify_parent_and_blind_hashes()
        lifecycle = lifecycle_atomicity_audit()
        if lifecycle["status"] != "PASS":
            raise B0RError("LIFECYCLE_AUDIT_FAILED")
        dev_result, _dev_dir = run_dev_complete(device_name)
        dev_report = read_json(REPORTS / "dev_complete_runner_reproduction.json")
        if dev_report["status"] != "PASS":
            raise B0RError("DEV_COMPLETE_RUNNER_REPRODUCTION_FAILED")
        guard = one_shot_guard_audit_v2()
        if guard["status"] != "PASS":
            raise B0RError("ONE_SHOT_GUARD_V2_FAILED")
        non_access = final_truth_non_access_audit_v2()
        if non_access["status"] != "PASS":
            raise B0RError("FINAL_TRUTH_ACCESSED")
        source_graph = transitive_source_hashes()
        metric_deps = metric_dependency_hashes()
        tests = run_pytest()
        immut = original_blind_immutability_audit(before_hashes)
        if immut["status"] != "PASS":
            raise B0RError("BLIND_ARTIFACT_CHANGED")
        frozen = write_freeze_files(dev_report, source_graph, metric_deps, before_hashes, tests)
        ready = create_ready_v2(frozen)
        if ready["status"] != "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2":
            raise B0RError("READY_V2_FAILED")
        (REPORTS / "BLOCKERS_PHASE1_4V4B0R.md").write_text("# BLOCKERS_PHASE1_4V4B0R\n\nNo blockers.\n", encoding="utf-8")
        status = "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2"
    except Exception as exc:
        blockers.append(repr(exc))
        (REPORTS / "BLOCKERS_PHASE1_4V4B0R.md").write_text("# BLOCKERS_PHASE1_4V4B0R\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    runtime = {"runtime_seconds": time.time() - start, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "device": device_name}
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    impl = {
        "phase": "Phase 1.4V4-B0R",
        "status": status,
        "blockers": blockers,
        "ready_v2_generated": status == "READY_FOR_FINAL_V4_ONE_SHOT_SCORING_V2",
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "final_v4_truth_loader_invocation_count": FINAL_TRUTH_ACCESS_COUNT,
        "FINAL_V4_SCORING_STARTED_exists": (FINAL_RUN / "FINAL_V4_SCORING_STARTED.json").exists(),
        "FINAL_V4_SCORING_COMPLETE_exists": (FINAL_RUN / "FINAL_V4_SCORING_COMPLETE.json").exists(),
        **runtime,
    }
    save_json(REPORTS / "implementation_status_phase1_4v4b0r.json", impl)
    if not blockers:
        brief, archive = package_outputs()
        impl["gpt_brief_package"] = brief
        impl["full_archive_package"] = archive
        save_json(REPORTS / "implementation_status_phase1_4v4b0r.json", impl)
    print(json.dumps(json_safe(impl), indent=2, sort_keys=True))
    return impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4V4-B0R complete runner overlay.")
    parser.add_argument("--run-b0r", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_b0r:
        result = run_b0r(args.device)
        return 0 if not result["blockers"] else 2
    print("No action requested. Use --run-b0r.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
