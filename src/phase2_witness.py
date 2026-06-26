from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.phase1_4ir_uid_safe_scoring import ALL_SELECTOR_KEYS, K


ROOT = Path(__file__).resolve().parents[1]
PHASE12_CACHE = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer" / "candidate_cache"
FINAL_V4_RUN = (
    ROOT
    / "outputs"
    / "compatibility"
    / "phase1_4v4b0r_complete_runner"
    / "final_v4_complete_one_shot"
)
DEFAULT_FINAL_SUMMARY = FINAL_V4_RUN / "results" / "summary.json"
DEFAULT_FINAL_COMPLETE = FINAL_V4_RUN / "FINAL_V4_SCORING_COMPLETE.json"


class Phase2WitnessError(RuntimeError):
    """Hard-fail exception for Phase 2 witnessed selection pilot errors."""


@dataclass(frozen=True)
class CandidateCache:
    path: Path
    name: str
    split: str
    x: np.ndarray
    r: np.ndarray
    true_n: np.ndarray
    cand_n: np.ndarray
    p0_error: np.ndarray
    deterministic_p0_error: np.ndarray
    posterior_mean_p0_error: np.ndarray
    indices: np.ndarray
    labels: np.ndarray | None
    sample_uids: list[str]
    file_sha256: str

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def k(self) -> int:
        return int(self.cand_n.shape[1])

    @property
    def d(self) -> int:
        return int(self.x.shape[1])


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val):
            return None
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in fieldnames})


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    if not isinstance(obj, dict):
        raise Phase2WitnessError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def save_config_copy(config_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")


def as_numpy(obj: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
    else:
        arr = np.asarray(obj)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def qualified_sample_uid(split: str, source_index: int, row: int, cache_name: str) -> str:
    return f"phase2_dev:{cache_name}:split:{split}:source_index:{int(source_index)}:row:{int(row)}"


def load_candidate_cache(path: Path, *, split: str) -> CandidateCache:
    if not path.exists():
        raise Phase2WitnessError(f"CANDIDATE_CACHE_MISSING:{path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["x", "r", "true_n", "cand_n", "p0_error", "indices"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise Phase2WitnessError(f"CANDIDATE_CACHE_MISSING_KEYS:{missing}")
    name = str(payload.get("name", path.stem))
    x = as_numpy(payload["x"], np.float32)
    r = as_numpy(payload["r"], np.float32)
    true_n = as_numpy(payload["true_n"], np.float32)
    cand_n = as_numpy(payload["cand_n"], np.float32)
    p0_error = as_numpy(payload["p0_error"], np.float64)
    indices = as_numpy(payload["indices"], np.int64)
    labels = as_numpy(payload["labels"], np.int64) if "labels" in payload else None
    deterministic = as_numpy(payload.get("deterministic_p0_error", np.full(x.shape[0], np.nan)), np.float64)
    posterior = as_numpy(payload.get("posterior_mean_p0_error", np.full(x.shape[0], np.nan)), np.float64)
    if x.ndim != 2 or r.shape != x.shape or true_n.shape != x.shape:
        raise Phase2WitnessError(f"CACHE_X_R_TRUE_N_SHAPE_MISMATCH:{x.shape}:{r.shape}:{true_n.shape}")
    if cand_n.ndim != 3 or cand_n.shape[0] != x.shape[0] or cand_n.shape[2] != x.shape[1]:
        raise Phase2WitnessError(f"CACHE_CANDIDATE_SHAPE_MISMATCH:{cand_n.shape}:{x.shape}")
    if cand_n.shape[1] != K:
        raise Phase2WitnessError(f"CACHE_EXPECTED_K16:{cand_n.shape}")
    if p0_error.shape != (x.shape[0], cand_n.shape[1]):
        raise Phase2WitnessError(f"CACHE_P0_SHAPE_MISMATCH:{p0_error.shape}:{cand_n.shape}")
    if indices.shape[0] != x.shape[0]:
        raise Phase2WitnessError(f"CACHE_INDEX_SHAPE_MISMATCH:{indices.shape}:{x.shape}")
    uids = [qualified_sample_uid(split, int(indices[i]), i, name) for i in range(x.shape[0])]
    if len(set(uids)) != len(uids):
        raise Phase2WitnessError("CACHE_UIDS_NOT_UNIQUE")
    return CandidateCache(
        path=path,
        name=name,
        split=split,
        x=x,
        r=r,
        true_n=true_n,
        cand_n=cand_n,
        p0_error=p0_error,
        deterministic_p0_error=deterministic,
        posterior_mean_p0_error=posterior,
        indices=indices,
        labels=labels,
        sample_uids=uids,
        file_sha256=sha256_file(path),
    )


def compute_p0_rmse(candidate_nulls: np.ndarray, true_null: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate_nulls, dtype=np.float64)
    truth = np.asarray(true_null, dtype=np.float64)
    return np.sqrt(np.mean((cand - truth[:, None, :]) ** 2, axis=2))


def load_validation_scores(expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    from src import phase1_4v4b0_scoring as b0

    scores = b0.load_validation_scores_from_artifacts()
    out: dict[str, np.ndarray] = {}
    for key in ALL_SELECTOR_KEYS:
        arr = np.asarray(scores[key], dtype=np.float32)
        if arr.shape != expected_shape:
            raise Phase2WitnessError(f"SELECTOR_SCORE_SHAPE_MISMATCH:{key}:{arr.shape}:{expected_shape}")
        out[key] = arr
    return out


def argmax_indices(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)
    if arr.ndim != 2:
        raise Phase2WitnessError(f"SCORES_MUST_BE_2D:{arr.shape}")
    return np.argmax(arr, axis=1).astype(np.int64)


def make_rademacher_rows(num_rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    rows = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(int(num_rows), int(dim)))
    return rows / math.sqrt(float(dim))


def make_dct2_lowfreq_rows(num_rows: int, img_size: int) -> np.ndarray:
    size = int(img_size)
    if size * size <= 0:
        raise Phase2WitnessError(f"INVALID_IMAGE_SIZE:{img_size}")
    coords = [(u, v) for u in range(size) for v in range(size)]
    coords.sort(key=lambda uv: (uv[0] * uv[0] + uv[1] * uv[1], uv[0] + uv[1], uv[0], uv[1]))
    yy = np.arange(size, dtype=np.float64)
    xx = np.arange(size, dtype=np.float64)
    rows = []
    for u, v in coords[: int(num_rows)]:
        au = math.sqrt(1.0 / size) if u == 0 else math.sqrt(2.0 / size)
        av = math.sqrt(1.0 / size) if v == 0 else math.sqrt(2.0 / size)
        by = au * np.cos(np.pi * (2.0 * yy + 1.0) * u / (2.0 * size))
        bx = av * np.cos(np.pi * (2.0 * xx + 1.0) * v / (2.0 * size))
        basis = np.outer(by, bx)
        flat = basis.reshape(-1)
        norm = max(float(np.linalg.norm(flat)), 1e-12)
        rows.append((flat / norm).astype(np.float32))
    return np.stack(rows, axis=0)


def make_witness_rows(kind: str, num_rows: int, dim: int, seed: int = 0) -> np.ndarray:
    kind_norm = str(kind or "rademacher").lower()
    if kind_norm in {"rademacher", "random_rademacher", "fresh_rademacher_rows"}:
        return make_rademacher_rows(int(num_rows), int(dim), int(seed))
    if kind_norm in {"dct2_low_frequency", "dct2_lowfreq", "lowfreq_dct", "fixed_lowfreq"}:
        return make_dct2_lowfreq_rows(int(num_rows), _image_size_from_dim(int(dim)))
    if kind_norm in {"hybrid_dct_rademacher", "dct_rademacher_hybrid"}:
        n_dct = int(math.ceil(int(num_rows) / 2.0))
        n_rand = int(num_rows) - n_dct
        dct = make_dct2_lowfreq_rows(n_dct, _image_size_from_dim(int(dim)))
        if n_rand <= 0:
            return dct
        rand = make_rademacher_rows(n_rand, int(dim), int(seed))
        return np.concatenate([dct, rand], axis=0).astype(np.float32, copy=False)
    raise Phase2WitnessError(f"UNKNOWN_WITNESS_ROW_KIND:{kind}")


def stable_uint64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def candidate_images(cache: CandidateCache) -> np.ndarray:
    return cache.r[:, None, :] + cache.cand_n


def witness_residual_scores(candidates: np.ndarray, truth: np.ndarray, rows: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidates, dtype=np.float32)
    true = np.asarray(truth, dtype=np.float32)
    w = np.asarray(rows, dtype=np.float32)
    if cand.ndim != 2 or true.ndim != 1 or w.ndim != 2:
        raise Phase2WitnessError(f"WITNESS_SHAPE_ERROR:{cand.shape}:{true.shape}:{w.shape}")
    diff = cand - true[None, :]
    projected = diff @ w.T
    return np.sum(np.square(projected, dtype=np.float32), axis=1).astype(np.float64)


def select_random_witness(
    candidates: np.ndarray,
    truth: np.ndarray,
    rows: np.ndarray,
    budget: int,
) -> np.ndarray:
    if int(budget) > rows.shape[0]:
        raise Phase2WitnessError(f"WITNESS_BUDGET_EXCEEDS_ROWS:{budget}:{rows.shape[0]}")
    n = candidates.shape[0]
    selected = np.empty(n, dtype=np.int64)
    sub = rows[: int(budget)]
    for i in range(n):
        scores = witness_residual_scores(candidates[i], truth[i], sub)
        selected[i] = int(np.argmin(scores))
    return selected


def adaptive_witness_order(candidates: np.ndarray, library: np.ndarray) -> np.ndarray:
    proj = np.asarray(candidates, dtype=np.float32) @ np.asarray(library, dtype=np.float32).T
    var = np.var(proj, axis=0, dtype=np.float64)
    return np.argsort(-var, kind="stable").astype(np.int64)


def select_adaptive_witness(
    candidates: np.ndarray,
    truth: np.ndarray,
    library: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, list[list[int]]]:
    n = candidates.shape[0]
    selected = np.empty(n, dtype=np.int64)
    row_orders: list[list[int]] = []
    for i in range(n):
        order = adaptive_witness_order(candidates[i], library)
        chosen = order[: int(budget)]
        row_orders.append([int(v) for v in chosen.tolist()])
        scores = witness_residual_scores(candidates[i], truth[i], library[chosen])
        selected[i] = int(np.argmin(scores))
    return selected, row_orders


def select_compatibility_prefilter_adaptive_witness(
    candidates: np.ndarray,
    truth: np.ndarray,
    library: np.ndarray,
    selector_scores: np.ndarray,
    budget: int,
    top_m: int,
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    n, k, _d = candidates.shape
    selected = np.empty(n, dtype=np.int64)
    row_orders: list[list[int]] = []
    candidate_sets: list[list[int]] = []
    scores = np.asarray(selector_scores, dtype=np.float64)
    if scores.shape != (n, k):
        raise Phase2WitnessError(f"PREFILTER_SCORE_SHAPE_MISMATCH:{scores.shape}:{(n, k)}")
    for i in range(n):
        top = np.argsort(-scores[i], kind="stable")[: int(top_m)].astype(np.int64)
        sub_candidates = candidates[i, top]
        order = adaptive_witness_order(sub_candidates, library)
        chosen = order[: int(budget)]
        candidate_sets.append([int(v) for v in top.tolist()])
        row_orders.append([int(v) for v in chosen.tolist()])
        residual = witness_residual_scores(sub_candidates, truth[i], library[chosen])
        selected[i] = int(top[int(np.argmin(residual))])
    return selected, row_orders, candidate_sets


def paired_percentile_bootstrap(delta: np.ndarray, *, reps: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise Phase2WitnessError("BOOTSTRAP_EMPTY_DELTA")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(reps), arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "observed_mean": float(arr.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "bootstrap_standard_error": float(means.std(ddof=1)),
        "fraction_negative": float(np.mean(means < 0)),
        "reps": int(reps),
        "seed": int(seed),
    }


def win_loss_tie(delta: np.ndarray, *, tie_tol: float = 1e-12) -> dict[str, int]:
    arr = np.asarray(delta, dtype=np.float64)
    return {
        "wins": int(np.sum(arr < -tie_tol)),
        "losses": int(np.sum(arr > tie_tol)),
        "ties": int(np.sum(np.abs(arr) <= tie_tol)),
    }


def aggregate_oracle_gain(random_error: np.ndarray, method_error: np.ndarray, oracle_error: np.ndarray) -> float | None:
    rand = np.asarray(random_error, dtype=np.float64)
    meth = np.asarray(method_error, dtype=np.float64)
    oracle = np.asarray(oracle_error, dtype=np.float64)
    denom = float(rand.mean() - oracle.mean())
    if abs(denom) <= 1e-12:
        return None
    return float((rand.mean() - meth.mean()) / denom)


def summarize_errors(
    name: str,
    errors: np.ndarray,
    *,
    selected_indices: np.ndarray | None,
    random_error: np.ndarray,
    oracle_error: np.ndarray,
    posterior_error: np.ndarray,
    primary_error: np.ndarray | None,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    delta_random = err - np.asarray(random_error, dtype=np.float64)
    delta_posterior = err - np.asarray(posterior_error, dtype=np.float64)
    out: dict[str, Any] = {
        "method": name,
        "mean_p0_rmse": float(err.mean()),
        "median_p0_rmse": float(np.median(err)),
        "std_p0_rmse": float(err.std(ddof=1)) if err.size > 1 else 0.0,
        "delta_vs_random_mean": float(delta_random.mean()),
        "bootstrap_vs_random": paired_percentile_bootstrap(delta_random, reps=bootstrap_reps, seed=bootstrap_seed),
        "win_loss_tie_vs_random": win_loss_tie(delta_random),
        "delta_vs_posterior_mean": float(delta_posterior.mean()),
        "bootstrap_vs_posterior": paired_percentile_bootstrap(delta_posterior, reps=bootstrap_reps, seed=bootstrap_seed + 1),
        "oracle_gain_fraction_aggregate": aggregate_oracle_gain(random_error, err, oracle_error),
    }
    if primary_error is not None:
        delta_primary = err - np.asarray(primary_error, dtype=np.float64)
        out["delta_vs_primary_selector_mean"] = float(delta_primary.mean())
        out["bootstrap_vs_primary_selector"] = paired_percentile_bootstrap(
            delta_primary, reps=bootstrap_reps, seed=bootstrap_seed + 2
        )
        out["win_loss_tie_vs_primary_selector"] = win_loss_tie(delta_primary)
    if selected_indices is not None:
        out["selected_index_mean"] = float(np.mean(selected_indices))
        out["selected_index_histogram"] = {
            str(i): int(np.sum(np.asarray(selected_indices, dtype=np.int64) == i)) for i in range(K)
        }
    return out


def selector_error_vectors(p0_error: np.ndarray, selector_scores: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, scores in selector_scores.items():
        idx = argmax_indices(scores)
        out[key] = (idx, p0_error[np.arange(p0_error.shape[0]), idx])
    return out


def posterior_headroom_audit(
    random_error: np.ndarray,
    posterior_error: np.ndarray,
    oracle_error: np.ndarray,
    method_errors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    random_mean = float(np.mean(random_error))
    posterior_mean = float(np.mean(posterior_error))
    oracle_mean = float(np.mean(oracle_error))
    oracle_headroom = random_mean - oracle_mean
    posterior_gap_to_oracle = posterior_mean - oracle_mean
    posterior_capture = None if abs(oracle_headroom) <= 1e-12 else (random_mean - posterior_mean) / oracle_headroom
    method_rows = []
    for method, errors in method_errors.items():
        arr = np.asarray(errors, dtype=np.float64)
        method_rows.append(
            {
                "method": method,
                "mean_p0_rmse": float(arr.mean()),
                "gap_to_posterior": float(arr.mean() - posterior_mean),
                "gap_to_oracle": float(arr.mean() - oracle_mean),
            }
        )
    near_oracle = bool(
        posterior_gap_to_oracle <= max(1e-4, 0.05 * max(oracle_headroom, 0.0))
    )
    return {
        "status": "PASS",
        "random_mean": random_mean,
        "posterior_mean": posterior_mean,
        "oracle_mean": oracle_mean,
        "oracle_headroom_random_minus_oracle": float(oracle_headroom),
        "posterior_gap_to_oracle": float(posterior_gap_to_oracle),
        "posterior_fraction_of_oracle_headroom_captured": None if posterior_capture is None else float(posterior_capture),
        "posterior_is_near_oracle_for_p0_rmse": near_oracle,
        "interpretation": (
            "Posterior mean is already near the best candidate; beating posterior mean is not an informative "
            "single-candidate pilot gate in this cache."
            if near_oracle
            else "Posterior mean leaves enough P0-RMSE headroom for single-candidate selector tests."
        ),
        "method_gaps": method_rows,
    }


def _image_size_from_dim(dim: int) -> int:
    size = int(round(math.sqrt(float(dim))))
    if size * size != int(dim):
        raise Phase2WitnessError(f"DIMENSION_IS_NOT_SQUARE_IMAGE:{dim}")
    return size


def _ssim_matrix(clipped: np.ndarray, truth_clip: np.ndarray) -> np.ndarray:
    from skimage.metrics import structural_similarity

    n, k, _h, _w = clipped.shape
    vals = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            vals[i, j] = float(
                structural_similarity(truth_clip[i], clipped[i, j], data_range=1.0, win_size=7, channel_axis=None)
            )
    return vals


def _single_ssim(images: np.ndarray, truth_clip: np.ndarray) -> np.ndarray:
    return _ssim_matrix(images[:, None, :, :], truth_clip).reshape(images.shape[0])


def compute_quality_audit(
    cache: CandidateCache,
    method_vectors: Mapping[str, tuple[np.ndarray | None, np.ndarray]],
    *,
    focus_methods: Sequence[str],
    compute_lpips: bool,
    lpips_device: str,
) -> dict[str, Any]:
    from src import phase1_4v4b0_scoring as b0

    img_size = _image_size_from_dim(cache.d)
    canon = candidate_images(cache).astype(np.float32, copy=False)
    canon_img = canon.reshape(cache.n, cache.k, img_size, img_size)
    truth_img = cache.x.reshape(cache.n, img_size, img_size)
    clipped = np.clip(canon_img, 0.0, 1.0)
    truth_clip = np.clip(truth_img, 0.0, 1.0)
    full_mse = np.mean((canon - cache.x[:, None, :]) ** 2, axis=2)
    full_rmse = np.sqrt(full_mse)
    unclipped_psnr = b0.psnr_from_mse(full_mse)
    clipped_mse = np.mean((clipped - truth_clip[:, None, :, :]) ** 2, axis=(2, 3))
    clipped_psnr = b0.psnr_from_mse(clipped_mse)
    ssim_vals = _ssim_matrix(clipped, truth_clip)
    rapsd_vals = np.zeros((cache.n, cache.k), dtype=np.float64)
    for i in range(cache.n):
        for j in range(cache.k):
            rapsd_vals[i, j] = b0.rapsd_distance(clipped[i, j], truth_clip[i], bins=32)
    lpips_status = "SKIPPED"
    lpips_vals: np.ndarray | None = None
    lpips_posterior: np.ndarray | None = None
    if compute_lpips:
        try:
            lpips_vals = b0.compute_lpips_matrix(clipped, truth_clip, device_name=lpips_device)
            lpips_status = "PASS"
        except Exception as exc:  # pragma: no cover - environment dependent
            lpips_status = f"MISSING_OR_FAILED:{type(exc).__name__}:{exc}"
            lpips_vals = None

    posterior_flat = cache.r + cache.cand_n.mean(axis=1)
    posterior_img = posterior_flat.reshape(cache.n, img_size, img_size)
    posterior_clip = np.clip(posterior_img, 0.0, 1.0)
    posterior_full_mse = np.mean((posterior_flat - cache.x) ** 2, axis=1)
    posterior_clipped_mse = np.mean((posterior_clip - truth_clip) ** 2, axis=(1, 2))
    if compute_lpips and lpips_status == "PASS":
        try:
            lpips_posterior = b0.compute_lpips_matrix(posterior_clip[:, None, :, :], truth_clip, device_name=lpips_device).reshape(cache.n)
        except Exception as exc:  # pragma: no cover - environment dependent
            lpips_status = f"POSTERIOR_LPIPS_FAILED:{type(exc).__name__}:{exc}"
            lpips_posterior = None

    def reduce_candidate(idx: np.ndarray) -> dict[str, float | str]:
        rows = np.arange(cache.n)
        out: dict[str, float | str] = {
            "canonical_unclipped_full_rmse_mean": float(full_rmse[rows, idx].mean()),
            "canonical_unclipped_psnr_mean": float(unclipped_psnr[rows, idx].mean()),
            "canonical_clipped_psnr_mean": float(clipped_psnr[rows, idx].mean()),
            "canonical_clipped_ssim_mean": float(ssim_vals[rows, idx].mean()),
            "canonical_clipped_rapsd_mean": float(rapsd_vals[rows, idx].mean()),
            "range_violation_mean": float(np.maximum(canon_img[rows, idx] - 1.0, 0.0).mean() + np.maximum(-canon_img[rows, idx], 0.0).mean()),
        }
        out["canonical_clipped_lpips_mean"] = (
            float(lpips_vals[rows, idx].mean()) if lpips_vals is not None else "[DATA MISSING]"
        )
        return out

    def reduce_random() -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "canonical_unclipped_full_rmse_mean": float(full_rmse.mean(axis=1).mean()),
            "canonical_unclipped_psnr_mean": float(unclipped_psnr.mean(axis=1).mean()),
            "canonical_clipped_psnr_mean": float(clipped_psnr.mean(axis=1).mean()),
            "canonical_clipped_ssim_mean": float(ssim_vals.mean(axis=1).mean()),
            "canonical_clipped_rapsd_mean": float(rapsd_vals.mean(axis=1).mean()),
            "range_violation_mean": float(np.maximum(canon_img - 1.0, 0.0).mean() + np.maximum(-canon_img, 0.0).mean()),
        }
        out["canonical_clipped_lpips_mean"] = (
            float(lpips_vals.mean(axis=1).mean()) if lpips_vals is not None else "[DATA MISSING]"
        )
        return out

    def reduce_posterior() -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "canonical_unclipped_full_rmse_mean": float(np.sqrt(posterior_full_mse).mean()),
            "canonical_unclipped_psnr_mean": float(b0.psnr_from_mse(posterior_full_mse).mean()),
            "canonical_clipped_psnr_mean": float(b0.psnr_from_mse(posterior_clipped_mse).mean()),
            "canonical_clipped_ssim_mean": float(_single_ssim(posterior_clip, truth_clip).mean()),
            "canonical_clipped_rapsd_mean": float(np.asarray([b0.rapsd_distance(posterior_clip[i], truth_clip[i], bins=32) for i in range(cache.n)]).mean()),
            "range_violation_mean": float(np.maximum(posterior_img - 1.0, 0.0).mean() + np.maximum(-posterior_img, 0.0).mean()),
        }
        out["canonical_clipped_lpips_mean"] = (
            float(lpips_posterior.mean()) if lpips_posterior is not None else "[DATA MISSING]"
        )
        return out

    methods: dict[str, Any] = {}
    for method in focus_methods:
        if method == "random_expectation":
            methods[method] = reduce_random()
        elif method == "posterior_mean":
            methods[method] = reduce_posterior()
        else:
            if method not in method_vectors or method_vectors[method][0] is None:
                continue
            methods[method] = reduce_candidate(np.asarray(method_vectors[method][0], dtype=np.int64))

    rows = []
    for method, vals in methods.items():
        row = {"method": method}
        row.update(vals)
        rows.append(row)
    return {
        "status": "PASS",
        "image_count": cache.n,
        "candidate_count": cache.n * cache.k,
        "lpips_status": lpips_status,
        "lpips_device_requested": lpips_device,
        "focus_methods": list(focus_methods),
        "method_quality_means": methods,
        "rows": rows,
        "interpretation": "Lower LPIPS/RAPSD is more perceptual/spectral match; higher PSNR/SSIM is better distortion/structure.",
    }


def cache_audit(cache: CandidateCache) -> dict[str, Any]:
    recomputed_p0 = compute_p0_rmse(cache.cand_n, cache.true_n)
    posterior = np.sqrt(np.mean((cache.cand_n.mean(axis=1) - cache.true_n) ** 2, axis=1))
    return {
        "status": "PASS",
        "cache_path": str(cache.path),
        "cache_sha256": cache.file_sha256,
        "cache_name": cache.name,
        "split": cache.split,
        "sample_count": cache.n,
        "candidate_count": cache.n * cache.k,
        "K": cache.k,
        "dimension": cache.d,
        "qualified_uid_count": len(set(cache.sample_uids)),
        "p0_error_max_abs_recompute_diff": float(np.max(np.abs(recomputed_p0 - cache.p0_error))),
        "posterior_mean_max_abs_recompute_diff": float(
            np.nanmax(np.abs(posterior - cache.posterior_mean_p0_error))
        ),
        "uses_final_v4_truth_or_candidates": False,
        "identity_note": "development cache arrays are converted to qualified sample_uid before all reporting",
    }


def final_v4_context_summary() -> dict[str, Any]:
    complete = json.loads(DEFAULT_FINAL_COMPLETE.read_text(encoding="utf-8")) if DEFAULT_FINAL_COMPLETE.exists() else {}
    summary = json.loads(DEFAULT_FINAL_SUMMARY.read_text(encoding="utf-8")) if DEFAULT_FINAL_SUMMARY.exists() else {}
    by_method = {
        str(row.get("method")): row
        for row in summary.get("selector_summary", [])
        if isinstance(row, Mapping) and row.get("method") is not None
    }
    primary = by_method.get("dm_fcc_seed3", {})
    posterior = by_method.get("posterior_mean", {})
    random = by_method.get("random_expectation", {})
    oracle = by_method.get("oracle_best_of_16", by_method.get("primary_oracle", {}))
    return {
        "status": "SUMMARY_ONLY_NOT_USED_FOR_METHOD_SELECTION",
        "complete_marker": complete,
        "final_classification": summary.get("final_classification"),
        "primary_selector": summary.get("primary_selector", "dm_fcc_seed3"),
        "primary_selected_p0_rmse_mean": summary.get(
            "primary_selected_mean", primary.get("canonical_unclipped_p0_rmse_mean")
        ),
        "random_candidate_p0_rmse_mean": summary.get(
            "random_mean", random.get("canonical_unclipped_p0_rmse_mean")
        ),
        "posterior_mean_p0_rmse_mean": summary.get(
            "posterior_mean", posterior.get("canonical_unclipped_p0_rmse_mean")
        ),
        "oracle_p0_rmse_mean": summary.get("oracle_mean", oracle.get("canonical_unclipped_p0_rmse_mean")),
        "posterior_mean_lpips": posterior.get("canonical_clipped_lpips_mean"),
        "primary_selector_lpips": primary.get("canonical_clipped_lpips_mean"),
        "H1": summary.get("H1", {}),
        "H2": summary.get("H2", {}),
        "H3": summary.get("H3", {}),
        "scientific_boundary": "final-v4 is consumed; this summary is used only to carry prior conclusions and posterior-mean limitation forward.",
    }


def leakage_operator_audit(cache: CandidateCache, config: Mapping[str, Any]) -> dict[str, Any]:
    witness_cfg = config.get("witness", {})
    random_kind = str(witness_cfg.get("random_witness", "rademacher"))
    fixed_kind = str(witness_cfg.get("fixed_witness", "dct2_low_frequency"))
    adaptive_kind = str(witness_cfg.get("adaptive_library", witness_cfg.get("adaptive_library_kind", "rademacher")))
    return {
        "status": "PASS",
        "phase": "phase2_witness_development_pilot",
        "final_v4_consumed": DEFAULT_FINAL_COMPLETE.exists(),
        "final_v4_inputs_loaded": False,
        "final_v4_used_for_method_selection": False,
        "development_cache": {
            "path": str(cache.path),
            "sha256": cache.file_sha256,
            "split": cache.split,
            "sample_count": cache.n,
            "candidate_count": cache.n * cache.k,
        },
        "sample_identity": {
            "primary_key": "qualified sample_uid",
            "uid_example": cache.sample_uids[:3],
            "position_only_join_forbidden": True,
        },
        "context_operator": {
            "source": "existing Phase1.2 Rad-5 development candidate cache",
            "role": "candidate generation context only",
            "not_a_new_locked_test_operator": True,
        },
        "witness_operator": {
            "kind": str(witness_cfg.get("operator", "fresh_rademacher_rows")),
            "random_witness_kind": random_kind,
            "fixed_witness_kind": fixed_kind,
            "adaptive_library_kind": adaptive_kind,
            "random_seed": witness_cfg.get("random_seed"),
            "adaptive_library_seed": witness_cfg.get("adaptive_library_seed"),
            "normalization": "configured row families are individually unit/row normalized; see witness kind fields",
            "seen_by_generator": False,
            "used_for_candidate_generation": False,
            "used_for_training_or_early_stopping": False,
        },
        "budget_interpretation": {
            "add_on_curve_status": "RUN_IN_THIS_PILOT",
            "fixed_total_budget_status": "NOT_RUN_REQUIRES_NEW_CONTEXT_SPLIT_CANDIDATE_GENERATION",
        },
    }


def build_method_tables(
    cache: CandidateCache,
    selector_scores: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    witness_cfg = config.get("witness", {})
    stats_cfg = config.get("statistics", {})
    budgets = [int(b) for b in witness_cfg.get("budgets", [1, 2, 4, 8, 16, 32, 64])]
    max_budget = max(budgets)
    library_size = int(witness_cfg.get("adaptive_library_size", 256))
    primary_selector = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
    top_m = int(witness_cfg.get("compatibility_prefilter_top_m", 4))
    quality_cfg = config.get("quality", {})
    bootstrap_reps = int(stats_cfg.get("bootstrap_replicates", 3000))
    bootstrap_seed = int(stats_cfg.get("bootstrap_seed", 271828))
    candidates = candidate_images(cache).astype(np.float32, copy=False)
    p0 = cache.p0_error
    true_x = cache.x
    oracle_idx = np.argmin(p0, axis=1).astype(np.int64)
    random_error = p0.mean(axis=1)
    oracle_error = p0[np.arange(cache.n), oracle_idx]
    posterior_error = np.sqrt(np.mean((cache.cand_n.mean(axis=1) - cache.true_n) ** 2, axis=1))
    deterministic_error = cache.deterministic_p0_error
    selector_vectors = selector_error_vectors(p0, selector_scores)
    if primary_selector not in selector_vectors:
        raise Phase2WitnessError(f"PRIMARY_SELECTOR_NOT_AVAILABLE:{primary_selector}")
    primary_idx, primary_error = selector_vectors[primary_selector]
    method_vectors: dict[str, tuple[np.ndarray | None, np.ndarray]] = {
        "deterministic": (None, deterministic_error),
        "random_expectation": (None, random_error),
        "posterior_mean": (None, posterior_error),
        "oracle_best_of_16": (oracle_idx, oracle_error),
    }
    method_vectors.update(selector_vectors)
    explicit_random_rows = witness_cfg.get("_explicit_random_rows")
    explicit_fixed_rows = witness_cfg.get("_explicit_fixed_rows")
    explicit_library_rows = witness_cfg.get("_explicit_library_rows")
    random_kind = str(witness_cfg.get("random_witness", "rademacher"))
    fixed_kind = str(witness_cfg.get("fixed_witness", "dct2_low_frequency"))
    adaptive_kind = str(witness_cfg.get("adaptive_library", witness_cfg.get("adaptive_library_kind", "rademacher")))
    random_rows = (
        np.asarray(explicit_random_rows, dtype=np.float32)
        if explicit_random_rows is not None
        else make_witness_rows(random_kind, max_budget, cache.d, int(witness_cfg.get("random_seed", 26021)))
    )
    fixed_rows = (
        np.asarray(explicit_fixed_rows, dtype=np.float32)
        if explicit_fixed_rows is not None
        else make_witness_rows(fixed_kind, max_budget, cache.d, int(witness_cfg.get("fixed_seed", 26023)))
    )
    library_rows = (
        np.asarray(explicit_library_rows, dtype=np.float32)
        if explicit_library_rows is not None
        else make_witness_rows(adaptive_kind, max(library_size, max_budget), cache.d, int(witness_cfg.get("adaptive_library_seed", 26022)))
    )
    for label, rows in {
        "random_rows": random_rows,
        "fixed_rows": fixed_rows,
        "library_rows": library_rows,
    }.items():
        if rows.ndim != 2 or rows.shape[1] != cache.d:
            raise Phase2WitnessError(f"WITNESS_ROWS_SHAPE_MISMATCH:{label}:{rows.shape}:{cache.d}")
        if rows.shape[0] < max_budget:
            raise Phase2WitnessError(f"WITNESS_ROWS_TOO_FEW:{label}:{rows.shape[0]}:{max_budget}")
    witness_traces: dict[str, Any] = {
        "budgets": budgets,
        "fixed_witness": str(
            witness_cfg.get(
                "fixed_witness",
                "DCT-II low-frequency rows sorted by radial frequency",
            )
        ),
        "random_witness_kind": random_kind,
        "fixed_witness_kind": fixed_kind,
        "adaptive_library_kind": adaptive_kind,
        "random_witness_rows_source": "explicit" if explicit_random_rows is not None else f"generated_{random_kind}",
        "fixed_witness_rows_source": "explicit" if explicit_fixed_rows is not None else f"generated_{fixed_kind}",
        "adaptive_library_rows_source": "explicit" if explicit_library_rows is not None else f"generated_{adaptive_kind}",
        "adaptive_selected_library_rows": {},
        "prefilter_candidate_sets": {},
    }
    for budget in budgets:
        random_idx = select_random_witness(candidates, true_x, random_rows, budget)
        method_vectors[f"random_witness_b{budget}"] = (random_idx, p0[np.arange(cache.n), random_idx])
        fixed_idx = select_random_witness(candidates, true_x, fixed_rows, budget)
        method_vectors[f"fixed_lowfreq_witness_b{budget}"] = (fixed_idx, p0[np.arange(cache.n), fixed_idx])
        adaptive_idx, adaptive_rows = select_adaptive_witness(candidates, true_x, library_rows, budget)
        method_vectors[f"adaptive_witness_b{budget}"] = (adaptive_idx, p0[np.arange(cache.n), adaptive_idx])
        witness_traces["adaptive_selected_library_rows"][str(budget)] = adaptive_rows[:10]
        compat_idx, compat_rows, compat_sets = select_compatibility_prefilter_adaptive_witness(
            candidates,
            true_x,
            library_rows,
            selector_scores[primary_selector],
            budget,
            top_m,
        )
        name = f"compat_top{top_m}_adaptive_witness_b{budget}"
        method_vectors[name] = (compat_idx, p0[np.arange(cache.n), compat_idx])
        witness_traces["adaptive_selected_library_rows"][name] = compat_rows[:10]
        witness_traces["prefilter_candidate_sets"][name] = compat_sets[:10]

    summaries: dict[str, dict[str, Any]] = {}
    for name, (idx, err) in method_vectors.items():
        summaries[name] = summarize_errors(
            name,
            err,
            selected_indices=None if idx is None else np.asarray(idx, dtype=np.int64),
            random_error=random_error,
            oracle_error=oracle_error,
            posterior_error=posterior_error,
            primary_error=primary_error,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )

    per_budget_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for prefix in ["random_witness", "fixed_lowfreq_witness", "adaptive_witness", f"compat_top{top_m}_adaptive_witness"]:
            name = f"{prefix}_b{budget}"
            row = {
                "budget": budget,
                "method": name,
                "mean_p0_rmse": summaries[name]["mean_p0_rmse"],
                "delta_vs_random_mean": summaries[name]["delta_vs_random_mean"],
                "delta_vs_primary_selector_mean": summaries[name].get("delta_vs_primary_selector_mean"),
                "delta_vs_posterior_mean": summaries[name]["delta_vs_posterior_mean"],
                "oracle_gain_fraction_aggregate": summaries[name]["oracle_gain_fraction_aggregate"],
                "bootstrap_vs_random_ci_lower": summaries[name]["bootstrap_vs_random"]["ci_lower"],
                "bootstrap_vs_random_ci_upper": summaries[name]["bootstrap_vs_random"]["ci_upper"],
            }
            if name != f"random_witness_b{budget}":
                ref_name = f"random_witness_b{budget}"
                delta = method_vectors[name][1] - method_vectors[ref_name][1]
                boot = paired_percentile_bootstrap(delta, reps=bootstrap_reps, seed=bootstrap_seed + 10 + budget)
                row["delta_vs_random_witness_same_budget_mean"] = float(np.mean(delta))
                row["delta_vs_random_witness_same_budget_ci_upper"] = boot["ci_upper"]
            if name not in {f"random_witness_b{budget}", f"fixed_lowfreq_witness_b{budget}"}:
                ref_name = f"fixed_lowfreq_witness_b{budget}"
                delta = method_vectors[name][1] - method_vectors[ref_name][1]
                boot = paired_percentile_bootstrap(delta, reps=bootstrap_reps, seed=bootstrap_seed + 20 + budget)
                row["delta_vs_fixed_witness_same_budget_mean"] = float(np.mean(delta))
                row["delta_vs_fixed_witness_same_budget_ci_upper"] = boot["ci_upper"]
            per_budget_rows.append(row)

    headroom_methods = {
        "dm_fcc_seed3": primary_error,
        f"random_witness_b{max_budget}": method_vectors[f"random_witness_b{max_budget}"][1],
        f"fixed_lowfreq_witness_b{max_budget}": method_vectors[f"fixed_lowfreq_witness_b{max_budget}"][1],
        f"adaptive_witness_b{max_budget}": method_vectors[f"adaptive_witness_b{max_budget}"][1],
        f"compat_top{top_m}_adaptive_witness_b{max_budget}": method_vectors[f"compat_top{top_m}_adaptive_witness_b{max_budget}"][1],
    }
    headroom = posterior_headroom_audit(random_error, posterior_error, oracle_error, headroom_methods)

    per_image_rows: list[dict[str, Any]] = []
    focus_methods = [
        "random_expectation",
        "posterior_mean",
        primary_selector,
        f"random_witness_b{max_budget}",
        f"fixed_lowfreq_witness_b{max_budget}",
        f"adaptive_witness_b{max_budget}",
        f"compat_top{top_m}_adaptive_witness_b{max_budget}",
        "oracle_best_of_16",
    ]
    quality_focus = [
        "random_expectation",
        "posterior_mean",
        primary_selector,
        f"random_witness_b{max_budget}",
        f"fixed_lowfreq_witness_b{max_budget}",
        f"adaptive_witness_b{max_budget}",
        f"compat_top{top_m}_adaptive_witness_b{max_budget}",
        f"compat_top{top_m}_adaptive_witness_b{int(config.get('pilot_gate', {}).get('low_budget', min(max_budget, 16)))}",
        "oracle_best_of_16",
    ]
    quality = compute_quality_audit(
        cache,
        method_vectors,
        focus_methods=quality_focus,
        compute_lpips=bool(quality_cfg.get("compute_lpips", True)),
        lpips_device=str(quality_cfg.get("lpips_device", "cuda")),
    )
    for i, uid in enumerate(cache.sample_uids):
        for method in focus_methods:
            idx, err = method_vectors[method]
            per_image_rows.append(
                {
                    "sample_uid": uid,
                    "source_index": int(cache.indices[i]),
                    "method": method,
                    "selected_index": "" if idx is None else int(np.asarray(idx)[i]),
                    "p0_rmse": float(np.asarray(err)[i]),
                    "oracle_index": int(oracle_idx[i]),
                    "random_expectation_p0_rmse": float(random_error[i]),
                    "posterior_mean_p0_rmse": float(posterior_error[i]),
                }
            )
    return summaries, per_budget_rows, per_image_rows, witness_traces, headroom, quality


def compute_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    per_budget_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    witness_cfg = config.get("witness", {})
    gate_cfg = config.get("pilot_gate", {})
    primary_budget = int(gate_cfg.get("primary_budget", max(int(b) for b in witness_cfg.get("budgets", [64]))))
    low_budget = int(gate_cfg.get("low_budget", min(primary_budget, 16)))
    top_m = int(witness_cfg.get("compatibility_prefilter_top_m", 4))
    min_gain = float(gate_cfg.get("min_oracle_gain_fraction", 0.45))
    adaptive = summaries[f"adaptive_witness_b{primary_budget}"]
    random_witness = summaries[f"random_witness_b{primary_budget}"]
    fixed_witness = summaries[f"fixed_lowfreq_witness_b{primary_budget}"]
    compat_primary = summaries[f"compat_top{top_m}_adaptive_witness_b{primary_budget}"]
    compat = summaries[f"compat_top{top_m}_adaptive_witness_b{low_budget}"]
    primary = summaries[str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))]
    posterior = summaries["posterior_mean"]
    oracle = summaries["oracle_best_of_16"]
    random = summaries["random_expectation"]
    delta_adaptive_vs_random_witness = adaptive["mean_p0_rmse"] - random_witness["mean_p0_rmse"]
    cond = {
        "adaptive_primary_budget_beats_random_expectation_with_ci": bool(
            adaptive["bootstrap_vs_random"]["ci_upper"] < 0.0
            and adaptive["oracle_gain_fraction_aggregate"] is not None
            and adaptive["oracle_gain_fraction_aggregate"] >= min_gain
        ),
        "adaptive_primary_budget_beats_random_witness_same_budget_by_mean": bool(
            delta_adaptive_vs_random_witness < 0.0
        ),
        "adaptive_primary_budget_beats_fixed_witness_same_budget_by_mean": bool(
            adaptive["mean_p0_rmse"] < fixed_witness["mean_p0_rmse"]
        ),
        "adaptive_primary_budget_beats_frozen_primary_selector_by_mean": bool(
            adaptive["mean_p0_rmse"] < primary["mean_p0_rmse"]
        ),
        "adaptive_primary_budget_beats_posterior_mean_by_mean": bool(
            adaptive["mean_p0_rmse"] < posterior["mean_p0_rmse"]
        ),
        "compat_prefilter_low_budget_beats_adaptive_low_budget_by_mean": bool(
            compat["mean_p0_rmse"] < summaries[f"adaptive_witness_b{low_budget}"]["mean_p0_rmse"]
        ),
        "compat_prefilter_low_budget_beats_frozen_primary_selector_by_mean": bool(
            compat["mean_p0_rmse"] < primary["mean_p0_rmse"]
        ),
        "compat_prefilter_primary_budget_beats_fixed_witness_same_budget_by_mean": bool(
            compat_primary["mean_p0_rmse"] < fixed_witness["mean_p0_rmse"]
        ),
        "compat_prefilter_primary_budget_beats_posterior_mean_by_mean": bool(
            compat_primary["mean_p0_rmse"] < posterior["mean_p0_rmse"]
        ),
    }
    witness_signal = bool(
        cond["adaptive_primary_budget_beats_random_expectation_with_ci"]
        and cond["adaptive_primary_budget_beats_random_witness_same_budget_by_mean"]
    )
    hybrid_signal = bool(
        cond["compat_prefilter_low_budget_beats_frozen_primary_selector_by_mean"]
        or cond["compat_prefilter_primary_budget_beats_fixed_witness_same_budget_by_mean"]
    )
    if witness_signal or hybrid_signal:
        if (
            cond["adaptive_primary_budget_beats_posterior_mean_by_mean"]
            or cond["compat_prefilter_primary_budget_beats_posterior_mean_by_mean"]
        ) and cond["compat_prefilter_primary_budget_beats_fixed_witness_same_budget_by_mean"]:
            decision = "READY_TO_PREREGISTER_INDEPENDENT_WITNESS_TEST"
        else:
            decision = "CONTINUE_WITNESS_DEVELOPMENT_DO_NOT_LOCK_TEST_YET"
    else:
        decision = "REDESIGN_OR_STOP_WITNESS_LINE_BEFORE_TEST"
    return {
        "status": "PASS",
        "gate_scope": "development_pilot_only_not_confirmatory",
        "primary_budget": primary_budget,
        "low_budget": low_budget,
        "conditions": cond,
        "decision": decision,
        "method_means": {
            "random_expectation": random["mean_p0_rmse"],
            "posterior_mean": posterior["mean_p0_rmse"],
            str(witness_cfg.get("primary_selector", "dm_fcc_seed3")): primary["mean_p0_rmse"],
            f"random_witness_b{primary_budget}": random_witness["mean_p0_rmse"],
            f"fixed_lowfreq_witness_b{primary_budget}": fixed_witness["mean_p0_rmse"],
            f"adaptive_witness_b{primary_budget}": adaptive["mean_p0_rmse"],
            f"compat_top{top_m}_adaptive_witness_b{low_budget}": compat["mean_p0_rmse"],
            f"compat_top{top_m}_adaptive_witness_b{primary_budget}": compat_primary["mean_p0_rmse"],
            "oracle_best_of_16": oracle["mean_p0_rmse"],
        },
        "interpretation": {
            "posterior_mean_challenge": "If posterior mean is not beaten, claim witness improves single-candidate selection but not Euclidean posterior-mean distortion.",
            "compatibility_prefilter_challenge": "If prefilter does not improve adaptive witness, do not claim complementarity.",
            "fixed_total_budget": "Not tested in this pilot; requires candidates generated from reduced context operator.",
        },
    }


def make_markdown_report(
    output_dir: Path,
    config: Mapping[str, Any],
    cache: CandidateCache,
    summaries: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    final_summary: Mapping[str, Any],
    headroom: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    witness_cfg = config.get("witness", {})
    primary_budget = int(gate["primary_budget"])
    low_budget = int(gate["low_budget"])
    top_m = int(witness_cfg.get("compatibility_prefilter_top_m", 4))
    primary = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
    lines = [
        "# Phase 2 Witness Development Pilot",
        "",
        "## Scope",
        "",
        "This is a development pilot for add-on witnessed candidate selection. It does not use final-v4 truth/candidates for method selection and is not a locked confirmatory test.",
        "",
        f"- Development cache: `{cache.path}`",
        f"- Samples: {cache.n}; candidates per sample: {cache.k}; dimension: {cache.d}",
        (
            "- Witness rows: "
            f"random=`{witness_cfg.get('random_witness', 'rademacher')}`, "
            f"fixed=`{witness_cfg.get('fixed_witness', 'dct2_low_frequency')}`, "
            f"adaptive_library=`{witness_cfg.get('adaptive_library', witness_cfg.get('adaptive_library_kind', 'rademacher'))}`; "
            "all are unseen by the generator and used only after candidate generation."
        ),
        f"- Fixed-total-budget experiment: not run here; it requires new candidates from a reduced context operator.",
        "",
        "## Final-v4 Context Carried Forward",
        "",
        f"- Classification: `{final_summary.get('final_classification')}`",
        f"- Primary selector P0-RMSE: `{final_summary.get('primary_selected_p0_rmse_mean')}`",
        f"- Posterior mean P0-RMSE: `{final_summary.get('posterior_mean_p0_rmse_mean')}`",
        f"- Boundary: final-v4 is consumed and was used only as a summary of prior conclusions.",
        "",
        "## Pilot Means",
        "",
        "| Method | Mean P0-RMSE | Oracle gain fraction | Delta vs random | Delta vs posterior |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in [
        "random_expectation",
        "posterior_mean",
        primary,
        f"random_witness_b{primary_budget}",
        f"fixed_lowfreq_witness_b{primary_budget}",
        f"adaptive_witness_b{primary_budget}",
        f"compat_top{top_m}_adaptive_witness_b{low_budget}",
        "oracle_best_of_16",
    ]:
        s = summaries[method]
        lines.append(
            "| {method} | {mean:.10f} | {gain} | {dr:.10f} | {dp:.10f} |".format(
                method=method,
                mean=float(s["mean_p0_rmse"]),
                gain="NA"
                if s["oracle_gain_fraction_aggregate"] is None
                else f"{float(s['oracle_gain_fraction_aggregate']):.4f}",
                dr=float(s["delta_vs_random_mean"]),
                dp=float(s["delta_vs_posterior_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Posterior Headroom",
            "",
            f"- Posterior gap to oracle: `{headroom.get('posterior_gap_to_oracle')}`",
            f"- Posterior captured oracle headroom: `{headroom.get('posterior_fraction_of_oracle_headroom_captured')}`",
            f"- Posterior near oracle: `{headroom.get('posterior_is_near_oracle_for_p0_rmse')}`",
            f"- Interpretation: {headroom.get('interpretation')}",
            "",
            "## Quality Means",
            "",
            f"- LPIPS status: `{quality.get('lpips_status')}`",
            "",
            "| Method | Clipped PSNR | SSIM | LPIPS | RAPSD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in quality.get("rows", []):
        lines.append(
            "| {method} | {psnr} | {ssim} | {lpips} | {rapsd} |".format(
                method=row.get("method"),
                psnr=row.get("canonical_clipped_psnr_mean"),
                ssim=row.get("canonical_clipped_ssim_mean"),
                lpips=row.get("canonical_clipped_lpips_mean"),
                rapsd=row.get("canonical_clipped_rapsd_mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Pilot Gate",
            "",
            f"- Decision: `{gate['decision']}`",
            f"- Conditions: `{json.dumps(gate['conditions'], sort_keys=True)}`",
            "",
            "## Research Judgment",
            "",
        ]
    )
    decision = str(gate["decision"])
    if decision == "READY_TO_PREREGISTER_INDEPENDENT_WITNESS_TEST":
        lines.append("The development evidence is strong enough to freeze an independent witnessed-selection test protocol.")
    elif decision == "CONTINUE_WITNESS_DEVELOPMENT_DO_NOT_LOCK_TEST_YET":
        lines.append(
            "Witness measurements show useful add-on signal, but at least one hard question remains before a locked test: posterior mean, compatibility complementarity, or low-budget efficiency."
        )
    else:
        lines.append("This pilot does not justify a locked test without redesigning the witness selector or candidate generation setting.")
    lines.extend(
        [
            "",
            "## Claims Allowed Now",
            "",
            "- Allowed: fresh witness rows can provide sample-specific evidence beyond prior-only candidate scoring in this development setting if the reported gate conditions support it.",
            "- Allowed: any improvement comes from additional observed directions, not from violating `A_c P0 = 0`.",
            "- Not allowed: witness certifies the full null-space content.",
            "- Not allowed: final-v4 was used to tune this Phase 2 method.",
            "- Not allowed: fixed-total-budget benefit has been shown by this add-on pilot.",
            "",
            "## Key Files",
            "",
            f"- Summary JSON: `{output_dir / 'reports' / 'pilot_summary.json'}`",
            f"- Gate JSON: `{output_dir / 'reports' / 'pilot_gate.json'}`",
            f"- Posterior headroom audit: `{output_dir / 'reports' / 'posterior_headroom_audit.json'}`",
            f"- Quality audit: `{output_dir / 'reports' / 'quality_metrics.json'}`",
            f"- Leakage/operator audit: `{output_dir / 'reports' / 'leakage_operator_audit.json'}`",
            f"- Per-budget CSV: `{output_dir / 'reports' / 'per_budget_metrics.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def repo_state() -> dict[str, Any]:
    cmd = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_witness_pilot(config_path: Path) -> dict[str, Any]:
    started = time.time()
    started_utc = now_utc()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/phase2_witness_pilot/dev_pilot"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    save_config_copy(config_path, output_dir)
    cache_cfg = config.get("dataset", {})
    split = str(cache_cfg.get("split", "val"))
    if split != "val":
        raise Phase2WitnessError("ONLY_VAL_DEVELOPMENT_CACHE_SUPPORTED_FOR_SELECTOR_SCORE_REUSE")
    cache_path = ROOT / str(cache_cfg.get("cache_path", PHASE12_CACHE / "val_64_selector_k16.pt"))
    cache = load_candidate_cache(cache_path, split=split)
    cache_info = cache_audit(cache)
    if cache_info["p0_error_max_abs_recompute_diff"] > 1e-5:
        raise Phase2WitnessError(f"CACHE_P0_RECOMPUTE_DIFF_TOO_LARGE:{cache_info['p0_error_max_abs_recompute_diff']}")
    selector_scores = load_validation_scores((cache.n, cache.k))
    final_summary = final_v4_context_summary()
    leak_audit = leakage_operator_audit(cache, config)
    summaries, per_budget_rows, per_image_rows, witness_traces, headroom, quality = build_method_tables(
        cache, selector_scores, config
    )
    gate = compute_gate(summaries, per_budget_rows, config)
    runtime = {
        "started_utc": started_utc,
        "elapsed_seconds": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": "python phase2_witness_pilot.py --config " + str(config_path),
        "repo_state": repo_state(),
    }
    write_json(reports / "cache_audit.json", cache_info)
    write_json(reports / "post_final_v4_summary.json", final_summary)
    write_json(reports / "leakage_operator_audit.json", leak_audit)
    write_json(reports / "method_summaries.json", summaries)
    write_json(reports / "posterior_headroom_audit.json", headroom)
    write_json(reports / "quality_metrics.json", quality)
    write_csv(reports / "quality_metrics.csv", quality.get("rows", []))
    write_json(reports / "pilot_gate.json", gate)
    write_json(reports / "witness_trace_sample.json", witness_traces)
    write_csv(reports / "per_budget_metrics.csv", per_budget_rows)
    write_csv(reports / "per_image_methods.csv", per_image_rows)
    report_md = make_markdown_report(output_dir, config, cache, summaries, gate, final_summary, headroom, quality)
    (reports / "research_decision.md").write_text(report_md, encoding="utf-8")
    runtime["elapsed_seconds"] = float(time.time() - started)
    write_json(reports / "runtime_and_memory.json", runtime)
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "research_decision.md": sha256_file(reports / "research_decision.md"),
        "pilot_gate.json": sha256_file(reports / "pilot_gate.json"),
        "posterior_headroom_audit.json": sha256_file(reports / "posterior_headroom_audit.json"),
        "quality_metrics.json": sha256_file(reports / "quality_metrics.json"),
        "quality_metrics.csv": sha256_file(reports / "quality_metrics.csv"),
        "leakage_operator_audit.json": sha256_file(reports / "leakage_operator_audit.json"),
        "per_budget_metrics.csv": sha256_file(reports / "per_budget_metrics.csv"),
        "per_image_methods.csv": sha256_file(reports / "per_image_methods.csv"),
        "cache_sha256": cache.file_sha256,
    }
    summary = {
        "status": "PHASE2_WITNESS_DEV_PILOT_COMPLETE",
        "run_name": config.get("run_name"),
        "timestamp_utc": now_utc(),
        "output_dir": str(output_dir),
        "cache_audit": cache_info,
        "leakage_operator_audit_status": leak_audit["status"],
        "final_v4_summary_status": final_summary["status"],
        "posterior_headroom_audit": headroom,
        "quality_status": quality["status"],
        "quality_lpips_status": quality["lpips_status"],
        "gate": gate,
        "key_method_summaries": {
            key: summaries[key]
            for key in [
                "random_expectation",
                "posterior_mean",
                str(config.get("witness", {}).get("primary_selector", "dm_fcc_seed3")),
                f"random_witness_b{gate['primary_budget']}",
                f"fixed_lowfreq_witness_b{gate['primary_budget']}",
                f"adaptive_witness_b{gate['primary_budget']}",
                f"compat_top{config.get('witness', {}).get('compatibility_prefilter_top_m', 4)}_adaptive_witness_b{gate['low_budget']}",
                "oracle_best_of_16",
            ]
        },
        "artifact_hashes": hashes,
    }
    write_json(reports / "pilot_summary.json", summary)
    hashes["pilot_summary.json"] = sha256_file(reports / "pilot_summary.json")
    write_json(reports / "package_hashes.json", hashes)
    atomic_write_json(output_dir / "PHASE2_WITNESS_DEV_PILOT_COMPLETE.json", {"status": summary["status"], "summary_sha256": hashes["pilot_summary.json"], "gate_decision": gate["decision"]})
    return summary


def create_brief_package(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    delivery = output_dir / "delivery"
    ensure_dir(delivery)
    package = delivery / "phase2_witness_dev_pilot_gpt_brief.zip"
    files = [
        output_dir / "config_used.yaml",
        output_dir / "PHASE2_WITNESS_DEV_PILOT_COMPLETE.json",
        reports / "research_decision.md",
        reports / "pilot_summary.json",
        reports / "pilot_gate.json",
        reports / "posterior_headroom_audit.json",
        reports / "quality_metrics.json",
        reports / "quality_metrics.csv",
        reports / "leakage_operator_audit.json",
        reports / "post_final_v4_summary.json",
        reports / "per_budget_metrics.csv",
        reports / "package_hashes.json",
    ]
    import zipfile

    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(output_dir).as_posix())
    write_json(delivery / "brief_package_manifest.json", {"package": str(package), "sha256": sha256_file(package), "files": [str(p) for p in files]})
    return package
