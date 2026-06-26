from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import phase1_2_rad5_64_pipeline as p12
from bayesian_witness_assimilation import best_grid_estimate
from src import phase1_4v4b0_scoring as b0
from src import phase73_overnight_gauge_gan_expansion as p73
from src.bayesian_witness import (
    barycenter_null,
    conditional_nullspace_audit,
    p0_rmse,
    posterior_entropy,
    standardize_scores,
)
from src.dc_balanced import (
    bias_variance_risk,
    build_dc_balanced_rows,
    centered_rmse,
    dc_row,
    dct_band_rmse,
    dct_lowfreq_non_dc_rows,
    full_rmse,
    hadamard_lowsequency_non_dc_rows,
    mean_abs_error,
    random_zero_mean_rows,
    row_audit,
    wp0_diagnostics,
)
from src.eval import make_measurement
from src.phase2_fresh_operator import build_fresh_split, resolve_device, score_frozen_selectors
from src.phase2_witness import (
    CandidateCache,
    atomic_write_json,
    cache_audit,
    load_candidate_cache,
    paired_percentile_bootstrap,
    repo_state,
    sha256_file,
    write_csv,
    write_json,
)
from src.projections import exact_data_anchor, get_exact_projector, relative_measurement_error
from src.utils import apply_experiment_defaults


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "dc_balanced_fixed_total_smoke.yaml"


class DCBalancedRunnerError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise DCBalancedRunnerError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().numpy())
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


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def sha256_numpy(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def copy_config(config_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def make_measurement_from_rows(
    rows: np.ndarray,
    *,
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    role: str,
    arm_id: str,
    device: torch.device,
):
    op_cfg = dict(config["operator"])
    n = int(op_cfg.get("img_size", 64)) ** 2
    arr = np.asarray(rows, dtype=np.float32)
    base = p73.regime_config("rad5", device)
    base.update(
        {
            "seed": int(run_config.get("operator_seed", op_cfg.get("seed", 0))),
            "sampling_ratio": float(arr.shape[0] / n),
            "pattern_type": "rademacher",
            "matrix_normalization": str(op_cfg.get("matrix_normalization", "orthonormal_rows")),
            "noise_std": float(op_cfg.get("noise_std", 0.0)),
            "use_final_dc_project": True,
            "output_range_mode": "clamp_eval_only",
            "num_workers": 0,
            "batch_size": int(config.get("batch_size", 8)),
        }
    )
    base = apply_experiment_defaults(base)
    measurement = make_measurement(base, device)
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    override = measurement.set_A_override(
        tensor,
        metadata={
            "phase": "dc_balanced_fixed_total",
            "role": role,
            "arm_id": arm_id,
            "row_sha256_float32": sha256_numpy(arr),
            "rows_shape": list(arr.shape),
            "dc_balanced_fixed_total": True,
        },
        rebuild_cache=True,
    )
    base["dc_balanced_override"] = {
        "role": role,
        "arm_id": arm_id,
        "rows_shape": list(arr.shape),
        "rows_sha256": sha256_numpy(arr),
        "override": override,
    }
    return measurement, base


def non_dc_rows(kind: str, count: int, *, dim: int, img_size: int, seed: int) -> np.ndarray:
    kind_norm = str(kind).lower()
    if int(count) <= 0:
        return np.zeros((0, int(dim)), dtype=np.float32)
    if kind_norm in {"random", "rademacher", "balanced_random"}:
        return random_zero_mean_rows(int(count), int(dim), int(seed))
    if kind_norm in {"dct", "dct2", "lowfreq_dct", "non_dc_lowfreq_dct"}:
        return dct_lowfreq_non_dc_rows(int(count), int(img_size))
    if kind_norm in {"hadamard", "lowsequency_hadamard", "non_dc_lowsequency_hadamard"}:
        return hadamard_lowsequency_non_dc_rows(int(count), int(dim))
    raise DCBalancedRunnerError(f"UNKNOWN_NON_DC_KIND:{kind}")


def build_arms(config: Mapping[str, Any], run_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    op_cfg = dict(config["operator"])
    img_size = int(op_cfg.get("img_size", 64))
    dim = img_size * img_size
    total_m = int(op_cfg.get("total_m", 41))
    non_dc_m = total_m - 1
    seed = int(run_config.get("operator_seed", op_cfg.get("seed", 0)))
    if total_m != 41:
        raise DCBalancedRunnerError(f"THIS_PROTOCOL_EXPECTS_TOTAL_M_41:{total_m}")

    arms: list[dict[str, Any]] = []
    for kind, arm_id in [
        ("random", "dc_plus_40_random"),
        ("dct", "dc_plus_40_non_dc_dct"),
        ("hadamard", "dc_plus_40_non_dc_hadamard"),
    ]:
        rows = build_dc_balanced_rows(kind, non_dc_m, dim=dim, img_size=img_size, seed=seed)
        arms.append(
            {
                "arm_id": arm_id,
                "family": kind,
                "budget": 0,
                "context_rows": rows,
                "witness_rows": np.zeros((0, dim), dtype=np.float32),
                "total_rows": rows,
                "description": f"DC + {non_dc_m} {kind} rows, all in context",
            }
        )

    witness_kind = str(config.get("mixed", {}).get("witness_kind", "dct"))
    budgets = [int(v) for v in config.get("mixed", {}).get("witness_budgets", [4, 8, 16, 32])]
    for b in budgets:
        if b <= 0 or b > non_dc_m:
            raise DCBalancedRunnerError(f"INVALID_MIXED_BUDGET:{b}")
        random_context = random_zero_mean_rows(non_dc_m - b, dim, seed + 1009 * b)
        witness = non_dc_rows(witness_kind, b, dim=dim, img_size=img_size, seed=seed + 2003 * b)
        context = np.concatenate([dc_row(dim)[None, :], random_context], axis=0).astype(np.float32)
        total = np.concatenate([context, witness], axis=0).astype(np.float32)
        arms.append(
            {
                "arm_id": f"mixed_random_context_{witness_kind}_witness_b{b:02d}",
                "family": f"mixed_{witness_kind}",
                "budget": int(b),
                "context_rows": context,
                "witness_rows": witness,
                "total_rows": total,
                "description": f"DC + {non_dc_m-b} random context + {b} non-DC {witness_kind} witness rows",
            }
        )
    return arms


def ridge_data_anchor(rows: np.ndarray, y: np.ndarray, *, lambda_: float, device: torch.device) -> np.ndarray:
    arr = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device=device, dtype=torch.float64)
    yy = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device=device, dtype=torch.float64)
    eye = torch.eye(arr.shape[0], device=device, dtype=torch.float64)
    gram = arr @ arr.T + float(lambda_) * eye
    try:
        coeff = torch.cholesky_solve(yy.T.contiguous(), torch.linalg.cholesky(gram)).T.contiguous()
    except RuntimeError:
        coeff = torch.linalg.solve(gram, yy.T.contiguous()).T.contiguous()
    return (coeff @ arr).detach().cpu().numpy().astype(np.float32)


def tv_baseline(
    rows: np.ndarray,
    y: np.ndarray,
    *,
    img_size: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    tv_cfg = dict(config.get("classic_tv", {}))
    if not bool(tv_cfg.get("enabled", False)):
        return np.zeros((0, rows.shape[1]), dtype=np.float32), {"status": "SKIPPED_BY_CONFIG"}
    iters = int(tv_cfg.get("iters", 30))
    lr = float(tv_cfg.get("lr", 0.05))
    tv_weight = float(tv_cfg.get("tv_weight", 1e-3))
    meas_weight = float(tv_cfg.get("measurement_weight", 1.0))
    batch_size = int(tv_cfg.get("batch_size", 8))
    anchor = ridge_data_anchor(rows, y, lambda_=float(tv_cfg.get("anchor_lambda", 1e-6)), device=device)
    A = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device=device, dtype=torch.float32)
    outputs: list[np.ndarray] = []
    losses: list[float] = []
    for start in range(0, y.shape[0], batch_size):
        yy = torch.from_numpy(np.asarray(y[start : start + batch_size], dtype=np.float32)).to(device)
        x0 = torch.from_numpy(anchor[start : start + batch_size]).to(device).reshape(-1, 1, img_size, img_size)
        z = x0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            flat = z.reshape(z.shape[0], -1)
            residual = flat @ A.T - yy
            dx = z[:, :, :, 1:] - z[:, :, :, :-1]
            dy = z[:, :, 1:, :] - z[:, :, :-1, :]
            loss = meas_weight * torch.mean(residual * residual) + tv_weight * (dx.abs().mean() + dy.abs().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach().cpu().item()))
        outputs.append(z.detach().cpu().reshape(z.shape[0], -1).numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0), {
        "status": "PASS",
        "iters": iters,
        "lr": lr,
        "tv_weight": tv_weight,
        "measurement_weight": meas_weight,
        "final_loss_mean": float(np.mean(losses)) if losses else None,
        "note": "Lightweight TV-regularized classical baseline; not used for method tuning.",
    }


def clean_measurements(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ np.asarray(rows, dtype=np.float32).T


def compute_residual(xhat: np.ndarray, rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros(xhat.shape[0], dtype=np.float64)
    pred = np.asarray(xhat, dtype=np.float64) @ np.asarray(rows, dtype=np.float64).T
    yy = np.asarray(y, dtype=np.float64)
    return np.linalg.norm(pred - yy, axis=1) / np.maximum(np.linalg.norm(yy, axis=1), 1e-12)


def psnr_from_full_rmse(vals: np.ndarray) -> np.ndarray:
    rmse = np.asarray(vals, dtype=np.float64)
    return -20.0 * np.log10(np.maximum(rmse, 1e-12))


def ssim_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int) -> np.ndarray:
    from skimage.metrics import structural_similarity

    pred = np.clip(np.asarray(xhat, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(np.asarray(x, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    return np.asarray(
        [structural_similarity(truth[i], pred[i], data_range=1.0, win_size=7, channel_axis=None) for i in range(pred.shape[0])],
        dtype=np.float64,
    )


def rapsd_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int) -> np.ndarray:
    pred = np.clip(np.asarray(xhat, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(np.asarray(x, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    return np.asarray([b0.rapsd_distance(pred[i], truth[i], bins=32) for i in range(pred.shape[0])], dtype=np.float64)


def lpips_values(xhat: np.ndarray, x: np.ndarray, *, img_size: int, device_name: str) -> np.ndarray | None:
    pred = np.clip(np.asarray(xhat, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    truth = np.clip(np.asarray(x, dtype=np.float32).reshape(-1, img_size, img_size), 0.0, 1.0)
    try:
        return b0.compute_lpips_matrix(pred[:, None, :, :], truth, device_name=device_name).reshape(pred.shape[0])
    except Exception:
        return None


def metric_rows_for_estimates(
    *,
    run_id: str,
    arm: Mapping[str, Any],
    cache: CandidateCache,
    estimates: Sequence[Mapping[str, Any]],
    context_rows: np.ndarray,
    witness_rows: np.ndarray,
    total_rows: np.ndarray,
    img_size: int,
    quality_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    x = np.asarray(cache.x, dtype=np.float32)
    y_context = clean_measurements(context_rows, x)
    y_witness = clean_measurements(witness_rows, x)
    y_total = clean_measurements(total_rows, x)
    method_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    lpips_cache: dict[str, np.ndarray | None] = {}
    for est in estimates:
        xhat = np.asarray(est["xhat"], dtype=np.float32)
        f_rmse = full_rmse(xhat, x)
        c_rmse = centered_rmse(xhat, x)
        mean_err = mean_abs_error(xhat, x)
        bands = dct_band_rmse(xhat, x, img_size=img_size, low_count=40)
        context_res = compute_residual(xhat, context_rows, y_context)
        witness_res = compute_residual(xhat, witness_rows, y_witness)
        combined_res = compute_residual(xhat, total_rows, y_total)
        true_n = np.asarray(cache.true_n, dtype=np.float64)
        context_null_est = xhat.astype(np.float64) - np.asarray(cache.r, dtype=np.float64)
        context_p0 = p0_rmse(context_null_est, true_n)
        psnr = psnr_from_full_rmse(f_rmse)
        ssim = ssim_values(xhat, x, img_size=img_size) if bool(quality_cfg.get("compute_ssim", True)) else None
        rapsd = rapsd_values(xhat, x, img_size=img_size) if bool(quality_cfg.get("compute_rapsd", True)) else None
        lpips = None
        if bool(quality_cfg.get("compute_lpips", False)):
            lpips = lpips_values(xhat, x, img_size=img_size, device_name=str(quality_cfg.get("lpips_device", "cuda")))
            lpips_cache[str(est["method"])] = lpips
        method_rows.append(
            {
                "run_id": run_id,
                "arm_id": arm["arm_id"],
                "budget": int(arm["budget"]),
                "context_m": int(context_rows.shape[0]),
                "witness_m": int(witness_rows.shape[0]),
                "total_m": int(total_rows.shape[0]),
                "method": est["method"],
                "family": arm["family"],
                "estimator": est.get("estimator", ""),
                "alpha": est.get("alpha", ""),
                "tau": est.get("tau", ""),
                "centered_rmse_mean": float(c_rmse.mean()),
                "full_rmse_mean": float(f_rmse.mean()),
                "context_p0_rmse_mean": float(context_p0.mean()),
                "global_mean_abs_error_mean": float(mean_err.mean()),
                "dct_non_dc_low_rmse_mean": float(bands["dct_non_dc_low_rmse"].mean()),
                "dct_mid_rmse_mean": float(bands["dct_mid_rmse"].mean()),
                "dct_high_rmse_mean": float(bands["dct_high_rmse"].mean()),
                "psnr_mean": float(psnr.mean()),
                "ssim_mean": "[DATA MISSING]" if ssim is None else float(ssim.mean()),
                "rapsd_mean": "[DATA MISSING]" if rapsd is None else float(rapsd.mean()),
                "lpips_mean": "[DATA MISSING]" if lpips is None else float(np.mean(lpips)),
                "context_residual_max": float(context_res.max()),
                "witness_residual_max": float(witness_res.max()),
                "combined_residual_max": float(combined_res.max()),
            }
        )
        for i, uid in enumerate(cache.sample_uids):
            per_image.append(
                {
                    "run_id": run_id,
                    "arm_id": arm["arm_id"],
                    "budget": int(arm["budget"]),
                    "context_m": int(context_rows.shape[0]),
                    "witness_m": int(witness_rows.shape[0]),
                    "source_index": int(cache.indices[i]),
                    "sample_uid": uid,
                    "pair_uid": f"{run_id}:{int(cache.indices[i])}",
                    "method": est["method"],
                    "estimator": est.get("estimator", ""),
                    "selected_index": "" if est.get("selected_indices") is None else int(est["selected_indices"][i]),
                    "centered_rmse": float(c_rmse[i]),
                    "full_rmse": float(f_rmse[i]),
                    "context_p0_rmse": float(context_p0[i]),
                    "global_mean_abs_error": float(mean_err[i]),
                    "dct_non_dc_low_rmse": float(bands["dct_non_dc_low_rmse"][i]),
                    "dct_mid_rmse": float(bands["dct_mid_rmse"][i]),
                    "dct_high_rmse": float(bands["dct_high_rmse"][i]),
                    "psnr": float(psnr[i]),
                    "ssim": "[DATA MISSING]" if ssim is None else float(ssim[i]),
                    "rapsd": "[DATA MISSING]" if rapsd is None else float(rapsd[i]),
                    "lpips": "[DATA MISSING]" if lpips is None else float(lpips[i]),
                    "context_residual": float(context_res[i]),
                    "witness_residual": float(witness_res[i]),
                    "combined_residual": float(combined_res[i]),
                }
            )
    return method_rows, per_image, {"lpips_computed_methods": sorted(lpips_cache)}


def add_estimate(
    estimates: list[dict[str, Any]],
    *,
    method: str,
    xhat: np.ndarray,
    estimator: str,
    selected_indices: np.ndarray | None = None,
    alpha: float | None = None,
    tau: float | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    estimates.append(
        {
            "method": method,
            "xhat": np.asarray(xhat, dtype=np.float32),
            "estimator": estimator,
            "selected_indices": selected_indices,
            "alpha": "" if alpha is None else float(alpha),
            "tau": "" if tau is None else float(tau),
            "diagnostics": {} if diagnostics is None else dict(diagnostics),
        }
    )


def build_estimates(
    *,
    cache: CandidateCache,
    context_projector,
    context_rows: np.ndarray,
    witness_rows: np.ndarray,
    total_rows: np.ndarray,
    selector_scores: Mapping[str, np.ndarray] | None,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    witness_cfg = dict(config.get("witness", {}))
    x = np.asarray(cache.x, dtype=np.float32)
    y_total = clean_measurements(total_rows, x)
    y_context = clean_measurements(context_rows, x)
    estimates: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    lambda_audit = float(witness_cfg.get("conditional_audit_lambda", 1e-5))
    ridge_lambda = float(config.get("classical", {}).get("tikhonov_lambda", lambda_audit))
    img_size = int(config["operator"].get("img_size", 64))

    minnorm = ridge_data_anchor(total_rows, y_total, lambda_=0.0, device=device)
    add_estimate(estimates, method="joint_minimum_norm", xhat=minnorm, estimator="linear_joint_minnorm")
    tikh = ridge_data_anchor(total_rows, y_total, lambda_=ridge_lambda, device=device)
    add_estimate(
        estimates,
        method="joint_tikhonov",
        xhat=tikh,
        estimator="linear_joint_tikhonov",
        diagnostics={"lambda": ridge_lambda},
    )
    tv_x, tv_diag = tv_baseline(total_rows, y_total, img_size=img_size, config=config, device=device)
    diagnostics["tv_baseline"] = tv_diag
    if tv_x.shape[0] == x.shape[0]:
        add_estimate(estimates, method="joint_tv_pgd", xhat=tv_x, estimator="tv_regularized_joint_pgd", diagnostics=tv_diag)

    zero_null = np.zeros_like(cache.true_n, dtype=np.float32)
    rows_by_image = [np.asarray(witness_rows, dtype=np.float32) for _ in range(cache.n)]
    anchor_audited, anchor_diag = conditional_nullspace_audit(
        zero_null,
        cache.true_n,
        rows_by_image,
        context_projector,
        lambda_=lambda_audit,
    )
    add_estimate(
        estimates,
        method="context_anchor_condaudit",
        xhat=(cache.r + anchor_audited).astype(np.float32),
        estimator="Ac_dagger_yc_then_conditional_audit",
        diagnostics=anchor_diag,
    )

    posterior_null = cache.cand_n.mean(axis=1)
    post_audited, post_diag = conditional_nullspace_audit(
        posterior_null,
        cache.true_n,
        rows_by_image,
        context_projector,
        lambda_=lambda_audit,
    )
    add_estimate(
        estimates,
        method="posterior_mean_condaudit",
        xhat=(cache.r + post_audited).astype(np.float32),
        estimator="uniform_candidate_barycenter_then_conditional_audit",
        alpha=0.0,
        diagnostics=post_diag,
    )

    if witness_rows.shape[0] > 0:
        tau_grid = [float(v) for v in witness_cfg.get("tau_grid", [0.1, 0.2, 0.5, 1.0])]
        alpha0_null, _idx0, weights0, alpha0, tau0, row0 = best_grid_estimate(
            cache=cache,
            rows_by_image=rows_by_image,
            prior_scores_z=None,
            alpha_grid=[0.0],
            tau_grid=tau_grid,
            estimator="barycenter",
            design="dc_balanced_mixed",
            budget=int(witness_rows.shape[0]),
            grid_rows=grid_rows,
            alpha_filter="zero",
        )
        alpha0_audited, alpha0_diag = conditional_nullspace_audit(
            alpha0_null,
            cache.true_n,
            rows_by_image,
            context_projector,
            lambda_=lambda_audit,
        )
        add_estimate(
            estimates,
            method="alpha0_likelihood_barycenter_condaudit",
            xhat=(cache.r + alpha0_audited).astype(np.float32),
            estimator="likelihood_weighted_barycenter_then_conditional_audit",
            alpha=alpha0,
            tau=tau0,
            diagnostics={"selected_grid": row0, "entropy_mean": float(posterior_entropy(weights0).mean()), "audit": alpha0_diag},
        )
        if selector_scores is not None:
            primary_selector = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
            if primary_selector not in selector_scores:
                raise DCBalancedRunnerError(f"PRIMARY_SELECTOR_MISSING:{primary_selector}")
            prior_scores_z = standardize_scores(np.asarray(selector_scores[primary_selector], dtype=np.float64))
            alpha_grid = [float(v) for v in witness_cfg.get("alpha_grid", [0.25, 0.5, 1.0])]
            soft_null, _idx1, weights1, alpha1, tau1, row1 = best_grid_estimate(
                cache=cache,
                rows_by_image=rows_by_image,
                prior_scores_z=prior_scores_z,
                alpha_grid=alpha_grid,
                tau_grid=tau_grid,
                estimator="barycenter",
                design="dc_balanced_mixed",
                budget=int(witness_rows.shape[0]),
                grid_rows=grid_rows,
                alpha_filter="positive",
            )
            soft_audited, soft_diag = conditional_nullspace_audit(
                soft_null,
                cache.true_n,
                rows_by_image,
                context_projector,
                lambda_=lambda_audit,
            )
            add_estimate(
                estimates,
                method="soft_fcc_barycenter_condaudit",
                xhat=(cache.r + soft_audited).astype(np.float32),
                estimator="soft_FCC_prior_likelihood_barycenter_then_conditional_audit",
                alpha=alpha1,
                tau=tau1,
                diagnostics={"selected_grid": row1, "entropy_mean": float(posterior_entropy(weights1).mean()), "audit": soft_diag},
            )
    else:
        if selector_scores is not None and bool(witness_cfg.get("include_prior_only_on_full_context", False)):
            primary_selector = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
            prior_scores_z = standardize_scores(np.asarray(selector_scores[primary_selector], dtype=np.float64))
            weights = np.exp(float(witness_cfg.get("prior_only_alpha", 0.5)) * prior_scores_z)
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
            soft_null = barycenter_null(cache.cand_n, weights)
            add_estimate(
                estimates,
                method="soft_fcc_prior_only_barycenter",
                xhat=(cache.r + soft_null).astype(np.float32),
                estimator="soft_FCC_prior_only_barycenter",
                alpha=float(witness_cfg.get("prior_only_alpha", 0.5)),
            )

    diagnostics["conditional_audit_lambda"] = lambda_audit
    diagnostics["ridge_lambda"] = ridge_lambda
    diagnostics["context_anchor_joint_minnorm_identity"] = identity_check_anchor_vs_joint(
        cache=cache,
        context_projector=context_projector,
        witness_rows=witness_rows,
        total_rows=total_rows,
        device=device,
    )
    diagnostics["bias_variance_risk"] = risk_from_wp0(witness_rows, context_projector, lambda_=lambda_audit)
    return estimates, grid_rows, diagnostics


def risk_from_wp0(witness_rows: np.ndarray, projector, *, lambda_: float) -> dict[str, Any]:
    if witness_rows.size == 0:
        return {"status": "NO_WITNESS_ROWS"}
    with torch.no_grad():
        w = torch.from_numpy(np.asarray(witness_rows, dtype=np.float64)).to(device=projector.device, dtype=projector.dtype)
        z = projector.null_project_flat(w).detach().cpu().numpy().astype(np.float64)
    s = np.asarray(witness_rows, dtype=np.float64) @ z.T
    eig = np.linalg.eigvalsh(0.5 * (s + s.T))
    return bias_variance_risk(eig, lambda_=lambda_, noise_variance=0.0) | {
        "eig_min": float(eig.min()) if eig.size else None,
        "eig_max": float(eig.max()) if eig.size else None,
    }


def identity_check_anchor_vs_joint(
    *,
    cache: CandidateCache,
    context_projector,
    witness_rows: np.ndarray,
    total_rows: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if witness_rows.size == 0:
        return {"status": "NO_WITNESS_ROWS", "max_abs_diff": 0.0}
    rows_by_image = [np.asarray(witness_rows, dtype=np.float32) for _ in range(cache.n)]
    zero_null = np.zeros_like(cache.true_n, dtype=np.float32)
    audited, diag = conditional_nullspace_audit(zero_null, cache.true_n, rows_by_image, context_projector, lambda_=0.0)
    x_anchor_plus = np.asarray(cache.r + audited, dtype=np.float32)
    y_total = clean_measurements(total_rows, cache.x)
    joint = ridge_data_anchor(total_rows, y_total, lambda_=0.0, device=device)
    diff = np.max(np.abs(x_anchor_plus.astype(np.float64) - joint.astype(np.float64)))
    return {
        "status": "PASS" if diff < 1e-5 else "WARN",
        "max_abs_diff": float(diff),
        "conditional_audit_diag": diag,
        "identity": "Ac^dagger yc plus exact lambda=0 conditional audit equals [Ac;W]^dagger y for noiseless full-row-rank rows.",
    }


def run_arm(
    *,
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    arm: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_id = str(run_config["run_id"])
    arm_id = str(arm["arm_id"])
    arm_dir = output_dir / "runs" / run_id / arm_id
    reports = arm_dir / "reports"
    ensure_dir(reports)
    context_rows = np.asarray(arm["context_rows"], dtype=np.float32)
    witness_rows = np.asarray(arm["witness_rows"], dtype=np.float32)
    total_rows = np.asarray(arm["total_rows"], dtype=np.float32)
    img_size = int(config["operator"].get("img_size", 64))

    context_measurement, base_config = make_measurement_from_rows(
        context_rows,
        config=config,
        run_config=run_config,
        role="context_for_candidate_generation",
        arm_id=arm_id,
        device=device,
    )
    generator, gen_config, _ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
        Path(config.get("checkpoint", p12.PHASE79_CKPT)), base_config, context_measurement, device
    )
    if missing or unexpected:
        raise DCBalancedRunnerError(f"GENERATOR_LOAD_NOT_STRICT:{run_id}:{arm_id}:{missing}:{unexpected}")

    run_cfg = copy.deepcopy(dict(config))
    run_cfg["split"] = copy.deepcopy(dict(config["split"]))
    run_cfg["split"].update(copy.deepcopy(dict(run_config.get("split", {}))))
    run_cfg["split"]["name"] = f"{run_cfg['split'].get('name', 'dc_balanced')}_{arm_id}"
    run_cfg["candidate_seed"] = int(run_config.get("candidate_seed", config.get("candidate_seed", 100700)))
    split = build_fresh_split(run_cfg, context_measurement, device)
    cache_path = arm_dir / "candidate_cache" / f"{split['name']}_k{int(config.get('candidate_k', 16))}.pt"
    if not cache_path.exists() or not bool(config.get("reuse_existing_cache", True)):
        p12.build_candidate_cache(
            generator,
            context_measurement,
            gen_config,
            split,
            out=arm_dir,
            k=int(config.get("candidate_k", 16)),
            seed=int(run_cfg["candidate_seed"]),
            device=device,
        )
    raw_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    selector_scores = None
    selector_audit: dict[str, Any] = {"status": "SKIPPED_BY_CONFIG"}
    if bool(config.get("witness", {}).get("use_fcc_scores", True)):
        selector_scores, selector_audit = score_frozen_selectors(raw_cache, device)
    cache = load_candidate_cache(cache_path, split=f"{run_id}:{arm_id}")
    context_projector = get_exact_projector(context_measurement, dtype=torch.float64, device=device)
    estimates, grid_rows, estimate_diag = build_estimates(
        cache=cache,
        context_projector=context_projector,
        context_rows=context_rows,
        witness_rows=witness_rows,
        total_rows=total_rows,
        selector_scores=selector_scores,
        config=config,
        device=device,
    )
    method_rows, per_image_rows, quality_status = metric_rows_for_estimates(
        run_id=run_id,
        arm=arm,
        cache=cache,
        estimates=estimates,
        context_rows=context_rows,
        witness_rows=witness_rows,
        total_rows=total_rows,
        img_size=img_size,
        quality_cfg=config.get("quality", {}),
    )
    row_reports = {
        "context": row_audit(context_rows, name=f"{arm_id}:context"),
        "witness": None if witness_rows.size == 0 else row_audit(witness_rows, name=f"{arm_id}:witness_non_dc_only"),
        "total": row_audit(total_rows, name=f"{arm_id}:total"),
        "wp0": wp0_diagnostics(witness_rows, context_projector, lambda_=float(config.get("witness", {}).get("conditional_audit_lambda", 1e-5))),
    }
    operator_audit = {
        "status": "PASS",
        "run_id": run_id,
        "arm_id": arm_id,
        "description": arm["description"],
        "checkpoint_state_key": state_key,
        "checkpoint_sha256": sha256_file(Path(config.get("checkpoint", p12.PHASE79_CKPT))),
        "context_rows_sha256": sha256_numpy(context_rows),
        "witness_rows_sha256": sha256_numpy(witness_rows),
        "total_rows_sha256": sha256_numpy(total_rows),
        "context_m": int(context_rows.shape[0]),
        "witness_m": int(witness_rows.shape[0]),
        "total_m": int(total_rows.shape[0]),
        "context_projector": context_projector.info_dict(),
        "row_audits": row_reports,
        "split_manifest": split.get("split_manifest", {}),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "candidate_seed": int(run_cfg["candidate_seed"]),
    }
    write_json(reports / "operator_and_row_audit.json", operator_audit)
    write_json(reports / "selector_transfer_audit.json", selector_audit)
    write_json(reports / "cache_audit.json", cache_audit(cache))
    write_json(reports / "estimate_diagnostics.json", estimate_diag)
    write_json(reports / "quality_status.json", quality_status)
    write_csv(reports / "grid_calibration_metrics.csv", grid_rows)
    write_csv(reports / "method_metrics.csv", method_rows)
    write_csv(reports / "per_image_metrics.csv", per_image_rows)
    np.savez_compressed(
        arm_dir / "estimates_focus.npz",
        **{str(est["method"]): np.asarray(est["xhat"], dtype=np.float32) for est in estimates},
    )
    summary = {
        "status": "PASS",
        "run_id": run_id,
        "arm_id": arm_id,
        "method_count": len(estimates),
        "best_centered_rmse_method": min(method_rows, key=lambda r: float(r["centered_rmse_mean"])),
        "operator_audit": operator_audit,
        "artifact_hashes": {
            "method_metrics.csv": sha256_file(reports / "method_metrics.csv"),
            "per_image_metrics.csv": sha256_file(reports / "per_image_metrics.csv"),
            "operator_and_row_audit.json": sha256_file(reports / "operator_and_row_audit.json"),
            "estimates_focus.npz": sha256_file(arm_dir / "estimates_focus.npz"),
        },
    }
    write_json(reports / "arm_summary.json", summary)
    return {
        "summary": summary,
        "method_rows": method_rows,
        "per_image_rows": per_image_rows,
        "grid_rows": grid_rows,
    }


def paired_metric_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    reference_method: str,
    metric: str,
    reps: int,
    seed: int,
    arm: str | None = None,
    reference_arm: str | None = None,
) -> dict[str, Any] | None:
    a: dict[str, float] = {}
    b: dict[str, float] = {}
    for row in rows:
        if str(row["method"]) == method and (arm is None or str(row["arm_id"]) == arm):
            a[str(row["pair_uid"])] = float(row[metric])
        if str(row["method"]) == reference_method and (reference_arm is None or str(row["arm_id"]) == reference_arm):
            b[str(row["pair_uid"])] = float(row[metric])
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    delta = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    boot = paired_percentile_bootstrap(delta, reps=reps, seed=seed)
    return {
        "method": method,
        "reference_method": reference_method,
        "arm": arm,
        "reference_arm": reference_arm,
        "metric": metric,
        "n": int(delta.shape[0]),
        "mean_delta": float(delta.mean()),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "wins": int(np.sum(delta < 0)),
        "losses": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
    }


def aggregate_outputs(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    run_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = output_dir / "reports"
    ensure_dir(reports)
    all_method = [dict(r) for out in run_outputs for r in out["method_rows"]]
    all_per = [dict(r) for out in run_outputs for r in out["per_image_rows"]]
    all_grid = [dict(r) for out in run_outputs for r in out["grid_rows"]]
    write_csv(reports / "all_method_metrics.csv", all_method)
    write_csv(reports / "all_per_image_metrics.csv", all_per)
    write_csv(reports / "all_grid_calibration_metrics.csv", all_grid)

    method_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_per:
        method_groups[(str(row["arm_id"]), str(row["method"]))].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (arm_id, method), rows in sorted(method_groups.items()):
        item = {
            "arm_id": arm_id,
            "method": method,
            "n": len(rows),
            "budget": int(rows[0]["budget"]),
            "context_m": int(rows[0]["context_m"]),
            "witness_m": int(rows[0]["witness_m"]),
        }
        for metric in [
            "centered_rmse",
            "full_rmse",
            "context_p0_rmse",
            "global_mean_abs_error",
            "dct_non_dc_low_rmse",
            "dct_mid_rmse",
            "dct_high_rmse",
            "psnr",
            "context_residual",
            "witness_residual",
            "combined_residual",
        ]:
            vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean())
        for metric in ["ssim", "rapsd", "lpips"]:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[metric]))
                except (TypeError, ValueError):
                    pass
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else "[DATA MISSING]"
        aggregate_rows.append(item)
    aggregate_rows.sort(key=lambda r: (r["centered_rmse_mean"], r["arm_id"], r["method"]))
    write_csv(reports / "aggregate_method_metrics.csv", aggregate_rows)

    reps = int(config.get("statistics", {}).get("bootstrap_replicates", 1000))
    seed = int(config.get("statistics", {}).get("bootstrap_seed", 20260625))
    comparisons: list[dict[str, Any]] = []
    random_ref_arm = "dc_plus_40_random"
    random_ref_method = "posterior_mean_condaudit"
    for row in aggregate_rows:
        arm_id = str(row["arm_id"])
        method = str(row["method"])
        for metric in ["centered_rmse", "full_rmse"]:
            cmp_random = paired_metric_delta(
                all_per,
                method=method,
                reference_method=random_ref_method,
                metric=metric,
                reps=reps,
                seed=seed + len(comparisons),
                arm=arm_id,
                reference_arm=random_ref_arm,
            )
            if cmp_random is not None:
                cmp_random["comparison"] = "vs_DC40_random_posterior_mean"
                comparisons.append(cmp_random)
            if method not in {"joint_minimum_norm", "joint_tikhonov"}:
                cmp_joint = paired_metric_delta(
                    all_per,
                    method=method,
                    reference_method="joint_minimum_norm",
                    metric=metric,
                    reps=reps,
                    seed=seed + 500 + len(comparisons),
                    arm=arm_id,
                    reference_arm=arm_id,
                )
                if cmp_joint is not None:
                    cmp_joint["comparison"] = "vs_same_total_joint_minimum_norm"
                    comparisons.append(cmp_joint)
    write_json(reports / "paired_bootstrap_comparisons.json", comparisons)

    mechanism = mechanism_attribution(aggregate_rows, comparisons)
    write_json(reports / "mechanism_attribution.json", mechanism)
    gate = gate_report(config=config, aggregate_rows=aggregate_rows, comparisons=comparisons, mechanism=mechanism)
    write_json(reports / "gate_report.json", gate)
    write_research_decision(reports / "research_decision.md", gate, mechanism)
    write_claim_ledger(reports / "claim_evidence_ledger.md", gate, mechanism)
    return gate


def _mean_lookup(rows: Sequence[Mapping[str, Any]], arm: str, method: str, metric: str) -> float | None:
    for row in rows:
        if str(row["arm_id"]) == arm and str(row["method"]) == method:
            val = row.get(f"{metric}_mean")
            return None if val is None or isinstance(val, str) else float(val)
    return None


def mechanism_attribution(
    aggregate_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    random_post = _mean_lookup(aggregate_rows, "dc_plus_40_random", "posterior_mean_condaudit", "centered_rmse")
    random_joint = _mean_lookup(aggregate_rows, "dc_plus_40_random", "joint_minimum_norm", "centered_rmse")
    dct_joint = _mean_lookup(aggregate_rows, "dc_plus_40_non_dc_dct", "joint_minimum_norm", "centered_rmse")
    had_joint = _mean_lookup(aggregate_rows, "dc_plus_40_non_dc_hadamard", "joint_minimum_norm", "centered_rmse")
    mixed_rows = [r for r in aggregate_rows if str(r["arm_id"]).startswith("mixed_")]
    best_mixed = min(mixed_rows, key=lambda r: float(r["centered_rmse_mean"])) if mixed_rows else None
    best_soft = min(
        [r for r in mixed_rows if str(r["method"]) == "soft_fcc_barycenter_condaudit"],
        key=lambda r: float(r["centered_rmse_mean"]),
        default=None,
    )
    best_alpha0 = min(
        [r for r in mixed_rows if str(r["method"]) == "alpha0_likelihood_barycenter_condaudit"],
        key=lambda r: float(r["centered_rmse_mean"]),
        default=None,
    )
    return {
        "random_reference_centered_rmse": random_post,
        "random_joint_centered_rmse": random_joint,
        "dct_joint_centered_rmse": dct_joint,
        "hadamard_joint_centered_rmse": had_joint,
        "best_mixed_centered": best_mixed,
        "best_soft_fcc_centered": best_soft,
        "best_alpha0_centered": best_alpha0,
        "dc_control_interpretation": "All arms include the same normalized DC row in context; centered RMSE removes residual global-mean gains.",
        "low_frequency_design_delta_vs_random_joint": None
        if random_joint is None or dct_joint is None
        else float(dct_joint - random_joint),
        "hadamard_design_delta_vs_random_joint": None
        if random_joint is None or had_joint is None
        else float(had_joint - random_joint),
        "fcc_increment_soft_minus_alpha0": None
        if best_soft is None or best_alpha0 is None
        else float(best_soft["centered_rmse_mean"] - best_alpha0["centered_rmse_mean"]),
    }


def gate_report(
    *,
    config: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    min_gain = float(config.get("gate", {}).get("min_centered_rmse_gain", 1e-4))
    context_res_limit = float(config.get("gate", {}).get("context_residual_max", 1e-4))
    candidate_success: list[dict[str, Any]] = []
    for cmp in comparisons:
        if cmp.get("metric") != "centered_rmse" or cmp.get("comparison") != "vs_DC40_random_posterior_mean":
            continue
        method = str(cmp["method"])
        arm = str(cmp["arm"])
        if method not in {"alpha0_likelihood_barycenter_condaudit", "soft_fcc_barycenter_condaudit", "posterior_mean_condaudit", "context_anchor_condaudit"}:
            continue
        same_joint = next(
            (
                c
                for c in comparisons
                if c.get("metric") == "centered_rmse"
                and c.get("comparison") == "vs_same_total_joint_minimum_norm"
                and c.get("method") == method
                and c.get("arm") == arm
            ),
            None,
        )
        row = next((r for r in aggregate_rows if str(r["arm_id"]) == arm and str(r["method"]) == method), None)
        conditions = {
            "beats_DC40_random_posterior_centered_CI": bool(cmp["mean_delta"] < -min_gain and cmp["ci_upper"] < 0.0),
            "beats_same_total_joint_minnorm_centered_CI": bool(
                same_joint is not None and same_joint["mean_delta"] < -min_gain and same_joint["ci_upper"] < 0.0
            ),
            "context_residual_ok": bool(row is not None and float(row["context_residual_mean"]) < context_res_limit),
        }
        if all(conditions.values()):
            candidate_success.append({"arm": arm, "method": method, "conditions": conditions, "vs_random": cmp, "vs_joint": same_joint})
    fcc_increment = mechanism.get("fcc_increment_soft_minus_alpha0")
    if candidate_success:
        decision = "READY_TO_PREREGISTER_LOCKED_TEST_FOR_MIXED_METHOD"
    elif fcc_increment is not None and float(fcc_increment) >= 0:
        decision = "NO_FCC_INCREMENT_DC_BALANCED_DEV_ATTRIBUTION_NEEDED"
    else:
        decision = "NO_PRIOR_ASSISTED_GAIN_YET_KEEP_DEVELOPMENT"
    return {
        "status": "PASS",
        "scope": "fresh development only, not locked",
        "total_m": int(config["operator"].get("total_m", 41)),
        "primary_endpoint": "centered full RMSE",
        "co_primary": "original full RMSE",
        "min_centered_rmse_gain": min_gain,
        "context_residual_limit": context_res_limit,
        "decision": decision,
        "successful_mixed_rows": candidate_success,
        "mechanism_attribution": mechanism,
        "locked_test_authorized": bool(candidate_success),
        "locked_test_note": "If authorized, freeze this code/config/hash first and run one external locked test without retuning.",
    }


def write_math_derivation(path: Path) -> None:
    text = r"""# DC-Balanced Fixed-Total Conditional Null-Space Assimilation

## Geometry

Let \(A_c\in\mathbb R^{m_c\times n}\) be the context operator and
\(P_0=I-A_c^\dagger A_c\).  A context-consistent estimate \(\bar x\) satisfies
\(A_c\bar x=y_c\).  For witness rows \(W\), define
\[
x^+=\bar x+P_0W^\top(WP_0W^\top+\lambda I)^{-1}(y_w-W\bar x).
\]

## Context preservation

\[
A_cx^+
=A_c\bar x+A_cP_0W^\top(WP_0W^\top+\lambda I)^{-1}(y_w-W\bar x).
\]
Since \(P_0=I-A_c^\dagger A_c\),
\[
A_cP_0=A_c-A_cA_c^\dagger A_c=A_c-A_c=0,
\]
where \(A_cA_c^\dagger A_c=A_c\) is the Moore--Penrose identity.  Therefore
\[
A_cx^+=A_c\bar x=y_c.
\]
Plain language: the update lives only in directions the context measurements cannot see.

## Exact noiseless update is an orthogonal projection

Assume noiseless witness \(y_w=Wx\), \(\lambda=0\), and \(\bar x\) is context
consistent.  Then \(e=\bar x-x\in\ker(A_c)\), so \(P_0e=e\).  Let
\[
Q=P_0W^\top(WP_0W^\top)^{-1}WP_0.
\]
For full-row-rank \(WP_0\) on its selected witness subspace, \(Q\) is symmetric
and idempotent:
\[
Q^\top=Q,\qquad
Q^2=P_0W^\top S^{-1}WP_0P_0W^\top S^{-1}WP_0
=P_0W^\top S^{-1}SS^{-1}WP_0=Q,
\]
where \(S=WP_0W^\top\).  Thus \(Q\) is the orthogonal projector onto
\(\operatorname{range}(P_0W^\top)\).  The error after the update is
\[
x^+-x=\bar x-x+P_0W^\top S^{-1}(Wx-W\bar x)
=e-P_0W^\top S^{-1}We=(I-Q)e.
\]
By Pythagoras for an orthogonal projector,
\[
\|x^+-x\|_2^2=\|(I-Q)e\|_2^2=\|e\|_2^2-\|Qe\|_2^2\le \|e\|_2^2.
\]
Plain language: exact witness assimilation can only remove the component of the context error that the witness rows observe.

## Minimum-norm equivalence

If \(\bar x=A_c^\dagger y_c\), then \(\bar x\in\operatorname{range}(A_c^\top)\) and is orthogonal to \(\ker(A_c)\).  The conditional update adds the minimum-norm vector
\(\delta\in\ker(A_c)\) satisfying \(W(\bar x+\delta)=y_w\):
\[
\delta=P_0W^\top(WP_0W^\top)^\dagger(y_w-W\bar x).
\]
The result satisfies both \(A_cx^+=y_c\) and \(Wx^+=y_w\), while remaining the
minimum-norm member because its row-space part is the minimum-norm context
anchor and its null-space part is the minimum-norm witness correction.  Hence
\[
x^+=[A_c;W]^\dagger[y_c;y_w].
\]
Plain language: when the starting point is the context backprojection, conditional audit is just the joint minimum-norm solution written in range-null coordinates.

## Noisy witness bias-variance expression

Let \(y_w=Wx+\epsilon\), \(S=WP_0W^\top\), and
\[
Q_\lambda=P_0W^\top(S+\lambda I)^{-1}WP_0.
\]
With \(e=\bar x-x\in\ker(A_c)\),
\[
e^+=x^+-x=(I-Q_\lambda)e+P_0W^\top(S+\lambda I)^{-1}\epsilon.
\]
The first term is bias left by soft shrinkage; the second is witness noise
amplification.  If \(S=U\operatorname{diag}(s_i)U^\top\) and noise variance is
\(\sigma^2 I\), the variance contribution is
\[
\sigma^2\operatorname{tr}\left[P_0W^\top(S+\lambda I)^{-2}WP_0\right]
=\sigma^2\sum_i \frac{s_i}{(s_i+\lambda)^2}.
\]
Plain language: smaller \(\lambda\) follows the witness harder; larger \(\lambda\) protects against noisy or ill-conditioned witness rows.

## DC-balanced fixed total

All arms contain the same normalized DC row \(1/\sqrt n\) in context and count it
inside \(M=41\).  Every other row is zero mean and unit norm.  The primary
endpoint is centered RMSE, so a method cannot win merely by correcting global
brightness.  Plain language: this experiment asks whether non-DC sensor design,
conditional assimilation, or learned prior remains useful after the mean-value
shortcut is removed.
"""
    write_text(path, text)


def write_research_decision(path: Path, gate: Mapping[str, Any], mechanism: Mapping[str, Any]) -> None:
    lines = [
        "# DC-Balanced Fixed-Total Development Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Primary endpoint: `{gate['primary_endpoint']}`",
        f"- Locked test authorized by dev gate: `{gate['locked_test_authorized']}`",
        "",
        "## Mechanism Snapshot",
        "",
        f"- Random reference centered RMSE: `{mechanism.get('random_reference_centered_rmse')}`",
        f"- DCT joint centered RMSE: `{mechanism.get('dct_joint_centered_rmse')}`",
        f"- Hadamard joint centered RMSE: `{mechanism.get('hadamard_joint_centered_rmse')}`",
        f"- Best mixed row: `{mechanism.get('best_mixed_centered')}`",
        f"- FCC increment soft minus alpha=0: `{mechanism.get('fcc_increment_soft_minus_alpha0')}`",
        "",
        "Interpretation rule: claim FCC only if the soft-FCC barycenter beats the alpha=0 likelihood barycenter after this same-rate DC-balanced control and after the frozen locked test.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_claim_ledger(path: Path, gate: Mapping[str, Any], mechanism: Mapping[str, Any]) -> None:
    rows = [
        ("Context preservation", "reports/math_derivation.md + tests/test_dc_balanced.py", "proved and numerically tested"),
        ("DC confound controlled", "operator_and_row_audit.json for every arm", "same DC row in context; centered RMSE primary"),
        ("Measurement design attribution", "reports/mechanism_attribution.json", "joint min-norm/Tikhonov rows separate design from prior"),
        ("Prior-assisted gain", "reports/paired_bootstrap_comparisons.json", "supported only if posterior/alpha0/soft beats joint baselines"),
        ("FCC increment", "reports/mechanism_attribution.json", "soft minus alpha=0, not claimed if non-negative"),
        ("Locked-test status", "reports/gate_report.json", str(gate.get("locked_test_authorized"))),
    ]
    text = "# Claim-Evidence Ledger\n\n| Claim | Evidence artifact | Status |\n|---|---|---|\n"
    text += "\n".join(f"| {a} | `{b}` | {c} |" for a, b, c in rows)
    text += "\n\n## Current Mechanism\n\n```json\n"
    text += json.dumps(json_safe(mechanism), indent=2, sort_keys=True)
    text += "\n```\n"
    write_text(path, text)


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/dc_balanced_fixed_total/smoke"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    copy_config(config_path, output_dir)
    write_math_derivation(reports / "math_derivation.md")
    device = resolve_device(str(config.get("device", "cuda")))
    run_outputs: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_config in config["runs"]:
        arms = build_arms(config, run_config)
        for arm in arms:
            if config.get("arms") and str(arm["arm_id"]) not in set(config["arms"]):
                continue
            out = run_arm(config=config, run_config=run_config, arm=arm, output_dir=output_dir, device=device)
            run_outputs.append(out)
            run_summaries.append(out["summary"])
    write_json(reports / "run_summaries.json", run_summaries)
    gate = aggregate_outputs(config=config, output_dir=output_dir, run_outputs=run_outputs)
    lineage = {
        "status": "PASS",
        "repo_state": repo_state(),
        "config": str(config_path),
        "final_v4_or_phase2_locked_used_for_tuning": False,
        "development_source": config.get("split", {}).get("source", "STL10 train+unlabeled"),
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
    }
    write_json(reports / "lineage_and_leakage_audit.json", lineage)
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "math_derivation.md": sha256_file(reports / "math_derivation.md"),
        "aggregate_method_metrics.csv": sha256_file(reports / "aggregate_method_metrics.csv"),
        "all_per_image_metrics.csv": sha256_file(reports / "all_per_image_metrics.csv"),
        "paired_bootstrap_comparisons.json": sha256_file(reports / "paired_bootstrap_comparisons.json"),
        "gate_report.json": sha256_file(reports / "gate_report.json"),
        "mechanism_attribution.json": sha256_file(reports / "mechanism_attribution.json"),
        "claim_evidence_ledger.md": sha256_file(reports / "claim_evidence_ledger.md"),
    }
    runtime = {
        "status": "PASS",
        "started_utc": now_utc(),
        "elapsed_seconds": float(time.time() - started),
        "artifact_hashes": hashes,
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "DC_BALANCED_FIXED_TOTAL_DEVELOPMENT_COMPLETE",
        "output_dir": str(output_dir),
        "gate": gate,
        "runtime": runtime,
        "key_artifacts": {
            "math_derivation": str(reports / "math_derivation.md"),
            "aggregate_metrics": str(reports / "aggregate_method_metrics.csv"),
            "per_image": str(reports / "all_per_image_metrics.csv"),
            "mechanism_attribution": str(reports / "mechanism_attribution.json"),
            "gate_report": str(reports / "gate_report.json"),
            "research_decision": str(reports / "research_decision.md"),
            "claim_evidence_ledger": str(reports / "claim_evidence_ledger.md"),
        },
    }
    write_json(reports / "summary.json", summary)
    atomic_write_json(
        output_dir / "DC_BALANCED_FIXED_TOTAL_DEVELOPMENT_COMPLETE.json",
        {
            "status": summary["status"],
            "decision": gate["decision"],
            "summary_sha256": sha256_file(reports / "summary.json"),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DC-balanced fixed-total witness attribution experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.config))
    print(
        json.dumps(
            json_safe(
                {
                    "status": summary["status"],
                    "output_dir": summary["output_dir"],
                    "decision": summary["gate"]["decision"],
                    "key_artifacts": summary["key_artifacts"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
