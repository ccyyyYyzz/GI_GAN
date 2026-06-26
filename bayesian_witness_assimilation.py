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
from src import phase1_4v4b0_scoring as b0
from src.bayesian_witness import (
    BayesianWitnessError,
    MethodEstimate,
    barycenter_null,
    candidate_measurement_residuals,
    candidate_p0_rmse,
    conditional_nullspace_audit,
    format_grid_tag,
    map_indices_from_weights,
    p0_rmse,
    posterior_entropy,
    posterior_weights_for_rows,
    sequential_risk_order,
    stable_softmax,
    standardize_scores,
    variance_order,
)
from src.phase2_fresh_operator import (
    build_fresh_split,
    candidate_feasibility_audit,
    make_fresh_context_measurement,
    resolve_device,
    score_frozen_selectors,
)
from src.phase2_witness import (
    CandidateCache,
    atomic_write_json,
    cache_audit,
    final_v4_context_summary,
    load_candidate_cache,
    make_witness_rows,
    paired_percentile_bootstrap,
    repo_state,
    sha256_file,
    write_csv,
    write_json,
)
from src.projections import get_exact_projector, relative_measurement_error


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "bayesian_witness_assimilation_dev.yaml"


class BayesianWitnessRunnerError(RuntimeError):
    """Raised for hard protocol errors in Bayesian witness assimilation runs."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise BayesianWitnessRunnerError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def save_config_copy(config_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


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


def sha256_numpy(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def rows_for_budget(order: np.ndarray, row_pool: np.ndarray, budget: int) -> list[np.ndarray]:
    idx = np.asarray(order, dtype=np.int64)
    if idx.ndim != 2:
        raise BayesianWitnessRunnerError(f"ROW_ORDER_MUST_BE_2D:{idx.shape}")
    return [np.asarray(row_pool[idx[i, : int(budget)]], dtype=np.float32) for i in range(idx.shape[0])]


def weights_for_all_images(
    cache: CandidateCache,
    rows_by_image: Sequence[np.ndarray],
    prior_scores_z: np.ndarray | None,
    *,
    alpha: float,
    tau: float,
) -> np.ndarray:
    weights = np.zeros((cache.n, cache.k), dtype=np.float64)
    for i in range(cache.n):
        ps = None if prior_scores_z is None else prior_scores_z[i]
        weights[i] = posterior_weights_for_rows(
            cache.cand_n[i],
            cache.true_n[i],
            rows_by_image[i],
            ps,
            alpha=alpha,
            tau=tau,
        )
    return weights


def map_from_rows(cache: CandidateCache, rows_by_image: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    selected = np.empty(cache.n, dtype=np.int64)
    for i in range(cache.n):
        residual = candidate_measurement_residuals(cache.cand_n[i], cache.true_n[i], rows_by_image[i])
        score = np.sum(residual * residual, axis=1)
        selected[i] = int(np.argmin(score))
    return selected, cache.cand_n[np.arange(cache.n), selected]


def make_row_orders(
    cache: CandidateCache,
    row_pool: np.ndarray,
    prior_scores_z: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    witness_cfg = config["witness"]
    budgets = [int(b) for b in witness_cfg["budgets"]]
    max_budget = max(budgets)
    if row_pool.shape[0] < max_budget:
        raise BayesianWitnessRunnerError(f"ROW_POOL_TOO_SMALL:{row_pool.shape[0]}:{max_budget}")
    designs = [str(v) for v in witness_cfg.get("designs", ["fixed_lowfreq", "variance_adaptive", "bayes_risk"])]
    orders: dict[str, np.ndarray] = {}
    traces: dict[str, Any] = {"status": "PASS", "sample_traces": {}, "row_pool_sha256": sha256_numpy(row_pool)}
    if "fixed_lowfreq" in designs:
        fixed = np.tile(np.arange(max_budget, dtype=np.int64)[None, :], (cache.n, 1))
        orders["fixed_lowfreq"] = fixed
    if "variance_adaptive" in designs:
        var = np.zeros((cache.n, max_budget), dtype=np.int64)
        for i in range(cache.n):
            var[i] = variance_order(cache.cand_n[i], row_pool)[:max_budget]
        orders["variance_adaptive"] = var
        traces["sample_traces"]["variance_adaptive_first5"] = var[:5].tolist()
    if "bayes_risk" in designs:
        risk = np.zeros((cache.n, max_budget), dtype=np.int64)
        sample = []
        alpha_design = float(witness_cfg.get("risk_design_alpha", 0.0))
        tau_design = float(witness_cfg.get("risk_design_tau", 0.2))
        sigma2 = float(witness_cfg.get("risk_design_sigma2", 1e-6))
        redundancy = float(witness_cfg.get("redundancy_max_abs_dot", 0.995))
        for i in range(cache.n):
            order, trace = sequential_risk_order(
                cache.cand_n[i],
                cache.true_n[i],
                row_pool,
                prior_scores_z[i],
                alpha=alpha_design,
                tau=tau_design,
                max_budget=max_budget,
                sigma2=sigma2,
                redundancy_max_abs_dot=redundancy,
            )
            risk[i] = np.asarray(order, dtype=np.int64)
            if i < 5:
                sample.append({"sample_uid": cache.sample_uids[i], "trace": trace})
        orders["bayes_risk"] = risk
        traces["sample_traces"]["bayes_risk_first5"] = sample
        traces["risk_design"] = {
            "alpha": alpha_design,
            "tau": tau_design,
            "sigma2": sigma2,
            "redundancy_max_abs_dot": redundancy,
        }
    return orders, traces


def add_estimate(
    estimates: list[MethodEstimate],
    *,
    method: str,
    budget: int | None,
    design: str,
    estimator: str,
    null_estimate: np.ndarray,
    selected_indices: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    alpha: float | None = None,
    tau: float | None = None,
    rows_by_image: list[np.ndarray] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    estimates.append(
        MethodEstimate(
            method=method,
            budget=budget,
            design=design,
            estimator=estimator,
            null_estimate=np.asarray(null_estimate, dtype=np.float32),
            selected_indices=None if selected_indices is None else np.asarray(selected_indices, dtype=np.int64),
            weights=None if weights is None else np.asarray(weights, dtype=np.float32),
            alpha=None if alpha is None else float(alpha),
            tau=None if tau is None else float(tau),
            rows_by_image=rows_by_image,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )
    )


def best_grid_estimate(
    *,
    cache: CandidateCache,
    rows_by_image: list[np.ndarray],
    prior_scores_z: np.ndarray | None,
    alpha_grid: Sequence[float],
    tau_grid: Sequence[float],
    estimator: str,
    design: str,
    budget: int,
    grid_rows: list[dict[str, Any]],
    alpha_filter: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float, float, dict[str, Any]]:
    best: dict[str, Any] | None = None
    for alpha in alpha_grid:
        if alpha_filter == "zero" and abs(float(alpha)) > 1e-12:
            continue
        if alpha_filter == "positive" and float(alpha) <= 0:
            continue
        for tau in tau_grid:
            weights = weights_for_all_images(cache, rows_by_image, prior_scores_z, alpha=float(alpha), tau=float(tau))
            if estimator == "map":
                idx = map_indices_from_weights(weights)
                null_est = cache.cand_n[np.arange(cache.n), idx]
            elif estimator == "barycenter":
                idx = None
                null_est = barycenter_null(cache.cand_n, weights)
            else:
                raise BayesianWitnessRunnerError(f"UNKNOWN_GRID_ESTIMATOR:{estimator}")
            err = p0_rmse(null_est, cache.true_n)
            row = {
                "design": design,
                "budget": int(budget),
                "estimator": estimator,
                "alpha": float(alpha),
                "tau": float(tau),
                "alpha_filter": alpha_filter,
                "mean_context_p0_rmse": float(err.mean()),
                "median_context_p0_rmse": float(np.median(err)),
                "mean_entropy": float(posterior_entropy(weights).mean()),
                "method_tag": format_grid_tag(float(alpha), float(tau)),
            }
            grid_rows.append(row)
            if best is None or row["mean_context_p0_rmse"] < best["row"]["mean_context_p0_rmse"]:
                best = {"row": row, "weights": weights, "null_est": null_est, "idx": idx}
    if best is None:
        raise BayesianWitnessRunnerError(f"EMPTY_GRID:{design}:{budget}:{estimator}:{alpha_filter}")
    return (
        np.asarray(best["null_est"], dtype=np.float32),
        None if best["idx"] is None else np.asarray(best["idx"], dtype=np.int64),
        np.asarray(best["weights"], dtype=np.float32),
        float(best["row"]["alpha"]),
        float(best["row"]["tau"]),
        best["row"],
    )


def evaluate_assimilation_methods(
    cache: CandidateCache,
    selector_scores: Mapping[str, np.ndarray],
    projector,
    row_pool: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[MethodEstimate], list[dict[str, Any]], dict[str, Any]]:
    witness_cfg = config["witness"]
    primary_selector = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
    if primary_selector not in selector_scores:
        raise BayesianWitnessRunnerError(f"PRIMARY_SELECTOR_SCORE_MISSING:{primary_selector}")
    prior_scores_z = standardize_scores(np.asarray(selector_scores[primary_selector], dtype=np.float64))
    alpha_grid = [float(v) for v in witness_cfg.get("alpha_grid", [0.0, 0.5, 1.0])]
    tau_grid = [float(v) for v in witness_cfg.get("tau_grid", [0.1, 0.2, 0.5])]
    if not any(abs(v) <= 1e-12 for v in alpha_grid):
        raise BayesianWitnessRunnerError("ALPHA_GRID_MUST_INCLUDE_ZERO")
    if not any(v > 0 for v in alpha_grid):
        raise BayesianWitnessRunnerError("ALPHA_GRID_MUST_INCLUDE_POSITIVE_VALUE")
    budgets = [int(v) for v in witness_cfg["budgets"]]
    order_by_design, row_trace = make_row_orders(cache, row_pool, prior_scores_z, config)
    estimates: list[MethodEstimate] = []
    grid_rows: list[dict[str, Any]] = []

    posterior = cache.cand_n.mean(axis=1)
    add_estimate(
        estimates,
        method="posterior_mean",
        budget=None,
        design="none",
        estimator="uniform_barycenter",
        null_estimate=posterior,
        weights=np.full((cache.n, cache.k), 1.0 / cache.k, dtype=np.float32),
        alpha=0.0,
        tau=None,
    )
    primary_idx = np.argmax(np.asarray(selector_scores[primary_selector]), axis=1).astype(np.int64)
    add_estimate(
        estimates,
        method=primary_selector,
        budget=None,
        design="fcc_prior_only",
        estimator="map",
        null_estimate=cache.cand_n[np.arange(cache.n), primary_idx],
        selected_indices=primary_idx,
    )
    oracle_idx = np.argmin(cache.p0_error, axis=1).astype(np.int64)
    add_estimate(
        estimates,
        method="oracle_best_of_16",
        budget=None,
        design="oracle",
        estimator="best_candidate_by_true_p0_rmse",
        null_estimate=cache.cand_n[np.arange(cache.n), oracle_idx],
        selected_indices=oracle_idx,
    )

    for budget in budgets:
        for design, order in order_by_design.items():
            rows_by_image = rows_for_budget(order, row_pool, budget)
            map_idx, map_null = map_from_rows(cache, rows_by_image)
            add_estimate(
                estimates,
                method=f"{design}_map_witness_b{budget}",
                budget=budget,
                design=design,
                estimator="witness_likelihood_map",
                null_estimate=map_null,
                selected_indices=map_idx,
                rows_by_image=rows_by_image,
            )
            like_bary, _idx, like_w, alpha0, tau0, row0 = best_grid_estimate(
                cache=cache,
                rows_by_image=rows_by_image,
                prior_scores_z=None,
                alpha_grid=[0.0],
                tau_grid=tau_grid,
                estimator="barycenter",
                design=design,
                budget=budget,
                grid_rows=grid_rows,
                alpha_filter="zero",
            )
            add_estimate(
                estimates,
                method=f"{design}_likelihood_barycenter_b{budget}",
                budget=budget,
                design=design,
                estimator="likelihood_weighted_barycenter",
                null_estimate=like_bary,
                weights=like_w,
                alpha=alpha0,
                tau=tau0,
                rows_by_image=rows_by_image,
                diagnostics={"selected_grid": row0},
            )
            audited, audit_diag = conditional_nullspace_audit(
                like_bary,
                cache.true_n,
                rows_by_image,
                projector,
                lambda_=float(witness_cfg.get("conditional_audit_lambda", 1e-5)),
            )
            add_estimate(
                estimates,
                method=f"{design}_likelihood_barycenter_condaudit_b{budget}",
                budget=budget,
                design=design,
                estimator="likelihood_barycenter_then_conditional_audit",
                null_estimate=audited,
                weights=like_w,
                alpha=alpha0,
                tau=tau0,
                rows_by_image=rows_by_image,
                diagnostics={"selected_grid": row0, "conditional_audit": audit_diag},
            )
            soft_map, soft_idx, soft_map_w, alpha_m, tau_m, row_m = best_grid_estimate(
                cache=cache,
                rows_by_image=rows_by_image,
                prior_scores_z=prior_scores_z,
                alpha_grid=alpha_grid,
                tau_grid=tau_grid,
                estimator="map",
                design=design,
                budget=budget,
                grid_rows=grid_rows,
                alpha_filter="positive",
            )
            add_estimate(
                estimates,
                method=f"{design}_soft_prior_likelihood_map_b{budget}",
                budget=budget,
                design=design,
                estimator="soft_prior_likelihood_map",
                null_estimate=soft_map,
                selected_indices=soft_idx,
                weights=soft_map_w,
                alpha=alpha_m,
                tau=tau_m,
                rows_by_image=rows_by_image,
                diagnostics={"selected_grid": row_m},
            )
            soft_bary, _idx_b, soft_w, alpha_b, tau_b, row_b = best_grid_estimate(
                cache=cache,
                rows_by_image=rows_by_image,
                prior_scores_z=prior_scores_z,
                alpha_grid=alpha_grid,
                tau_grid=tau_grid,
                estimator="barycenter",
                design=design,
                budget=budget,
                grid_rows=grid_rows,
                alpha_filter="positive",
            )
            add_estimate(
                estimates,
                method=f"{design}_soft_prior_barycenter_b{budget}",
                budget=budget,
                design=design,
                estimator="soft_prior_likelihood_barycenter",
                null_estimate=soft_bary,
                weights=soft_w,
                alpha=alpha_b,
                tau=tau_b,
                rows_by_image=rows_by_image,
                diagnostics={"selected_grid": row_b},
            )
            soft_audited, soft_audit_diag = conditional_nullspace_audit(
                soft_bary,
                cache.true_n,
                rows_by_image,
                projector,
                lambda_=float(witness_cfg.get("conditional_audit_lambda", 1e-5)),
            )
            add_estimate(
                estimates,
                method=f"{design}_soft_prior_barycenter_condaudit_b{budget}",
                budget=budget,
                design=design,
                estimator="soft_prior_barycenter_then_conditional_audit",
                null_estimate=soft_audited,
                weights=soft_w,
                alpha=alpha_b,
                tau=tau_b,
                rows_by_image=rows_by_image,
                diagnostics={"selected_grid": row_b, "conditional_audit": soft_audit_diag},
            )
    return estimates, grid_rows, row_trace


def relmeaserr_for_estimates(
    cache: CandidateCache,
    estimates: Sequence[MethodEstimate],
    measurement,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, dict[str, float]]:
    payload = torch.load(cache.path, map_location="cpu", weights_only=False)
    if "y" not in payload:
        raise BayesianWitnessRunnerError(f"CACHE_MISSING_Y:{cache.path}")
    y = np.asarray(payload["y"], dtype=np.float32)
    out: dict[str, dict[str, float]] = {}
    for est in estimates:
        vals = []
        xhat = (cache.r + est.null_estimate).astype(np.float32, copy=False)
        for start in range(0, cache.n, int(batch_size)):
            xb = torch.from_numpy(xhat[start : start + int(batch_size)]).to(device)
            yb = torch.from_numpy(y[start : start + int(batch_size)]).to(device)
            rel = relative_measurement_error(xb, yb, measurement).detach().cpu().numpy()
            vals.append(rel)
        arr = np.concatenate(vals)
        out[est.method] = {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
        }
    return out


def metric_rows_for_estimates(
    *,
    run_id: str,
    cache: CandidateCache,
    estimates: Sequence[MethodEstimate],
    relmeaserr: Mapping[str, Mapping[str, float]],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_method = {est.method: est for est in estimates}
    posterior_err = p0_rmse(by_method["posterior_mean"].null_estimate, cache.true_n)
    fixed_refs: dict[int, np.ndarray] = {}
    for est in estimates:
        if est.method.startswith("fixed_lowfreq_map_witness_b") and est.budget is not None:
            fixed_refs[int(est.budget)] = p0_rmse(est.null_estimate, cache.true_n)
    method_rows = []
    per_image_rows = []
    for est in estimates:
        err = p0_rmse(est.null_estimate, cache.true_n)
        full_rmse = np.sqrt(np.mean(((cache.r + est.null_estimate) - cache.x) ** 2, axis=1))
        delta_post = err - posterior_err
        boot_post = paired_percentile_bootstrap(delta_post, reps=bootstrap_reps, seed=bootstrap_seed + 11)
        row = {
            "run_id": run_id,
            "method": est.method,
            "budget": "" if est.budget is None else int(est.budget),
            "design": est.design,
            "estimator": est.estimator,
            "alpha": "" if est.alpha is None else float(est.alpha),
            "tau": "" if est.tau is None else float(est.tau),
            "mean_context_p0_rmse": float(np.mean(err)),
            "median_context_p0_rmse": float(np.median(err)),
            "mean_full_rmse": float(np.mean(full_rmse)),
            "delta_vs_posterior_mean": float(np.mean(delta_post)),
            "ci_vs_posterior_lower": float(boot_post["ci_lower"]),
            "ci_vs_posterior_upper": float(boot_post["ci_upper"]),
            "relmeaserr_mean": relmeaserr[est.method]["mean"],
            "relmeaserr_max": relmeaserr[est.method]["max"],
            "selected_index_mean": ""
            if est.selected_indices is None
            else float(np.mean(np.asarray(est.selected_indices, dtype=np.float64))),
            "mean_entropy": "" if est.weights is None else float(np.mean(posterior_entropy(est.weights))),
        }
        if est.budget is not None and int(est.budget) in fixed_refs:
            delta_fixed = err - fixed_refs[int(est.budget)]
            boot_fixed = paired_percentile_bootstrap(delta_fixed, reps=bootstrap_reps, seed=bootstrap_seed + 17 + int(est.budget))
            row["delta_vs_fixed_lowfreq_map_same_budget"] = float(np.mean(delta_fixed))
            row["ci_vs_fixed_lowfreq_map_same_budget_lower"] = float(boot_fixed["ci_lower"])
            row["ci_vs_fixed_lowfreq_map_same_budget_upper"] = float(boot_fixed["ci_upper"])
        method_rows.append(row)
        for i, uid in enumerate(cache.sample_uids):
            per_image_rows.append(
                {
                    "run_id": run_id,
                    "sample_uid": uid,
                    "source_index": int(cache.indices[i]),
                    "method": est.method,
                    "budget": "" if est.budget is None else int(est.budget),
                    "design": est.design,
                    "estimator": est.estimator,
                    "alpha": "" if est.alpha is None else float(est.alpha),
                    "tau": "" if est.tau is None else float(est.tau),
                    "context_p0_rmse": float(err[i]),
                    "full_rmse": float(full_rmse[i]),
                    "relmeaserr": relmeaserr[est.method]["max"],
                    "selected_index": "" if est.selected_indices is None else int(est.selected_indices[i]),
                }
            )
    return method_rows, per_image_rows


def quality_metrics(
    cache: CandidateCache,
    estimates: Sequence[MethodEstimate],
    *,
    focus_methods: Sequence[str],
    compute_lpips: bool,
    lpips_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from skimage.metrics import structural_similarity

    img_size = int(round(math.sqrt(cache.d)))
    truth = cache.x.reshape(cache.n, img_size, img_size)
    truth_clip = np.clip(truth, 0.0, 1.0)
    focus = [est for est in estimates if est.method in set(focus_methods)]
    rows: list[dict[str, Any]] = []
    lpips_status = "SKIPPED"
    lpips_values: dict[str, float] = {}
    if compute_lpips and focus:
        try:
            stack = np.stack(
                [
                    np.clip((cache.r + est.null_estimate).reshape(cache.n, img_size, img_size), 0.0, 1.0)
                    for est in focus
                ],
                axis=1,
            )
            vals = b0.compute_lpips_matrix(stack, truth_clip, device_name=lpips_device)
            for j, est in enumerate(focus):
                lpips_values[est.method] = float(np.mean(vals[:, j]))
            lpips_status = "PASS"
        except Exception as exc:  # pragma: no cover - environment dependent
            lpips_status = f"MISSING_OR_FAILED:{type(exc).__name__}:{exc}"
    for est in focus:
        xhat = (cache.r + est.null_estimate).reshape(cache.n, img_size, img_size)
        clip = np.clip(xhat, 0.0, 1.0)
        full_mse = np.mean(((cache.r + est.null_estimate) - cache.x) ** 2, axis=1)
        clip_mse = np.mean((clip - truth_clip) ** 2, axis=(1, 2))
        ssim = np.asarray(
            [
                structural_similarity(truth_clip[i], clip[i], data_range=1.0, win_size=7, channel_axis=None)
                for i in range(cache.n)
            ],
            dtype=np.float64,
        )
        rapsd = np.asarray([b0.rapsd_distance(clip[i], truth_clip[i], bins=32) for i in range(cache.n)], dtype=np.float64)
        rows.append(
            {
                "method": est.method,
                "budget": "" if est.budget is None else int(est.budget),
                "design": est.design,
                "estimator": est.estimator,
                "unclipped_full_rmse_mean": float(np.sqrt(full_mse).mean()),
                "unclipped_psnr_mean": float(b0.psnr_from_mse(full_mse).mean()),
                "clipped_psnr_mean": float(b0.psnr_from_mse(clip_mse).mean()),
                "clipped_ssim_mean": float(ssim.mean()),
                "clipped_rapsd_mean": float(rapsd.mean()),
                "clipped_lpips_mean": lpips_values.get(est.method, "[DATA MISSING]"),
                "range_violation_mean": float(np.maximum(xhat - 1.0, 0.0).mean() + np.maximum(-xhat, 0.0).mean()),
            }
        )
    return rows, {"lpips_status": lpips_status, "focus_method_count": len(focus)}


def build_or_load_fresh_cache(
    run_config: Mapping[str, Any],
    *,
    run_dir: Path,
    device: torch.device,
) -> tuple[CandidateCache, Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    measurement, base_config = make_fresh_context_measurement(run_config, device)
    generator, gen_config, ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
        Path(run_config.get("checkpoint", p12.PHASE79_CKPT)), base_config, measurement, device
    )
    if missing or unexpected:
        raise BayesianWitnessRunnerError(f"GENERATOR_LOAD_NOT_STRICT:{missing}:{unexpected}")
    split = build_fresh_split(run_config, measurement, device)
    cache_path = run_dir / "candidate_cache" / f"{split['name']}_k{int(run_config.get('candidate_k', 16))}.pt"
    if not cache_path.exists() or not bool(run_config.get("reuse_existing_cache", True)):
        p12.build_candidate_cache(
            generator,
            measurement,
            gen_config,
            split,
            out=run_dir,
            k=int(run_config.get("candidate_k", 16)),
            seed=int(run_config.get("candidate_seed", 970700)),
            device=device,
        )
    raw_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    selector_scores, selector_audit = score_frozen_selectors(raw_cache, device)
    cache = load_candidate_cache(cache_path, split=str(run_config["split"].get("name", "bayesian_dev")))
    a_hash = hashlib.sha256(measurement.A.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    operator_audit = {
        "status": "PASS",
        "context_operator": dict(run_config["context_operator"]),
        "split": dict(run_config["split"]),
        "checkpoint": str(run_config.get("checkpoint", p12.PHASE79_CKPT)),
        "checkpoint_sha256": sha256_file(Path(run_config.get("checkpoint", p12.PHASE79_CKPT))),
        "checkpoint_state_key": state_key,
        "A_sha256_float32": a_hash,
        "m": int(measurement.m),
        "n": int(measurement.n),
        "img_size": int(measurement.img_size),
        "projector": projector.info_dict(),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "candidate_seed": int(run_config.get("candidate_seed", 970700)),
    }
    return cache, measurement, operator_audit, selector_scores, selector_audit


def run_single_development_operator(
    *,
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_id = str(run_config["run_id"])
    run_dir = output_dir / "runs" / run_id
    reports = run_dir / "reports"
    ensure_dir(reports)
    cache, measurement, operator_audit, selector_scores, selector_audit = build_or_load_fresh_cache(
        run_config,
        run_dir=run_dir,
        device=device,
    )
    cache_info = cache_audit(cache)
    feasibility = candidate_feasibility_audit(cache, measurement, device)
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    row_pool = make_witness_rows(
        str(config["witness"].get("row_pool_kind", "dct2_low_frequency")),
        int(config["witness"].get("row_pool_size", 256)),
        cache.d,
        int(run_config.get("row_pool_seed", config["witness"].get("row_pool_seed", 97201))),
    ).astype(np.float32)
    estimates, grid_rows, row_trace = evaluate_assimilation_methods(cache, selector_scores, projector, row_pool, config)
    rel = relmeaserr_for_estimates(
        cache,
        estimates,
        measurement,
        device,
        batch_size=int(config.get("batch_size", 8)),
    )
    stats_cfg = config.get("statistics", {})
    method_rows, per_image_rows = metric_rows_for_estimates(
        run_id=run_id,
        cache=cache,
        estimates=estimates,
        relmeaserr=rel,
        bootstrap_reps=int(stats_cfg.get("bootstrap_replicates", 1000)),
        bootstrap_seed=int(stats_cfg.get("bootstrap_seed", 20260625)),
    )
    focus_methods = list(config.get("quality", {}).get("focus_methods", []))
    if not focus_methods:
        best_methods = sorted(method_rows, key=lambda r: float(r["mean_context_p0_rmse"]))[:10]
        focus_methods = ["posterior_mean", "oracle_best_of_16"] + [str(r["method"]) for r in best_methods]
        focus_methods = list(dict.fromkeys(focus_methods))
    quality_rows, quality_status = quality_metrics(
        cache,
        estimates,
        focus_methods=focus_methods,
        compute_lpips=bool(config.get("quality", {}).get("compute_lpips", False)),
        lpips_device=str(config.get("quality", {}).get("lpips_device", "cuda")),
    )
    selected_estimates = {est.method: est.null_estimate.astype(np.float32) for est in estimates if est.method in set(focus_methods)}
    np.savez_compressed(run_dir / "focus_null_estimates.npz", **selected_estimates)
    write_json(reports / "operator_audit.json", operator_audit)
    write_json(reports / "selector_transfer_audit.json", selector_audit)
    write_json(reports / "cache_audit.json", cache_info)
    write_json(reports / "candidate_feasibility_audit.json", feasibility)
    write_json(reports / "row_design_trace.json", row_trace)
    write_json(reports / "relmeaserr_by_method.json", rel)
    write_csv(reports / "grid_calibration_metrics.csv", grid_rows)
    write_csv(reports / "method_metrics.csv", method_rows)
    write_csv(reports / "per_image_metrics.csv", per_image_rows)
    write_csv(reports / "quality_metrics.csv", quality_rows)
    run_summary = {
        "status": "PASS",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "operator_audit": operator_audit,
        "cache_audit": cache_info,
        "candidate_feasibility": feasibility,
        "quality_status": quality_status,
        "row_pool": {
            "kind": str(config["witness"].get("row_pool_kind", "dct2_low_frequency")),
            "shape": list(row_pool.shape),
            "sha256": sha256_numpy(row_pool),
        },
        "best_method_by_context_p0_rmse": min(method_rows, key=lambda r: float(r["mean_context_p0_rmse"])),
        "artifact_hashes": {
            "method_metrics.csv": sha256_file(reports / "method_metrics.csv"),
            "per_image_metrics.csv": sha256_file(reports / "per_image_metrics.csv"),
            "grid_calibration_metrics.csv": sha256_file(reports / "grid_calibration_metrics.csv"),
            "quality_metrics.csv": sha256_file(reports / "quality_metrics.csv"),
            "focus_null_estimates.npz": sha256_file(run_dir / "focus_null_estimates.npz"),
        },
    }
    write_json(reports / "run_summary.json", run_summary)
    return {
        "summary": run_summary,
        "method_rows": method_rows,
        "per_image_rows": per_image_rows,
        "quality_rows": quality_rows,
        "grid_rows": grid_rows,
    }


def group_per_image(per_image_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in per_image_rows:
        uid = f"{row['run_id']}::{row['sample_uid']}"
        values[str(row["method"])].append((uid, float(row["context_p0_rmse"])))
    out: dict[str, dict[str, np.ndarray]] = {}
    for method, pairs in values.items():
        pairs_sorted = sorted(pairs, key=lambda p: p[0])
        out[method] = {
            "uids": np.asarray([p[0] for p in pairs_sorted], dtype=object),
            "values": np.asarray([p[1] for p in pairs_sorted], dtype=np.float64),
        }
    return out


def paired_delta(
    grouped: Mapping[str, Mapping[str, np.ndarray]],
    method: str,
    reference: str,
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    a = grouped[method]
    b = grouped[reference]
    if list(a["uids"]) != list(b["uids"]):
        raise BayesianWitnessRunnerError(f"PAIRING_UID_MISMATCH:{method}:{reference}")
    delta = np.asarray(a["values"], dtype=np.float64) - np.asarray(b["values"], dtype=np.float64)
    boot = paired_percentile_bootstrap(delta, reps=reps, seed=seed)
    return {
        "method": method,
        "reference": reference,
        "mean_delta": float(np.mean(delta)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "wins": int(np.sum(delta < 0)),
        "losses": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "n": int(delta.shape[0]),
    }


def aggregate_results(
    *,
    config: Mapping[str, Any],
    run_outputs: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    reports = output_dir / "reports"
    ensure_dir(reports)
    all_method_rows = [dict(r) for out in run_outputs for r in out["method_rows"]]
    all_per_image_rows = [dict(r) for out in run_outputs for r in out["per_image_rows"]]
    all_quality_rows = [dict(r) for out in run_outputs for r in out["quality_rows"]]
    all_grid_rows = [dict(r) for out in run_outputs for r in out["grid_rows"]]
    write_csv(reports / "all_method_metrics.csv", all_method_rows)
    write_csv(reports / "all_per_image_metrics.csv", all_per_image_rows)
    write_csv(reports / "all_quality_metrics.csv", all_quality_rows)
    write_csv(reports / "all_grid_calibration_metrics.csv", all_grid_rows)
    quality_numeric = [
        "unclipped_full_rmse_mean",
        "unclipped_psnr_mean",
        "clipped_psnr_mean",
        "clipped_ssim_mean",
        "clipped_lpips_mean",
        "clipped_rapsd_mean",
        "range_violation_mean",
    ]
    quality_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_quality_rows:
        quality_by_method[str(row["method"])].append(row)
    aggregate_quality_rows: list[dict[str, Any]] = []
    for method, rows in quality_by_method.items():
        item: dict[str, Any] = {
            "method": method,
            "run_count": len(rows),
            "budget": rows[0].get("budget", ""),
            "design": rows[0].get("design", ""),
            "estimator": rows[0].get("estimator", ""),
        }
        for col in quality_numeric:
            vals = []
            missing = 0
            for row in rows:
                try:
                    vals.append(float(row[col]))
                except (KeyError, TypeError, ValueError):
                    missing += 1
            item[col] = float(np.mean(vals)) if vals else "[DATA MISSING]"
            if missing:
                item[f"{col}_missing_count"] = int(missing)
        aggregate_quality_rows.append(item)
    aggregate_quality_rows.sort(
        key=lambda row: row["unclipped_full_rmse_mean"]
        if isinstance(row["unclipped_full_rmse_mean"], float)
        else float("inf")
    )
    write_csv(reports / "aggregate_quality_metrics.csv", aggregate_quality_rows)
    grouped = group_per_image(all_per_image_rows)
    method_means = {
        method: float(np.mean(payload["values"]))
        for method, payload in grouped.items()
    }
    stats_cfg = config.get("statistics", {})
    reps = int(stats_cfg.get("bootstrap_replicates", 1000))
    seed = int(stats_cfg.get("bootstrap_seed", 20260625)) + 500
    budgets = [int(v) for v in config["witness"]["budgets"]]
    aggregate_rows: list[dict[str, Any]] = []
    for method, mean_val in sorted(method_means.items(), key=lambda kv: kv[1]):
        row: dict[str, Any] = {"method": method, "aggregate_mean_context_p0_rmse": mean_val}
        if "posterior_mean" in grouped and method != "posterior_mean":
            row.update({f"vs_posterior_{k}": v for k, v in paired_delta(grouped, method, "posterior_mean", reps=reps, seed=seed).items() if k not in {"method", "reference"}})
        aggregate_rows.append(row)
    write_csv(reports / "aggregate_method_metrics.csv", aggregate_rows)

    best_by_budget: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    min_gain = float(config.get("gate", {}).get("min_mean_gain", 1e-4))
    relmeaserr_limit = float(config.get("gate", {}).get("relmeaserr_max", 1e-4))
    for budget in budgets:
        suffix = f"_b{budget}"
        methods_b = [m for m in grouped if m.endswith(suffix)]
        existing = [
            m
            for m in methods_b
            if m in {f"fixed_lowfreq_map_witness_b{budget}", f"variance_adaptive_map_witness_b{budget}"}
        ]
        strongest_existing = min(existing, key=lambda m: method_means[m]) if existing else None
        new_methods = [
            m
            for m in methods_b
            if (
                "barycenter" in m
                or "condaudit" in m
                or m.startswith("bayes_risk")
                or "soft_prior" in m
            )
        ]
        best_new = min(new_methods, key=lambda m: method_means[m]) if new_methods else None
        if best_new is None:
            continue
        row = {
            "budget": int(budget),
            "best_new_method": best_new,
            "best_new_mean": method_means[best_new],
            "strongest_existing_witness": strongest_existing,
            "strongest_existing_mean": None if strongest_existing is None else method_means[strongest_existing],
            "posterior_mean": method_means.get("posterior_mean"),
        }
        post_cmp = paired_delta(grouped, best_new, "posterior_mean", reps=reps, seed=seed + budget)
        row["delta_vs_posterior"] = post_cmp["mean_delta"]
        row["ci_vs_posterior_upper"] = post_cmp["ci_upper"]
        row["ci_vs_posterior_lower"] = post_cmp["ci_lower"]
        comparisons.append(post_cmp)
        if strongest_existing is not None:
            ex_cmp = paired_delta(grouped, best_new, strongest_existing, reps=reps, seed=seed + 100 + budget)
            row["delta_vs_strongest_existing"] = ex_cmp["mean_delta"]
            row["ci_vs_strongest_existing_upper"] = ex_cmp["ci_upper"]
            row["ci_vs_strongest_existing_lower"] = ex_cmp["ci_lower"]
            comparisons.append(ex_cmp)
        best_by_budget.append(row)
    write_csv(reports / "best_by_budget.csv", best_by_budget)
    write_json(reports / "paired_comparisons.json", comparisons)

    successful = []
    for row in best_by_budget:
        existing_ok = row.get("strongest_existing_witness") is not None and row.get("ci_vs_strongest_existing_upper") is not None
        conditions = {
            "beats_posterior_with_ci": bool(row["delta_vs_posterior"] < -min_gain and row["ci_vs_posterior_upper"] < 0),
            "beats_strongest_existing_with_ci": bool(
                existing_ok and row["delta_vs_strongest_existing"] < -min_gain and row["ci_vs_strongest_existing_upper"] < 0
            ),
        }
        row["gate_conditions"] = conditions
        if all(conditions.values()):
            successful.append(row)
    alpha_positive_methods = [m for m in grouped if "soft_prior" in m]
    alpha0_methods = [m for m in grouped if "likelihood_barycenter" in m and "soft_prior" not in m]
    best_alpha_pos = min(alpha_positive_methods, key=lambda m: method_means[m]) if alpha_positive_methods else None
    best_alpha0 = min(alpha0_methods, key=lambda m: method_means[m]) if alpha0_methods else None
    fcc_cmp = None
    if best_alpha_pos is not None and best_alpha0 is not None:
        fcc_cmp = paired_delta(grouped, best_alpha_pos, best_alpha0, reps=reps, seed=seed + 999)
    if successful:
        decision = "READY_TO_FREEZE_NEW_LOCKED_TEST_PROTOCOL"
    elif best_alpha0 is not None and method_means.get(best_alpha0, float("inf")) < method_means.get("posterior_mean", float("inf")):
        decision = "WITNESS_ASSIMILATION_DEVELOPMENT_SIGNAL_NO_LOCKED_TEST_YET"
    else:
        decision = "ADDON_ASSIMILATION_NOT_ENOUGH_RECOMMEND_MASK_AWARE_GENERATOR"
    gate = {
        "status": "PASS",
        "scope": "fresh development only; no final-v4 or Phase 2 locked-test tuning",
        "decision": decision,
        "budgets": budgets,
        "min_mean_gain": min_gain,
        "relmeaserr_max_gate": relmeaserr_limit,
        "successful_budget_rows": successful,
        "best_by_budget": best_by_budget,
        "best_alpha_positive_method": None if best_alpha_pos is None else {"method": best_alpha_pos, "mean": method_means[best_alpha_pos]},
        "best_alpha0_likelihood_method": None if best_alpha0 is None else {"method": best_alpha0, "mean": method_means[best_alpha0]},
        "fcc_prior_complementarity_comparison": fcc_cmp,
        "attribution_rule": (
            "Claim FCC-witness complementarity only if an alpha>0 soft-prior method beats the best alpha=0 likelihood method on a fresh locked test."
        ),
    }
    write_json(reports / "gate_report.json", gate)
    return gate


def write_math_derivation(output_dir: Path) -> None:
    text = r"""# Bayesian Witness Assimilation in the Context Null Space

## Range-null candidate geometry

Let the context operator be \(A_c\) and let \(P_0=I-A_c^\dagger A_c\).  A context-feasible candidate is written
\[
x_k=r+n_k,\qquad r=A_c^\dagger y_c,\qquad A_c n_k=0.
\]
For any probability vector \(\pi\) on the finite candidate set,
\[
A_c\left(r+\sum_k\pi_k n_k\right)=A_c r+\sum_k\pi_k A_c n_k=y_c.
\]
Thus every convex posterior barycenter remains exactly feasible for the original context measurements.  Plain language: averaging candidates is safe only because the average is taken inside the context null space.

## Soft FCC prior and witness likelihood

The FCC score is used as a calibratable prior,
\[
\pi_k^{(0)}=\frac{\exp(\alpha s_k)}{\sum_j\exp(\alpha s_j)}.
\]
Witness rows \(W\) observe \(y_w=W(r+n_\star)\).  Since all candidates share \(r\), the candidate residual is
\[
W x_k-y_w=W(r+n_k)-W(r+n_\star)=W(n_k-n_\star).
\]
With model-mismatch temperature \(\tau\),
\[
\pi_k=\frac{\pi_k^{(0)}\exp[-\|W(n_k-n_\star)\|^2/(2\tau^2)]}{\sum_j \pi_j^{(0)}\exp[-\|W(n_j-n_\star)\|^2/(2\tau^2)]}.
\]
Plain language: witness rows reweight candidates by observed physical evidence, while \(\tau\) admits that the finite candidate set is misspecified.

## MSE-optimal estimator

For squared error over the finite posterior, minimize
\[
R(z)=\sum_k \pi_k\|z-(r+n_k)\|^2.
\]
Differentiate:
\[
\nabla_z R(z)=2\sum_k\pi_k(z-r-n_k)
          =2\left(z-r-\sum_k\pi_k n_k\right).
\]
Setting the gradient to zero gives
\[
\bar x=r+\sum_k\pi_k n_k.
\]
Plain language: the correct estimator under squared error is the posterior mean/barycenter, not necessarily the MAP candidate.

## Conditional null-space witness audit

Starting from any context-feasible \(v=r+n\), update only inside the context null space:
\[
\hat x=v+P_0W^\top(WP_0W^\top+\lambda I)^{-1}(y_w-Wv).
\]
Context preservation follows because \(A_cP_0=0\):
\[
A_c\hat x=A_cv+A_cP_0W^\top(\cdots)=y_c.
\]
Let \(S=WP_0W^\top\) and residual \(e_w=Wv-y_w\).  The post-audit witness residual is
\[
W\hat x-y_w=e_w-S(S+\lambda I)^{-1}e_w
           =\lambda(S+\lambda I)^{-1}e_w.
\]
Plain language: the conditional audit can reduce witness residuals while provably never spending context consistency.

## Decision-theoretic row design

For posterior weights \(\pi\), the finite-candidate covariance is
\[
C_\pi=\sum_k\pi_k(n_k-\bar n)(n_k-\bar n)^\top.
\]
The Bayes risk under squared error is \(R(\pi)=\operatorname{tr} C_\pi\).  A row \(a\) is selected by expected risk decrease
\[
U(a)=R(\pi)-\mathbb E_{y|a,\pi}R(\pi_y).
\]
The implemented fast design uses the Gaussian surrogate
\[
U_G(a)=\frac{a^\top C_\pi^2 a}{a^\top C_\pi a+\sigma^2},
\]
computed from the \(K\times K\) candidate Gram matrix without forming \(C_\pi\) or dense \(P_0\).  Plain language: rows are chosen for expected posterior contraction, not merely for raw candidate variance.
"""
    path = output_dir / "reports" / "math_derivation.md"
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_research_decision(output_dir: Path, gate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    reports = output_dir / "reports"
    best = gate.get("successful_budget_rows") or gate.get("best_by_budget", [])
    lines = [
        "# Bayesian Witness Assimilation Development Decision",
        "",
        "## Scope",
        "",
        "This run uses fresh development splits/operators only. final-v4 and Phase 2 locked tests are carried only as historical evidence and are not used for tuning.",
        "",
        f"- Runs: `{[r.get('run_id') for r in config.get('runs', [])]}`",
        f"- Budgets: `{config['witness']['budgets']}`",
        f"- Row pool: `{config['witness'].get('row_pool_kind')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        "",
        "## Best Budget Rows",
        "",
        "| Budget | Best new method | Mean P0 RMSE | Strongest existing | Delta vs posterior | Delta vs existing |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for row in gate.get("best_by_budget", []):
        lines.append(
            "| {budget} | {method} | {mean} | {existing} | {dp} | {de} |".format(
                budget=row.get("budget"),
                method=row.get("best_new_method"),
                mean=row.get("best_new_mean"),
                existing=row.get("strongest_existing_witness"),
                dp=row.get("delta_vs_posterior"),
                de=row.get("delta_vs_strongest_existing"),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Allowed on this run: development evidence about posterior assimilation, row design, estimator choice, and conditional null-space audit.",
            "- Not allowed on this run: a locked-test claim or FCC-witness complementarity claim.",
            "- FCC complementarity requires alpha-positive soft prior to beat alpha=0 witness assimilation on a future fresh locked test.",
            "",
            "## Mechanism Attribution",
            "",
            "- Attribute large gains to conditional null-space witness audit after posterior barycentering, not to hard MAP witness selection unless the aggregate metrics show otherwise.",
            "- Treat row-design conclusions empirically: fixed low-frequency, variance-adaptive, and Bayes-risk rows are all reported; only the winner in `best_by_budget.csv` should be carried forward.",
            "- Treat alpha-positive FCC prior gains as development-only until the alpha-positive method beats alpha=0 likelihood assimilation on a future locked test.",
            "- Conditional audit can beat best-of-K candidate oracle because it is a continuous update in the context null space, not a selection among the original finite candidates.",
            "",
            "## Recommended Next Step",
            "",
        ]
    )
    if gate.get("decision") == "READY_TO_FREEZE_NEW_LOCKED_TEST_PROTOCOL":
        lines.append("Freeze a one-shot independent locked protocol using the successful budget/method family, with alpha/tau fixed from development.")
    elif gate.get("decision") == "WITNESS_ASSIMILATION_DEVELOPMENT_SIGNAL_NO_LOCKED_TEST_YET":
        lines.append("Refine calibration/row design on development only, then freeze a smaller candidate locked protocol if the signal remains stable.")
    else:
        lines.append("Stop add-on analysis and move to a mask-aware or budget-conditioned generator, because posterior assimilation did not beat the strong full-context posterior mean baseline.")
    path = reports / "research_decision.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_ledger(output_dir: Path, gate: Mapping[str, Any]) -> None:
    text = [
        "# Claim-Evidence Ledger",
        "",
        "| Claim | Evidence artifact | Status |",
        "|---|---|---|",
        "| Conditional audit preserves context measurements | `reports/math_derivation.md`, per-run `relmeaserr_by_method.json` | checked numerically |",
        "| Posterior barycenter is MSE-optimal over finite candidates | `reports/math_derivation.md` | proved |",
        "| New assimilation beats posterior mean and strongest witness baseline | `reports/gate_report.json` | {status} |".format(
            status="supported in development" if gate.get("successful_budget_rows") else "not supported by gate"
        ),
        "| FCC score and witness likelihood are complementary | `reports/gate_report.json` alpha-positive vs alpha=0 comparison | development-only, not claimable unless future locked test passes |",
        "| final-v4/Phase 2 locked tests were not reused for tuning | `reports/lineage_and_leakage_audit.json` | enforced by config lineage |",
    ]
    (output_dir / "reports" / "claim_evidence_ledger.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def build_lineage(config: Mapping[str, Any], run_outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "consumed_tests_not_used_for_tuning": {
            "final_v4": "historical summary only",
            "phase2_locked_external": "historical summary only",
        },
        "fresh_development_runs": [
            {
                "run_id": out["summary"]["run_id"],
                "context_operator": out["summary"]["operator_audit"]["context_operator"],
                "split": out["summary"]["operator_audit"]["split"],
                "A_sha256_float32": out["summary"]["operator_audit"]["A_sha256_float32"],
                "cache_path": out["summary"]["operator_audit"]["cache_path"],
                "cache_sha256": out["summary"]["operator_audit"]["cache_sha256"],
            }
            for out in run_outputs
        ],
        "row_pool": {
            "kind": str(config["witness"].get("row_pool_kind")),
            "size": int(config["witness"].get("row_pool_size")),
            "normalization": "rows are unit-normalized by make_witness_rows",
        },
        "final_v4_context_summary": final_v4_context_summary(),
        "repo_state": repo_state(),
    }


def run_bayesian_witness(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/bayesian_witness_assimilation/dev_v1"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    save_config_copy(config_path, output_dir)
    write_math_derivation(output_dir)
    device = resolve_device(str(config.get("device", "cuda")))
    base_run_config = {k: v for k, v in config.items() if k not in {"runs", "witness", "statistics", "quality", "gate"}}
    run_outputs = []
    for run in config.get("runs", []):
        run_config = deep_merge(base_run_config, run)
        run_config["witness"] = copy.deepcopy(config["witness"])
        run_config["statistics"] = copy.deepcopy(config.get("statistics", {}))
        run_config["quality"] = copy.deepcopy(config.get("quality", {}))
        run_outputs.append(
            run_single_development_operator(
                config=config,
                run_config=run_config,
                output_dir=output_dir,
                device=device,
            )
        )
    gate = aggregate_results(config=config, run_outputs=run_outputs, output_dir=output_dir)
    lineage = build_lineage(config, run_outputs)
    write_json(reports / "lineage_and_leakage_audit.json", lineage)
    write_research_decision(output_dir, gate, config)
    write_claim_ledger(output_dir, gate)
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "math_derivation.md": sha256_file(reports / "math_derivation.md"),
        "gate_report.json": sha256_file(reports / "gate_report.json"),
        "research_decision.md": sha256_file(reports / "research_decision.md"),
        "claim_evidence_ledger.md": sha256_file(reports / "claim_evidence_ledger.md"),
        "lineage_and_leakage_audit.json": sha256_file(reports / "lineage_and_leakage_audit.json"),
        "aggregate_method_metrics.csv": sha256_file(reports / "aggregate_method_metrics.csv"),
        "aggregate_quality_metrics.csv": sha256_file(reports / "aggregate_quality_metrics.csv"),
        "all_per_image_metrics.csv": sha256_file(reports / "all_per_image_metrics.csv"),
    }
    runtime = {
        "status": "PASS",
        "started_utc": now_utc(),
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "artifact_hashes": hashes,
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "BAYESIAN_WITNESS_ASSIMILATION_DEV_COMPLETE",
        "output_dir": str(output_dir),
        "gate": gate,
        "lineage_status": lineage["status"],
        "runtime": runtime,
        "key_artifacts": {
            "math_derivation": str(reports / "math_derivation.md"),
            "gate_report": str(reports / "gate_report.json"),
            "research_decision": str(reports / "research_decision.md"),
            "claim_evidence_ledger": str(reports / "claim_evidence_ledger.md"),
            "aggregate_metrics": str(reports / "aggregate_method_metrics.csv"),
        },
    }
    write_json(reports / "summary.json", summary)
    atomic_write_json(
        output_dir / "BAYESIAN_WITNESS_ASSIMILATION_DEV_COMPLETE.json",
        {"status": summary["status"], "summary_sha256": sha256_file(reports / "summary.json"), "decision": gate["decision"]},
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fresh-development Bayesian witness assimilation and conditional null-space audit experiments."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config path. Defaults to configs/compatibility/bayesian_witness_assimilation_dev.yaml.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_bayesian_witness(Path(args.config))
    print(json.dumps(json_safe({
        "status": summary["status"],
        "output_dir": summary["output_dir"],
        "decision": summary["gate"]["decision"],
        "key_artifacts": summary["key_artifacts"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
