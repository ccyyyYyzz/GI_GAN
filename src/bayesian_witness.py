from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


class BayesianWitnessError(RuntimeError):
    """Raised when Bayesian witness assimilation receives inconsistent inputs."""


def _as_float64(arr: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = _as_float64(logits)
    maxv = np.max(arr, axis=axis, keepdims=True)
    expv = np.exp(arr - maxv)
    denom = np.sum(expv, axis=axis, keepdims=True)
    return expv / np.maximum(denom, 1e-300)


def standardize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = _as_float64(scores)
    if arr.ndim != 2:
        raise BayesianWitnessError(f"PRIOR_SCORES_MUST_BE_2D:{arr.shape}")
    mu = np.mean(arr, axis=1, keepdims=True)
    sd = np.std(arr, axis=1, keepdims=True)
    return (arr - mu) / np.maximum(sd, float(eps))


def p0_rmse(null_estimate: np.ndarray, true_null: np.ndarray) -> np.ndarray:
    est = _as_float64(null_estimate)
    truth = _as_float64(true_null)
    if est.shape != truth.shape:
        raise BayesianWitnessError(f"P0_RMSE_SHAPE_MISMATCH:{est.shape}:{truth.shape}")
    return np.sqrt(np.mean((est - truth) ** 2, axis=1))


def candidate_p0_rmse(candidate_nulls: np.ndarray, true_null: np.ndarray) -> np.ndarray:
    cand = _as_float64(candidate_nulls)
    truth = _as_float64(true_null)
    if cand.ndim != 3 or truth.ndim != 2 or cand.shape[0] != truth.shape[0] or cand.shape[2] != truth.shape[1]:
        raise BayesianWitnessError(f"CANDIDATE_RMSE_SHAPE_MISMATCH:{cand.shape}:{truth.shape}")
    return np.sqrt(np.mean((cand - truth[:, None, :]) ** 2, axis=2))


def candidate_measurement_residuals(candidate_nulls_i: np.ndarray, true_null_i: np.ndarray, rows: np.ndarray) -> np.ndarray:
    cand = _as_float64(candidate_nulls_i)
    truth = _as_float64(true_null_i)
    w = _as_float64(rows)
    if cand.ndim != 2 or truth.ndim != 1 or w.ndim != 2 or cand.shape[1] != truth.shape[0] or w.shape[1] != truth.shape[0]:
        raise BayesianWitnessError(f"RESIDUAL_SHAPE_MISMATCH:{cand.shape}:{truth.shape}:{w.shape}")
    return (cand - truth[None, :]) @ w.T


def posterior_weights_for_rows(
    candidate_nulls_i: np.ndarray,
    true_null_i: np.ndarray,
    rows: np.ndarray,
    prior_scores_i: np.ndarray | None,
    *,
    alpha: float,
    tau: float,
) -> np.ndarray:
    cand = _as_float64(candidate_nulls_i)
    if cand.ndim != 2:
        raise BayesianWitnessError(f"CANDIDATES_MUST_BE_2D:{cand.shape}")
    k = cand.shape[0]
    if prior_scores_i is None:
        log_prior = np.zeros(k, dtype=np.float64)
    else:
        score = _as_float64(prior_scores_i).reshape(-1)
        if score.shape[0] != k:
            raise BayesianWitnessError(f"PRIOR_SCORE_LENGTH_MISMATCH:{score.shape}:{k}")
        log_prior = float(alpha) * score
    if rows.size == 0:
        return stable_softmax(log_prior, axis=0)
    if tau <= 0:
        raise BayesianWitnessError(f"TAU_MUST_BE_POSITIVE:{tau}")
    residual = candidate_measurement_residuals(cand, true_null_i, rows)
    nll = np.sum(residual * residual, axis=1) / (2.0 * float(tau) * float(tau))
    return stable_softmax(log_prior - nll, axis=0)


def barycenter_null(candidate_nulls: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cand = _as_float64(candidate_nulls)
    w = _as_float64(weights)
    if cand.ndim != 3 or w.ndim != 2 or cand.shape[:2] != w.shape:
        raise BayesianWitnessError(f"BARYCENTER_SHAPE_MISMATCH:{cand.shape}:{w.shape}")
    return np.einsum("nk,nkd->nd", w, cand, optimize=True)


def map_indices_from_weights(weights: np.ndarray) -> np.ndarray:
    w = _as_float64(weights)
    if w.ndim != 2:
        raise BayesianWitnessError(f"WEIGHTS_MUST_BE_2D:{w.shape}")
    return np.argmax(w, axis=1).astype(np.int64)


def posterior_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.clip(_as_float64(weights), 1e-300, 1.0)
    return -np.sum(w * np.log(w), axis=1)


def risk_trace(candidate_nulls_i: np.ndarray, weights_i: np.ndarray) -> float:
    cand = _as_float64(candidate_nulls_i)
    q = _as_float64(weights_i).reshape(-1)
    mean = q @ cand
    diff = cand - mean[None, :]
    return float(np.sum(q * np.sum(diff * diff, axis=1)))


def gaussian_risk_utilities(
    candidate_nulls_i: np.ndarray,
    weights_i: np.ndarray,
    row_pool: np.ndarray,
    *,
    sigma2: float,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    cand = _as_float64(candidate_nulls_i)
    q = _as_float64(weights_i).reshape(-1)
    rows = _as_float64(row_pool)
    if cand.ndim != 2 or rows.ndim != 2 or cand.shape[1] != rows.shape[1] or q.shape[0] != cand.shape[0]:
        raise BayesianWitnessError(f"UTILITY_SHAPE_MISMATCH:{cand.shape}:{q.shape}:{rows.shape}")
    mean = q @ cand
    diff = cand - mean[None, :]
    projected = diff @ rows.T
    gram = diff @ diff.T
    weighted_projected = q[:, None] * projected
    numerator = np.sum(weighted_projected * (gram @ weighted_projected), axis=0)
    denominator = np.sum(q[:, None] * projected * projected, axis=0) + float(sigma2)
    utilities = numerator / np.maximum(denominator, 1e-300)
    if blocked is not None:
        utilities = utilities.copy()
        utilities[np.asarray(blocked, dtype=bool)] = -np.inf
    return utilities


def _redundancy_block_mask(
    row_pool: np.ndarray,
    selected_indices: Sequence[int],
    *,
    max_abs_dot: float,
) -> np.ndarray:
    rows = _as_float64(row_pool)
    blocked = np.zeros(rows.shape[0], dtype=bool)
    if selected_indices:
        blocked[np.asarray(selected_indices, dtype=np.int64)] = True
        selected = rows[np.asarray(selected_indices, dtype=np.int64)]
        pool_norm = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        sel_norm = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-12)
        corr = np.max(np.abs(pool_norm @ sel_norm.T), axis=1)
        blocked |= corr >= float(max_abs_dot)
    return blocked


def sequential_risk_order(
    candidate_nulls_i: np.ndarray,
    true_null_i: np.ndarray,
    row_pool: np.ndarray,
    prior_scores_i: np.ndarray | None,
    *,
    alpha: float,
    tau: float,
    max_budget: int,
    sigma2: float,
    redundancy_max_abs_dot: float = 0.999,
) -> tuple[list[int], list[dict[str, float]]]:
    cand = _as_float64(candidate_nulls_i)
    pool = _as_float64(row_pool)
    if int(max_budget) > pool.shape[0]:
        raise BayesianWitnessError(f"RISK_BUDGET_EXCEEDS_ROW_POOL:{max_budget}:{pool.shape[0]}")
    if prior_scores_i is None:
        logq = np.zeros(cand.shape[0], dtype=np.float64)
    else:
        logq = float(alpha) * _as_float64(prior_scores_i).reshape(-1)
    selected: list[int] = []
    trace: list[dict[str, float]] = []
    for step in range(int(max_budget)):
        q = stable_softmax(logq, axis=0)
        before = risk_trace(cand, q)
        blocked = _redundancy_block_mask(pool, selected, max_abs_dot=redundancy_max_abs_dot)
        utilities = gaussian_risk_utilities(cand, q, pool, sigma2=sigma2, blocked=blocked)
        if not np.isfinite(utilities).any():
            blocked = np.zeros(pool.shape[0], dtype=bool)
            if selected:
                blocked[np.asarray(selected, dtype=np.int64)] = True
            utilities = gaussian_risk_utilities(cand, q, pool, sigma2=sigma2, blocked=blocked)
        row_idx = int(np.argmax(utilities))
        selected.append(row_idx)
        residual = candidate_measurement_residuals(cand, true_null_i, pool[row_idx : row_idx + 1]).reshape(-1)
        logq = logq - (residual * residual) / (2.0 * float(tau) * float(tau))
        q_after = stable_softmax(logq, axis=0)
        trace.append(
            {
                "step": float(step + 1),
                "row_index": float(row_idx),
                "utility": float(utilities[row_idx]),
                "risk_before": before,
                "risk_after_observed_update": risk_trace(cand, q_after),
                "posterior_entropy_after": float(posterior_entropy(q_after[None, :])[0]),
            }
        )
    return selected, trace


def variance_order(candidate_nulls_i: np.ndarray, row_pool: np.ndarray, weights_i: np.ndarray | None = None) -> np.ndarray:
    cand = _as_float64(candidate_nulls_i)
    rows = _as_float64(row_pool)
    projected = cand @ rows.T
    if weights_i is None:
        score = np.var(projected, axis=0)
    else:
        q = _as_float64(weights_i).reshape(-1)
        mu = q @ projected
        score = q @ ((projected - mu[None, :]) ** 2)
    return np.argsort(-score, kind="stable").astype(np.int64)


def conditional_nullspace_audit(
    null_estimate: np.ndarray,
    true_null: np.ndarray,
    rows_by_image: Sequence[np.ndarray],
    projector,
    *,
    lambda_: float,
    batch_project_rows: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    est = _as_float64(null_estimate)
    truth = _as_float64(true_null)
    if est.shape != truth.shape:
        raise BayesianWitnessError(f"CONDITIONAL_AUDIT_SHAPE_MISMATCH:{est.shape}:{truth.shape}")
    updates = np.zeros_like(est)
    max_context_update = 0.0
    max_witness_residual_before = 0.0
    max_witness_residual_after = 0.0
    max_condition = 0.0
    solver_fallbacks = 0
    for i, rows in enumerate(rows_by_image):
        w = _as_float64(rows)
        if w.size == 0:
            continue
        with torch.no_grad():
            wt = torch.from_numpy(w).to(device=projector.device, dtype=projector.dtype)
            z = projector.null_project_flat(wt).detach().cpu().numpy().astype(np.float64)
            au = projector.A_forward(torch.from_numpy(z).to(device=projector.device, dtype=projector.dtype))
            max_context_update = max(max_context_update, float(torch.linalg.norm(au).detach().cpu().item()))
        s_mat = w @ z.T
        s_mat = 0.5 * (s_mat + s_mat.T)
        matrix = s_mat + float(lambda_) * np.eye(w.shape[0], dtype=np.float64)
        residual = (truth[i] - est[i]) @ w.T
        max_witness_residual_before = max(max_witness_residual_before, float(np.linalg.norm(residual)))
        try:
            coeff = np.linalg.solve(matrix, residual)
        except np.linalg.LinAlgError:
            solver_fallbacks += 1
            coeff = np.linalg.pinv(matrix, rcond=1e-10) @ residual
        updates[i] = coeff @ z
        after = (truth[i] - (est[i] + updates[i])) @ w.T
        max_witness_residual_after = max(max_witness_residual_after, float(np.linalg.norm(after)))
        if matrix.size:
            max_condition = max(max_condition, float(np.linalg.cond(matrix)))
    audited = est + updates
    diagnostics = {
        "lambda": float(lambda_),
        "max_context_A_update_norm": max_context_update,
        "max_witness_residual_before": max_witness_residual_before,
        "max_witness_residual_after": max_witness_residual_after,
        "residual_shrink_ratio_max_norm": None
        if max_witness_residual_before <= 0
        else float(max_witness_residual_after / max_witness_residual_before),
        "max_linear_system_condition": max_condition,
        "solver_fallbacks": int(solver_fallbacks),
        "implementation": "matrix_free_P0_rows_via_projector.null_project_flat; no dense n-by-n P0 constructed",
    }
    return audited, diagnostics


@dataclass(frozen=True)
class MethodEstimate:
    method: str
    budget: int | None
    design: str
    estimator: str
    null_estimate: np.ndarray
    selected_indices: np.ndarray | None
    weights: np.ndarray | None
    alpha: float | None
    tau: float | None
    rows_by_image: list[np.ndarray] | None
    diagnostics: Mapping[str, Any]


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    arr = _as_float64(values).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def format_grid_tag(alpha: float, tau: float) -> str:
    a = f"{float(alpha):.4g}".replace(".", "p").replace("-", "m")
    t = f"{float(tau):.4g}".replace(".", "p").replace("-", "m")
    return f"a{a}_t{t}"
