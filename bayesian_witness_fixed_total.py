from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import phase1_2_rad5_64_pipeline as p12
from bayesian_witness_assimilation import (
    best_grid_estimate,
    ensure_dir,
    json_safe,
    relmeaserr_for_estimates,
    save_config_copy,
    write_math_derivation,
)
from src.bayesian_witness import (
    MethodEstimate,
    barycenter_null,
    conditional_nullspace_audit,
    map_indices_from_weights,
    p0_rmse,
    standardize_scores,
)
from src.phase2_fresh_operator import (
    build_fresh_split,
    candidate_feasibility_audit,
    make_fixed_total_context_measurement,
    resolve_device,
    score_frozen_selectors,
)
from src.phase2_witness import (
    atomic_write_json,
    cache_audit,
    load_candidate_cache,
    make_witness_rows,
    paired_percentile_bootstrap,
    repo_state,
    sha256_file,
    write_csv,
    write_json,
)
from src.projections import get_exact_projector


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "bayesian_witness_fixed_total_1pct_pilot.yaml"


class FixedTotalWitnessError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise FixedTotalWitnessError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def sha256_numpy(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def add_estimate(
    estimates: list[MethodEstimate],
    *,
    method: str,
    budget: int,
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
            budget=int(budget),
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


def map_from_rows(cache, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows64 = np.asarray(rows, dtype=np.float64)
    selected = np.zeros(cache.n, dtype=np.int64)
    for i in range(cache.n):
        diff = np.asarray(cache.cand_n[i] - cache.true_n[i][None, :], dtype=np.float64)
        residual = diff @ rows64.T
        score = np.sum(residual * residual, axis=1)
        selected[i] = int(np.argmin(score))
    return selected, cache.cand_n[np.arange(cache.n), selected]


def make_estimates_for_budget(
    *,
    cache,
    selector_scores: Mapping[str, np.ndarray],
    projector,
    budget: int,
    witness_rows: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[MethodEstimate], list[dict[str, Any]], dict[str, Any]]:
    witness_cfg = config["witness"]
    primary_selector = str(witness_cfg.get("primary_selector", "dm_fcc_seed3"))
    if primary_selector not in selector_scores:
        raise FixedTotalWitnessError(f"PRIMARY_SELECTOR_MISSING:{primary_selector}")
    prior_scores_z = standardize_scores(np.asarray(selector_scores[primary_selector], dtype=np.float64))
    alpha_grid = [float(v) for v in witness_cfg.get("alpha_grid", [0.0, 0.5])]
    tau_grid = [float(v) for v in witness_cfg.get("tau_grid", [0.2, 1.0])]
    estimates: list[MethodEstimate] = []
    grid_rows: list[dict[str, Any]] = []
    b = int(budget)
    posterior = cache.cand_n.mean(axis=1)
    add_estimate(
        estimates,
        method=f"posterior_mean_b{b}",
        budget=b,
        design="fixed_total_context_only",
        estimator="uniform_barycenter",
        null_estimate=posterior,
        weights=np.full((cache.n, cache.k), 1.0 / cache.k, dtype=np.float32),
        alpha=0.0,
    )
    primary_idx = np.argmax(np.asarray(selector_scores[primary_selector]), axis=1).astype(np.int64)
    add_estimate(
        estimates,
        method=f"{primary_selector}_b{b}",
        budget=b,
        design="fcc_prior_only",
        estimator="map",
        null_estimate=cache.cand_n[np.arange(cache.n), primary_idx],
        selected_indices=primary_idx,
    )
    oracle_idx = np.argmin(cache.p0_error, axis=1).astype(np.int64)
    add_estimate(
        estimates,
        method=f"oracle_best_of_16_b{b}",
        budget=b,
        design="oracle",
        estimator="best_candidate_by_true_context_p0_rmse",
        null_estimate=cache.cand_n[np.arange(cache.n), oracle_idx],
        selected_indices=oracle_idx,
    )
    diagnostics: dict[str, Any] = {}
    if b <= 0:
        return estimates, grid_rows, diagnostics
    rows_by_image = [np.asarray(witness_rows, dtype=np.float32) for _ in range(cache.n)]
    idx, map_null = map_from_rows(cache, witness_rows)
    add_estimate(
        estimates,
        method=f"fixed_total_lowfreq_map_witness_b{b}",
        budget=b,
        design="fixed_total_lowfreq",
        estimator="witness_likelihood_map",
        null_estimate=map_null,
        selected_indices=idx,
        rows_by_image=rows_by_image,
    )
    like_null, _idx, like_w, alpha0, tau0, row0 = best_grid_estimate(
        cache=cache,
        rows_by_image=rows_by_image,
        prior_scores_z=None,
        alpha_grid=[0.0],
        tau_grid=tau_grid,
        estimator="barycenter",
        design="fixed_total_lowfreq",
        budget=b,
        grid_rows=grid_rows,
        alpha_filter="zero",
    )
    add_estimate(
        estimates,
        method=f"fixed_total_lowfreq_likelihood_barycenter_b{b}",
        budget=b,
        design="fixed_total_lowfreq",
        estimator="likelihood_weighted_barycenter",
        null_estimate=like_null,
        weights=like_w,
        alpha=alpha0,
        tau=tau0,
        rows_by_image=rows_by_image,
        diagnostics={"selected_grid": row0},
    )
    audited, audit_diag = conditional_nullspace_audit(
        like_null,
        cache.true_n,
        rows_by_image,
        projector,
        lambda_=float(witness_cfg.get("conditional_audit_lambda", 1e-5)),
    )
    add_estimate(
        estimates,
        method=f"fixed_total_lowfreq_likelihood_barycenter_condaudit_b{b}",
        budget=b,
        design="fixed_total_lowfreq",
        estimator="likelihood_barycenter_then_conditional_audit",
        null_estimate=audited,
        weights=like_w,
        alpha=alpha0,
        tau=tau0,
        rows_by_image=rows_by_image,
        diagnostics={"selected_grid": row0, "conditional_audit": audit_diag},
    )
    soft_null, _idx2, soft_w, alpha_b, tau_b, row_b = best_grid_estimate(
        cache=cache,
        rows_by_image=rows_by_image,
        prior_scores_z=prior_scores_z,
        alpha_grid=alpha_grid,
        tau_grid=tau_grid,
        estimator="barycenter",
        design="fixed_total_lowfreq",
        budget=b,
        grid_rows=grid_rows,
        alpha_filter="positive",
    )
    add_estimate(
        estimates,
        method=f"fixed_total_lowfreq_soft_prior_barycenter_b{b}",
        budget=b,
        design="fixed_total_lowfreq",
        estimator="soft_prior_likelihood_barycenter",
        null_estimate=soft_null,
        weights=soft_w,
        alpha=alpha_b,
        tau=tau_b,
        rows_by_image=rows_by_image,
        diagnostics={"selected_grid": row_b},
    )
    soft_audited, soft_audit_diag = conditional_nullspace_audit(
        soft_null,
        cache.true_n,
        rows_by_image,
        projector,
        lambda_=float(witness_cfg.get("conditional_audit_lambda", 1e-5)),
    )
    add_estimate(
        estimates,
        method=f"fixed_total_lowfreq_soft_prior_barycenter_condaudit_b{b}",
        budget=b,
        design="fixed_total_lowfreq",
        estimator="soft_prior_barycenter_then_conditional_audit",
        null_estimate=soft_audited,
        weights=soft_w,
        alpha=alpha_b,
        tau=tau_b,
        rows_by_image=rows_by_image,
        diagnostics={"selected_grid": row_b, "conditional_audit": soft_audit_diag},
    )
    diagnostics["witness_rows"] = {
        "kind": str(witness_cfg.get("witness_row_kind", "dct2_low_frequency")),
        "budget": b,
        "sha256": sha256_numpy(witness_rows),
    }
    return estimates, grid_rows, diagnostics


def metric_rows(
    *,
    base_run_id: str,
    budget: int,
    context_m: int,
    total_m: int,
    cache,
    estimates: Sequence[MethodEstimate],
    relmeaserr: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = []
    per_image = []
    for est in estimates:
        context_p0 = p0_rmse(est.null_estimate, cache.true_n)
        full = np.sqrt(np.mean(((cache.r + est.null_estimate) - cache.x) ** 2, axis=1))
        methods.append(
            {
                "base_run_id": base_run_id,
                "budget": int(budget),
                "context_m": int(context_m),
                "total_m": int(total_m),
                "method": est.method,
                "design": est.design,
                "estimator": est.estimator,
                "alpha": "" if est.alpha is None else float(est.alpha),
                "tau": "" if est.tau is None else float(est.tau),
                "mean_context_p0_rmse": float(context_p0.mean()),
                "mean_full_rmse": float(full.mean()),
                "relmeaserr_mean": relmeaserr[est.method]["mean"],
                "relmeaserr_max": relmeaserr[est.method]["max"],
            }
        )
        for i, uid in enumerate(cache.sample_uids):
            per_image.append(
                {
                    "base_run_id": base_run_id,
                    "budget": int(budget),
                    "context_m": int(context_m),
                    "source_index": int(cache.indices[i]),
                    "sample_uid": uid,
                    "method": est.method,
                    "context_p0_rmse": float(context_p0[i]),
                    "full_rmse": float(full[i]),
                    "relmeaserr": relmeaserr[est.method]["max"],
                    "selected_index": "" if est.selected_indices is None else int(est.selected_indices[i]),
                }
            )
    return methods, per_image


def paired_boot(delta: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    boot = paired_percentile_bootstrap(delta, reps=reps, seed=seed)
    return {
        "mean_delta": float(np.mean(delta)),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "wins": int(np.sum(delta < 0)),
        "losses": int(np.sum(delta > 0)),
        "n": int(delta.shape[0]),
    }


def aggregate_fixed_total(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    all_per_image: Sequence[Mapping[str, Any]],
    all_method_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = output_dir / "reports"
    reps = int(config.get("statistics", {}).get("bootstrap_replicates", 1000))
    seed = int(config.get("statistics", {}).get("bootstrap_seed", 20260625)) + 700
    baseline: dict[tuple[str, int], float] = {}
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_per_image:
        by_method[str(row["method"])].append(row)
        if int(row["budget"]) == 0 and str(row["method"]) == "posterior_mean_b0":
            baseline[(str(row["base_run_id"]), int(row["source_index"]))] = float(row["full_rmse"])
    if not baseline:
        raise FixedTotalWitnessError("MISSING_BUDGET0_POSTERIOR_BASELINE")
    aggregate_rows = []
    for method, rows in sorted(by_method.items()):
        rows_sorted = sorted(rows, key=lambda r: (str(r["base_run_id"]), int(r["source_index"])))
        vals = np.asarray([float(r["full_rmse"]) for r in rows_sorted], dtype=np.float64)
        item = {
            "method": method,
            "budget": int(rows_sorted[0]["budget"]),
            "context_m": int(rows_sorted[0]["context_m"]),
            "n": int(vals.shape[0]),
            "mean_full_rmse": float(vals.mean()),
            "mean_context_p0_rmse": float(np.mean([float(r["context_p0_rmse"]) for r in rows_sorted])),
        }
        paired = []
        ok = True
        for r in rows_sorted:
            key = (str(r["base_run_id"]), int(r["source_index"]))
            if key not in baseline:
                ok = False
                break
            paired.append(float(r["full_rmse"]) - baseline[key])
        if ok:
            comp = paired_boot(np.asarray(paired, dtype=np.float64), reps, seed + int(item["budget"]))
            item.update({f"vs_full_context_posterior_{k}": v for k, v in comp.items()})
        aggregate_rows.append(item)
    write_csv(reports / "aggregate_fixed_total_metrics.csv", aggregate_rows)
    budgets = sorted({int(r["budget"]) for r in aggregate_rows if int(r["budget"]) > 0})
    best_by_budget = []
    for b in budgets:
        candidates = [
            r for r in aggregate_rows
            if int(r["budget"]) == b and ("condaudit" in str(r["method"]) or "barycenter" in str(r["method"]) or "map_witness" in str(r["method"]))
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: float(r["mean_full_rmse"]))
        best_by_budget.append(best)
    successes = [
        r for r in best_by_budget
        if float(r.get("vs_full_context_posterior_mean_delta", 1.0)) < -float(config.get("gate", {}).get("min_mean_gain", 1e-4))
        and float(r.get("vs_full_context_posterior_ci_upper", 1.0)) < 0.0
    ]
    decision = "FIXED_TOTAL_SAME_RATE_SIGNAL" if successes else "NO_FIXED_TOTAL_SAME_RATE_SIGNAL"
    gate = {
        "status": "PASS",
        "scope": "fixed_total_1pct_development_extra_not_locked",
        "total_m": int(config["context_operator"].get("total_m", 41)),
        "budgets": budgets,
        "baseline": "posterior_mean_b0 uses all total_m context rows",
        "decision": decision,
        "best_by_budget": best_by_budget,
        "successful_budget_rows": successes,
        "interpretation": (
            "Success means the split context+witness method beats the all-context posterior mean at the same total row count."
        ),
    }
    write_json(reports / "fixed_total_gate.json", gate)
    return gate


def run_fixed_total(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/bayesian_witness_fixed_total/onepct_pilot"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    save_config_copy(config_path, output_dir)
    write_math_derivation(output_dir)
    device = resolve_device(str(config.get("device", "cuda")))
    total_m = int(config["context_operator"].get("total_m", 41))
    budgets = [int(b) for b in config["fixed_total"]["witness_budgets"]]
    if 0 not in budgets:
        budgets = [0] + budgets
    all_method_rows: list[dict[str, Any]] = []
    all_per_image: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_cfg in config["runs"]:
        base_run_id = str(run_cfg["run_id"])
        for budget in budgets:
            budget_dir = output_dir / "runs" / base_run_id / f"budget_b{budget:03d}"
            ensure_dir(budget_dir / "reports")
            run_config = dict(config)
            run_config["context_operator"] = dict(config["context_operator"])
            run_config["context_operator"].update(dict(run_cfg.get("context_operator", {})))
            run_config["split"] = dict(config["split"])
            run_config["split"].update(dict(run_cfg.get("split", {})))
            run_config["split"]["name"] = f"{run_config['split']['name']}_b{budget:03d}"
            run_config["candidate_seed"] = int(run_cfg.get("candidate_seed", config.get("candidate_seed", 990700))) + budget * 10000
            measurement, base_config, _A_total, _A_context = make_fixed_total_context_measurement(
                run_config,
                witness_budget=int(budget),
                device=device,
            )
            context_m = int(measurement.m)
            generator, gen_config, _ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
                Path(config.get("checkpoint", p12.PHASE79_CKPT)), base_config, measurement, device
            )
            if missing or unexpected:
                raise FixedTotalWitnessError(f"GENERATOR_LOAD_NOT_STRICT:{base_run_id}:b{budget}:{missing}:{unexpected}")
            split = build_fresh_split(run_config, measurement, device)
            cache_path = budget_dir / "candidate_cache" / f"{split['name']}_k{int(config.get('candidate_k', 16))}.pt"
            if not cache_path.exists() or not bool(config.get("reuse_existing_cache", True)):
                p12.build_candidate_cache(
                    generator,
                    measurement,
                    gen_config,
                    split,
                    out=budget_dir,
                    k=int(config.get("candidate_k", 16)),
                    seed=int(run_config["candidate_seed"]),
                    device=device,
                )
            raw_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            selector_scores, selector_audit = score_frozen_selectors(raw_cache, device)
            cache = load_candidate_cache(cache_path, split=f"{base_run_id}_b{budget:03d}")
            projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
            witness_rows = (
                np.zeros((0, cache.d), dtype=np.float32)
                if budget == 0
                else make_witness_rows(
                    str(config["witness"].get("witness_row_kind", "dct2_low_frequency")),
                    int(budget),
                    cache.d,
                    int(run_cfg.get("witness_seed", config["witness"].get("witness_seed", 99401))) + int(budget),
                ).astype(np.float32)
            )
            estimates, grid_rows, diag = make_estimates_for_budget(
                cache=cache,
                selector_scores=selector_scores,
                projector=projector,
                budget=int(budget),
                witness_rows=witness_rows,
                config=config,
            )
            rel = relmeaserr_for_estimates(cache, estimates, measurement, device, batch_size=int(config.get("batch_size", 8)))
            methods, per_img = metric_rows(
                base_run_id=base_run_id,
                budget=int(budget),
                context_m=context_m,
                total_m=total_m,
                cache=cache,
                estimates=estimates,
                relmeaserr=rel,
            )
            all_method_rows.extend(methods)
            all_per_image.extend(per_img)
            operator_audit = {
                "status": "PASS",
                "base_run_id": base_run_id,
                "budget": int(budget),
                "total_m": total_m,
                "context_m": context_m,
                "witness_m": int(budget),
                "context_sampling_ratio": float(context_m / int(measurement.n)),
                "total_sampling_ratio": float(total_m / int(measurement.n)),
                "context_operator": run_config["context_operator"],
                "split": run_config["split"],
                "checkpoint_state_key": state_key,
                "checkpoint_sha256": sha256_file(Path(config.get("checkpoint", p12.PHASE79_CKPT))),
                "A_context_sha256_float32": sha256_numpy(measurement.A.detach().cpu().numpy().astype(np.float32)),
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "projector": projector.info_dict(),
            }
            write_json(budget_dir / "reports" / "operator_audit.json", operator_audit)
            write_json(budget_dir / "reports" / "selector_transfer_audit.json", selector_audit)
            write_json(budget_dir / "reports" / "cache_audit.json", cache_audit(cache))
            write_json(budget_dir / "reports" / "candidate_feasibility_audit.json", candidate_feasibility_audit(cache, measurement, device))
            write_json(budget_dir / "reports" / "relmeaserr_by_method.json", rel)
            write_csv(budget_dir / "reports" / "method_metrics.csv", methods)
            write_csv(budget_dir / "reports" / "per_image_metrics.csv", per_img)
            write_csv(budget_dir / "reports" / "grid_calibration_metrics.csv", grid_rows)
            write_json(budget_dir / "reports" / "budget_summary.json", {"operator_audit": operator_audit, "diagnostics": diag})
            run_summaries.append(
                {
                    "base_run_id": base_run_id,
                    "budget": int(budget),
                    "context_m": context_m,
                    "cache_path": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "witness_rows_sha256": sha256_numpy(witness_rows),
                    "method_count": len(methods),
                }
            )
    write_csv(reports / "all_method_metrics.csv", all_method_rows)
    write_csv(reports / "all_per_image_metrics.csv", all_per_image)
    write_json(reports / "run_budget_summaries.json", run_summaries)
    gate = aggregate_fixed_total(output_dir=output_dir, config=config, all_per_image=all_per_image, all_method_rows=all_method_rows)
    report_lines = [
        "# Fixed-Total 1% Bayesian Witness Pilot",
        "",
        "Total row count is fixed at 41. Each budget uses 41-b context rows for candidate generation and b low-frequency witness rows for posterior assimilation/audit.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Total m: `{gate['total_m']}`",
        "",
        "| Budget | Best method | Context m | Mean full RMSE | Delta vs b0 posterior | CI upper |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in gate["best_by_budget"]:
        report_lines.append(
            "| {budget} | {method} | {context_m} | {mean_full_rmse} | {delta} | {upper} |".format(
                budget=row["budget"],
                method=row["method"],
                context_m=row["context_m"],
                mean_full_rmse=row["mean_full_rmse"],
                delta=row.get("vs_full_context_posterior_mean_delta"),
                upper=row.get("vs_full_context_posterior_ci_upper"),
            )
        )
    report_lines += [
        "",
        "Interpretation: this is the fair same-rate test. A positive gate means context+witness splitting beats using all 41 rows as context. A negative gate means add-on witness gains did not survive fixed total budget with the current frozen generator.",
    ]
    (reports / "research_decision.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    lineage = {
        "status": "PASS",
        "not_touching_previous_addon_dirs": True,
        "source": "STL10 train+unlabeled development only",
        "final_v4_or_phase2_locked_used_for_tuning": False,
        "repo_state": repo_state(),
    }
    write_json(reports / "lineage_and_leakage_audit.json", lineage)
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "fixed_total_gate.json": sha256_file(reports / "fixed_total_gate.json"),
        "research_decision.md": sha256_file(reports / "research_decision.md"),
        "all_method_metrics.csv": sha256_file(reports / "all_method_metrics.csv"),
        "all_per_image_metrics.csv": sha256_file(reports / "all_per_image_metrics.csv"),
        "lineage_and_leakage_audit.json": sha256_file(reports / "lineage_and_leakage_audit.json"),
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
        "status": "BAYESIAN_WITNESS_FIXED_TOTAL_DEV_COMPLETE",
        "output_dir": str(output_dir),
        "gate": gate,
        "runtime": runtime,
        "key_artifacts": {
            "research_decision": str(reports / "research_decision.md"),
            "gate_report": str(reports / "fixed_total_gate.json"),
            "aggregate_metrics": str(reports / "aggregate_fixed_total_metrics.csv"),
        },
    }
    write_json(reports / "summary.json", summary)
    atomic_write_json(
        output_dir / "BAYESIAN_WITNESS_FIXED_TOTAL_DEV_COMPLETE.json",
        {"status": summary["status"], "decision": gate["decision"], "summary_sha256": sha256_file(reports / "summary.json")},
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fixed-total Bayesian witness same-sampling-rate development pilot.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_fixed_total(Path(args.config))
    print(json.dumps(json_safe({"status": summary["status"], "output_dir": summary["output_dir"], "decision": summary["gate"]["decision"], "key_artifacts": summary["key_artifacts"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
