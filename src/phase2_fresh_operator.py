from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

import phase1_2_rad5_64_pipeline as p12
import phase1_3r_recovery_and_relock as p13
from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src import phase69B_controlled_gauge_cgan_pilot as p69b
from src import phase73_overnight_gauge_gan_expansion as p73
from src.compatibility_model import CompatibilityCritic
from src.eval import make_measurement
from src.measurement import create_fixed_measurement_matrix
from src.phase1_4ir_uid_safe_scoring import ALL_SELECTOR_KEYS
from src.phase2_witness import (
    CandidateCache,
    as_numpy,
    atomic_write_json,
    build_method_tables,
    cache_audit,
    compute_gate,
    final_v4_context_summary,
    leakage_operator_audit,
    load_candidate_cache,
    make_dct2_lowfreq_rows,
    make_markdown_report,
    make_rademacher_rows,
    repo_state,
    sha256_file,
    write_csv,
    write_json,
)
from src.projections import get_exact_projector, relative_measurement_error
from src.utils import apply_experiment_defaults


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock" / "recovered_selector_artifacts"
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "phase2_fresh_operator_smoke.yaml"
DEFAULT_FIXED_TOTAL_CONFIG = ROOT / "configs" / "compatibility" / "phase2_fixed_total_smoke.yaml"


class Phase2FreshOperatorError(RuntimeError):
    """Hard-fail exception for Phase 2 fresh-operator smoke/protocol work."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise Phase2FreshOperatorError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def save_config_copy(config_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_fresh_context_measurement(config: Mapping[str, Any], device: torch.device):
    op_cfg = dict(config["context_operator"])
    base = p73.regime_config("rad5", device)
    base.update(
        {
            "seed": int(op_cfg["seed"]),
            "sampling_ratio": float(op_cfg.get("sampling_ratio", 0.05)),
            "pattern_type": str(op_cfg.get("pattern_type", "rademacher")),
            "matrix_normalization": str(op_cfg.get("matrix_normalization", base.get("matrix_normalization", "legacy_sqrt_m"))),
            "noise_std": float(op_cfg.get("noise_std", 0.0)),
            "use_final_dc_project": True,
            "output_range_mode": "clamp_eval_only",
            "num_workers": 0,
            "batch_size": int(config.get("batch_size", 8)),
        }
    )
    base = apply_experiment_defaults(base)
    measurement = make_measurement(base, device)
    return measurement, base


def _stable_index_order(indices: np.ndarray, salt: str) -> np.ndarray:
    import hashlib

    arr = np.asarray(indices, dtype=np.int64)
    keyed = [
        (
            hashlib.sha256(f"{salt}:{int(idx)}".encode("utf-8")).hexdigest(),
            int(idx),
        )
        for idx in arr.tolist()
    ]
    keyed.sort()
    return np.asarray([idx for _key, idx in keyed], dtype=np.int64)


def _select_fresh_split_indices(split_cfg: Mapping[str, Any]) -> tuple[np.ndarray, Any, dict[str, Any]]:
    source = str(split_cfg.get("source", "STL10 train+unlabeled"))
    count = int(split_cfg.get("sample_count", 16))
    if source == "STL10 train+unlabeled":
        offset = int(split_cfg.get("train_unlabeled_offset", 4096))
        train_indices_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
        if offset + count > train_indices_full.shape[0]:
            raise Phase2FreshOperatorError("FRESH_SPLIT_EXCEEDS_TRAIN_UNLABELED_INDEX_POOL")
        indices = train_indices_full[offset : offset + count]
        dataset = p69a.stl10_dataset("train+unlabeled")
        manifest = {
            "source": source,
            "offset": offset,
            "sample_count": count,
            "indices_sha256": p69a.sha256_np(indices),
            "full_train_sorted_sha256": p69a.sha256_np(train_indices_full, sort_int64=True),
            "purpose": "Phase 2 fresh-operator development or locked protocol according to config status.",
        }
        return np.asarray(indices, dtype=np.int64), dataset, manifest
    if source == "STL10 test":
        dataset = p69a.stl10_dataset("test")
        total = int(split_cfg.get("test_pool_count", len(dataset)))
        if total > len(dataset):
            raise Phase2FreshOperatorError(f"TEST_POOL_COUNT_EXCEEDS_DATASET:{total}:{len(dataset)}")
        pool = np.arange(total, dtype=np.int64)
        exclude_path = Path(
            str(
                split_cfg.get(
                    "exclude_indices_npy",
                    ROOT
                    / "outputs"
                    / "compatibility"
                    / "phase1_4ir_incident_recovery"
                    / "manifests"
                    / "final_locked_test_64_v4_indices.npy",
                )
            )
        )
        excluded = np.load(exclude_path).astype(np.int64) if exclude_path.exists() else np.zeros(0, dtype=np.int64)
        candidates = np.setdiff1d(pool, excluded, assume_unique=False)
        ordered = _stable_index_order(candidates, str(split_cfg.get("selection_salt", "PHASE2_LOCKED_TEST_SELECTION_V1")))
        offset = int(split_cfg.get("test_selection_offset", 0))
        if offset + count > ordered.shape[0]:
            raise Phase2FreshOperatorError(f"FRESH_TEST_SPLIT_EXCEEDS_AVAILABLE_POOL:{offset}:{count}:{ordered.shape[0]}")
        indices = ordered[offset : offset + count]
        overlap = int(np.intersect1d(indices, excluded).shape[0])
        if overlap:
            raise Phase2FreshOperatorError(f"FRESH_TEST_SPLIT_OVERLAPS_EXCLUDED_INDICES:{overlap}")
        manifest = {
            "source": source,
            "selection_mode": "stable_hash_ordered_test_minus_exclusions",
            "selection_salt": str(split_cfg.get("selection_salt", "PHASE2_LOCKED_TEST_SELECTION_V1")),
            "test_selection_offset": offset,
            "sample_count": count,
            "indices_sha256": p69a.sha256_np(indices),
            "test_pool_count": total,
            "excluded_indices_path": str(exclude_path),
            "excluded_indices_count": int(excluded.shape[0]),
            "excluded_overlap_count": overlap,
            "purpose": "Phase 2 locked test split; final-v4 source indices excluded.",
        }
        return np.asarray(indices, dtype=np.int64), dataset, manifest
    raise Phase2FreshOperatorError(f"UNSUPPORTED_PHASE2_SPLIT_SOURCE:{source}")


def make_fixed_total_context_measurement(
    config: Mapping[str, Any],
    *,
    witness_budget: int,
    device: torch.device,
):
    op_cfg = dict(config["context_operator"])
    total_m = int(op_cfg.get("total_m", 205))
    n = int(op_cfg.get("img_size", 64)) ** 2
    budget = int(witness_budget)
    if budget < 0 or budget >= total_m:
        raise Phase2FreshOperatorError(f"INVALID_FIXED_TOTAL_WITNESS_BUDGET:{budget}:{total_m}")
    context_m = total_m - budget
    total_ratio = float(total_m / n)
    A_total, total_meta = create_fixed_measurement_matrix(
        img_size=int(op_cfg.get("img_size", 64)),
        sampling_ratio=total_ratio,
        pattern_type=str(op_cfg.get("pattern_type", "rademacher")),
        device=device,
        seed=int(op_cfg["seed"]),
        matrix_normalization=str(op_cfg.get("matrix_normalization", "legacy_sqrt_m")),
        return_metadata=True,
    )
    A_context = A_total[:context_m].contiguous()
    base = p73.regime_config("rad5", device)
    base.update(
        {
            "seed": int(op_cfg["seed"]),
            "sampling_ratio": float(context_m / n),
            "pattern_type": str(op_cfg.get("pattern_type", "rademacher")),
            "matrix_normalization": str(op_cfg.get("matrix_normalization", base.get("matrix_normalization", "legacy_sqrt_m"))),
            "noise_std": float(op_cfg.get("noise_std", 0.0)),
            "use_final_dc_project": True,
            "output_range_mode": "clamp_eval_only",
            "num_workers": 0,
            "batch_size": int(config.get("batch_size", 8)),
        }
    )
    base = apply_experiment_defaults(base)
    measurement = make_measurement(base, device)
    override = measurement.set_A_override(
        A_context,
        metadata={
            "phase": "phase2_fixed_total_smoke",
            "role": "context_rows_only_candidate_generation",
            "total_m": total_m,
            "context_m": context_m,
            "withheld_witness_budget": budget,
            "total_A_sha256": __import__("hashlib").sha256(
                A_total.detach().cpu().numpy().astype(np.float32).tobytes()
            ).hexdigest(),
            "source_total_operator_seed": int(op_cfg["seed"]),
            "source_total_operator_pattern_type": str(op_cfg.get("pattern_type", "rademacher")),
        },
        rebuild_cache=True,
    )
    base["fixed_total_budget"] = {
        "total_m": total_m,
        "context_m": context_m,
        "witness_budget": budget,
        "context_sampling_ratio": float(context_m / n),
        "total_sampling_ratio": total_ratio,
        "total_operator_metadata": total_meta,
        "override": override,
    }
    return measurement, base, A_total.detach().cpu().numpy().astype(np.float32), A_context.detach().cpu().numpy().astype(np.float32)


def build_fresh_split(config: Mapping[str, Any], measurement, device: torch.device) -> dict[str, Any]:
    split_cfg = config["split"]
    count = int(split_cfg.get("sample_count", 16))
    indices, dataset, split_manifest = _select_fresh_split_indices(split_cfg)
    subset = p69b.source_subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=int(config.get("batch_size", 8)), shuffle=False, num_workers=0)
    xs, labels, seen_indices = [], [], []
    cursor = 0
    for batch in loader:
        x, label = batch
        bsz = int(x.shape[0])
        xs.append(x.float())
        labels.append(torch.as_tensor(label).long())
        seen_indices.append(torch.from_numpy(indices[cursor : cursor + bsz].astype(np.int64)))
        cursor += bsz
    x = torch.cat(xs, dim=0)
    labels_t = torch.cat(labels, dim=0)
    indices_t = torch.cat(seen_indices, dim=0)
    y_rows = []
    for start in range(0, x.shape[0], int(config.get("batch_size", 8))):
        xb = x[start : start + int(config.get("batch_size", 8))].to(device)
        y_rows.append(measurement.A_forward(measurement.flatten_img(xb)).detach().cpu())
    split = {
        "name": str(split_cfg.get("name", "phase2_fresh_dev_smoke")),
        "x": x,
        "y": torch.cat(y_rows, dim=0),
        "labels": labels_t,
        "indices": indices_t,
        "split_manifest": split_manifest,
    }
    return split


def score_torch_selector(artifact_path: Path, cache: dict[str, Any], device: torch.device) -> np.ndarray:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    cfg = artifact["model_config"]
    model = CompatibilityCritic(
        embed_dim=int(cfg["embed_dim"]),
        base_channels=int(cfg["base_channels"]),
        temperature=float(cfg["temperature"]),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    return p13.score_ranker(model, cache, device, artifact["training_recipe"]["preprocessing_mode"])


def score_sklearn_selector(artifact_path: Path, cache: dict[str, Any]) -> np.ndarray:
    artifact = joblib.load(artifact_path)
    x, names = p13.feature_matrix_for_cache(cache, artifact["mode"])
    if list(names) != list(artifact["feature_order"]):
        raise Phase2FreshOperatorError(f"SCALAR_FEATURE_ORDER_MISMATCH:{artifact_path}")
    scores = artifact["selected_model"].predict(x).astype(np.float32)
    return scores.reshape(cache["p0_error"].shape)


def score_frozen_selectors(cache: dict[str, Any], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {"status": "PASS", "artifact_dir": str(ARTIFACT_DIR), "selectors": {}}
    for key in ALL_SELECTOR_KEYS:
        if key.endswith("selector"):
            path = ARTIFACT_DIR / f"{key}.joblib"
            scores[key] = score_sklearn_selector(path, cache)
            kind = "sklearn"
        else:
            path = ARTIFACT_DIR / f"{key}.pt"
            scores[key] = score_torch_selector(path, cache, device)
            kind = "torch"
        if scores[key].shape != tuple(cache["p0_error"].shape):
            raise Phase2FreshOperatorError(f"SELECTOR_SCORE_SHAPE_MISMATCH:{key}:{scores[key].shape}:{tuple(cache['p0_error'].shape)}")
        audit["selectors"][key] = {
            "kind": kind,
            "path": str(path),
            "sha256": sha256_file(path),
            "score_shape": list(scores[key].shape),
            "score_sha256": __import__("hashlib").sha256(np.ascontiguousarray(scores[key]).tobytes()).hexdigest(),
        }
    return scores, audit


def candidate_feasibility_audit(cache: CandidateCache, measurement, device: torch.device) -> dict[str, Any]:
    canon = torch.from_numpy(cache.r[:, None, :] + cache.cand_n).reshape(cache.n * cache.k, cache.d).to(device)
    payload = torch.load(cache.path, map_location="cpu", weights_only=False)
    if "y" not in payload:
        raise Phase2FreshOperatorError(f"CANDIDATE_CACHE_MISSING_Y_FOR_FEASIBILITY:{cache.path}")
    y_arr = as_numpy(payload["y"], np.float32)
    if y_arr.shape[0] != cache.n:
        raise Phase2FreshOperatorError(f"CANDIDATE_CACHE_Y_SHAPE_MISMATCH:{y_arr.shape}:{cache.n}")
    y_rep = torch.from_numpy(np.repeat(y_arr, cache.k, axis=0)).to(device)
    rel = relative_measurement_error(canon.float(), y_rep.float(), measurement).detach().cpu().numpy()
    return {
        "status": "PASS" if float(np.max(rel)) < 1e-4 else "FAIL",
        "canonical_relmeaserr_max": float(np.max(rel)),
        "canonical_relmeaserr_mean": float(np.mean(rel)),
        "sample_count": cache.n,
        "candidate_count": cache.n * cache.k,
    }


def create_preregistration_text(config: Mapping[str, Any], gate: Mapping[str, Any], headroom: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2 Witnessed Candidate Selection Preregistration v0",
            "",
            "Status: development draft, not a locked test protocol.",
            "",
            "## Scientific Question",
            "",
            "Can additional witness measurements, unseen by the generator, improve single-candidate selection among context-feasible candidates without claiming to verify the entire context null space?",
            "",
            "## Primary Endpoint",
            "",
            "- Single selected candidate canonical P0-RMSE, reported with posterior mean and oracle best-of-K in the same table.",
            "- Secondary perception/spectral endpoints: clipped LPIPS, RAPSD, PSNR, SSIM.",
            "- Do not define success as beating posterior mean when posterior mean is within 5% of oracle headroom; instead classify this as posterior-near-oracle regime.",
            "",
            "## Candidate Protocol",
            "",
            f"- Context operator family: `{config['context_operator'].get('pattern_type')}` with seed `{config['context_operator'].get('seed')}` for this smoke.",
            "- Candidate generation only sees context measurements.",
            "- Witness rows are generated independently and used only for selection after candidate generation.",
            "- All sample linkage uses qualified sample_uid plus hashes; final-v4 is excluded from method development.",
            "",
            "## Comparisons",
            "",
            "random, posterior mean, scalar, sum-image, scratch dual, frozen compatibility selector, random witness, fixed low-frequency witness, adaptive witness, compatibility + witness, oracle.",
            "",
            "## Current Development Gate",
            "",
            f"- Gate decision: `{gate.get('decision')}`.",
            f"- Posterior near oracle: `{headroom.get('posterior_is_near_oracle_for_p0_rmse')}`.",
            "",
            "## Locked Test Requirement",
            "",
            "Before locked test: freeze operator seeds, split UIDs, candidate K, witness budgets, selectors, endpoint definitions, bootstrap seed, and one-shot scorer. The locked test must use fresh samples/operators not touched during this development smoke.",
            "",
        ]
    )


def run_fresh_operator_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.time()
    started_utc = now_utc()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/phase2_fresh_operator_smoke/dev_smoke_v1"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    save_config_copy(config_path, output_dir)
    device = resolve_device(str(config.get("device", "cuda")))
    measurement, base_config = make_fresh_context_measurement(config, device)
    generator, gen_config, ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
        Path(config.get("checkpoint", p12.PHASE79_CKPT)), base_config, measurement, device
    )
    if missing or unexpected:
        raise Phase2FreshOperatorError(f"GENERATOR_LOAD_NOT_STRICT:{missing}:{unexpected}")
    split = build_fresh_split(config, measurement, device)
    cache = p12.build_candidate_cache(
        generator,
        measurement,
        gen_config,
        split,
        out=output_dir,
        k=int(config.get("candidate_k", 16)),
        seed=int(config.get("candidate_seed", 920700)),
        device=device,
    )
    cache_path = output_dir / "candidate_cache" / f"{split['name']}_k{int(config.get('candidate_k', 16))}.pt"
    selector_scores, selector_audit = score_frozen_selectors(cache, device)
    cache_obj = load_candidate_cache(cache_path, split="fresh_dev_smoke")
    cache_info = cache_audit(cache_obj)
    feasibility = candidate_feasibility_audit(cache_obj, measurement, device)
    witness_config = {
        "witness": config["witness"],
        "statistics": config["statistics"],
        "quality": config.get("quality", {"compute_lpips": False}),
        "pilot_gate": config["pilot_gate"],
    }
    summaries, per_budget_rows, per_image_rows, witness_traces, headroom, quality = build_method_tables(
        cache_obj, selector_scores, witness_config
    )
    gate = compute_gate(summaries, per_budget_rows, witness_config)
    final_summary = final_v4_context_summary()
    leak = leakage_operator_audit(cache_obj, witness_config)
    leak["context_operator"].update(
        {
            "source": "fresh operator generated for Phase 2 smoke",
            "not_a_new_locked_test_operator": True,
            "A_sha256": __import__("hashlib").sha256(measurement.A.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest(),
        }
    )
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    operator_audit = {
        "status": "PASS",
        "img_size": int(measurement.img_size),
        "m": int(measurement.m),
        "n": int(measurement.n),
        "sampling_ratio": float(measurement.sampling_ratio),
        "pattern_type": str(measurement.pattern_type),
        "matrix_normalization": str(measurement.matrix_normalization),
        "seed": int(config["context_operator"]["seed"]),
        "A_sha256_float32": leak["context_operator"]["A_sha256"],
        "projector": projector.info_dict(),
        "checkpoint_state_key": state_key,
        "checkpoint_hash": sha256_file(Path(config.get("checkpoint", p12.PHASE79_CKPT))),
    }
    write_json(reports / "fresh_operator_audit.json", operator_audit)
    write_json(reports / "selector_transfer_audit.json", selector_audit)
    write_json(reports / "cache_audit.json", cache_info)
    write_json(reports / "candidate_feasibility_audit.json", feasibility)
    write_json(reports / "leakage_operator_audit.json", leak)
    write_json(reports / "post_final_v4_summary.json", final_summary)
    write_json(reports / "method_summaries.json", summaries)
    write_json(reports / "posterior_headroom_audit.json", headroom)
    write_json(reports / "quality_metrics.json", quality)
    write_json(reports / "pilot_gate.json", gate)
    write_csv(reports / "per_budget_metrics.csv", per_budget_rows)
    write_csv(reports / "per_image_methods.csv", per_image_rows)
    write_csv(reports / "quality_metrics.csv", quality.get("rows", []))
    write_json(reports / "witness_trace_sample.json", witness_traces)
    prereg = create_preregistration_text(config, gate, headroom)
    (reports / "phase2_preregistration_v0.md").write_text(prereg, encoding="utf-8")
    report = make_markdown_report(output_dir, witness_config, cache_obj, summaries, gate, final_summary, headroom, quality)
    report = report.replace("Phase 2 Witness Development Pilot", "Phase 2 Fresh-Operator Witness Smoke")
    (reports / "research_decision.md").write_text(report, encoding="utf-8")
    runtime = {
        "status": "PASS",
        "started_utc": started_utc,
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "repo_state": repo_state(),
    }
    write_json(reports / "runtime_and_memory.json", runtime)
    hashes = {
        path.name if path.parent == reports else path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in [
            output_dir / "config_used.yaml",
            reports / "research_decision.md",
            reports / "phase2_preregistration_v0.md",
            reports / "fresh_operator_audit.json",
            reports / "selector_transfer_audit.json",
            reports / "pilot_gate.json",
            reports / "posterior_headroom_audit.json",
            reports / "quality_metrics.csv",
            reports / "per_budget_metrics.csv",
            reports / "leakage_operator_audit.json",
        ]
    }
    write_json(reports / "package_hashes.json", hashes)
    summary = {
        "status": "PHASE2_FRESH_OPERATOR_SMOKE_COMPLETE",
        "output_dir": str(output_dir),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "operator_audit": operator_audit,
        "candidate_feasibility": feasibility,
        "gate": gate,
        "posterior_headroom_audit": headroom,
        "quality_lpips_status": quality.get("lpips_status"),
        "key_method_means": gate["method_means"],
        "artifact_hashes": hashes,
    }
    write_json(reports / "pilot_summary.json", summary)
    hashes["pilot_summary.json"] = sha256_file(reports / "pilot_summary.json")
    write_json(reports / "package_hashes.json", hashes)
    atomic_write_json(
        output_dir / "PHASE2_FRESH_OPERATOR_SMOKE_COMPLETE.json",
        {"status": summary["status"], "gate_decision": gate["decision"], "summary_sha256": hashes["pilot_summary.json"]},
    )
    return summary


def _method_quality_value(quality: Mapping[str, Any], method: str, field: str) -> float | None:
    val = quality.get("method_quality_means", {}).get(method, {}).get(field)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return None


def _method_p0_value(summaries: Mapping[str, Mapping[str, Any]], method: str) -> float | None:
    val = summaries.get(method, {}).get("mean_p0_rmse")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return None


def _fixed_total_focus_methods(budget: int, primary_selector: str, top_m: int) -> list[str]:
    if int(budget) <= 0:
        return ["random_expectation", "posterior_mean", primary_selector, "oracle_best_of_16"]
    return [
        "random_expectation",
        "posterior_mean",
        primary_selector,
        f"random_witness_b{budget}",
        f"fixed_lowfreq_witness_b{budget}",
        f"adaptive_witness_b{budget}",
        f"compat_top{top_m}_adaptive_witness_b{budget}",
        "oracle_best_of_16",
    ]


def _summarize_fixed_total_budget(
    *,
    budget: int,
    context_m: int,
    total_m: int,
    cache_obj: CandidateCache,
    summaries: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Any],
    primary_selector: str,
    top_m: int,
    cache_path: Path,
    cache_sha256: str,
    feasibility: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _fixed_total_focus_methods(budget, primary_selector, top_m):
        rows.append(
            {
                "witness_budget": int(budget),
                "context_m": int(context_m),
                "total_m": int(total_m),
                "method": method,
                "mean_context_p0_rmse": _method_p0_value(summaries, method),
                "canonical_unclipped_full_rmse_mean": _method_quality_value(
                    quality, method, "canonical_unclipped_full_rmse_mean"
                ),
                "canonical_clipped_psnr_mean": _method_quality_value(quality, method, "canonical_clipped_psnr_mean"),
                "canonical_clipped_ssim_mean": _method_quality_value(quality, method, "canonical_clipped_ssim_mean"),
                "canonical_clipped_lpips_mean": _method_quality_value(quality, method, "canonical_clipped_lpips_mean"),
                "canonical_clipped_rapsd_mean": _method_quality_value(quality, method, "canonical_clipped_rapsd_mean"),
                "cache_path": str(cache_path),
                "cache_sha256": cache_sha256,
                "sample_count": cache_obj.n,
                "candidate_count": cache_obj.n * cache_obj.k,
                "context_relmeaserr_max": feasibility.get("canonical_relmeaserr_max"),
            }
        )
    return rows


def fixed_total_gate(summary_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(r) for r in summary_rows]

    def row_for(budget: int, method: str) -> dict[str, Any] | None:
        for row in rows:
            if int(row["witness_budget"]) == int(budget) and str(row["method"]) == method:
                return row
        return None

    def full_rmse(row: Mapping[str, Any] | None) -> float | None:
        if row is None:
            return None
        val = row.get("canonical_unclipped_full_rmse_mean")
        return float(val) if isinstance(val, (int, float, np.integer, np.floating)) else None

    primary = str(config.get("witness", {}).get("primary_selector", "dm_fcc_seed3"))
    top_m = int(config.get("witness", {}).get("compatibility_prefilter_top_m", 4))
    budget_values = sorted({int(r["witness_budget"]) for r in rows})
    eval_budgets = [b for b in budget_values if b > 0]
    baseline = {
        "random_expectation": full_rmse(row_for(0, "random_expectation")),
        "posterior_mean": full_rmse(row_for(0, "posterior_mean")),
        primary: full_rmse(row_for(0, primary)),
        "oracle_best_of_16": full_rmse(row_for(0, "oracle_best_of_16")),
    }
    witness_candidates: list[dict[str, Any]] = []
    for budget in eval_budgets:
        for method in [
            f"random_witness_b{budget}",
            f"fixed_lowfreq_witness_b{budget}",
            f"adaptive_witness_b{budget}",
            f"compat_top{top_m}_adaptive_witness_b{budget}",
        ]:
            row = row_for(budget, method)
            val = full_rmse(row)
            if row is not None and val is not None:
                item = dict(row)
                item["full_rmse"] = val
                item["same_context_random_full_rmse"] = full_rmse(row_for(budget, "random_expectation"))
                item["same_context_posterior_full_rmse"] = full_rmse(row_for(budget, "posterior_mean"))
                witness_candidates.append(item)
    best = min(witness_candidates, key=lambda r: float(r["full_rmse"])) if witness_candidates else None

    def less(a: float | None, b: float | None) -> bool:
        return a is not None and b is not None and a < b

    adaptive_better_random = []
    compat_better_adaptive = []
    witness_better_same_context_random = []
    witness_better_full_context_random = []
    witness_better_full_context_posterior = []
    for budget in eval_budgets:
        adaptive = full_rmse(row_for(budget, f"adaptive_witness_b{budget}"))
        random_w = full_rmse(row_for(budget, f"random_witness_b{budget}"))
        compat = full_rmse(row_for(budget, f"compat_top{top_m}_adaptive_witness_b{budget}"))
        same_random = full_rmse(row_for(budget, "random_expectation"))
        for method in [
            f"random_witness_b{budget}",
            f"fixed_lowfreq_witness_b{budget}",
            f"adaptive_witness_b{budget}",
            f"compat_top{top_m}_adaptive_witness_b{budget}",
        ]:
            val = full_rmse(row_for(budget, method))
            witness_better_same_context_random.append(less(val, same_random))
            witness_better_full_context_random.append(less(val, baseline["random_expectation"]))
            witness_better_full_context_posterior.append(less(val, baseline["posterior_mean"]))
        adaptive_better_random.append(less(adaptive, random_w))
        compat_better_adaptive.append(less(compat, adaptive))

    best_full = None if best is None else float(best["full_rmse"])
    conditions = {
        "any_witness_beats_same_context_random_full_rmse": bool(any(witness_better_same_context_random)),
        "any_witness_beats_full_context_random_full_rmse": bool(any(witness_better_full_context_random)),
        "any_witness_beats_full_context_posterior_full_rmse": bool(any(witness_better_full_context_posterior)),
        "any_adaptive_beats_random_witness_same_budget_full_rmse": bool(any(adaptive_better_random)),
        "any_compat_prefilter_beats_adaptive_same_budget_full_rmse": bool(any(compat_better_adaptive)),
        "best_fixed_total_beats_full_context_posterior_full_rmse": bool(
            less(best_full, baseline["posterior_mean"])
        ),
        "best_fixed_total_beats_full_context_oracle_full_rmse": bool(less(best_full, baseline["oracle_best_of_16"])),
    }
    if conditions["best_fixed_total_beats_full_context_posterior_full_rmse"] and (
        conditions["any_adaptive_beats_random_witness_same_budget_full_rmse"]
        or conditions["any_compat_prefilter_beats_adaptive_same_budget_full_rmse"]
    ):
        decision = "FIXED_TOTAL_SIGNAL_READY_FOR_LARGER_DEVELOPMENT_PROTOCOL"
    elif conditions["any_witness_beats_same_context_random_full_rmse"] or conditions["any_witness_beats_full_context_random_full_rmse"]:
        decision = "CONTINUE_FIXED_TOTAL_DEVELOPMENT_DO_NOT_LOCK_TEST_YET"
    else:
        decision = "REDESIGN_FIXED_TOTAL_WITNESS_BEFORE_MORE_COMPUTE"
    return {
        "status": "PASS",
        "gate_scope": "fixed_total_development_smoke_only_not_confirmatory",
        "baseline_budget0_full_rmse": baseline,
        "budgets_evaluated": eval_budgets,
        "conditions": conditions,
        "best_fixed_total_method": None
        if best is None
        else {
            "witness_budget": int(best["witness_budget"]),
            "context_m": int(best["context_m"]),
            "method": str(best["method"]),
            "full_rmse": best_full,
            "delta_vs_full_context_posterior_full_rmse": None
            if baseline["posterior_mean"] is None
            else float(best_full - baseline["posterior_mean"]),
            "delta_vs_full_context_random_full_rmse": None
            if baseline["random_expectation"] is None
            else float(best_full - baseline["random_expectation"]),
        },
        "decision": decision,
        "interpretation": {
            "primary_cross_budget_metric": "canonical_unclipped_full_rmse_mean",
            "p0_rmse_note": "Context P0-RMSE is reported within each operator but is not the cross-budget net-benefit endpoint.",
            "measurement_claim_boundary": "Witness rows add observed directions; remaining joint null space is still unverifiable.",
        },
    }


def fixed_total_markdown_report(
    output_dir: Path,
    config: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    final_summary: Mapping[str, Any],
) -> str:
    rows = [dict(r) for r in summary_rows]
    lines = [
        "# Phase 2 Fixed-Total Witness Smoke",
        "",
        "## Scope",
        "",
        "This development smoke tests context/witness splitting under a fixed total row budget. It does not use final-v4 for method selection and is not a locked test.",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Total measurement rows: `{config['context_operator'].get('total_m', 205)}`",
        f"- Witness row source: `{config.get('fixed_total', {}).get('witness_row_source', 'method_specific_rows')}`",
        f"- Budgets evaluated: `{gate.get('budgets_evaluated')}`",
        f"- Final-v4 carried only as prior conclusion: `{final_summary.get('final_classification')}`",
        "",
        "## Cross-Budget Endpoint",
        "",
        "The cross-budget endpoint is canonical unclipped full RMSE. Context P0-RMSE is not directly compared across context operators because the null space changes when context rows are removed.",
        "",
        "## Full-RMSE Summary",
        "",
        "| Budget | Context m | Method | Full RMSE | Context P0-RMSE | LPIPS | RAPSD |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (int(r["witness_budget"]), str(r["method"]))):
        lines.append(
            "| {budget} | {context} | {method} | {rmse} | {p0} | {lpips} | {rapsd} |".format(
                budget=int(row["witness_budget"]),
                context=int(row["context_m"]),
                method=row["method"],
                rmse=row.get("canonical_unclipped_full_rmse_mean"),
                p0=row.get("mean_context_p0_rmse"),
                lpips=row.get("canonical_clipped_lpips_mean"),
                rapsd=row.get("canonical_clipped_rapsd_mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Decision: `{gate.get('decision')}`",
            f"- Best fixed-total method: `{gate.get('best_fixed_total_method')}`",
            f"- Conditions: `{json.dumps(gate.get('conditions', {}), sort_keys=True)}`",
            "",
            "## Claims",
            "",
            "- Allowed: this smoke can diagnose whether sacrificing context rows for witness rows has development signal on full-image distortion/perception metrics.",
            "- Not allowed: this smoke confirms fixed-total benefit, verifies complete P0, or justifies tuning final-v4.",
            "- Not allowed: context P0-RMSE differences across budgets are method-level net benefit without the full-RMSE/quality endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def run_fixed_total_smoke(config_path: Path = DEFAULT_FIXED_TOTAL_CONFIG) -> dict[str, Any]:
    started = time.time()
    started_utc = now_utc()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/phase2_fixed_total_smoke/dev_fixed_total_v1"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    save_config_copy(config_path, output_dir)
    device = resolve_device(str(config.get("device", "cuda")))
    total_m = int(config["context_operator"].get("total_m", 205))
    fixed_cfg = dict(config.get("fixed_total", {}))
    witness_row_source = str(fixed_cfg.get("witness_row_source", "method_specific_rows"))
    if witness_row_source not in {"method_specific_rows", "withheld_total_rows"}:
        raise Phase2FreshOperatorError(f"UNKNOWN_FIXED_TOTAL_WITNESS_ROW_SOURCE:{witness_row_source}")
    budgets = [0] + [int(b) for b in fixed_cfg.get("witness_budgets", [8, 16, 32])]
    primary_selector = str(config.get("witness", {}).get("primary_selector", "dm_fcc_seed3"))
    top_m = int(config.get("witness", {}).get("compatibility_prefilter_top_m", 4))
    all_rows: list[dict[str, Any]] = []
    budget_reports: list[dict[str, Any]] = []
    for budget in budgets:
        budget_dir = output_dir / f"budget_b{int(budget):03d}"
        ensure_dir(budget_dir / "reports")
        budget_config = dict(config)
        budget_config["split"] = dict(config["split"])
        budget_config["split"]["name"] = f"{config['split'].get('name', 'phase2_fixed_total_dev')}_b{int(budget):03d}"
        measurement, base_config, A_total, A_context = make_fixed_total_context_measurement(
            budget_config,
            witness_budget=int(budget),
            device=device,
        )
        context_m = int(measurement.m)
        generator, gen_config, _ckpt, state_key, missing, unexpected = p12.load_phase79_generator(
            Path(config.get("checkpoint", p12.PHASE79_CKPT)), base_config, measurement, device
        )
        if missing or unexpected:
            raise Phase2FreshOperatorError(f"GENERATOR_LOAD_NOT_STRICT_BUDGET_{budget}:{missing}:{unexpected}")
        split = build_fresh_split(budget_config, measurement, device)
        cache = p12.build_candidate_cache(
            generator,
            measurement,
            gen_config,
            split,
            out=budget_dir,
            k=int(config.get("candidate_k", 16)),
            seed=int(config.get("candidate_seed", 930700)) + int(budget) * 10000,
            device=device,
        )
        cache_path = budget_dir / "candidate_cache" / f"{split['name']}_k{int(config.get('candidate_k', 16))}.pt"
        selector_scores, selector_audit = score_frozen_selectors(cache, device)
        cache_obj = load_candidate_cache(cache_path, split=f"fixed_total_b{int(budget):03d}")
        cache_info = cache_audit(cache_obj)
        feasibility = candidate_feasibility_audit(cache_obj, measurement, device)
        if cache_info["p0_error_max_abs_recompute_diff"] > 1e-5:
            raise Phase2FreshOperatorError(f"FIXED_TOTAL_CACHE_P0_RECOMPUTE_DIFF_TOO_LARGE:{budget}:{cache_info['p0_error_max_abs_recompute_diff']}")
        witness_budget = max(1, int(budget))
        img_size = int(measurement.img_size)
        if witness_row_source == "withheld_total_rows" and int(budget) > 0:
            witness_rows = np.asarray(A_total[context_m:total_m], dtype=np.float32)
            if witness_rows.shape != (int(budget), cache_obj.d):
                raise Phase2FreshOperatorError(
                    f"WITHHELD_ROWS_SHAPE_MISMATCH:{budget}:{witness_rows.shape}:{(int(budget), cache_obj.d)}"
                )
            explicit_random_rows = witness_rows
            explicit_fixed_rows = witness_rows
            explicit_library_rows = witness_rows
            fixed_witness_label = "withheld rows from the same fresh total operator; random/fixed/adaptive variants share the same row pool"
        else:
            explicit_random_rows = make_rademacher_rows(
                witness_budget,
                cache_obj.d,
                int(config["witness"].get("random_seed", 93101)) + int(budget),
            )
            explicit_fixed_rows = make_dct2_lowfreq_rows(witness_budget, img_size)
            explicit_library_rows = make_rademacher_rows(
                max(int(config["witness"].get("adaptive_library_size", 256)), witness_budget),
                cache_obj.d,
                int(config["witness"].get("adaptive_library_seed", 93102)) + int(budget),
            )
            fixed_witness_label = "DCT-II low-frequency rows for fixed-total mixed-pattern comparison"
        witness_config = {
            "witness": {
                **dict(config["witness"]),
                "budgets": [witness_budget],
                "fixed_witness": fixed_witness_label,
                "fixed_total_witness_row_source": witness_row_source,
                "_explicit_random_rows": explicit_random_rows,
                "_explicit_fixed_rows": explicit_fixed_rows,
                "_explicit_library_rows": explicit_library_rows,
            },
            "statistics": config["statistics"],
            "quality": config.get("quality", {"compute_lpips": False}),
            "pilot_gate": {
                **dict(config["pilot_gate"]),
                "primary_budget": witness_budget,
                "low_budget": witness_budget,
            },
        }
        summaries, per_budget_rows, per_image_rows, witness_traces, headroom, quality = build_method_tables(
            cache_obj, selector_scores, witness_config
        )
        gate = compute_gate(summaries, per_budget_rows, witness_config)
        budget_rows = _summarize_fixed_total_budget(
            budget=int(budget),
            context_m=context_m,
            total_m=total_m,
            cache_obj=cache_obj,
            summaries=summaries,
            quality=quality,
            primary_selector=primary_selector,
            top_m=top_m,
            cache_path=cache_path,
            cache_sha256=sha256_file(cache_path),
            feasibility=feasibility,
        )
        all_rows.extend(budget_rows)
        operator_audit = {
            "status": "PASS",
            "budget": int(budget),
            "total_m": total_m,
            "context_m": context_m,
            "witness_budget": int(budget),
            "context_A_sha256_float32": __import__("hashlib").sha256(A_context.astype(np.float32).tobytes()).hexdigest(),
            "total_A_sha256_float32": __import__("hashlib").sha256(A_total.astype(np.float32).tobytes()).hexdigest(),
            "checkpoint_hash": sha256_file(Path(config.get("checkpoint", p12.PHASE79_CKPT))),
            "checkpoint_state_key": state_key,
            "projector": get_exact_projector(measurement, dtype=torch.float64, device=device).info_dict(),
        }
        budget_report = {
            "budget": int(budget),
            "context_m": context_m,
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "operator_audit": operator_audit,
            "cache_audit": cache_info,
            "candidate_feasibility": feasibility,
            "selector_transfer_audit": selector_audit,
            "method_summaries": summaries,
            "posterior_headroom_audit": headroom,
            "quality": quality,
            "pilot_gate": gate,
        }
        budget_reports.append(
            {
                "budget": int(budget),
                "context_m": context_m,
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "candidate_feasibility": feasibility,
                "pilot_gate_decision": gate["decision"],
            }
        )
        write_json(budget_dir / "reports" / "budget_report.json", budget_report)
        write_json(budget_dir / "reports" / "operator_audit.json", operator_audit)
        write_json(budget_dir / "reports" / "candidate_feasibility_audit.json", feasibility)
        write_json(budget_dir / "reports" / "selector_transfer_audit.json", selector_audit)
        write_json(budget_dir / "reports" / "method_summaries.json", summaries)
        write_json(budget_dir / "reports" / "posterior_headroom_audit.json", headroom)
        write_json(budget_dir / "reports" / "quality_metrics.json", quality)
        write_json(budget_dir / "reports" / "witness_trace_sample.json", witness_traces)
        write_csv(budget_dir / "reports" / "per_budget_metrics.csv", per_budget_rows)
        write_csv(budget_dir / "reports" / "per_image_methods.csv", per_image_rows)
    fixed_gate = fixed_total_gate(all_rows, config)
    final_summary = final_v4_context_summary()
    leak = {
        "status": "PASS",
        "phase": "phase2_fixed_total_development_smoke",
        "final_v4_consumed": final_summary.get("complete_marker", {}) != {},
        "final_v4_inputs_loaded": False,
        "final_v4_used_for_method_selection": False,
        "split": config["split"],
        "sample_identity": {
            "primary_key": "qualified sample_uid",
            "position_only_join_forbidden": True,
        },
        "budget_interpretation": {
            "fixed_total_budget_status": "RUN_IN_THIS_PILOT",
            "add_on_curve_status": "NOT_THE_SCOPE_OF_THIS_RUN",
        },
        "fixed_total_witness_row_source": witness_row_source,
        "row_source_interpretation": (
            "Strict same-pool split: witness rows are withheld from the same total operator used to define context."
            if witness_row_source == "withheld_total_rows"
            else "Mixed-pattern row allocation: candidate generation uses reduced random context rows; witness methods may allocate their remaining rows with different legal pattern families."
        ),
        "candidate_generation": "Each witness budget regenerates candidates from A_c only with context_m=total_m-budget.",
        "witness_use": "Witness rows are generated after candidate generation for selection only; they are not inputs, normalization stats, training labels, or early stopping signals.",
    }
    report_text = fixed_total_markdown_report(output_dir, config, all_rows, fixed_gate, final_summary)
    write_csv(reports / "fixed_total_summary.csv", all_rows)
    write_json(reports / "fixed_total_gate.json", fixed_gate)
    write_json(reports / "budget_reports_index.json", budget_reports)
    write_json(reports / "post_final_v4_summary.json", final_summary)
    write_json(reports / "leakage_operator_audit.json", leak)
    (reports / "research_decision_fixed_total.md").write_text(report_text, encoding="utf-8")
    runtime = {
        "status": "PASS",
        "started_utc": started_utc,
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "repo_state": repo_state(),
    }
    write_json(reports / "runtime_and_memory.json", runtime)
    hashes = {
        path.name if path.parent == reports else path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in [
            output_dir / "config_used.yaml",
            reports / "research_decision_fixed_total.md",
            reports / "fixed_total_gate.json",
            reports / "fixed_total_summary.csv",
            reports / "budget_reports_index.json",
            reports / "leakage_operator_audit.json",
        ]
    }
    write_json(reports / "package_hashes.json", hashes)
    summary = {
        "status": "PHASE2_FIXED_TOTAL_SMOKE_COMPLETE",
        "output_dir": str(output_dir),
        "gate": fixed_gate,
        "budget_reports": budget_reports,
        "artifact_hashes": hashes,
    }
    write_json(reports / "pilot_summary.json", summary)
    hashes["pilot_summary.json"] = sha256_file(reports / "pilot_summary.json")
    write_json(reports / "package_hashes.json", hashes)
    atomic_write_json(
        output_dir / "PHASE2_FIXED_TOTAL_SMOKE_COMPLETE.json",
        {"status": summary["status"], "gate_decision": fixed_gate["decision"], "summary_sha256": hashes["pilot_summary.json"]},
    )
    return summary


def create_fresh_brief_package(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    delivery = output_dir / "delivery"
    ensure_dir(delivery)
    package = delivery / "phase2_fresh_operator_smoke_gpt_brief.zip"
    files = [
        output_dir / "config_used.yaml",
        output_dir / "PHASE2_FRESH_OPERATOR_SMOKE_COMPLETE.json",
        reports / "research_decision.md",
        reports / "phase2_preregistration_v0.md",
        reports / "pilot_summary.json",
        reports / "pilot_gate.json",
        reports / "fresh_operator_audit.json",
        reports / "selector_transfer_audit.json",
        reports / "candidate_feasibility_audit.json",
        reports / "posterior_headroom_audit.json",
        reports / "quality_metrics.csv",
        reports / "per_budget_metrics.csv",
        reports / "leakage_operator_audit.json",
        reports / "package_hashes.json",
    ]
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(output_dir).as_posix())
    write_json(delivery / "fresh_brief_package_manifest.json", {"package": str(package), "sha256": sha256_file(package), "files": [str(p) for p in files]})
    return package


def create_fixed_total_brief_package(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    delivery = output_dir / "delivery"
    ensure_dir(delivery)
    package = delivery / "phase2_fixed_total_smoke_gpt_brief.zip"
    files = [
        output_dir / "config_used.yaml",
        output_dir / "PHASE2_FIXED_TOTAL_SMOKE_COMPLETE.json",
        reports / "research_decision_fixed_total.md",
        reports / "pilot_summary.json",
        reports / "fixed_total_gate.json",
        reports / "fixed_total_summary.csv",
        reports / "budget_reports_index.json",
        reports / "post_final_v4_summary.json",
        reports / "leakage_operator_audit.json",
        reports / "package_hashes.json",
    ]
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(output_dir).as_posix())
    write_json(
        delivery / "fixed_total_brief_package_manifest.json",
        {"package": str(package), "sha256": sha256_file(package), "files": [str(p) for p in files]},
    )
    return package
