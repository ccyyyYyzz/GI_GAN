from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.phase2_fresh_operator import (
    Phase2FreshOperatorError,
    candidate_feasibility_audit,
    create_preregistration_text,
    fixed_total_gate,
    resolve_device,
)
from src.phase2_witness import CandidateCache


class TinyOperator:
    def __init__(self) -> None:
        self.A = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        self.m = 1
        self.n = 2
        self.img_size = 1

    def A_forward(self, flat: torch.Tensor) -> torch.Tensor:
        return flat @ self.A.to(device=flat.device, dtype=flat.dtype).T

    def flatten_img(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def unflatten_img(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], 1, 1, 2)


def _tiny_cache(path: Path) -> CandidateCache:
    return CandidateCache(
        path=path,
        name="tiny",
        split="unit",
        x=np.array([[1.0, 2.0]], dtype=np.float32),
        r=np.array([[1.0, 0.0]], dtype=np.float32),
        true_n=np.array([[0.0, 2.0]], dtype=np.float32),
        cand_n=np.array([[[0.0, 1.0], [0.0, 3.0]]], dtype=np.float32),
        p0_error=np.array([[1.0, 1.0]], dtype=np.float64),
        deterministic_p0_error=np.array([1.0], dtype=np.float64),
        posterior_mean_p0_error=np.array([0.0], dtype=np.float64),
        indices=np.array([7], dtype=np.int64),
        labels=np.array([1], dtype=np.int64),
        sample_uids=["phase2_dev:tiny:split:unit:source_index:7:row:0"],
        file_sha256="unit",
    )


def test_resolve_device_falls_back_when_cuda_unavailable() -> None:
    device = resolve_device("cuda")
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_preregistration_text_declares_development_scope() -> None:
    config = {
        "context_operator": {"pattern_type": "rademacher", "seed": 123},
    }
    text = create_preregistration_text(
        config,
        {"decision": "CONTINUE_WITNESS_DEVELOPMENT_DO_NOT_LOCK_TEST_YET"},
        {"posterior_is_near_oracle_for_p0_rmse": False},
    )
    assert "development draft" in text
    assert "not a locked test protocol" in text
    assert "final-v4 is excluded from method development" in text


def test_candidate_feasibility_audit_uses_cache_y(tmp_path: Path) -> None:
    path = tmp_path / "cache.pt"
    torch.save({"y": torch.tensor([[1.0]], dtype=torch.float32)}, path)
    report = candidate_feasibility_audit(_tiny_cache(path), TinyOperator(), torch.device("cpu"))
    assert report["status"] == "PASS"
    assert report["candidate_count"] == 2
    assert report["canonical_relmeaserr_max"] < 1e-6


def test_candidate_feasibility_audit_requires_y(tmp_path: Path) -> None:
    path = tmp_path / "cache.pt"
    torch.save({"x": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}, path)
    try:
        candidate_feasibility_audit(_tiny_cache(path), TinyOperator(), torch.device("cpu"))
    except Phase2FreshOperatorError as exc:
        assert "CANDIDATE_CACHE_MISSING_Y_FOR_FEASIBILITY" in str(exc)
    else:
        raise AssertionError("candidate_feasibility_audit should fail when cache y is absent")


def test_fixed_total_gate_uses_full_rmse_cross_budget_endpoint() -> None:
    rows = [
        {
            "witness_budget": 0,
            "context_m": 205,
            "method": "random_expectation",
            "canonical_unclipped_full_rmse_mean": 0.30,
        },
        {
            "witness_budget": 0,
            "context_m": 205,
            "method": "posterior_mean",
            "canonical_unclipped_full_rmse_mean": 0.28,
        },
        {
            "witness_budget": 0,
            "context_m": 205,
            "method": "dm_fcc_seed3",
            "canonical_unclipped_full_rmse_mean": 0.29,
        },
        {
            "witness_budget": 0,
            "context_m": 205,
            "method": "oracle_best_of_16",
            "canonical_unclipped_full_rmse_mean": 0.27,
        },
        {
            "witness_budget": 8,
            "context_m": 197,
            "method": "random_expectation",
            "canonical_unclipped_full_rmse_mean": 0.31,
        },
        {
            "witness_budget": 8,
            "context_m": 197,
            "method": "random_witness_b8",
            "canonical_unclipped_full_rmse_mean": 0.285,
        },
        {
            "witness_budget": 8,
            "context_m": 197,
            "method": "fixed_lowfreq_witness_b8",
            "canonical_unclipped_full_rmse_mean": 0.286,
        },
        {
            "witness_budget": 8,
            "context_m": 197,
            "method": "adaptive_witness_b8",
            "canonical_unclipped_full_rmse_mean": 0.279,
        },
        {
            "witness_budget": 8,
            "context_m": 197,
            "method": "compat_top4_adaptive_witness_b8",
            "canonical_unclipped_full_rmse_mean": 0.278,
        },
    ]
    gate = fixed_total_gate(rows, {"witness": {"primary_selector": "dm_fcc_seed3", "compatibility_prefilter_top_m": 4}})
    assert gate["conditions"]["best_fixed_total_beats_full_context_posterior_full_rmse"]
    assert gate["conditions"]["any_compat_prefilter_beats_adaptive_same_budget_full_rmse"]
    assert gate["decision"] == "FIXED_TOTAL_SIGNAL_READY_FOR_LARGER_DEVELOPMENT_PROTOCOL"
