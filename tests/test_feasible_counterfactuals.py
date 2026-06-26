from __future__ import annotations

from dataclasses import dataclass

import torch

from src.compatibility_data import decompose_split, make_derangement, make_semihard_donors, verify_feasible_pairs
from src.measurement import GhostMeasurementOperator


@dataclass
class DummySplit:
    name: str
    x: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor


def _components(pattern: str, normalization: str):
    op = GhostMeasurementOperator(
        img_size=16,
        sampling_ratio=0.25,
        pattern_type=pattern,
        matrix_normalization=normalization,
        noise_std=0.0,
        lambda_dc=1e-3,
        device="cpu",
        seed=44,
    )
    gen = torch.Generator().manual_seed(444)
    x = torch.rand(24, 1, 16, 16, generator=gen) * 1.4 - 0.2
    split = DummySplit("train", x, torch.arange(24) % 10, torch.arange(100, 124))
    return op, decompose_split(split, op, device=torch.device("cpu"), batch_size=8, dtype=torch.float64)


def test_feasible_counterfactuals_rad_random_and_semihard() -> None:
    op, comp = _components("rademacher", "legacy_sqrt_m")
    random_donors = make_derangement(comp.size, seed=7)
    semihard = make_semihard_donors(comp, seed=8, pool_size=12)
    random_report = verify_feasible_pairs(comp, op, random_donors, device=torch.device("cpu"), max_pairs=24)
    semihard_report = verify_feasible_pairs(comp, op, semihard, device=torch.device("cpu"), max_pairs=24)
    assert random_report["pass_float32_proxy"]
    assert semihard_report["pass_float32_proxy"]
    assert int((random_donors == torch.arange(comp.size)).sum().item()) == 0


def test_feasible_counterfactuals_scrambled_hadamard() -> None:
    op, comp = _components("scrambled_hadamard", "orthonormal_rows")
    donors = make_derangement(comp.size, seed=11)
    report = verify_feasible_pairs(comp, op, donors, device=torch.device("cpu"), max_pairs=24)
    assert report["pass_float32_proxy"]
