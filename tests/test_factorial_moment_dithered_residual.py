from __future__ import annotations

import torch

from src.factorial_moment_dithered_residual import (
    clip_to_line_box,
    compile_dithered_phase_bank,
    estimate_factorial_moments,
    line_box_interval,
    positive_part_risk_shrink,
)


def test_dithered_bank_has_prescribed_population_mean() -> None:
    phase = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64) / 2.0
    bank = compile_dithered_phase_bank(phase, pairs=200000, rho=0.6, seed=4)
    empirical = bank.rows.mean(dim=1)
    expected = bank.coherent_scale[:, None] * phase
    torch.testing.assert_close(empirical, expected, atol=4.0e-3, rtol=0.0)


def test_factorial_estimator_removes_known_shot_variance() -> None:
    torch.manual_seed(5)
    batch, replicates, pairs = 2, 4000, 16
    scale = torch.tensor([0.25, 0.5], dtype=torch.float64)
    beta = torch.tensor([0.8, -0.4], dtype=torch.float64)
    nuisance = torch.tensor([0.09, 0.04], dtype=torch.float64)
    shot = torch.tensor([0.16, 0.25], dtype=torch.float64)
    noise = torch.randn(batch, replicates, pairs, dtype=torch.float64)
    values = scale[:, None, None] * beta[:, None, None] + noise * torch.sqrt(
        (nuisance + shot)[:, None, None]
    )
    shot_tensor = shot[:, None, None].expand_as(values)
    estimate = estimate_factorial_moments(values, shot_tensor, scale)
    torch.testing.assert_close(estimate.beta.mean(dim=1), beta, atol=0.01, rtol=0.0)
    torch.testing.assert_close(
        estimate.nuisance_variance.mean(dim=1), nuisance, atol=0.01, rtol=0.0
    )


def test_line_box_interval_and_risk_shrink_are_valid() -> None:
    anchor = torch.tensor([[0.2, 0.6, 0.4]], dtype=torch.float64)
    phase = torch.tensor([[0.5, -0.5, 0.25]], dtype=torch.float64)
    lower, upper = line_box_interval(anchor, phase)
    coefficient = clip_to_line_box(torch.tensor([[8.0, -8.0]]), lower, upper)
    images = anchor[:, None] + coefficient[:, :, None] * phase[:, None]
    assert float(images.min()) >= -1.0e-12
    assert float(images.max()) <= 1.0 + 1.0e-12
    shrunk = positive_part_risk_shrink(
        torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(shrunk, torch.tensor([1.5, 0.0]))
