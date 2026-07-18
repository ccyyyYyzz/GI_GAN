from __future__ import annotations

import math

import torch

from src.complementary_poisson import (
    compile_equal_reference_photon_schedule,
    expected_complementary_rates,
    sample_complementary_poisson,
    split_signed_rows,
)


def test_signed_row_split_is_exact_and_nonnegative() -> None:
    rows = torch.tensor([[0.5, -0.25, 0.0, -0.75]], dtype=torch.float64)
    positive, negative = split_signed_rows(rows)
    assert torch.all(positive >= 0)
    assert torch.all(negative >= 0)
    assert torch.equal(positive - negative, rows)


def test_reference_schedule_spends_exact_fixed_budget() -> None:
    rows = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]],
        dtype=torch.float64,
    ) / 2.0
    reference = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    schedule = compile_equal_reference_photon_schedule(
        rows, reference, total_signal_photons=2.0e4, background_fraction=0.02
    )
    positive_rate, negative_rate = expected_complementary_rates(
        rows, reference[None, :], schedule
    )
    signal_total = (
        positive_rate
        + negative_rate
        - schedule.background_fraction * schedule.signal_photons_per_pair
    )
    assert torch.allclose(
        signal_total,
        torch.full_like(signal_total, schedule.signal_photons_per_pair),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert math.isclose(
        schedule.signal_photons_per_pair * rows.shape[0],
        schedule.total_signal_photons,
    )


def test_balanced_binary_rows_share_reference_exposure() -> None:
    rows = torch.tensor(
        [[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]],
        dtype=torch.float64,
    ) / 2.0
    reference = torch.tensor([0.2, 0.7, 0.5, 0.9], dtype=torch.float64)
    schedule = compile_equal_reference_photon_schedule(
        rows, reference, total_signal_photons=1.0e4
    )
    assert torch.allclose(
        schedule.exposure[0], schedule.exposure[1], rtol=0.0, atol=0.0
    )


def test_image_shaped_reference_is_accepted_as_one_object() -> None:
    rows = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float64)
    reference = torch.tensor([[0.2, 0.7], [0.5, 0.9]], dtype=torch.float64)
    schedule = compile_equal_reference_photon_schedule(
        rows, reference, total_signal_photons=1.0e4
    )
    assert schedule.reference_flux.shape == (1,)


def test_appending_one_pair_shortens_old_exposures_at_fixed_total_photons() -> None:
    old_rows = torch.tensor(
        [[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]],
        dtype=torch.float64,
    ) / 2.0
    new_row = torch.tensor([[1.0, -1.0, -1.0, 1.0]], dtype=torch.float64) / 2.0
    reference = torch.tensor([0.2, 0.7, 0.5, 0.9], dtype=torch.float64)
    old = compile_equal_reference_photon_schedule(
        old_rows, reference, total_signal_photons=1.0e5
    )
    augmented = compile_equal_reference_photon_schedule(
        torch.cat([old_rows, new_row]),
        reference,
        total_signal_photons=1.0e5,
    )
    assert torch.allclose(
        augmented.exposure[: old_rows.shape[0]],
        old.exposure * (old_rows.shape[0] / (old_rows.shape[0] + 1)),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_sample_mean_variance_and_seed_are_correct() -> None:
    rows = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float64)
    image = torch.tensor([[0.2, 0.4, 0.8, 0.1]], dtype=torch.float64)
    schedule = compile_equal_reference_photon_schedule(
        rows,
        image,
        total_signal_photons=2.0e3,
        background_fraction=0.01,
    )
    first = sample_complementary_poisson(
        rows, image, schedule, replicates=40000, seed=20260718
    )
    second = sample_complementary_poisson(
        rows, image, schedule, replicates=40000, seed=20260718
    )
    assert torch.equal(first.positive_counts, second.positive_counts)
    assert torch.equal(first.negative_counts, second.negative_counts)

    empirical_mean = first.bucket.mean()
    expected_mean = first.expected_bucket[0, 0, 0]
    analytic_variance = first.conditional_variance[0, 0, 0]
    standard_error = torch.sqrt(analytic_variance / first.bucket.numel())
    assert torch.abs(empirical_mean - expected_mean) <= 5.0 * standard_error

    empirical_variance = first.bucket.var(unbiased=True)
    assert torch.allclose(
        empirical_variance, analytic_variance, rtol=0.04, atol=0.0
    )
