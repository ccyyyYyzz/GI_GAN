"""Photon-accounted complementary measurements for signed GI/SPI rows.

A signed row ``a`` is implemented by two non-negative exposures
``a_+ = max(a, 0)`` and ``a_- = max(-a, 0)``.  Their independent Poisson
counts retain the physically relevant information that is lost when noise is
added directly to an already differenced bucket value.

The exposure compiler uses a training/reference mean image, never the test
truth, and gives every complementary pair the same expected signal-photon
allocation on that reference.  Consequently changing an arm from ``m`` to
``m+1`` pairs automatically shortens all exposures under a fixed total budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class ComplementaryPoissonError(ValueError):
    """Raised when rows, images, or a photon schedule are physically invalid."""


@dataclass(frozen=True)
class ComplementaryExposureSchedule:
    """Truth-free exposure schedule for a bank of signed rows."""

    exposure: torch.Tensor
    reference_flux: torch.Tensor
    signal_photons_per_pair: float
    total_signal_photons: float
    gain: float
    background_fraction: float


@dataclass(frozen=True)
class ComplementaryPoissonSample:
    """Counts, calibrated signed buckets, and their conditional moments."""

    positive_counts: torch.Tensor
    negative_counts: torch.Tensor
    bucket: torch.Tensor
    expected_bucket: torch.Tensor
    conditional_variance: torch.Tensor
    positive_rate: torch.Tensor
    negative_rate: torch.Tensor


def _rows_2d(rows: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(rows)
    if value.ndim != 2:
        raise ComplementaryPoissonError(f"ROWS_MUST_BE_2D:{tuple(value.shape)}")
    if value.numel() == 0:
        raise ComplementaryPoissonError("ROWS_MUST_BE_NONEMPTY")
    if not bool(torch.isfinite(value).all()):
        raise ComplementaryPoissonError("ROWS_MUST_BE_FINITE")
    return value


def _images_2d(images: torch.Tensor, pixels: int) -> torch.Tensor:
    value = torch.as_tensor(images)
    if value.numel() == 0 or value.numel() % int(pixels) != 0:
        raise ComplementaryPoissonError(
            f"IMAGE_PIXEL_MISMATCH:{value.numel()}:{int(pixels)}"
        )
    flat = value.reshape(-1, int(pixels))
    if not bool(torch.isfinite(flat).all()):
        raise ComplementaryPoissonError("IMAGES_MUST_BE_FINITE")
    if bool((flat < 0).any()):
        raise ComplementaryPoissonError("POISSON_OBJECT_MUST_BE_NONNEGATIVE")
    return flat


def split_signed_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the positive and negative non-negative patterns for signed rows."""

    value = _rows_2d(rows)
    return value.clamp_min(0), (-value).clamp_min(0)


def compile_equal_reference_photon_schedule(
    rows: torch.Tensor,
    reference_image: torch.Tensor,
    *,
    total_signal_photons: float,
    gain: float = 1.0,
    background_fraction: float = 0.01,
    minimum_reference_flux: float = 1.0e-12,
) -> ComplementaryExposureSchedule:
    """Compile row exposures without using the evaluated object's truth.

    ``reference_image`` may be one image or a batch.  A batch is averaged before
    scheduling and should normally be the training-set mean.  If there are
    ``m`` row pairs, every pair receives ``total_signal_photons / m`` expected
    signal photons on that reference image.
    """

    value = _rows_2d(rows)
    if float(total_signal_photons) <= 0:
        raise ComplementaryPoissonError("TOTAL_SIGNAL_PHOTONS_MUST_BE_POSITIVE")
    if float(gain) <= 0:
        raise ComplementaryPoissonError("GAIN_MUST_BE_POSITIVE")
    if float(background_fraction) < 0:
        raise ComplementaryPoissonError("BACKGROUND_FRACTION_MUST_BE_NONNEGATIVE")
    if float(minimum_reference_flux) <= 0:
        raise ComplementaryPoissonError("MINIMUM_REFERENCE_FLUX_MUST_BE_POSITIVE")

    reference = torch.as_tensor(
        reference_image, device=value.device, dtype=value.dtype
    )
    if reference.ndim == 1:
        reference = reference[None, :]
    reference_flat = _images_2d(reference, value.shape[1]).mean(dim=0)
    reference_flux = (value.abs() * reference_flat[None, :]).sum(dim=1)
    if bool((reference_flux <= float(minimum_reference_flux)).any()):
        bad = torch.nonzero(
            reference_flux <= float(minimum_reference_flux), as_tuple=False
        ).flatten()
        raise ComplementaryPoissonError(
            "REFERENCE_FLUX_TOO_SMALL:" + ",".join(str(int(i)) for i in bad)
        )

    per_pair = float(total_signal_photons) / int(value.shape[0])
    exposure = per_pair / (float(gain) * reference_flux)
    return ComplementaryExposureSchedule(
        exposure=exposure,
        reference_flux=reference_flux,
        signal_photons_per_pair=per_pair,
        total_signal_photons=float(total_signal_photons),
        gain=float(gain),
        background_fraction=float(background_fraction),
    )


def expected_complementary_rates(
    rows: torch.Tensor,
    images: torch.Tensor,
    schedule: ComplementaryExposureSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditional Poisson rates with shape ``[batch, rows]``."""

    value = _rows_2d(rows)
    flat = _images_2d(images, value.shape[1]).to(
        device=value.device, dtype=value.dtype
    )
    exposure = torch.as_tensor(
        schedule.exposure, device=value.device, dtype=value.dtype
    ).reshape(-1)
    if exposure.shape[0] != value.shape[0]:
        raise ComplementaryPoissonError(
            f"EXPOSURE_ROW_MISMATCH:{exposure.shape[0]}:{value.shape[0]}"
        )
    if not bool(torch.isfinite(exposure).all()) or bool((exposure <= 0).any()):
        raise ComplementaryPoissonError("EXPOSURES_MUST_BE_FINITE_POSITIVE")

    positive, negative = split_signed_rows(value)
    scale = float(schedule.gain) * exposure[None, :]
    background_each = (
        0.5
        * float(schedule.background_fraction)
        * float(schedule.signal_photons_per_pair)
    )
    positive_rate = scale * (flat @ positive.T) + background_each
    negative_rate = scale * (flat @ negative.T) + background_each
    return positive_rate, negative_rate


@torch.no_grad()
def sample_complementary_poisson(
    rows: torch.Tensor,
    images: torch.Tensor,
    schedule: ComplementaryExposureSchedule,
    *,
    replicates: int = 1,
    seed: int = 0,
) -> ComplementaryPoissonSample:
    """Sample complementary counts and calibrate their signed difference."""

    if int(replicates) <= 0:
        raise ComplementaryPoissonError("REPLICATES_MUST_BE_POSITIVE")
    value = _rows_2d(rows)
    flat = _images_2d(images, value.shape[1]).to(
        device=value.device, dtype=value.dtype
    )
    positive_rate, negative_rate = expected_complementary_rates(
        value, flat, schedule
    )
    expanded_positive = positive_rate[:, None, :].expand(
        -1, int(replicates), -1
    )
    expanded_negative = negative_rate[:, None, :].expand(
        -1, int(replicates), -1
    )
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    positive_counts = torch.poisson(expanded_positive, generator=generator)
    negative_counts = torch.poisson(expanded_negative, generator=generator)

    exposure = torch.as_tensor(
        schedule.exposure, device=value.device, dtype=value.dtype
    ).reshape(1, 1, -1)
    calibration = float(schedule.gain) * exposure
    bucket = (positive_counts - negative_counts) / calibration
    expected_bucket = (flat @ value.T)[:, None, :].expand_as(bucket)
    conditional_variance = (
        (positive_rate + negative_rate)[:, None, :] / calibration.square()
    ).expand_as(bucket)
    return ComplementaryPoissonSample(
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        bucket=bucket,
        expected_bucket=expected_bucket,
        conditional_variance=conditional_variance,
        positive_rate=positive_rate,
        negative_rate=negative_rate,
    )
