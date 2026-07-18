from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DitheredPhaseBank:
    rows: torch.Tensor
    coherent_scale: torch.Tensor
    crest_factor: torch.Tensor
    maximum_entry: torch.Tensor


@dataclass(frozen=True)
class FactorialMomentEstimate:
    beta: torch.Tensor
    nuisance_variance: torch.Tensor
    beta_variance: torch.Tensor
    sample_variance: torch.Tensor
    mean_shot_variance: torch.Tensor


def compile_dithered_phase_bank(
    phase: torch.Tensor,
    *,
    pairs: int,
    rho: float,
    seed: int,
) -> DitheredPhaseBank:
    """Compile sample-specific complementary binary rows around a unit phase."""

    if phase.ndim != 2:
        raise ValueError("PHASE_MUST_BE_BATCH_BY_PIXEL")
    if int(pairs) < 2:
        raise ValueError("AT_LEAST_TWO_DITHER_PAIRS_REQUIRED")
    if not 0.0 < float(rho) < 1.0:
        raise ValueError("RHO_MUST_LIE_STRICTLY_BETWEEN_ZERO_AND_ONE")
    norm = torch.linalg.vector_norm(phase, dim=1)
    if bool((norm <= 1.0e-8).any()):
        raise ValueError("PHASE_NORM_TOO_SMALL")
    unit = phase / norm[:, None]
    maximum = unit.abs().amax(dim=1)
    pixels = unit.shape[1]
    crest = float(pixels) ** 0.5 * maximum
    probability = 0.5 * (1.0 + float(rho) * unit / maximum[:, None])
    generator = torch.Generator(device=phase.device).manual_seed(int(seed))
    uniform = torch.rand(
        phase.shape[0], int(pairs), pixels,
        generator=generator,
        device=phase.device,
        dtype=phase.dtype,
    )
    signs = torch.where(
        uniform < probability[:, None], torch.ones_like(uniform), -torch.ones_like(uniform)
    )
    rows = signs / float(pixels) ** 0.5
    coherent_scale = float(rho) / crest
    return DitheredPhaseBank(
        rows=rows,
        coherent_scale=coherent_scale,
        crest_factor=crest,
        maximum_entry=maximum,
    )


def estimate_factorial_moments(
    centered_bucket: torch.Tensor,
    shot_variance: torch.Tensor,
    coherent_scale: torch.Tensor,
) -> FactorialMomentEstimate:
    """Estimate coherent coefficient and shot-corrected nuisance variance."""

    if centered_bucket.ndim != 3 or shot_variance.shape != centered_bucket.shape:
        raise ValueError("EXPECTED_BATCH_REPLICATE_PAIR_TENSORS")
    if coherent_scale.shape != (centered_bucket.shape[0],):
        raise ValueError("COHERENT_SCALE_SHAPE_MISMATCH")
    sample_variance = centered_bucket.var(dim=2, unbiased=True)
    mean_shot_variance = shot_variance.mean(dim=2)
    scale = coherent_scale[:, None]
    beta = centered_bucket.mean(dim=2) / scale
    beta_variance = sample_variance / (centered_bucket.shape[2] * scale.square())
    nuisance = sample_variance - mean_shot_variance
    return FactorialMomentEstimate(
        beta=beta,
        nuisance_variance=nuisance,
        beta_variance=beta_variance,
        sample_variance=sample_variance,
        mean_shot_variance=mean_shot_variance,
    )


def positive_part_risk_shrink(
    coefficient: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    factor = 1.0 - variance / coefficient.square().clamp_min(1.0e-12)
    return coefficient * factor.clamp_min(0.0)


def line_box_interval(
    anchor: torch.Tensor, phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact coefficient interval keeping anchor + alpha*phase in [0,1]."""

    if anchor.shape != phase.shape or anchor.ndim != 2:
        raise ValueError("ANCHOR_PHASE_SHAPE_MISMATCH")
    positive = phase > 1.0e-12
    negative = phase < -1.0e-12
    lower = torch.full_like(anchor, float("-inf"))
    upper = torch.full_like(anchor, float("inf"))
    lower = torch.where(positive, -anchor / phase.clamp_min(1.0e-12), lower)
    upper = torch.where(positive, (1.0 - anchor) / phase.clamp_min(1.0e-12), upper)
    safe_negative = phase.clamp_max(-1.0e-12)
    lower = torch.where(negative, (1.0 - anchor) / safe_negative, lower)
    upper = torch.where(negative, -anchor / safe_negative, upper)
    return lower.amax(dim=1), upper.amin(dim=1)


def clip_to_line_box(
    coefficient: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    while lower.ndim < coefficient.ndim:
        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)
    return torch.maximum(torch.minimum(coefficient, upper), lower)
