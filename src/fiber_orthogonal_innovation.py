from __future__ import annotations

import torch


def fiber_orthogonal_innovation(
    structural_direction: torch.Tensor,
    filtered_innovation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Remove only the per-sample component parallel to the structural direction."""

    if structural_direction.shape != filtered_innovation.shape:
        raise ValueError("STRUCTURAL_INNOVATION_SHAPE_MISMATCH")
    if structural_direction.ndim != 2:
        raise ValueError("FLAT_BATCH_REQUIRED")
    denominator = structural_direction.square().sum(dim=1, keepdim=True)
    numerator = (filtered_innovation * structural_direction).sum(dim=1, keepdim=True)
    zero_threshold = torch.finfo(denominator.dtype).eps * structural_direction.shape[1]
    beta = torch.where(
        denominator > zero_threshold,
        numerator / denominator.clamp_min(zero_threshold),
        torch.zeros_like(numerator),
    )
    parallel = beta * structural_direction
    orthogonal = filtered_innovation - parallel
    return orthogonal, beta, {
        "parallel": parallel,
        "parallel_energy_fraction": parallel.square().sum(dim=1)
        / filtered_innovation.square().sum(dim=1).clamp_min(zero_threshold),
        "relative_orthogonality_residual": (
            (orthogonal * structural_direction).sum(dim=1).abs()
            / (
                orthogonal.norm(dim=1) * structural_direction.norm(dim=1)
            ).clamp_min(zero_threshold)
        ),
    }
