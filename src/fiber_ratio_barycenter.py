from __future__ import annotations

import torch


def radial_bound(residual: torch.Tensor, radius: float) -> torch.Tensor:
    """Project each flattened residual into the closed Euclidean radius ball."""

    if residual.ndim < 2:
        raise ValueError("RESIDUAL_REQUIRES_BATCH_DIMENSION")
    flat = residual.flatten(1)
    norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    scale = torch.clamp(float(radius) / norm.clamp_min(1.0e-12), max=1.0)
    return (flat * scale).reshape_as(residual)


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of batched vectors onto the probability simplex."""

    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError("SIMPLEX_INPUT_MUST_BE_BATCH_BY_COMPONENT")
    ordered, _ = torch.sort(value, dim=1, descending=True)
    cumulative = ordered.cumsum(dim=1) - 1.0
    divisor = torch.arange(
        1, value.shape[1] + 1, device=value.device, dtype=value.dtype
    ).unsqueeze(0)
    positive = ordered - cumulative / divisor > 0.0
    rho = positive.sum(dim=1).clamp_min(1) - 1
    theta = cumulative.gather(1, rho[:, None]) / (rho.to(value.dtype) + 1.0)[:, None]
    projected = torch.clamp(value - theta, min=0.0)
    return projected / projected.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


@torch.no_grad()
def oracle_simplex_weights(
    particles: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 1024,
    tolerance: float = 1.0e-9,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Truth-only ceiling for a convex barycenter of fixed residual particles.

    ``particles`` has shape ``[batch, particle, coordinate]`` and ``target`` has
    shape ``[batch, coordinate]``.  This helper is diagnostic only; deployed
    AFRB weights never receive the target.
    """

    if particles.ndim != 3 or target.ndim != 2:
        raise ValueError("ORACLE_EXPECTS_BKN_AND_BN")
    if particles.shape[0] != target.shape[0] or particles.shape[2] != target.shape[1]:
        raise ValueError("ORACLE_SHAPE_MISMATCH")
    if int(iterations) < 1:
        raise ValueError("ORACLE_ITERATIONS_MUST_BE_POSITIVE")
    work = particles.to(torch.float64)
    target_work = target.to(device=work.device, dtype=work.dtype)
    gram = torch.bmm(work, work.transpose(1, 2))
    linear = torch.bmm(work, target_work.unsqueeze(-1)).squeeze(-1)
    lipschitz = 2.0 * torch.linalg.eigvalsh(gram).amax(dim=1).clamp_min(1.0e-12)
    weights = torch.full(
        (work.shape[0], work.shape[1]),
        1.0 / float(work.shape[1]),
        device=work.device,
        dtype=work.dtype,
    )
    completed = 0
    max_change = float("inf")
    for step in range(int(iterations)):
        gradient = 2.0 * (torch.bmm(gram, weights.unsqueeze(-1)).squeeze(-1) - linear)
        updated = project_simplex(weights - gradient / lipschitz[:, None])
        max_change = float((updated - weights).abs().max().detach().cpu())
        weights = updated
        completed = step + 1
        if max_change <= float(tolerance):
            break
    fitted = torch.einsum("bk,bkn->bn", weights, work)
    uniform = work.mean(dim=1)
    return weights, {
        "iterations": completed,
        "max_weight_change": max_change,
        "mean_oracle_null_mse": float((fitted - target_work).square().mean().cpu()),
        "mean_uniform_null_mse": float((uniform - target_work).square().mean().cpu()),
        "mean_effective_particle_count": float(
            (weights.square().sum(dim=1).reciprocal()).mean().cpu()
        ),
    }


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("WEIGHTS_MUST_BE_BATCH_BY_PARTICLE")
    normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return normalized.square().sum(dim=1).reciprocal()
