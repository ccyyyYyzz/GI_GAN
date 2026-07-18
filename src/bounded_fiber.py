from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundedFiberProjection:
    image_flat: torch.Tensor
    iterations: int
    max_measurement_residual: float
    max_box_violation: float
    converged: bool


def _box_violation(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.relu(-x).amax(), torch.relu(x - 1.0).amax())


def project_box_and_fiber(
    proposal_flat: torch.Tensor,
    y: torch.Tensor,
    projector,
    *,
    max_iterations: int = 100,
    measurement_tolerance: float = 1e-8,
    box_tolerance: float = 1e-8,
    stop_early: bool = True,
) -> BoundedFiberProjection:
    """Dykstra projection onto ``{x: A x = y} intersect [0, 1]^n``.

    ``projector`` is an :class:`ExactRangeNullProjector`.  The calculation is
    performed in its registered dtype (normally float64).  With
    ``stop_early=False`` the loop count is fixed, so gradients can flow through
    the finite unrolling during a training diagnostic.  Production training is
    expected to use a soft box penalty and reserve this exact bounded projection
    for validation and delivered images.
    """
    if proposal_flat.ndim != 2 or proposal_flat.shape[1] != projector.n:
        raise ValueError(
            f"Expected proposals [B,{projector.n}], got {tuple(proposal_flat.shape)}."
        )
    if y.ndim != 2 or y.shape[1] != projector.m:
        raise ValueError(f"Expected measurements [B,{projector.m}], got {tuple(y.shape)}.")
    if proposal_flat.shape[0] != y.shape[0]:
        raise ValueError("Proposal and measurement batch sizes differ.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")

    x = proposal_flat.to(device=projector.device, dtype=projector.dtype)
    y_exact = y.to(device=projector.device, dtype=projector.dtype)
    affine_correction = torch.zeros_like(x)
    box_correction = torch.zeros_like(x)
    converged = False
    iterations = 0

    for iteration in range(int(max_iterations)):
        affine_input = x + affine_correction
        on_fiber = projector.audit_flat(affine_input, y_exact)
        affine_correction = affine_input - on_fiber

        box_input = on_fiber + box_correction
        x = box_input.clamp(0.0, 1.0)
        box_correction = box_input - x
        iterations = iteration + 1

        if stop_early:
            with torch.no_grad():
                measurement_residual = (projector.A_forward(x) - y_exact).abs().amax()
                box_residual = _box_violation(x)
                converged = bool(
                    measurement_residual <= float(measurement_tolerance)
                    and box_residual <= float(box_tolerance)
                )
            if converged:
                break

    measurement_residual = float(
        (projector.A_forward(x) - y_exact).abs().amax().detach().cpu()
    )
    box_residual = float(_box_violation(x).detach().cpu())
    if not stop_early:
        converged = (
            measurement_residual <= float(measurement_tolerance)
            and box_residual <= float(box_tolerance)
        )
    return BoundedFiberProjection(
        image_flat=x,
        iterations=iterations,
        max_measurement_residual=measurement_residual,
        max_box_violation=box_residual,
        converged=bool(converged),
    )
