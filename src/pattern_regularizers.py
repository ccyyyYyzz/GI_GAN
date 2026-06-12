from __future__ import annotations

import torch
import torch.nn.functional as F


def energy_loss(P: torch.Tensor, target_transmission: float = 0.5) -> torch.Tensor:
    row_mean = P.mean(dim=1)
    target = torch.as_tensor(target_transmission, dtype=P.dtype, device=P.device)
    return torch.mean((row_mean - target) ** 2)


def decorrelation_loss(A_eff: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A_norm = A_eff / A_eff.norm(dim=1, keepdim=True).clamp_min(eps)
    gram = A_norm @ A_norm.T
    offdiag = gram - torch.diag_embed(torch.diagonal(gram))
    denom = max(1, A_eff.shape[0] * (A_eff.shape[0] - 1))
    return torch.sum(offdiag**2) / denom


def binary_loss(P_soft: torch.Tensor) -> torch.Tensor:
    return torch.mean(P_soft * (1.0 - P_soft))


def contrast_loss(P: torch.Tensor, target_contrast: float = 0.45) -> torch.Tensor:
    row_std = P.std(dim=1, unbiased=False)
    target = torch.as_tensor(target_contrast, dtype=P.dtype, device=P.device)
    return torch.mean((row_std - target) ** 2)


def bounded_contrast_loss(
    P: torch.Tensor,
    min_contrast: float = 0.05,
    max_contrast: float = 0.5,
) -> torch.Tensor:
    row_std = P.std(dim=1, unbiased=False)
    lo = torch.as_tensor(min_contrast, dtype=P.dtype, device=P.device)
    hi = torch.as_tensor(max_contrast, dtype=P.dtype, device=P.device)
    return torch.mean(F.relu(lo - row_std) ** 2 + F.relu(row_std - hi) ** 2)


def smoothness_loss(P: torch.Tensor, img_size: int) -> torch.Tensor:
    if img_size <= 1:
        return torch.zeros((), dtype=P.dtype, device=P.device)
    patterns = P.reshape(P.shape[0], 1, int(img_size), int(img_size))
    dx = torch.mean(torch.abs(patterns[:, :, :, 1:] - patterns[:, :, :, :-1]))
    dy = torch.mean(torch.abs(patterns[:, :, 1:, :] - patterns[:, :, :-1, :]))
    return dx + dy


def flip_margin_loss(logits: torch.Tensor, target_margin: float = 0.05) -> torch.Tensor:
    margin = logits.abs()
    target = torch.as_tensor(target_margin, dtype=logits.dtype, device=logits.device)
    return torch.mean(F.relu(margin - target) ** 2)


def flip_rate_loss(
    P_initial_hard: torch.Tensor,
    P_current_hard: torch.Tensor,
    min_target: float = 0.001,
    max_target: float = 0.05,
) -> torch.Tensor:
    flip_fraction = (P_initial_hard != P_current_hard).float().mean()
    lo = torch.as_tensor(min_target, dtype=flip_fraction.dtype, device=flip_fraction.device)
    hi = torch.as_tensor(max_target, dtype=flip_fraction.dtype, device=flip_fraction.device)
    return F.relu(lo - flip_fraction) ** 2 + F.relu(flip_fraction - hi) ** 2


def soft_flip_proxy_loss(
    P_soft: torch.Tensor,
    P_initial_hard: torch.Tensor,
    target_soft_delta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    soft_delta = torch.mean(torch.abs(P_soft - P_initial_hard))
    target = torch.as_tensor(target_soft_delta, dtype=P_soft.dtype, device=P_soft.device)
    return (soft_delta - target) ** 2, soft_delta


def secant_rip_loss(A_eff: torch.Tensor, x_batch: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x_batch.ndim != 4:
        raise ValueError("x_batch must have shape [B, C, H, W].")
    X = x_batch.reshape(x_batch.shape[0], -1)
    if X.shape[0] < 2:
        return torch.zeros((), dtype=X.dtype, device=X.device)
    perm = torch.roll(torch.randperm(X.shape[0], device=X.device), shifts=1)
    d = X - X[perm]
    d_norm = F.normalize(d, p=2, dim=1, eps=eps)
    Ad = d_norm @ A_eff.T
    energy = torch.sum(Ad**2, dim=1)
    return torch.mean((energy - 1.0) ** 2)


def total_pattern_loss(
    pattern_bank,
    x_batch: torch.Tensor,
    lambda_energy: float = 1.0,
    lambda_decorrelation: float = 0.1,
    lambda_binary: float = 0.01,
    lambda_secrip: float = 0.1,
    lambda_contrast: float = 0.0,
    target_contrast: float = 0.45,
    lambda_bounded_contrast: float = 0.0,
    continuous_min_contrast: float = 0.05,
    continuous_max_contrast: float = 0.5,
    lambda_smoothness: float = 0.0,
    lambda_flip_margin: float = 0.0,
    flip_margin_target: float = 0.05,
    lambda_soft_flip: float = 0.0,
    target_soft_flip_delta: float = 0.01,
    initial_pattern_state: dict | None = None,
    min_flip_fraction_target: float = 0.001,
    max_flip_fraction_target: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    P = pattern_bank.get_physical_patterns()
    P_soft = pattern_bank.get_soft_patterns()
    A_eff = pattern_bank.get_effective_A()

    energy = energy_loss(P, pattern_bank.target_transmission)
    decorrelation = decorrelation_loss(A_eff)
    binary = binary_loss(P_soft)
    secrip = secant_rip_loss(A_eff, x_batch)
    contrast = contrast_loss(P, target_contrast=target_contrast)
    bounded_contrast = bounded_contrast_loss(
        P,
        min_contrast=continuous_min_contrast,
        max_contrast=continuous_max_contrast,
    )
    smoothness = smoothness_loss(P, getattr(pattern_bank, "img_size", 0))
    flip_margin = flip_margin_loss(pattern_bank.logits, target_margin=flip_margin_target)
    soft_flip = torch.zeros((), dtype=P.dtype, device=P.device)
    soft_flip_delta = torch.zeros((), dtype=P.dtype, device=P.device)
    flip_rate = torch.zeros((), dtype=P.dtype, device=P.device)
    hard_flip_fraction = torch.zeros((), dtype=P.dtype, device=P.device)
    if initial_pattern_state and "P_initial_hard" in initial_pattern_state:
        P0 = initial_pattern_state["P_initial_hard"].detach().to(device=P.device, dtype=P.dtype)
        soft_flip, soft_flip_delta = soft_flip_proxy_loss(
            P_soft,
            P0,
            target_soft_delta=target_soft_flip_delta,
        )
        with torch.no_grad():
            P_hard = pattern_bank.get_hard_patterns().detach()
            hard_flip_fraction = (P_hard != P0).float().mean()
        flip_rate = flip_rate_loss(
            P0,
            pattern_bank.get_hard_patterns().detach(),
            min_target=min_flip_fraction_target,
            max_target=max_flip_fraction_target,
        ).to(dtype=P.dtype, device=P.device)

    total = (
        float(lambda_energy) * energy
        + float(lambda_decorrelation) * decorrelation
        + float(lambda_binary) * binary
        + float(lambda_secrip) * secrip
        + float(lambda_contrast) * contrast
        + float(lambda_bounded_contrast) * bounded_contrast
        + float(lambda_smoothness) * smoothness
        + float(lambda_flip_margin) * flip_margin
        + float(lambda_soft_flip) * soft_flip
    )
    details = {
        "pattern_energy_loss": energy,
        "pattern_decorrelation_loss": decorrelation,
        "pattern_binary_loss": binary,
        "pattern_secrip_loss": secrip,
        "pattern_contrast_loss": contrast,
        "pattern_bounded_contrast_loss": bounded_contrast,
        "pattern_smoothness_loss": smoothness,
        "pattern_flip_margin_loss": flip_margin,
        "pattern_soft_flip_loss": soft_flip,
        "pattern_soft_flip_delta": soft_flip_delta,
        "pattern_flip_rate_loss": flip_rate,
        "pattern_hard_flip_fraction": hard_flip_fraction,
        "pattern_total_loss": total,
    }
    return total, details
