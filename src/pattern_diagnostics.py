from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .utils import ensure_dir


def _detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


@torch.no_grad()
def _hard_patterns(pattern_bank) -> torch.Tensor:
    if hasattr(pattern_bank, "get_hard_patterns"):
        return pattern_bank.get_hard_patterns().detach()
    p_soft = pattern_bank.get_soft_patterns()
    return (p_soft > 0.5).float().detach()


@torch.no_grad()
def capture_initial_pattern_state(pattern_bank) -> dict[str, Any]:
    """Capture the initial learnable illumination state without n x n matrices."""

    was_training = bool(pattern_bank.training)
    pattern_bank.eval()
    A_initial = _detach_cpu(pattern_bank.get_effective_A())
    if was_training:
        pattern_bank.train()
    return {
        "P_initial_hard": _detach_cpu(_hard_patterns(pattern_bank)),
        "P_initial_soft": _detach_cpu(pattern_bank.get_soft_patterns()),
        "A_initial": A_initial,
        "logits_initial": _detach_cpu(pattern_bank.logits),
        "pattern_mode": pattern_bank.pattern_mode,
        "effective_A_mode": pattern_bank.effective_A_mode,
        "pattern_init": pattern_bank.init_type,
        "pattern_logit_abs_init": float(pattern_bank.pattern_logit_abs_init),
        "tau": float(pattern_bank.tau),
    }


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.detach().to(device=device, dtype=torch.float32)


def _scalar(value: torch.Tensor | float | int | str | None) -> float | int | str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return "missing"
        return float(value.detach().cpu().item())
    return float(value)


def _offdiag_corr(A: torch.Tensor) -> float:
    if A.ndim != 2 or A.shape[0] <= 1:
        return 0.0
    row_norm = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = row_norm @ row_norm.T
    offdiag = gram - torch.diag_embed(torch.diagonal(gram))
    denom = max(1, A.shape[0] * (A.shape[0] - 1))
    return float((offdiag.abs().sum() / denom).detach().cpu())


def _row_corr(A0: torch.Tensor, A1: torch.Tensor) -> torch.Tensor:
    A0n = A0 / A0.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A1n = A1 / A1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (A0n * A1n).sum(dim=1)


def _deterministic_secant_rip(A: torch.Tensor, x_batch: torch.Tensor, eps: float = 1e-12) -> float:
    if x_batch.ndim != 4 or x_batch.shape[0] < 2:
        return 0.0
    X = x_batch.reshape(x_batch.shape[0], -1)
    d = X - torch.roll(X, shifts=1, dims=0)
    d_norm = torch.nn.functional.normalize(d, p=2, dim=1, eps=eps)
    Ad = d_norm @ A.T
    energy = torch.sum(Ad**2, dim=1)
    return float(torch.mean((energy - 1.0) ** 2).detach().cpu())


@torch.no_grad()
def compare_pattern_states(
    pattern_bank,
    initial_state: dict[str, Any] | None,
    *,
    secant_batch: torch.Tensor | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Compare current pattern state against a captured initial state."""

    if not initial_state:
        return {
            "status": "missing_initial_pattern_state",
            "pattern_attribution_note": "Initial pattern state is missing; cannot audit pattern causality.",
        }

    device = pattern_bank.logits.device
    P0_hard = _to_device(initial_state["P_initial_hard"], device)
    P0_soft = _to_device(initial_state["P_initial_soft"], device)
    A0 = _to_device(initial_state["A_initial"], device)
    L0 = _to_device(initial_state["logits_initial"], device)

    P1_hard = _hard_patterns(pattern_bank).to(device)
    P1_soft = pattern_bank.get_soft_patterns().detach()
    was_training = bool(pattern_bank.training)
    pattern_bank.eval()
    A1 = pattern_bank.get_effective_A().detach()
    if was_training:
        pattern_bank.train()
    L1 = pattern_bank.logits.detach()

    flips = (P1_hard != P0_hard).float()
    row_flips = flips.mean(dim=1)
    hard_flip_count = int(flips.sum().detach().cpu().item())
    hard_flip_fraction = float(flips.mean().detach().cpu())
    A_delta = A1 - A0
    A_rel_fro_delta = float(A_delta.norm().detach().cpu() / A0.norm().clamp_min(1e-12).detach().cpu())
    A_max_abs_delta = float(A_delta.abs().max().detach().cpu())
    A_cos = float(
        ((A0.reshape(-1) * A1.reshape(-1)).sum() / (A0.norm() * A1.norm()).clamp_min(1e-12))
        .detach()
        .cpu()
    )
    row_corr = _row_corr(A0, A1)

    offdiag0 = _offdiag_corr(A0)
    offdiag1 = _offdiag_corr(A1)
    secant_initial: float | str = "missing"
    secant_final: float | str = "missing"
    if secant_batch is not None:
        batch = secant_batch.to(device=device, dtype=torch.float32)
        secant_initial = _deterministic_secant_rip(A0, batch)
        secant_final = _deterministic_secant_rip(A1, batch)

    soft_delta = P1_soft - P0_soft
    logits_delta = L1 - L0
    margin = (P1_soft - 0.5).abs()
    soft_flip_delta = torch.mean(torch.abs(P1_soft - P0_hard))
    attribution = (
        "No hard pattern change detected; improvement cannot be attributed to physical binary pattern changes."
        if hard_flip_fraction < 1e-6 and A_rel_fro_delta < 1e-6
        else "Pattern changed; compare against G-only control to estimate pattern-specific contribution."
    )

    result: dict[str, Any] = {
        "status": "ok",
        "pattern_mode": getattr(pattern_bank, "pattern_mode", ""),
        "effective_A_mode": getattr(pattern_bank, "effective_A_mode", ""),
        "pattern_physical_type": pattern_bank.get_pattern_stats().get("pattern_physical_type", "missing"),
        "pattern_init": getattr(pattern_bank, "init_type", ""),
        "pattern_logit_abs_init": float(getattr(pattern_bank, "pattern_logit_abs_init", 0.0)),
        "tau_initial": _scalar(initial_state.get("tau")),
        "tau_final": float(getattr(pattern_bank, "tau", 0.0)),
        "hard_flip_count": hard_flip_count,
        "hard_flip_fraction": hard_flip_fraction,
        "hard_hamming_distance": hard_flip_count,
        "row_flip_mean": float(row_flips.mean().detach().cpu()),
        "row_flip_max": float(row_flips.max().detach().cpu()),
        "row_flip_min": float(row_flips.min().detach().cpu()),
        "rows_with_any_flip_fraction": float((row_flips > 0).float().mean().detach().cpu()),
        "soft_l1_delta": float(soft_delta.abs().sum().detach().cpu()),
        "soft_l2_delta": float(soft_delta.norm().detach().cpu()),
        "soft_flip_delta": float(soft_flip_delta.detach().cpu()),
        "soft_max_abs_delta": float(soft_delta.abs().max().detach().cpu()),
        "logits_l2_delta": float(logits_delta.norm().detach().cpu()),
        "logits_max_abs_delta": float(logits_delta.abs().max().detach().cpu()),
        "margin_to_threshold_mean": float(margin.mean().detach().cpu()),
        "margin_to_threshold_min": float(margin.min().detach().cpu()),
        "near_threshold_fraction_0p05": float((margin < 0.05).float().mean().detach().cpu()),
        "near_threshold_fraction_0p10": float((margin < 0.10).float().mean().detach().cpu()),
        "A_rel_fro_delta": A_rel_fro_delta,
        "A_max_abs_delta": A_max_abs_delta,
        "A_cosine_initial_final": A_cos,
        "row_corr_initial_final_mean": float(row_corr.mean().detach().cpu()),
        "row_corr_initial_final_min": float(row_corr.min().detach().cpu()),
        "secant_rip_initial": secant_initial,
        "secant_rip_final": secant_final,
        "secant_rip_delta": (
            float(secant_final - secant_initial)
            if isinstance(secant_initial, float) and isinstance(secant_final, float)
            else "missing"
        ),
        "offdiag_corr_initial": offdiag0,
        "offdiag_corr_final": offdiag1,
        "offdiag_corr_delta": offdiag1 - offdiag0,
        "pattern_attribution_note": attribution,
    }
    if config:
        result.update(
            {
                "freeze_patterns": bool(config.get("freeze_patterns", False)),
                "freeze_generator_all": bool(config.get("freeze_generator_all", False)),
                "freeze_discriminator_all": bool(config.get("freeze_discriminator_all", False)),
                "pattern_trainable": not (
                    bool(config.get("freeze_patterns", False))
                    or not bool(config.get("pattern_requires_grad", True))
                ),
                "lr_patterns": float(config.get("lr_patterns", 0.0)),
                "pattern_update_every": int(config.get("pattern_update_every", 1)),
            }
        )
    return result


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_pattern_diagnostics_json(diagnostics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(diagnostics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


@torch.no_grad()
def save_pattern_change_visualization(
    initial_P: torch.Tensor,
    final_P: torch.Tensor,
    path: str | Path,
    img_size: int,
    max_patterns: int = 32,
) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    P0 = initial_P.detach().cpu().float()
    P1 = final_P.detach().cpu().float()
    flips = (P0 != P1).float()
    count = min(int(max_patterns), P0.shape[0])
    fig, axes = plt.subplots(count, 3, figsize=(6.0, max(2.0, count * 1.25)))
    if count == 1:
        axes = axes.reshape(1, 3)
    titles = ["Initial Pattern", "Final Pattern", "Flip Map"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9)
    for idx in range(count):
        for col, data in enumerate((P0, P1, flips)):
            cmap = "magma" if col == 2 else "gray"
            axes[idx, col].imshow(data[idx].reshape(img_size, img_size), cmap=cmap, vmin=0.0, vmax=1.0)
            axes[idx, col].set_xticks([])
            axes[idx, col].set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
