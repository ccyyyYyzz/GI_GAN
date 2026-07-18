from __future__ import annotations

import math

import torch
from torch import nn

from src.vqae_centered_residual_adapter import radial_frequency_masks


def _initial_logit(initial_mix: float, maximum_mix: float) -> float:
    probability = min(
        max(float(initial_mix) / float(maximum_mix), 1.0e-4),
        1.0 - 1.0e-4,
    )
    return math.log(probability / (1.0 - probability))


class FiberResidualSpectralFusionGate(nn.Module):
    """Per-image spectral gate from a structural fiber solution to a proposal."""

    def __init__(
        self,
        *,
        channels: int = 32,
        bands: int = 6,
        maximum_mix: float = 1.0,
        initial_mix: float = 0.02,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.bands = int(bands)
        self.maximum_mix = float(maximum_mix)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(2 * c, self.bands)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(
            self.head.bias,
            _initial_logit(float(initial_mix), self.maximum_mix),
        )

    def forward(
        self,
        base: torch.Tensor,
        reference_correction: torch.Tensor,
        proposal_correction: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference = proposal_correction - reference_correction
        features = self.encoder(
            torch.cat([base, reference_correction, difference, anchor], dim=1)
        ).mean(dim=(2, 3))
        weights = self.maximum_mix * torch.sigmoid(self.head(features))
        masks = radial_frequency_masks(
            difference.shape[-2],
            difference.shape[-1],
            self.bands,
            device=difference.device,
            dtype=difference.dtype,
        )
        multiplier = torch.einsum("bk,khw->bhw", weights, masks)
        spectrum = torch.fft.rfft2(difference.float(), norm="ortho")
        admitted = torch.fft.irfft2(
            spectrum * multiplier[:, None],
            s=difference.shape[-2:],
            norm="ortho",
        ).to(difference.dtype)
        return reference_correction + admitted, weights
