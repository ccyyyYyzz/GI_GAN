from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _ConditionEncoder(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, base: torch.Tensor, direction: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([base, direction, direction.abs(), anchor], dim=1))


def _initial_logit(initial_weight: float, maximum_weight: float) -> float:
    probability = min(max(float(initial_weight) / float(maximum_weight), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(probability / (1.0 - probability))


def radial_frequency_masks(
    height: int, width: int, bands: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    fy = torch.fft.fftfreq(int(height), device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(int(width), device=device, dtype=dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    normalized = radius / radius.max().clamp_min(1.0e-12)
    edges = torch.linspace(0.0, 1.0, int(bands) + 1, device=device, dtype=dtype)
    masks = []
    for index in range(int(bands)):
        if index + 1 == int(bands):
            mask = (normalized >= edges[index]) & (normalized <= edges[index + 1])
        else:
            mask = (normalized >= edges[index]) & (normalized < edges[index + 1])
        masks.append(mask.to(dtype))
    result = torch.stack(masks)
    if not torch.allclose(result.sum(dim=0), torch.ones_like(normalized)):
        raise RuntimeError("RADIAL_MASK_PARTITION_FAILED")
    return result


class VQAECenteredResidualAdapter(nn.Module):
    """Small gate that can only reweight a supplied prior residual direction."""

    def __init__(
        self,
        *,
        architecture: str,
        maximum_weight: float = 0.35,
        initial_weight: float = 0.10,
        channels: int = 32,
        bands: int = 6,
    ) -> None:
        super().__init__()
        if architecture not in {"spatial", "spectral", "global"}:
            raise ValueError(f"UNKNOWN_ADAPTER_ARCHITECTURE:{architecture}")
        self.architecture = architecture
        self.maximum_weight = float(maximum_weight)
        self.bands = int(bands)
        self.encoder = _ConditionEncoder(channels=int(channels))
        output_channels = 1 if architecture in {"spatial", "global"} else self.bands
        self.head = nn.Conv2d(2 * int(channels), output_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(
            self.head.bias, _initial_logit(float(initial_weight), self.maximum_weight)
        )

    def forward(
        self, base: torch.Tensor, direction: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(base, direction, anchor)
        logits = self.head(features)
        if self.architecture == "spatial":
            weight = self.maximum_weight * torch.sigmoid(
                F.interpolate(logits, size=base.shape[-2:], mode="bilinear", align_corners=False)
            )
            return weight * direction, weight
        pooled = logits.mean(dim=(2, 3))
        weight = self.maximum_weight * torch.sigmoid(pooled)
        if self.architecture == "global":
            return weight[:, :, None, None] * direction, weight
        masks = radial_frequency_masks(
            direction.shape[-2],
            direction.shape[-1],
            self.bands,
            device=direction.device,
            dtype=direction.dtype,
        )
        multiplier = torch.einsum("bk,khw->bhw", weight, masks)
        spectrum = torch.fft.rfft2(direction.float(), norm="ortho")
        filtered = torch.fft.irfft2(
            spectrum * multiplier[:, None],
            s=direction.shape[-2:],
            norm="ortho",
        ).to(direction.dtype)
        return filtered, weight
