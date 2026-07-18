from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _initial_logit(initial_weight: float, maximum_weight: float) -> float:
    probability = min(
        max(float(initial_weight) / float(maximum_weight), 1.0e-4),
        1.0 - 1.0e-4,
    )
    return math.log(probability / (1.0 - probability))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(int(channels))
        self.net = nn.Sequential(
            nn.GroupNorm(groups, int(channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(channels), int(channels), 3, padding=1),
            nn.GroupNorm(groups, int(channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(channels), int(channels), 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class FiberResidualPhaseGenerator(nn.Module):
    """Identity-initialized local rotation of a supplied fiber residual.

    The module does not perform the physical null projection itself.  Callers
    project its raw correction into the measurement null space before use.
    """

    def __init__(
        self,
        *,
        channels: int = 32,
        maximum_weight: float = 0.35,
        initial_weight: float = 0.10,
        rotation_scale: float = 0.50,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.maximum_weight = float(maximum_weight)
        self.rotation_scale = float(rotation_scale)
        self.stem = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.SiLU(inplace=True),
            _ResidualBlock(c),
        )
        self.down = nn.Sequential(
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(2 * c), 2 * c),
            nn.SiLU(inplace=True),
            _ResidualBlock(2 * c),
            _ResidualBlock(2 * c),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c, c, 3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.SiLU(inplace=True),
            _ResidualBlock(c),
        )
        self.weight_head = nn.Conv2d(c, 1, 1)
        self.phase_head = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(
            self.weight_head.bias,
            _initial_logit(float(initial_weight), self.maximum_weight),
        )
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)

    def forward(
        self,
        base: torch.Tensor,
        direction: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = torch.cat([base, direction, direction.abs(), anchor], dim=1)
        shallow = self.stem(inputs)
        deep = self.down(shallow)
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        features = self.fuse(torch.cat([shallow, deep], dim=1))
        weight = self.maximum_weight * torch.sigmoid(self.weight_head(features))
        direction_rms = direction.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1.0e-6)
        rotation = (
            self.rotation_scale
            * direction_rms
            * torch.tanh(self.phase_head(features))
        )
        correction = weight * direction + rotation
        return correction, {
            "weight": weight,
            "rotation": rotation,
            "direction_rms": direction_rms,
        }


def high_pass(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = int(kernel_size) // 2
    return image - F.avg_pool2d(
        image,
        kernel_size=int(kernel_size),
        stride=1,
        padding=padding,
    )


class ConditionalHighPassDiscriminator(nn.Module):
    """Patch discriminator for texture conditional on the stable GI estimate."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(2, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(2 * c), 2 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(4 * c), 4 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * c, 1, 3, padding=1),
        )

    def forward(self, base: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([base, high_pass(image)], dim=1))


def hinge_discriminator_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
