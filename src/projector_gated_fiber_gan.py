from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from src.gauge_geometry import GaugeGeometry


class FiberGANShapeError(ValueError):
    pass


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, int(channels))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, int(channels))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ProjectorQueryStep(nn.Module):
    """The single shared U-Net cell specified for PQBF-GAN."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
        )
        self.query_head = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.correction_head = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        last = self.correction_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        current: torch.Tensor,
        anchor: torch.Tensor,
        uncertainty: torch.Tensor,
        tau: torch.Tensor,
        geometry: GaugeGeometry,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e0 = self.stem(torch.cat([current, anchor, uncertainty, tau], dim=1))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        u1 = F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e1], dim=1))
        u2 = F.interpolate(u1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        trunk = self.up2(torch.cat([u2, e0], dim=1))
        probes = torch.tanh(self.query_head(trunk))
        visible = geometry.project_feature_maps(probes, null=False)
        invisible = geometry.project_feature_maps(probes, null=True)
        raw_correction = torch.tanh(
            self.correction_head(torch.cat([trunk, visible, invisible], dim=1))
        )
        return raw_correction, {
            "probe_visible_rms": visible.square().mean().sqrt(),
            "probe_invisible_rms": invisible.square().mean().sqrt(),
        }


@dataclass(frozen=True)
class FiberGeneratorOutput:
    raw_image: torch.Tensor
    null_residual: torch.Tensor
    raw_corrections: tuple[torch.Tensor, ...]
    raw_states: tuple[torch.Tensor, ...]
    diagnostics: dict[str, torch.Tensor]


class ProjectorGatedFiberGenerator(nn.Module):
    """Three-step gauge-invariant projector-query generator."""

    def __init__(
        self,
        geometry: GaugeGeometry,
        *,
        steps: int = 3,
        step_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        self.geometry = geometry
        self.steps = int(steps)
        self.step_scale = float(step_scale)
        self.shared_step = ProjectorQueryStep()

    def forward(
        self,
        anchor: torch.Tensor,
        uncertainty: torch.Tensor,
        *,
        geometry: GaugeGeometry | None = None,
    ) -> FiberGeneratorOutput:
        if anchor.ndim != 4 or anchor.shape[1] != 1:
            raise FiberGANShapeError(f"EXPECTED_ANCHOR_B1HW:{tuple(anchor.shape)}")
        if uncertainty.shape[0] == 1 and anchor.shape[0] != 1:
            uncertainty = uncertainty.expand(anchor.shape[0], -1, -1, -1)
        if uncertainty.shape != anchor.shape:
            raise FiberGANShapeError("uncertainty must match anchor shape")
        active_geometry = geometry or self.geometry
        if anchor.shape[-2] * anchor.shape[-1] != active_geometry.n:
            raise FiberGANShapeError("anchor spatial size does not match geometry")

        current = anchor
        corrections: list[torch.Tensor] = []
        states: list[torch.Tensor] = [anchor]
        diagnostics: dict[str, torch.Tensor] = {}
        for index in range(self.steps):
            tau_value = 0.0 if self.steps == 1 else index / float(self.steps - 1)
            tau = torch.full_like(anchor, tau_value)
            raw, step_diagnostics = self.shared_step(
                current, anchor, uncertainty, tau, active_geometry
            )
            null_update = active_geometry.project_feature_maps(raw, null=True)
            current = current + self.step_scale * null_update
            corrections.append(raw)
            states.append(current)
            diagnostics.update(
                {f"step{index + 1}_{key}": value for key, value in step_diagnostics.items()}
            )
            diagnostics[f"step{index + 1}_null_rms"] = null_update.square().mean().sqrt()
        return FiberGeneratorOutput(
            raw_image=current,
            null_residual=current - anchor,
            raw_corrections=tuple(corrections),
            raw_states=tuple(states),
            diagnostics=diagnostics,
        )


class FiberConditionalDiscriminator(nn.Module):
    """Single spectral-normalized PatchGAN conditioned on the bounded anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    spectral_norm(nn.Conv2d(3, 32, 4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
                    nn.GroupNorm(8, 64),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
                    nn.GroupNorm(8, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
                    nn.GroupNorm(8, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            ]
        )
        self.logits = spectral_norm(nn.Conv2d(128, 1, kernel_size=3, padding=1))

    def forward(
        self,
        anchor: torch.Tensor,
        image: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if anchor.shape != image.shape:
            raise FiberGANShapeError("anchor and image shapes differ")
        h = torch.cat([anchor, image, image - anchor], dim=1)
        features: list[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            features.append(h)
        logits = self.logits(h)
        if return_features:
            return logits, tuple(features)
        return logits


class ComplementaryDCTBucketDiscriminator(nn.Module):
    """Critic for the unmeasured part of an orthonormal DCT bucket spectrum.

    The acquired low-frequency DCT coefficients are not presented as fake
    evidence: they remain available only through the measurement-conditioned
    anchor.  The critic therefore learns the conditional distribution of the
    complementary (virtual) bucket coefficients instead of judging images in
    the detector-blind spatial domain.
    """

    def __init__(self, *, img_size: int, acquired_non_dc: int) -> None:
        super().__init__()
        size = int(img_size)
        if size <= 0:
            raise FiberGANShapeError(f"INVALID_DCT_SIZE:{size}")
        coords = [
            (u, v)
            for u in range(size)
            for v in range(size)
            if not (u == 0 and v == 0)
        ]
        coords.sort(
            key=lambda uv: (
                uv[0] * uv[0] + uv[1] * uv[1],
                uv[0] + uv[1],
                uv[0],
                uv[1],
            )
        )
        if not 0 <= int(acquired_non_dc) < len(coords):
            raise FiberGANShapeError(
                f"INVALID_ACQUIRED_DCT_COUNT:{acquired_non_dc}:{len(coords)}"
            )
        index = torch.arange(size, dtype=torch.float64)
        frequency = torch.arange(size, dtype=torch.float64)[:, None]
        scale = torch.full((size, 1), (2.0 / size) ** 0.5, dtype=torch.float64)
        scale[0, 0] = (1.0 / size) ** 0.5
        dct = scale * torch.cos(
            torch.pi * (2.0 * index[None, :] + 1.0) * frequency / (2.0 * size)
        )
        complement = torch.ones((1, 1, size, size), dtype=torch.float32)
        complement[0, 0, 0, 0] = 0.0
        for u, v in coords[: int(acquired_non_dc)]:
            complement[0, 0, int(u), int(v)] = 0.0
        self.register_buffer("dct", dct.float(), persistent=True)
        self.register_buffer("complement", complement, persistent=True)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    spectral_norm(nn.Conv2d(4, 32, 4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
                    nn.GroupNorm(8, 64),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
                    nn.GroupNorm(8, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
                    nn.GroupNorm(8, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            ]
        )
        self.logits = spectral_norm(nn.Conv2d(128, 1, kernel_size=3, padding=1))

    def _dct2(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise FiberGANShapeError(f"EXPECTED_IMAGE_B1HW:{tuple(image.shape)}")
        if image.shape[-2:] != self.complement.shape[-2:]:
            raise FiberGANShapeError(
                f"DCT_IMAGE_SIZE_MISMATCH:{tuple(image.shape[-2:])}:"
                f"{tuple(self.complement.shape[-2:])}"
            )
        matrix = self.dct.to(dtype=image.dtype)
        return torch.matmul(torch.matmul(matrix, image[:, 0]), matrix.t()).unsqueeze(1)

    @staticmethod
    def _compress(coefficients: torch.Tensor) -> torch.Tensor:
        # DCT buckets have a large DC-to-detail dynamic range.  asinh is
        # monotone, sign preserving, and linear near zero.
        return torch.asinh(coefficients)

    def forward(
        self,
        anchor: torch.Tensor,
        image: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if anchor.shape != image.shape:
            raise FiberGANShapeError("anchor and image shapes differ")
        anchor_coeff = self._dct2(anchor)
        image_coeff = self._dct2(image)
        mask = self.complement.to(dtype=image.dtype).expand(image.shape[0], -1, -1, -1)
        h = torch.cat(
            [
                self._compress(anchor_coeff),
                self._compress(mask * image_coeff),
                self._compress(mask * (image_coeff - anchor_coeff)),
                mask,
            ],
            dim=1,
        )
        features: list[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            features.append(h)
        logits = self.logits(h)
        if return_features:
            return logits, tuple(features)
        return logits


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
