from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class OperatorConditionedNullspaceError(RuntimeError):
    """Raised when operator-conditioned null-space components are inconsistent."""


class MatrixFreeNullProjector(nn.Module):
    """Differentiable range/null projection without constructing dense P0.

    For a fixed row operator A, this module computes

        P_R v = A^T (A A^T)^dagger A v
        P_0 v = v - P_R v

    using only A and an m x m pseudoinverse/regularized inverse.  It never
    forms an n x n projector.
    """

    def __init__(self, rows: torch.Tensor, *, rcond: float = 1e-10) -> None:
        super().__init__()
        if rows.ndim != 2:
            raise OperatorConditionedNullspaceError(f"ROWS_MUST_BE_2D:{tuple(rows.shape)}")
        a = rows.detach().clone().to(dtype=torch.float32)
        gram = a @ a.T
        gram_pinv = torch.linalg.pinv(gram.to(torch.float64), rcond=float(rcond)).to(torch.float32)
        self.register_buffer("A", a.contiguous())
        self.register_buffer("gram_pinv", gram_pinv.contiguous())

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def measurement(self, flat: torch.Tensor) -> torch.Tensor:
        return flat @ self.A.T

    def row_project(self, flat: torch.Tensor) -> torch.Tensor:
        y = self.measurement(flat)
        coeff = y @ self.gram_pinv.T
        return coeff @ self.A

    def null_project(self, flat: torch.Tensor) -> torch.Tensor:
        return flat - self.row_project(flat)

    def data_anchor(self, y: torch.Tensor) -> torch.Tensor:
        coeff = y @ self.gram_pinv.T
        return coeff @ self.A

    def relmeaserr(self, flat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.measurement(flat) - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)

    def diagnostics(self) -> dict[str, Any]:
        with torch.no_grad():
            gram = self.A @ self.A.T
            eig = torch.linalg.eigvalsh(gram.to(torch.float64)).clamp_min(0)
            row_norm = torch.linalg.norm(self.A, dim=1)
        vals = eig.detach().cpu()
        positive = vals[vals > 1e-12]
        return {
            "m": self.m,
            "n": self.n,
            "row_norm_min": float(row_norm.min().detach().cpu()),
            "row_norm_max": float(row_norm.max().detach().cpu()),
            "gram_eig_min": float(vals.min()) if vals.numel() else None,
            "gram_eig_max": float(vals.max()) if vals.numel() else None,
            "gram_condition": None if positive.numel() == 0 else float(positive.max() / positive.min()),
            "implementation": "A plus m-by-m Gram pseudoinverse; no dense n-by-n P0",
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SmallNullspaceUNet(nn.Module):
    """Compact canary network for r -> P0 residual prediction."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32, blocks: int = 2) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), nn.SiLU(inplace=True), *[ResidualBlock(c) for _ in range(blocks)])
        self.enc2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(inplace=True), *[ResidualBlock(c * 2) for _ in range(blocks)])
        self.enc3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.SiLU(inplace=True), *[ResidualBlock(c * 4) for _ in range(blocks)])
        self.mid = nn.Sequential(*[ResidualBlock(c * 4) for _ in range(max(1, blocks))])
        self.up2 = nn.Sequential(nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(c * 2))
        self.up1 = nn.Sequential(nn.Conv2d(c * 2 + c, c, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(c))
        self.out = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, r_img: torch.Tensor, cond_img: torch.Tensor | None = None) -> torch.Tensor:
        if cond_img is None:
            cond_img = torch.zeros_like(r_img)
        x = torch.cat([r_img, cond_img], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        mid = self.mid(e3)
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


@dataclass(frozen=True)
class NullspaceForward:
    x_hat: torch.Tensor
    null_hat: torch.Tensor
    raw_residual: torch.Tensor
    relmeaserr: torch.Tensor


def reconstruct_with_projected_residual(
    model: nn.Module,
    projector: MatrixFreeNullProjector,
    r_flat: torch.Tensor,
    y: torch.Tensor,
    *,
    img_size: int,
    cond_scalar: float = 0.0,
    cond_features: torch.Tensor | None = None,
) -> NullspaceForward:
    r_img = r_flat.reshape(r_flat.shape[0], 1, int(img_size), int(img_size))
    if cond_features is None:
        cond = torch.full_like(r_img, float(cond_scalar))
    else:
        cond = cond_features.to(device=r_flat.device, dtype=r_flat.dtype)
        if cond.ndim != 4:
            raise OperatorConditionedNullspaceError(f"COND_FEATURES_MUST_BE_NCHW:{tuple(cond.shape)}")
        if cond.shape[0] == 1 and r_img.shape[0] != 1:
            cond = cond.expand(r_img.shape[0], -1, -1, -1)
        if cond.shape[0] != r_img.shape[0] or cond.shape[-2:] != r_img.shape[-2:]:
            raise OperatorConditionedNullspaceError(f"COND_FEATURES_SHAPE_MISMATCH:{tuple(cond.shape)}:{tuple(r_img.shape)}")
    raw = model(r_img, cond)
    raw_flat = raw.reshape(raw.shape[0], -1)
    null = projector.null_project(raw_flat)
    x_hat = r_flat + null
    return NullspaceForward(
        x_hat=x_hat,
        null_hat=null,
        raw_residual=raw_flat,
        relmeaserr=projector.relmeaserr(x_hat, y),
    )
