from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LightImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, c * 6, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 6), c * 6),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(c * 6, int(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}.")
        z = self.net(x).flatten(1)
        return F.normalize(self.proj(z), dim=1, eps=1e-8)


class CompatibilityCritic(nn.Module):
    """Dual-encoder compatibility critic for (range, null) pairs."""

    def __init__(
        self,
        embed_dim: int = 128,
        base_channels: int = 24,
        temperature: float = 0.07,
        learn_temperature: bool = False,
        use_joint_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.f_r = LightImageEncoder(embed_dim=embed_dim, base_channels=base_channels)
        self.f_n = LightImageEncoder(embed_dim=embed_dim, base_channels=base_channels)
        self.learn_temperature = bool(learn_temperature)
        init_log_temp = math.log(float(temperature))
        if self.learn_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_temp, dtype=torch.float32))
        else:
            self.register_buffer("log_temperature", torch.tensor(init_log_temp, dtype=torch.float32))
        self.use_joint_mlp = bool(use_joint_mlp)
        self.joint_mlp = (
            nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim),
                nn.SiLU(inplace=True),
                nn.Linear(embed_dim, 1),
            )
            if self.use_joint_mlp
            else None
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 1.0)

    def encode_r(self, r: torch.Tensor) -> torch.Tensor:
        return self.f_r(r)

    def encode_n(self, n: torch.Tensor) -> torch.Tensor:
        return self.f_n(n)

    def forward_embeddings(self, r: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_r(r), self.encode_n(n)

    def score_matrix(self, r: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        z_r, z_n = self.forward_embeddings(r, n)
        logits = z_r @ z_n.T / self.temperature
        if self.joint_mlp is None:
            return logits
        # Joint MLP is only defined for aligned pairs; keep matrix path pure.
        return logits

    def score_pairs(self, r: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        z_r, z_n = self.forward_embeddings(r, n)
        dot = (z_r * z_n).sum(dim=1) / self.temperature
        if self.joint_mlp is None:
            return dot
        joint = torch.cat([z_r, z_n, z_r * z_n, (z_r - z_n).abs()], dim=1)
        return dot + self.joint_mlp(joint).squeeze(1)

    def forward(self, r: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(r, n)


def symmetric_infonce_loss(score_matrix: torch.Tensor) -> torch.Tensor:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError(f"Expected square score matrix, got {tuple(score_matrix.shape)}.")
    labels = torch.arange(score_matrix.shape[0], device=score_matrix.device)
    row_loss = F.cross_entropy(score_matrix, labels)
    col_loss = F.cross_entropy(score_matrix.T, labels)
    return 0.5 * (row_loss + col_loss)


def margin_ranking_from_scores(pos: torch.Tensor, neg: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    target = torch.ones_like(pos)
    return F.margin_ranking_loss(pos, neg, target, margin=float(margin))
