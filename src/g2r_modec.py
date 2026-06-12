"""Mode C posterior sampler components for the g2r_ run series.

Sampler head:  x_hat(z) = x_star + P0 @ G_theta(z, x_data)
  * P0 is the exact null-space projector loaded from the verified float32
    artifact (never rebuilt here);
  * x_star is the audited deterministic output (default) or the audited
    anchor — both measurement-consistent, so the range component of x_hat is
    frozen by construction and G_theta fills null space only.

Discriminator HARD RULE: D receives exactly concat(candidate, x_data) and
NOTHING else. No Av-y, no RelMeasErr, no B_lambda(Av-y), no residual-derived
feature, in any form. Keep it that way.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from .models import HQRefiner, HQTwoStageReconstructor


# ---------------------------------------------------------------------------
# P0 artifact loading (no rebuilding)
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_p0_artifact(
    path: str | Path,
    device: torch.device | str = "cpu",
    expected_sha256: str | None = None,
) -> torch.Tensor:
    """Load the verified float32 P0 artifact; optionally check its SHA-256."""
    path = Path(path)
    if expected_sha256:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"P0 artifact hash mismatch for {path}: expected {expected_sha256}, got {actual}."
            )
    payload = torch.load(path, map_location="cpu")
    P0 = payload["P0"] if isinstance(payload, dict) else payload
    if P0.dtype != torch.float32 or P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
        raise ValueError(f"P0 artifact must be a square float32 matrix, got {P0.dtype} {tuple(P0.shape)}.")
    return P0.to(device)


# ---------------------------------------------------------------------------
# z-injected second-stage refiner
# ---------------------------------------------------------------------------

class _ZFuse(nn.Module):
    """Concat a spatially-broadcast z embedding and fuse back to c channels.

    The 1x1 fusion conv is identity-initialized on the feature half and zero
    on the z half, so a warm-started backbone is exactly preserved at step 0.
    """

    def __init__(self, channels: int, z_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(channels + z_channels, channels, kernel_size=1)
        with torch.no_grad():
            self.fuse.weight.zero_()
            for i in range(channels):
                self.fuse.weight[i, i, 0, 0] = 1.0
            self.fuse.bias.zero_()

    def forward(self, feat: torch.Tensor, z_map: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([feat, z_map], dim=1))


class ZInjectedRefiner(nn.Module):
    """HQRefiner with z concatenated at the trunk bottleneck.

    The base refiner trunk is net[0] (RCB 3->c), net[1] (RDB), net[2] (RDB),
    net[3] (1x1 conv -> 1). z is injected between net[1] and net[2]; the
    optional per-scale flag adds a second injection between net[2] and net[3].
    """

    def __init__(
        self,
        base_channels: int = 64,
        z_dim: int = 64,
        z_channels: int = 64,
        per_scale_injection: bool = False,
    ) -> None:
        super().__init__()
        self.base = HQRefiner(base_channels=base_channels)
        self.z_dim = int(z_dim)
        self.z_channels = int(z_channels)
        self.per_scale_injection = bool(per_scale_injection)
        self.z_embed = nn.Sequential(
            nn.Linear(self.z_dim, self.z_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.z_channels, self.z_channels),
        )
        c = int(base_channels)
        self.fuse1 = _ZFuse(c, self.z_channels)
        self.fuse2 = _ZFuse(c, self.z_channels) if self.per_scale_injection else None

    def load_warm_refiner(self, refiner_state: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(refiner_state)

    def forward(self, x_data: torch.Tensor, x_stage1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z with shape [B, {self.z_dim}], got {tuple(z.shape)}.")
        proxy = torch.abs(x_stage1 - x_data)
        h = torch.cat([x_data, x_stage1, proxy], dim=1)
        h = self.base.net[0](h)
        h = self.base.net[1](h)
        z_map = self.z_embed(z)[:, :, None, None].expand(-1, -1, h.shape[-2], h.shape[-1])
        h = self.fuse1(h, z_map)
        h = self.base.net[2](h)
        if self.fuse2 is not None:
            h = self.fuse2(h, z_map)
        return self.base.net[3](h)


# ---------------------------------------------------------------------------
# Mode C sampler
# ---------------------------------------------------------------------------

class ModeCSampler(nn.Module):
    """x_hat(z) = x_star + P0 @ G_theta(z, x_data), computed in float32.

    G_theta = warm-started stage1 (frozen by default) + z-injected refiner.
    In delta mode (default) the projected term is P0 @ (g_out - x_star),
    a reparameterization of the same family that makes x_hat == x_star at
    warm start; the literal form P0 @ g_out is available via the flag.
    """

    def __init__(
        self,
        backbone: HQTwoStageReconstructor,
        P0: torch.Tensor,
        *,
        z_dim: int = 64,
        z_channels: int = 64,
        per_scale_injection: bool = False,
        freeze_stage1: bool = True,
        delta_mode: bool = True,
    ) -> None:
        super().__init__()
        base_channels = backbone.stage1.out.in_channels
        self.stage1 = copy.deepcopy(backbone.stage1)
        self.refiner = ZInjectedRefiner(
            base_channels=base_channels,
            z_dim=z_dim,
            z_channels=z_channels,
            per_scale_injection=per_scale_injection,
        )
        self.refiner.load_warm_refiner(backbone.refiner.state_dict())
        self.freeze_stage1 = bool(freeze_stage1)
        if self.freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad_(False)
        self.delta_mode = bool(delta_mode)
        self.z_dim = int(z_dim)
        self.register_buffer("P0", P0, persistent=False)  # loaded from artifact, not checkpointed

    def trainable_parameters(self):
        params = list(self.refiner.parameters())
        if not self.freeze_stage1:
            params += list(self.stage1.parameters())
        return params

    def _stage1_proposal(self, x_data: torch.Tensor) -> torch.Tensor:
        # Deterministic stage1: zero noise map so x_tilde does not jitter
        # between samples; all sample diversity enters through z.
        noise_map = torch.zeros_like(x_data)
        if self.freeze_stage1:
            with torch.no_grad():
                residual = self.stage1(x_data, noise_map)
        else:
            residual = self.stage1(x_data, noise_map)
        return x_data + residual

    def project_null(self, v_img: torch.Tensor) -> torch.Tensor:
        flat = v_img.reshape(v_img.shape[0], -1)
        return (flat @ self.P0).reshape_as(v_img)

    def forward(self, x_data: torch.Tensor, x_star: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x_data, x_star: [B,1,H,W] already K-expanded to match z [B, z_dim]."""
        x_data = x_data.float()
        x_star = x_star.float()
        # Backbone runs under the ambient autocast (AMP-friendly)...
        x_tilde = self._stage1_proposal(x_data)
        delta = self.refiner(x_data, x_tilde, z)
        # ...but the sampler head is pinned to float32: the P0 projection and
        # the x_star addition set the measurement-consistency floor.
        with torch.autocast(device_type=x_data.device.type, enabled=False):
            g_out = x_tilde.float() + delta.float()
            inner = (g_out - x_star) if self.delta_mode else g_out
            return x_star + self.project_null(inner)


# ---------------------------------------------------------------------------
# Conditional PatchGAN discriminator (spectral norm)
# ---------------------------------------------------------------------------

class CondPatchGAN(nn.Module):
    """4-layer conditional PatchGAN with spectral norm.

    Input is EXACTLY concat(candidate, x_data) along channels. The forward
    signature makes it structurally impossible to feed residual-derived
    features: the module itself performs the concat.
    """

    def __init__(self, base_channels: int = 64, pack: int = 1) -> None:
        super().__init__()
        self.pack = int(pack)
        in_ch = self.pack + 1  # pack candidates + conditioning anchor
        c = int(base_channels)
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, c, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c, c * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 4, c * 8, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 8, 1, 3, stride=1, padding=1)),
        )

    def forward(self, candidate: torch.Tensor, x_data: torch.Tensor) -> torch.Tensor:
        if candidate.shape[1] != self.pack:
            raise ValueError(f"Expected {self.pack} packed candidate channel(s), got {candidate.shape[1]}.")
        if x_data.shape[1] != 1:
            raise ValueError("Conditioning input must be the single-channel anchor x_data.")
        return self.net(torch.cat([candidate, x_data], dim=1))


# ---------------------------------------------------------------------------
# Losses and schedules
# ---------------------------------------------------------------------------

def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def rcgan_std_reward(x_hat: torch.Tensor, k: int) -> torch.Tensor:
    """Bendel/rcGAN-style standard-deviation reward.

    x_hat: [B, K, 1, H, W]. Returns sqrt(pi/(2K(K-1))) * E_b,pix sum_k
    |x_hat_k - mean_k|, an unbiased per-pixel std estimate under Gaussianity.
    Subtract beta_SD * reward from the generator loss (it is a REWARD).
    """
    if k < 2:
        raise ValueError("rcGAN std reward requires K >= 2 samples.")
    mean = x_hat.mean(dim=1, keepdim=True)
    coeff = math.sqrt(math.pi / (2.0 * k * (k - 1)))
    return coeff * (x_hat - mean).abs().sum(dim=1).mean()


def adv_ramp(step: int, total_steps: int, ramp_fraction: float = 0.15) -> float:
    """Linear 0 -> 1 ramp over the first ramp_fraction of training."""
    ramp_steps = max(1, int(total_steps * ramp_fraction))
    return min(1.0, step / ramp_steps)


def r1_penalty(d: CondPatchGAN, real: torch.Tensor, x_data: torch.Tensor) -> torch.Tensor:
    real = real.detach().requires_grad_(True)
    logits = d(real, x_data)
    (grad,) = torch.autograd.grad(logits.sum(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(dim=1).mean()


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return math.sqrt(total)


def exact_consistency_audit(measurement: Any, v_img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact (lambda=0) data-consistency audit for orthonormal-row operators.

    Pi_y_exact(v) = v - A^T (A v - y), valid when A A^T = I (scrambled/lowfreq
    Hadamard ensembles). The published B_lambda audit (lambda=1e-3) leaves a
    residual ~1e-3 x pre-audit (rel ~5e-3 on scr5 because the refiner emits
    large range-space corrections); this exact audit drives the unclipped
    RelMeasErr to the float32 floor (~1e-7), and the float64 eval recompute
    reaches the G-CERT regime.
    """
    A = measurement.get_current_A()
    with torch.no_grad():
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        orth_err = float(torch.linalg.norm(A @ A.T - eye))
    if orth_err > 1e-3:
        raise ValueError(
            f"exact_consistency_audit requires orthonormal rows (||AA^T - I||_F = {orth_err:.3e}); "
            "for non-orthonormal ensembles use a pinv-based audit instead."
        )
    flat = measurement.flatten_img(v_img.float()) if v_img.ndim == 4 else v_img.float()
    resid = flat @ A.T - y.float()
    return measurement.unflatten_img(flat - resid @ A)


def unclipped_rel_meas_err(measurement: Any, x_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample RelMeasErr on the UNCLIPPED vector (float32 runtime check)."""
    flat = measurement.flatten_img(x_hat.float())
    resid = measurement.A_forward(flat) - y.float()
    return torch.linalg.norm(resid, dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)
