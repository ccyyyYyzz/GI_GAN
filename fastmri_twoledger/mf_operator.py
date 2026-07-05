# -*- coding: utf-8 -*-
"""Masked-Fourier measurement operator for single-coil (emulated single-coil) MRI,
and the exact row/null projectors of the two-ledger accountability framework.

Forward model (single-coil):  y = A x = M . F x,  x complex image, F = centered
orthonormal 2D DFT, M = binary Cartesian undersampling mask (1D along the readout-
orthogonal / phase-encode axis, broadcast over the other axis).

Because F is unitary and M is a 0/1 diagonal in k-space:
    A^dagger = A^H = F^{-1} M        (zero-filled reconstruction)
    P_R = A^dagger A = F^{-1} M F     (row/measured ledger)
    P_0 = I - P_R                     (null ledger, = unsampled k-space)
    A A^H = M                         (=> singular values in {0,1}; measured modes sigma_i = 1)
    Pi_lambda(v) = v - A^H (A A^H + lambda I)^{-1}(A v - y)
                 = v - F^{-1} [ (M/(M+lambda)) (M F v - y) ]
    => measured residual contracts by exactly lambda/(lambda + sigma_i^2) = lambda/(lambda+1).

All k-space tensors are kept full-size with zeros at unsampled locations (so "y" and
"A x" live in the same C^{H x W} space); the mask selects the measured support.
Everything is complex; metrics/labels live on the magnitude image (a nonlinear map),
so certification is stated in the complex/linear domain (see paper honesty clause).
"""
from __future__ import annotations
import numpy as np
import torch


# ----------------------------- centered orthonormal FFT -----------------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT over the last two dims (fastMRI convention)."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT over the last two dims."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


# ----------------------------- Cartesian undersampling masks -----------------------------
def equispaced_mask(width: int, acceleration: int = 4, center_fraction: float = 0.08,
                    device="cpu") -> torch.Tensor:
    """Deterministic 1D Cartesian mask over `width` columns: a fully-sampled central
    ACS block plus equispaced outer lines. Returns a (width,) float64 0/1 vector.
    Matches the fastMRI EquispacedMaskFraction family (reproducible, no RNG)."""
    num_center = int(round(width * center_fraction))
    mask = np.zeros(width, dtype=np.float64)
    c0 = (width - num_center + 1) // 2
    mask[c0:c0 + num_center] = 1.0
    # equispaced outer lines targeting the requested acceleration
    adjusted = (acceleration * (num_center - width)) / (num_center * acceleration - width)
    if adjusted <= 0 or not np.isfinite(adjusted):
        adjusted = acceleration
    offset = 0
    idx = np.arange(offset, width, adjusted)
    mask[np.round(idx).astype(int).clip(0, width - 1)] = 1.0
    return torch.from_numpy(mask).to(device)


def random_mask(width: int, acceleration: int = 4, center_fraction: float = 0.08,
                seed: int = 0, device="cpu") -> torch.Tensor:
    """1D random Cartesian mask (fastMRI RandomMaskFunc style), reproducible by seed."""
    rng = np.random.default_rng(seed)
    num_center = int(round(width * center_fraction))
    prob = (width / acceleration - num_center) / (width - num_center)
    mask = (rng.uniform(size=width) < prob).astype(np.float64)
    c0 = (width - num_center + 1) // 2
    mask[c0:c0 + num_center] = 1.0
    return torch.from_numpy(mask).to(device)


class MaskedFourier:
    """Exact single-coil masked-Fourier operator + its row/null projectors.

    mask1d: (W,) 0/1 vector along the phase-encode (column) axis; broadcast over rows.
    All operations are complex; pass complex128 for the 1e-15 certification.
    """

    def __init__(self, mask1d: torch.Tensor, shape: tuple[int, int]):
        H, W = shape
        assert mask1d.shape[-1] == W, "mask length must equal image width"
        self.shape = (H, W)
        self.mask = mask1d.reshape(1, W).to(torch.float64)          # (1, W), broadcast over H
        self.m = float(self.mask.sum().item()) * H                  # number of measured real k-lines*H
        self.sampling_rate = self.mask.mean().item()

    def to(self, device):
        self.mask = self.mask.to(device)
        return self

    # forward / adjoint / pseudo-inverse
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """y = M . F x  (full-size k-space, zeros off support)."""
        return fft2c(x) * self.mask.to(x.real.dtype)

    def A_adj(self, k: torch.Tensor) -> torch.Tensor:
        """A^H k = F^{-1} (M k). For single-coil this equals A^dagger (zero-filled recon)."""
        return ifft2c(k * self.mask.to(k.real.dtype))

    A_dagger = A_adj  # unitary F + 0/1 mask => A^H == A^dagger

    # projectors
    def P_R(self, x: torch.Tensor) -> torch.Tensor:
        """Row/measured ledger: F^{-1} M F x."""
        return ifft2c(fft2c(x) * self.mask.to(x.real.dtype))

    def P_0(self, x: torch.Tensor) -> torch.Tensor:
        """Null ledger: x - P_R x."""
        return x - self.P_R(x)

    # GT-free record-consistency audit
    def audit(self, v: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        """Pi_lambda(v) = v - A^H (A A^H + lam I)^{-1}(A v - y).
        For masked-Fourier this is v - F^{-1}[ M/(M+lam) (M F v - y) ]."""
        resid = self.A(v) - y                              # supported on mask
        corr_k = resid * (self.mask / (self.mask + lam)).to(resid.real.dtype)
        return v - ifft2c(corr_k)

    def rel_meas_err(self, v: torch.Tensor, y: torch.Tensor) -> float:
        num = torch.linalg.vector_norm(self.A(v) - y)
        den = torch.linalg.vector_norm(y)
        return float((num / den).item())

    # feasible-but-wrong witness (Proposition 1)
    def witness(self, x_donor: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """u = x_donor - A^dagger(A x_donor - y_target): matches y_target exactly,
        carries donor's null content P_0 x_donor."""
        return x_donor - self.A_dagger(self.A(x_donor) - y_target)


def center_crop(img: torch.Tensor, size=(320, 320)) -> torch.Tensor:
    H, W = img.shape[-2:]
    th, tw = size
    top, left = (H - th) // 2, (W - tw) // 2
    return img[..., top:top + th, left:left + tw]
