# -*- coding: utf-8 -*-
"""CASSI (coded-aperture snapshot spectral imaging) measurement operator and the exact
row/null projectors of the two-ledger accountability framework.

Forward model (SD-CASSI, MST/TSA-Net convention): a 28-band cube x [nC,H,W] is spectrally
sheared (band i shifted by step*i columns), modulated by the coded aperture, and summed
along wavelength onto a 2D detector:
    y[r,c] = sum_i  Phi[i,r,c] * shift(x)[i,r,c],     Phi = shift(tile(mask, nC)),  y in R^{H x (W+(nC-1)step)}.

The decisive structural fact (used by MST's own GAP/DAUHST data-consistency step): because
every voxel lands on exactly one detector pixel,
    A A^T = diag(Phi_s),   Phi_s = sum_i Phi[i]^2   (per detector pixel),
so A^dagger, P_R, P_0 and the WHOLE singular spectrum are closed-form and O(N) -- no SVD,
no iterative solver, no unmatched-adjoint risk. Unlike masked-Fourier MRI (all sigma_i = 1),
the CASSI singular values sigma_j = sqrt(Phi_s[j]) are genuinely NON-UNIFORM, so the audit's
per-mode contraction lambda/(lambda+sigma_j^2) varies across the detector -- the empirical
face of the modal-contraction theorem.
"""
from __future__ import annotations
import numpy as np
import torch


def shift(x: torch.Tensor, step: int = 2) -> torch.Tensor:
    """[..., nC, H, W] -> [..., nC, H, W+(nC-1)step]; band i placed at columns [step*i, step*i+W)."""
    *b, nC, H, W = x.shape
    out = x.new_zeros(*b, nC, H, W + (nC - 1) * step)
    for i in range(nC):
        out[..., i, :, step * i:step * i + W] = x[..., i, :, :]
    return out


def shift_back(y3: torch.Tensor, W: int, step: int = 2) -> torch.Tensor:
    """[..., nC, H, Ws] -> [..., nC, H, W]; inverse placement of `shift`."""
    *b, nC, H, Ws = y3.shape
    out = y3.new_zeros(*b, nC, H, W)
    for i in range(nC):
        out[..., i, :, :] = y3[..., i, :, step * i:step * i + W]
    return out


class CASSI:
    def __init__(self, mask: torch.Tensor, nC: int = 28, step: int = 2):
        H, W = mask.shape
        self.nC, self.step, self.H, self.W = nC, step, H, W
        self.Ws = W + (nC - 1) * step
        mask3d = mask.reshape(1, H, W).repeat(nC, 1, 1)          # [nC,H,W] same aperture per band
        self.Phi = shift(mask3d, step)                          # [nC,H,Ws]
        Phi_s = (self.Phi ** 2).sum(0)                          # [H,Ws] = diag(A A^T)
        self.Phi_s = Phi_s
        self.Phi_s_safe = torch.where(Phi_s > 0, Phi_s, torch.ones_like(Phi_s))
        self.support = (Phi_s > 0)                              # measured detector pixels

    def to(self, device):
        self.Phi = self.Phi.to(device); self.Phi_s = self.Phi_s.to(device)
        self.Phi_s_safe = self.Phi_s_safe.to(device); self.support = self.support.to(device)
        return self

    # forward / adjoint / pseudo-inverse
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """cube [nC,H,W] -> snapshot [H,Ws]."""
        return (self.Phi * shift(x, self.step)).sum(-3)

    def At(self, y: torch.Tensor) -> torch.Tensor:
        """snapshot [H,Ws] -> cube [nC,H,W] (adjoint)."""
        return shift_back(y.unsqueeze(-3) * self.Phi, self.W, self.step)

    def A_dagger(self, y: torch.Tensor) -> torch.Tensor:
        """A^dagger = A^T (A A^T)^{-1} = At(y / Phi_s) -- exact min-norm inverse."""
        return self.At(y / self.Phi_s_safe)

    # projectors
    def P_R(self, x: torch.Tensor) -> torch.Tensor:
        return self.A_dagger(self.A(x))

    def P_0(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.P_R(x)

    # GT-free record-consistency audit with per-mode (per-detector-pixel) contraction
    def audit(self, v: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        resid = self.A(v) - y
        corr = self.At(resid / (self.Phi_s + lam))            # (Phi_s/(Phi_s+lam)) per pixel
        return v - corr

    def rel_meas_err(self, v: torch.Tensor, y: torch.Tensor) -> float:
        num = torch.linalg.vector_norm(self.A(v) - y)
        den = torch.linalg.vector_norm(y)
        return float((num / den).item())

    def witness(self, x_donor: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """u = x_donor - A^dagger(A x_donor - y_target): matches y_target, carries donor null."""
        return x_donor - self.A_dagger(self.A(x_donor) - y_target)

    # singular spectrum (closed form): sigma_j = sqrt(Phi_s[j]) over measured detector pixels
    def singular_values(self) -> torch.Tensor:
        return torch.sqrt(self.Phi_s[self.support])

    def contraction(self, lam: float) -> torch.Tensor:
        s2 = self.Phi_s[self.support]
        return lam / (lam + s2)


def load_mask(path, device="cpu", dtype=torch.float64):
    import scipy.io as sio
    m = sio.loadmat(path)["mask"].astype(np.float64)
    return torch.from_numpy(m).to(dtype).to(device)
