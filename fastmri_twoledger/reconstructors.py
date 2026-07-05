# -*- coding: utf-8 -*-
"""Reconstructors-under-audit for the fastMRI two-ledger study. Each returns a
FULL-resolution complex image so the exact operator/projectors apply. Metrics are
taken on the center-cropped magnitude (the fastMRI target domain).

  zero_filled : A^dagger y  ==  P_R x_gt exactly (noiseless) -> the RANGE CEILING.
  wavelet_cs  : POCS-ISTA with a wavelet-L1 prior; fills the null space classically.
  unet        : the official fastMRI single-coil U-Net (chans=256), a magnitude
                post-processor; lifted to full complex via zero-filled phase.
Also: data_consistent(x) = A^dagger y + P_0 x  (project any recon onto the record fiber).
"""
import numpy as np
import torch
import pywt
from mf_operator import ifft2c, fft2c, center_crop

_UNET = None


def zero_filled(op, y):
    return op.A_dagger(y)


def data_consistent(op, x, y):
    """Project x onto the exact record fiber: A^dagger y + P_0 x (keeps x's null content)."""
    return x - op.A_dagger(op.A(x) - y)


# ------------------------------- wavelet compressed sensing -------------------------------
def _soft(c, t):
    return np.sign(c) * np.maximum(np.abs(c) - t, 0.0)


def wavelet_cs(op, y, thresh=1e-7, iters=60, wavelet="db4", level=3):
    """POCS-ISTA: alternate exact fiber projection with wavelet soft-thresholding on
    real & imaginary parts. Data-consistent by construction at the final projection."""
    x = op.A_dagger(y).clone()
    for _ in range(iters):
        # wavelet soft-threshold (prior step) on real & imag
        arr = x.detach().cpu().numpy()
        out = np.zeros_like(arr)
        for part in ("real", "imag"):
            a = getattr(arr, part)
            coeffs = pywt.wavedec2(a, wavelet, level=level, mode="periodization")
            coeffs = [coeffs[0]] + [tuple(_soft(d, thresh) for d in det) for det in coeffs[1:]]
            rec = pywt.waverec2(coeffs, wavelet, mode="periodization")
            rec = rec[:a.shape[0], :a.shape[1]]
            if part == "real":
                out = out.astype(np.complex128) + rec
            else:
                out = out + 1j * rec
        x = torch.from_numpy(out).to(x.dtype)
        x = data_consistent(op, x, y)          # exact data consistency
    return x


# ------------------------------- fastMRI U-Net -------------------------------
def _load_unet(device):
    global _UNET
    if _UNET is None:
        from fastmri.models import Unet
        net = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
        sd = torch.load(r"E:/GAN_FCC_WORK/data_warehouse/fastmri_knee_sc/knee_sc_leaderboard_state_dict.pt",
                        map_location="cpu", weights_only=False)
        net.load_state_dict(sd)
        net.eval().to(device)
        _UNET = net
    return _UNET


def _norm(x, eps=1e-11):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps), m, s


def unet(op, y, device="cuda", crop=(320, 320)):
    """Run the official single-coil U-Net on the zero-filled magnitude (its native
    input), return (mag_320, x_full_complex). x_full lifts the U-Net magnitude back
    to full-res complex using the zero-filled phase; outside the crop keeps zero-filled."""
    net = _load_unet(device)
    x_zf = op.A_dagger(y)                                # full complex
    mag_full = x_zf.abs()
    mag_c = center_crop(mag_full, crop).to(torch.float32).to(device)
    xin, m, s = _norm(mag_c)
    with torch.no_grad():
        out = net(xin[None, None])[0, 0]
    out = out * s + m                                    # unnormalize
    out = out.clamp_min(0).double().cpu()               # U-Net magnitude on the crop
    # lift to full complex: crop region = unet-mag * zf-phase; outside = zf
    x_full = x_zf.clone()
    H, W = x_zf.shape[-2:]; th, tw = crop
    top, left = (H - th) // 2, (W - tw) // 2
    phase_c = center_crop(x_zf.angle(), crop)
    x_full[top:top + th, left:left + tw] = out * torch.exp(1j * phase_c)
    return out, x_full


# ------------------------------- metrics -------------------------------
def psnr_mag(x_complex_full, gt_mag, crop=(320, 320)):
    """PSNR of |crop(x)| against gt magnitude, normalized by gt max (fastMRI-style)."""
    m = center_crop(x_complex_full.abs(), crop) if x_complex_full.dim() >= 2 and torch.is_complex(x_complex_full) else x_complex_full
    m = m.to(torch.float64); g = gt_mag.to(torch.float64)
    data_range = float(g.max())
    mse = float(((m - g) ** 2).mean())
    return 10 * np.log10(data_range ** 2 / max(mse, 1e-30))
