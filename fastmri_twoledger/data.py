# -*- coding: utf-8 -*-
"""fastMRI single-coil val loader for the two-ledger study.
val h5 has fully-sampled `kspace` (S,H,W) complex + `reconstruction_esc` (S,320,320)
= emulated-single-coil magnitude ground truth. We take the complex ESC image
x = ifft2c(kspace) as the true scene, retrospectively undersample with our own mask."""
import glob, os
import numpy as np
import torch
import h5py
from mf_operator import ifft2c, center_crop

VAL_DIR = r"E:/GAN_FCC_WORK/data_warehouse/fastmri_knee_sc/val/singlecoil_val"


def list_volumes(val_dir=VAL_DIR, require_gt=True, match_shape=None, dominant_shape=False):
    """List val h5 volumes that open cleanly and (optionally) carry ground truth.
    Skips files still being written. fastMRI knee widths vary (368/372/...), so a
    single shared operator needs one common (H,W): pass match_shape=(H,W), or
    dominant_shape=True to auto-select the most common shape."""
    cand = []
    for p in sorted(glob.glob(os.path.join(val_dir, "*.h5"))):
        try:
            with h5py.File(p, "r") as f:
                if "kspace" not in f:
                    continue
                if require_gt and "reconstruction_esc" not in f:
                    continue
                shp = tuple(f["kspace"].shape[1:])   # (H, W)
            cand.append((p, shp))
        except Exception:
            continue
    if dominant_shape and match_shape is None:
        from collections import Counter
        match_shape = Counter(s for _, s in cand).most_common(1)[0][0]
    if match_shape is not None:
        cand = [(p, s) for p, s in cand if s == tuple(match_shape)]
    return [p for p, _ in cand]


def load_slice(path, sl=None, dtype=torch.complex128):
    """Return (x_complex (H,W), esc_gt (320,320), attrs). If sl is None use middle slice."""
    with h5py.File(path, "r") as f:
        ksp = f["kspace"]
        S = ksp.shape[0]
        s = S // 2 if sl is None else sl
        k = np.asarray(ksp[s]).astype(np.complex128)
        esc = np.asarray(f["reconstruction_esc"][s]) if "reconstruction_esc" in f else None
        attrs = dict(f.attrs)
    x = ifft2c(torch.from_numpy(k).to(dtype))          # complex ESC image, full (H,W)
    esc_t = torch.from_numpy(esc).to(torch.float64) if esc is not None else None
    return x, esc_t, attrs


def esc_from_complex(x):
    """The fastMRI ESC target = center-cropped magnitude of the complex image."""
    return center_crop(x.abs(), (320, 320))
