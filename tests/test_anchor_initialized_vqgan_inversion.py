from pathlib import Path

import numpy as np
import torch

import anchor_initialized_vqgan_inversion as inv
import gan_high_quality_gi as hq
import measurement_conditioned_vqgan as mc
from src.projections import relative_measurement_error


def test_anchor_latent_refiner_shapes_and_zero_init():
    refiner = inv.AnchorLatentRefiner(z_dim=8, codebook_size=16, base=8, delta_scale=0.25, logit_scale=1.0)
    x0 = torch.rand(2, 1, 64, 64)
    uncertainty = torch.rand(2, 1, 64, 64)
    z0 = torch.rand(2, 8, 8, 8)
    dz, dlogits = refiner(x0, uncertainty, z0)
    assert dz.shape == z0.shape
    assert dlogits.shape == (2, 16, 8, 8)
    assert float(dz.abs().max()) == 0.0
    assert float(dlogits.abs().max()) == 0.0


def test_null_blend_preserves_context_measurement():
    device = torch.device("cpu")
    rows = np.eye(64, dtype=np.float32)[:12]
    measurement = hq.make_measurement_operator(rows, img_size=8, device=device, lambda_solver=1e-10)
    x0 = torch.rand(3, 1, 8, 8)
    y0 = measurement.A_forward(measurement.flatten_img(x0))
    xg = torch.rand_like(x0)
    blended = inv.null_blend(x0, xg, 1.0, measurement)
    rel = relative_measurement_error(blended, y0, measurement)
    assert float(rel.max()) < 1e-6


def test_quantize_from_logits_shapes_and_entropy():
    model = mc.VQAutoencoder(codebook_size=16, z_dim=8, base=8)
    prior = inv.PriorPack("test", model, None, Path("dummy.pt"))
    logits = torch.randn(2, 16, 8, 8)
    zq, idx, entropy = inv.quantize_from_logits(prior, logits, soft_temperature=1.0, straight_through=True)
    assert zq.shape == (2, 8, 8, 8)
    assert idx.shape == (2, 8, 8)
    assert torch.isfinite(entropy)
