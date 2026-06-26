import numpy as np
import torch

import gan_high_quality_gi as hq
import measurement_conditioned_vqgan as mc
from src.projections import relative_measurement_error


def test_vector_quantizer_shapes_and_usage():
    vq = mc.VectorQuantizer(codebook_size=16, embed_dim=8)
    z = torch.randn(2, 8, 4, 4)
    zq, idx, loss, stats = vq(z)
    assert zq.shape == z.shape
    assert idx.shape == (2, 4, 4)
    assert torch.isfinite(loss)
    assert 0 <= stats["dead_code_fraction"] <= 1


def test_vq_autoencoder_forward_and_decode_logits():
    model = mc.VQAutoencoder(codebook_size=16, z_dim=8, base=8)
    x = torch.rand(2, 1, 64, 64)
    recon, idx, loss, _stats = model(x)
    logits = torch.randn(2, 16, idx.shape[1], idx.shape[2])
    soft = model.decode_logits(logits)
    assert recon.shape == x.shape
    assert soft.shape == x.shape
    assert torch.isfinite(loss)


def test_exact_audit_preserves_measurement():
    device = torch.device("cpu")
    rows = np.eye(64, dtype=np.float32)[:8]
    measurement = hq.make_measurement_operator(rows, img_size=8, device=device, lambda_solver=1e-8)
    truth = torch.rand(3, 1, 8, 8)
    y = measurement.A_forward(measurement.flatten_img(truth))
    proposal = torch.rand_like(truth)
    audited = mc.audit_image(proposal, y, measurement)
    rel = relative_measurement_error(audited, y, measurement)
    assert float(rel.max()) < 1e-6
