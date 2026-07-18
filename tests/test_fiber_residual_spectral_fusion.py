import torch

from src.fiber_residual_spectral_fusion import FiberResidualSpectralFusionGate


def test_gate_starts_near_reference_with_uniform_band_mix() -> None:
    torch.manual_seed(11)
    model = FiberResidualSpectralFusionGate(
        channels=8,
        bands=4,
        maximum_mix=1.0,
        initial_mix=0.02,
    )
    base = torch.rand(3, 1, 32, 32)
    reference = torch.randn_like(base)
    proposal = torch.randn_like(base)
    anchor = torch.rand_like(base)
    correction, weights = model(base, reference, proposal, anchor)
    expected = reference + 0.02 * (proposal - reference)
    assert correction.shape == reference.shape
    assert weights.shape == (3, 4)
    assert torch.allclose(weights, torch.full_like(weights, 0.02), atol=1.0e-6)
    assert torch.allclose(correction, expected, atol=1.0e-5)
