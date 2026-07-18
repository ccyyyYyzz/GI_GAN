import torch

from diagnose_fiber_residual_frequency_fusion import (
    dominates,
    smooth_radial_high_pass,
)


def test_smooth_high_pass_suppresses_constant() -> None:
    image = torch.ones(2, 1, 32, 32)
    filtered = smooth_radial_high_pass(image, cutoff=0.2, transition=0.01)
    assert float(filtered.abs().max()) < 1.0e-6


def test_smooth_high_pass_preserves_shape_and_finite_values() -> None:
    image = torch.randn(3, 1, 24, 20)
    filtered = smooth_radial_high_pass(image, cutoff=0.15)
    assert filtered.shape == image.shape
    assert torch.isfinite(filtered).all()


def test_dominance_requires_all_three_metrics() -> None:
    reference = {"psnr": 20.0, "ssim": 0.5, "lpips": 0.3}
    assert dominates({"psnr": 20.1, "ssim": 0.51, "lpips": 0.29}, reference)
    assert not dominates({"psnr": 20.1, "ssim": 0.49, "lpips": 0.29}, reference)
