import torch

from src.fiber_residual_phase_gan import (
    ConditionalHighPassDiscriminator,
    FiberResidualPhaseGenerator,
    high_pass,
    hinge_discriminator_loss,
)


def test_generator_starts_as_fixed_residual_weight() -> None:
    torch.manual_seed(7)
    model = FiberResidualPhaseGenerator(
        channels=8,
        maximum_weight=0.4,
        initial_weight=0.1,
        rotation_scale=0.5,
    )
    base = torch.rand(3, 1, 16, 16)
    direction = torch.randn_like(base)
    anchor = torch.rand_like(base)
    correction, audit = model(base, direction, anchor)
    assert torch.allclose(correction, 0.1 * direction, atol=1.0e-6)
    assert torch.allclose(audit["rotation"], torch.zeros_like(direction))
    assert torch.allclose(audit["weight"], torch.full_like(direction, 0.1), atol=1.0e-6)


def test_discriminator_and_hinge_loss_are_finite() -> None:
    discriminator = ConditionalHighPassDiscriminator(channels=8)
    base = torch.rand(4, 1, 32, 32)
    real = torch.rand_like(base)
    fake = torch.rand_like(base)
    real_score = discriminator(base, real)
    fake_score = discriminator(base, fake)
    loss = hinge_discriminator_loss(real_score, fake_score)
    assert real_score.shape == fake_score.shape == (4, 1, 4, 4)
    assert torch.isfinite(loss)


def test_high_pass_removes_constant_interior() -> None:
    image = torch.ones(1, 1, 16, 16)
    detail = high_pass(image)
    assert torch.allclose(detail[..., 2:-2, 2:-2], torch.zeros(1, 1, 12, 12))
