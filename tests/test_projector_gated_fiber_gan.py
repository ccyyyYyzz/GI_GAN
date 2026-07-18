from __future__ import annotations

import torch

from src.gauge_geometry import GaugeGeometry
from src.projector_gated_fiber_gan import (
    ComplementaryDCTBucketDiscriminator,
    FiberConditionalDiscriminator,
    ProjectorGatedFiberGenerator,
    parameter_count,
)


def test_complementary_dct_bucket_critic_masks_acquired_coefficients() -> None:
    critic = ComplementaryDCTBucketDiscriminator(img_size=8, acquired_non_dc=7)
    assert critic.complement.shape == (1, 1, 8, 8)
    assert int((critic.complement == 0).sum()) == 8
    probe = torch.randn(2, 1, 8, 8)
    coeff = critic._dct2(probe)
    assert torch.allclose(
        coeff.square().sum(dim=(1, 2, 3)),
        probe.square().sum(dim=(1, 2, 3)),
        rtol=1e-5,
        atol=1e-5,
    )
    logits = critic(probe, probe)
    assert logits.shape[0] == probe.shape[0]


def _geometry(seed: int = 1) -> GaugeGeometry:
    generator = torch.Generator().manual_seed(seed)
    return GaugeGeometry(torch.randn(9, 64, generator=generator, dtype=torch.float64))


def test_generator_states_stay_on_the_anchor_measurement_fiber() -> None:
    torch.manual_seed(3)
    geometry = _geometry()
    truth = torch.rand(2, 64, dtype=torch.float64)
    z = truth @ geometry.Q.T
    anchor = geometry.affine_project_flat(torch.rand_like(truth), z).float().reshape(2, 1, 8, 8)
    model = ProjectorGatedFiberGenerator(geometry, steps=3)
    torch.nn.init.normal_(model.shared_step.correction_head[-1].weight, std=1e-2)

    output = model(anchor, torch.zeros_like(anchor))

    for state in output.raw_states:
        error = geometry.relative_record_error(state.reshape(2, 64).double(), z)
        assert error.max().item() < 1e-5
    assert len(output.raw_corrections) == 3
    assert output.raw_image.shape == anchor.shape


def test_generator_and_conditional_discriminator_backpropagate() -> None:
    torch.manual_seed(5)
    geometry = _geometry()
    truth = torch.rand(2, 64, dtype=torch.float64)
    z = truth @ geometry.Q.T
    anchor = geometry.affine_project_flat(torch.rand_like(truth), z).float().reshape(2, 1, 8, 8)
    generator = ProjectorGatedFiberGenerator(geometry, steps=3)
    discriminator = FiberConditionalDiscriminator()

    output = generator(anchor, torch.zeros_like(anchor))
    logits = discriminator(anchor, output.raw_image)
    loss = output.raw_image.sub(truth.float().reshape_as(output.raw_image)).square().mean() - logits.mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())
    assert parameter_count(generator) == 787_107
    assert parameter_count(discriminator) == 314_977


def test_output_is_invariant_to_measurement_row_reparameterization() -> None:
    torch.manual_seed(13)
    rows = torch.randn(7, 64, dtype=torch.float64)
    transform = torch.randn(7, 7, dtype=torch.float64) + 2.0 * torch.eye(7, dtype=torch.float64)
    geometry = GaugeGeometry(rows)
    changed_geometry = GaugeGeometry(transform @ rows)
    truth = torch.rand(2, 64, dtype=torch.float64)
    z = truth @ geometry.Q.T
    anchor = geometry.affine_project_flat(torch.rand_like(truth), z).float().reshape(2, 1, 8, 8)
    model = ProjectorGatedFiberGenerator(geometry, steps=3)
    torch.nn.init.normal_(model.shared_step.correction_head[-1].weight, std=1e-2)

    native = model(anchor, torch.zeros_like(anchor))
    changed = model(anchor, torch.zeros_like(anchor), geometry=changed_geometry)

    assert torch.max(torch.abs(native.raw_image - changed.raw_image)).item() < 1e-5
