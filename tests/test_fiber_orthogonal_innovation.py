import torch

from src.fiber_orthogonal_innovation import fiber_orthogonal_innovation


def test_removes_only_parallel_component_without_renormalizing() -> None:
    structural = torch.tensor([[2.0, 0.0, 0.0]])
    innovation = torch.tensor([[3.0, 4.0, 0.0]])
    orthogonal, beta, audit = fiber_orthogonal_innovation(structural, innovation)
    assert torch.allclose(beta, torch.tensor([[1.5]]))
    assert torch.allclose(orthogonal, torch.tensor([[0.0, 4.0, 0.0]]))
    assert torch.allclose((orthogonal * structural).sum(dim=1), torch.zeros(1))
    assert torch.allclose(audit["parallel_energy_fraction"], torch.tensor([9.0 / 25.0]))


def test_zero_structural_direction_preserves_innovation() -> None:
    structural = torch.zeros(2, 4)
    innovation = torch.randn(2, 4)
    orthogonal, beta, audit = fiber_orthogonal_innovation(structural, innovation)
    assert torch.equal(orthogonal, innovation)
    assert torch.equal(beta, torch.zeros(2, 1))
    assert torch.equal(audit["parallel_energy_fraction"], torch.zeros(2))
