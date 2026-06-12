from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_rowspace_basis(A: torch.Tensor, *, rtol: float | None = None) -> torch.Tensor:
    """Return Q with orthonormal columns spanning Range(A^T)."""
    A = A.float()
    _U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if rtol is None:
        rtol = max(A.shape) * torch.finfo(S.dtype).eps
    rank = int((S > float(rtol) * S.max().clamp_min(1e-12)).sum().item())
    if rank <= 0:
        raise ValueError("A appears rank deficient with rank 0.")
    return Vh[:rank].T.contiguous()


def project_row(u: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    flat = u.reshape(1, -1) if u.ndim == 1 else u
    projected = (flat @ Q) @ Q.T
    return projected.reshape_as(u)


def project_null(u: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return u - project_row(u, Q)


def project_null_hadamard(u: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Exact null projection for orthonormal-row Hadamard-like A."""
    flat = u.reshape(1, -1) if u.ndim == 1 else u
    projected = (flat @ A.T.float()) @ A.float()
    out = flat - projected
    return out.reshape_as(u)


def verify_projector(A: torch.Tensor, Q: torch.Tensor) -> dict[str, Any]:
    A = A.float()
    Q = Q.float()
    n = A.shape[1]
    Iq = torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    qtq_err = torch.linalg.norm(Q.T @ Q - Iq).item()
    gen = torch.Generator(device=A.device)
    gen.manual_seed(530)
    probe = torch.randn(min(64, n), n, device=A.device, dtype=A.dtype, generator=gen)
    P0_probe = project_null(probe, Q)
    ap0_norm = (torch.linalg.norm(A @ P0_probe.T) / torch.linalg.norm(probe).clamp_min(1e-12)).item()
    idem_err = (torch.linalg.norm(project_null(P0_probe, Q) - P0_probe) / torch.linalg.norm(probe).clamp_min(1e-12)).item()
    S = torch.linalg.svdvals(A)
    rank = int(torch.linalg.matrix_rank(A).item())
    row_gram = A @ A.T
    row_I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    row_orth_err = torch.linalg.norm(row_gram - row_I).item()
    return {
        "m": int(A.shape[0]),
        "n": int(A.shape[1]),
        "row_rank": rank,
        "basis_cols": int(Q.shape[1]),
        "qtq_minus_I_norm": qtq_err,
        "A_P0_norm": ap0_norm,
        "P0_idempotence_norm": idem_err,
        "singular_min": float(S.min().item()),
        "singular_max": float(S.max().item()),
        "singular_mean": float(S.mean().item()),
        "row_orthonormality_norm": row_orth_err,
    }


def save_basis(Q: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Q": Q.detach().cpu()}, path)
    return path


def load_basis(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "Q" in obj:
        return obj["Q"].to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    raise ValueError(f"Could not load row-space basis from {path}")
