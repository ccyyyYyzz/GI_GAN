from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch


def _a_hash(A: torch.Tensor) -> str:
    cpu = A.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _as_flat(x: torch.Tensor, operator) -> tuple[torch.Tensor, bool]:
    if x.ndim == 4:
        return operator.flatten_img(x), True
    if x.ndim == 2 and x.shape[1] == operator.n:
        return x, False
    raise ValueError(f"Expected image [B,1,H,W] or flat [B,{operator.n}], got {tuple(x.shape)}.")


def _restore_shape(v: torch.Tensor, was_image: bool, operator) -> torch.Tensor:
    return operator.unflatten_img(v) if was_image else v


@dataclass(frozen=True)
class ExactProjectorInfo:
    a_sha256: str
    m: int
    n: int
    img_size: int
    dtype: str
    device: str
    solver: str
    rcond: float
    rank_estimate: int
    singular_min: float
    singular_max: float
    condition_estimate: float
    row_orthonormal_max_abs: float
    row_orthonormal_fro: float
    fallback_reason: str


class ExactRangeNullProjector:
    """Matrix-free exact range/null decomposition for full-row-rank A.

    The projection is PR(v) = A^T (A A^T)^dagger A v and P0(v) = v - PR(v).
    No n x n projector is formed.  A Cholesky solve is used when AA^T is
    positive definite; otherwise a pseudoinverse fallback records the rank and
    conditioning in ``info``.
    """

    def __init__(
        self,
        operator,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        rcond: float = 1e-12,
        row_orthonormal_tol: float = 1e-7,
    ) -> None:
        self.operator = operator
        self.dtype = dtype
        self.device = torch.device(device or operator.A.device)
        self.rcond = float(rcond)
        self.A = operator.A.detach().to(device=self.device, dtype=dtype).contiguous()
        self.m = int(self.A.shape[0])
        self.n = int(self.A.shape[1])
        self.G = self.A @ self.A.T
        eye = torch.eye(self.m, device=self.device, dtype=dtype)
        gram_delta = self.G - eye
        row_orth_max = float(gram_delta.abs().max().detach().cpu().item())
        row_orth_fro = float(torch.linalg.norm(gram_delta).detach().cpu().item())

        self.solver = "cholesky"
        self.fallback_reason = ""
        self._chol: torch.Tensor | None = None
        self._pinv: torch.Tensor | None = None
        self._row_orthonormal = row_orth_max <= float(row_orthonormal_tol)
        if self._row_orthonormal:
            self.solver = "row_orthonormal"
        else:
            try:
                self._chol = torch.linalg.cholesky(self.G)
            except RuntimeError as exc:
                self.solver = "pinv"
                self.fallback_reason = f"cholesky_failed:{exc.__class__.__name__}"
                self._pinv = torch.linalg.pinv(self.G, rcond=self.rcond)

        try:
            eig = torch.linalg.eigvalsh(self.G)
            eig = eig.clamp_min(0)
            svals = torch.sqrt(eig)
        except RuntimeError:
            svals = torch.linalg.svdvals(self.A)
        smax = svals.max().clamp_min(torch.finfo(dtype).tiny)
        rank = int((svals > self.rcond * smax).sum().detach().cpu().item())
        smin = float(svals.min().detach().cpu().item())
        smax_f = float(smax.detach().cpu().item())
        cond = float("inf") if smin <= 0 else float(smax_f / smin)
        self.info = ExactProjectorInfo(
            a_sha256=_a_hash(operator.A),
            m=self.m,
            n=self.n,
            img_size=int(operator.img_size),
            dtype=str(dtype).replace("torch.", ""),
            device=str(self.device),
            solver=self.solver,
            rcond=self.rcond,
            rank_estimate=rank,
            singular_min=smin,
            singular_max=smax_f,
            condition_estimate=cond,
            row_orthonormal_max_abs=row_orth_max,
            row_orthonormal_fro=row_orth_fro,
            fallback_reason=self.fallback_reason,
        )

    def solve_gram(self, rhs: torch.Tensor) -> torch.Tensor:
        rhs = rhs.to(device=self.device, dtype=self.dtype).contiguous()
        if rhs.ndim != 2 or rhs.shape[1] != self.m:
            raise ValueError(f"Expected RHS [B,{self.m}], got {tuple(rhs.shape)}.")
        if self.solver == "row_orthonormal":
            return rhs
        if self.solver == "cholesky" and self._chol is not None:
            return torch.cholesky_solve(rhs.T.contiguous(), self._chol).T.contiguous()
        if self._pinv is None:
            self._pinv = torch.linalg.pinv(self.G, rcond=self.rcond)
        return rhs @ self._pinv.T

    def A_forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.to(device=self.device, dtype=self.dtype) @ self.A.T

    def AT_forward(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(device=self.device, dtype=self.dtype) @ self.A

    def row_project_flat(self, v: torch.Tensor) -> torch.Tensor:
        y = self.A_forward(v)
        return self.AT_forward(self.solve_gram(y))

    def null_project_flat(self, v: torch.Tensor) -> torch.Tensor:
        v_exact = v.to(device=self.device, dtype=self.dtype)
        return v_exact - self.row_project_flat(v_exact)

    def data_anchor_flat(self, y: torch.Tensor) -> torch.Tensor:
        return self.AT_forward(self.solve_gram(y))

    def audit_flat(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v_exact = v.to(device=self.device, dtype=self.dtype)
        residual = self.A_forward(v_exact) - y.to(device=self.device, dtype=self.dtype)
        return v_exact - self.AT_forward(self.solve_gram(residual))

    def info_dict(self) -> dict[str, Any]:
        return asdict(self.info)


_PROJECTOR_CACHE: dict[tuple[str, str, str, float, float], ExactRangeNullProjector] = {}


def get_exact_projector(
    operator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    rcond: float = 1e-12,
    row_orthonormal_tol: float = 1e-7,
    use_cache: bool = True,
) -> ExactRangeNullProjector:
    device_obj = torch.device(device or operator.A.device)
    key = (_a_hash(operator.A), str(dtype), str(device_obj), float(rcond), float(row_orthonormal_tol))
    if use_cache and key in _PROJECTOR_CACHE:
        return _PROJECTOR_CACHE[key]
    projector = ExactRangeNullProjector(
        operator,
        dtype=dtype,
        device=device_obj,
        rcond=rcond,
        row_orthonormal_tol=row_orthonormal_tol,
    )
    if use_cache:
        _PROJECTOR_CACHE[key] = projector
    return projector


def exact_row_project(
    x: torch.Tensor,
    operator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    rcond: float = 1e-12,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    flat, was_image = _as_flat(x, operator)
    projector = get_exact_projector(operator, dtype=dtype, device=device or x.device, rcond=rcond)
    out = projector.row_project_flat(flat).to(device=x.device, dtype=x.dtype)
    out = _restore_shape(out, was_image, operator)
    return (out, projector.info_dict()) if return_info else out


def exact_null_project(
    x: torch.Tensor,
    operator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    rcond: float = 1e-12,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    flat, was_image = _as_flat(x, operator)
    projector = get_exact_projector(operator, dtype=dtype, device=device or x.device, rcond=rcond)
    out = projector.null_project_flat(flat).to(device=x.device, dtype=x.dtype)
    out = _restore_shape(out, was_image, operator)
    return (out, projector.info_dict()) if return_info else out


def exact_data_anchor(
    y: torch.Tensor,
    operator,
    *,
    as_image: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    rcond: float = 1e-12,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if y.ndim != 2 or y.shape[1] != operator.m:
        raise ValueError(f"Expected measurement [B,{operator.m}], got {tuple(y.shape)}.")
    projector = get_exact_projector(operator, dtype=dtype, device=device or y.device, rcond=rcond)
    out = projector.data_anchor_flat(y).to(device=y.device, dtype=y.dtype)
    if as_image:
        out = operator.unflatten_img(out)
    return (out, projector.info_dict()) if return_info else out


def exact_audit(
    x: torch.Tensor,
    y: torch.Tensor,
    operator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    rcond: float = 1e-12,
) -> torch.Tensor:
    flat, was_image = _as_flat(x, operator)
    projector = get_exact_projector(operator, dtype=dtype, device=device or x.device, rcond=rcond)
    out = projector.audit_flat(flat, y).to(device=x.device, dtype=x.dtype)
    return _restore_shape(out, was_image, operator)


def soft_audit(
    x: torch.Tensor,
    y: torch.Tensor,
    operator,
    lambda_: float | None = None,
) -> torch.Tensor:
    flat, was_image = _as_flat(x, operator)
    if lambda_ is None or abs(float(lambda_) - float(operator.lambda_dc)) <= 0.0:
        out = operator.dc_project(flat, y)
    else:
        A = operator.A.to(device=flat.device, dtype=flat.dtype)
        eye = torch.eye(operator.m, device=flat.device, dtype=flat.dtype)
        K = A @ A.T + float(lambda_) * eye
        rhs = (flat @ A.T - y).T.contiguous()
        try:
            sol = torch.cholesky_solve(rhs, torch.linalg.cholesky(K)).T.contiguous()
        except RuntimeError:
            sol = torch.linalg.solve(K, rhs).T.contiguous()
        out = flat - sol @ A
    return _restore_shape(out, was_image, operator)


def relative_measurement_error(x: torch.Tensor, y: torch.Tensor, operator) -> torch.Tensor:
    flat, _was_image = _as_flat(x, operator)
    pred = operator.A_forward(flat)
    return torch.linalg.norm(pred - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)


def row_project(x: torch.Tensor, operator, *, exact: bool = True, lambda_: float | None = None) -> torch.Tensor:
    if exact:
        return exact_row_project(x, operator)
    flat, was_image = _as_flat(x, operator)
    if lambda_ is None or abs(float(lambda_) - float(operator.lambda_dc)) <= 0.0:
        out = operator.AT_forward(operator.solve_K(operator.A_forward(flat)))
    else:
        zero_y = torch.zeros(flat.shape[0], operator.m, device=flat.device, dtype=flat.dtype)
        out = flat - soft_audit(flat, zero_y, operator, lambda_)
    return _restore_shape(out, was_image, operator)


def null_project(x: torch.Tensor, operator, *, exact: bool = True, lambda_: float | None = None) -> torch.Tensor:
    if exact:
        return exact_null_project(x, operator)
    flat, was_image = _as_flat(x, operator)
    if lambda_ is None or abs(float(lambda_) - float(operator.lambda_dc)) <= 0.0:
        out = operator.null_project(flat)
    else:
        out = flat - row_project(flat, operator, exact=False, lambda_=lambda_)
    return _restore_shape(out, was_image, operator)
