from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


_PIQP_MATRIX_CACHE: dict[str, Any] = {}


class GaugeGeometryError(RuntimeError):
    pass


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class GaugeGeometryInfo:
    rows_sha256: str
    m: int
    n: int
    rank: int
    relative_cutoff: float
    singular_max: float
    singular_min_retained: float
    singular_min_all: float


class GaugeGeometry(nn.Module):
    """Intrinsic orthonormal row-space coordinates for a linear GI operator."""

    def __init__(self, rows: torch.Tensor, *, relative_cutoff: float = 1e-12) -> None:
        super().__init__()
        if rows.ndim != 2:
            raise GaugeGeometryError(f"ROWS_MUST_BE_2D:{tuple(rows.shape)}")
        rows64 = rows.detach().to(dtype=torch.float64, device="cpu").contiguous()
        u, singular, vh = torch.linalg.svd(rows64, full_matrices=False)
        threshold = float(relative_cutoff) * float(singular.max())
        keep = singular > threshold
        rank = int(keep.sum())
        if rank < 1:
            raise GaugeGeometryError("OPERATOR_HAS_ZERO_NUMERICAL_RANK")
        u_r = u[:, keep].contiguous()
        s_r = singular[keep].contiguous()
        q = vh[keep].contiguous()
        self.register_buffer("U_r", u_r, persistent=True)
        self.register_buffer("singular", s_r, persistent=True)
        self.register_buffer("Q", q, persistent=True)
        self.info = GaugeGeometryInfo(
            rows_sha256=_tensor_sha256(rows.detach().to(dtype=torch.float32)),
            m=int(rows.shape[0]),
            n=int(rows.shape[1]),
            rank=rank,
            relative_cutoff=float(relative_cutoff),
            singular_max=float(singular.max()),
            singular_min_retained=float(s_r.min()),
            singular_min_all=float(singular.min()),
        )

    @classmethod
    def from_rows_qr(
        cls, rows: torch.Tensor, *, relative_cutoff: float = 1e-12
    ) -> "GaugeGeometry":
        """Build the same row-space geometry without a direct wide-matrix SVD."""

        if rows.ndim != 2:
            raise GaugeGeometryError(f"ROWS_MUST_BE_2D:{tuple(rows.shape)}")
        if rows.shape[0] > rows.shape[1]:
            raise GaugeGeometryError("QR_CONSTRUCTOR_REQUIRES_ROWS_NOT_EXCEED_COLUMNS")
        rows64 = rows.detach().to(dtype=torch.float64, device="cpu").contiguous()
        q_columns, triangular = torch.linalg.qr(rows64.T, mode="reduced")
        u_small, singular, vh_small = torch.linalg.svd(
            triangular.T, full_matrices=False
        )
        threshold = float(relative_cutoff) * float(singular.max())
        keep = singular > threshold
        rank = int(keep.sum())
        if rank < 1:
            raise GaugeGeometryError("OPERATOR_HAS_ZERO_NUMERICAL_RANK")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        u_r = u_small[:, keep].contiguous()
        s_r = singular[keep].contiguous()
        q = (vh_small[keep] @ q_columns.T).contiguous()
        instance.register_buffer("U_r", u_r, persistent=True)
        instance.register_buffer("singular", s_r, persistent=True)
        instance.register_buffer("Q", q, persistent=True)
        instance.info = GaugeGeometryInfo(
            rows_sha256=_tensor_sha256(rows.detach().to(dtype=torch.float32)),
            m=int(rows.shape[0]),
            n=int(rows.shape[1]),
            rank=rank,
            relative_cutoff=float(relative_cutoff),
            singular_max=float(singular.max()),
            singular_min_retained=float(s_r.min()),
            singular_min_all=float(singular.min()),
        )
        return instance

    @property
    def m(self) -> int:
        return int(self.info.m)

    @property
    def n(self) -> int:
        return int(self.info.n)

    @property
    def rank(self) -> int:
        return int(self.info.rank)

    def info_dict(self) -> dict[str, Any]:
        return asdict(self.info)

    def intrinsic_record(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2 or y.shape[1] != self.m:
            raise GaugeGeometryError(f"EXPECTED_Y_BM:{tuple(y.shape)}:{self.m}")
        y64 = y.to(device=self.Q.device, dtype=torch.float64)
        return (y64 @ self.U_r) / self.singular.unsqueeze(0)

    def row_project_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.ndim != 2 or flat.shape[1] != self.n:
            raise GaugeGeometryError(f"EXPECTED_FLAT_BN:{tuple(flat.shape)}:{self.n}")
        q = self.Q.to(dtype=flat.dtype)
        return (flat @ q.T) @ q

    def null_project_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return flat - self.row_project_flat(flat)

    def project_feature_maps(
        self, maps: torch.Tensor, *, null: bool, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if maps.ndim != 4 or maps.shape[-2] * maps.shape[-1] != self.n:
            raise GaugeGeometryError(f"EXPECTED_MAPS_WITH_N_PIXELS:{tuple(maps.shape)}:{self.n}")
        batch, channels, height, width = maps.shape
        with torch.cuda.amp.autocast(enabled=False):
            flat = maps.reshape(batch * channels, self.n).to(dtype=dtype)
            projected = self.null_project_flat(flat) if null else self.row_project_flat(flat)
        return projected.to(dtype=maps.dtype).reshape(batch, channels, height, width)

    def affine_project_flat(self, flat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if flat.ndim != 2 or flat.shape[1] != self.n:
            raise GaugeGeometryError(f"EXPECTED_FLAT_BN:{tuple(flat.shape)}:{self.n}")
        if z.ndim != 2 or z.shape != (flat.shape[0], self.rank):
            raise GaugeGeometryError(
                f"EXPECTED_INTRINSIC_BR:{tuple(z.shape)}:{flat.shape[0]}:{self.rank}"
            )
        q = self.Q.to(device=flat.device, dtype=flat.dtype)
        z_cast = z.to(device=flat.device, dtype=flat.dtype)
        return flat + (z_cast - flat @ q.T) @ q

    def relative_record_error(self, flat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        q = self.Q.to(device=flat.device, dtype=flat.dtype)
        z_cast = z.to(device=flat.device, dtype=flat.dtype)
        return torch.linalg.norm(flat @ q.T - z_cast, dim=1) / torch.linalg.norm(
            z_cast, dim=1
        ).clamp_min(1e-12)


class GaugeEmpiricalAnchor(nn.Module):
    """Gauge-invariant empirical LMMSE anchor with a precomputed n-by-r gain."""

    def __init__(
        self,
        *,
        mu: torch.Tensor,
        gain: torch.Tensor,
        posterior_std: torch.Tensor,
        lambda_: float,
        rows_sha256: str,
    ) -> None:
        super().__init__()
        self.register_buffer("mu", mu.detach().to(torch.float32).reshape(1, -1))
        self.register_buffer("gain", gain.detach().to(torch.float32))
        self.register_buffer("posterior_std", posterior_std.detach().to(torch.float32))
        self.lambda_ = float(lambda_)
        self.rows_sha256 = str(rows_sha256)

    @classmethod
    def fit(
        cls,
        train_images: np.ndarray,
        geometry: GaugeGeometry,
        *,
        lambda_: float,
    ) -> "GaugeEmpiricalAnchor":
        x = np.asarray(train_images, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != geometry.n:
            raise GaugeGeometryError(f"TRAIN_MATRIX_SHAPE_MISMATCH:{x.shape}:{geometry.n}")
        q = geometry.Q.detach().cpu().numpy()
        mu = x.mean(axis=0)
        z_scaled = (x - mu[None, :]) / math.sqrt(max(1, x.shape[0] - 1))
        w = z_scaled @ q.T
        gram = w.T @ w + float(lambda_) * np.eye(geometry.rank, dtype=np.float64)
        inv_gram = np.linalg.inv(gram)
        cross = w.T @ z_scaled
        gain = cross.T @ inv_gram
        diag_c = np.sum(z_scaled * z_scaled, axis=0)
        posterior = diag_c - np.einsum(
            "ij,ik,kj->j", cross, inv_gram, cross, optimize=True
        )
        posterior = np.sqrt(np.maximum(posterior, 0.0))
        return cls(
            mu=torch.from_numpy(mu),
            gain=torch.from_numpy(gain),
            posterior_std=torch.from_numpy(posterior),
            lambda_=float(lambda_),
            rows_sha256=geometry.info.rows_sha256,
        )

    def forward(
        self, y: torch.Tensor, geometry: GaugeGeometry
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = geometry.intrinsic_record(y)
        q64 = geometry.Q.to(device=y.device, dtype=torch.float64)
        mu64 = self.mu.to(device=y.device, dtype=torch.float64)
        delta = z - mu64 @ q64.T
        pred = self.mu.to(device=y.device) + delta.to(torch.float32) @ self.gain.to(
            device=y.device
        ).T
        audited = geometry.affine_project_flat(pred.to(torch.float64), z)
        return audited, z

    def normalized_posterior_map(
        self,
        *,
        img_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        array = self.posterior_std.detach().cpu().numpy().astype(np.float32)
        low, high = float(np.percentile(array, 1)), float(np.percentile(array, 99))
        normalized = np.clip((array - low) / max(high - low, 1e-8), 0.0, 1.0)
        return torch.from_numpy(normalized).to(device=device, dtype=dtype).reshape(
            1, 1, int(img_size), int(img_size)
        )


@dataclass(frozen=True)
class GaugeDykstraResult:
    image_flat: torch.Tensor
    iterations: int
    converged: bool
    max_relative_record_error: float
    max_step_change: float
    max_box_violation: float
    max_stationarity_residual: float = 0.0
    max_intrinsic_infinity_residual: float = 0.0
    max_proximal_residual: float = 0.0
    max_complementarity_residual: float = 0.0


def _lbfgs_box_fiber_projection(
    proposal: torch.Tensor,
    target: torch.Tensor,
    geometry: GaugeGeometry,
    initial_dual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust dual fallback for active-set degeneracy at box corners."""
    try:
        from scipy.optimize import least_squares, minimize
    except Exception as exc:  # pragma: no cover - environment guard
        raise GaugeGeometryError(f"SCIPY_LBFGS_FALLBACK_UNAVAILABLE:{exc!r}") from exc

    q_np = geometry.Q.detach().cpu().numpy().astype(np.float64, copy=False)
    proposal_np = proposal.detach().cpu().numpy().astype(np.float64, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    initial_np = initial_dual.detach().cpu().numpy().astype(np.float64, copy=False)
    outputs: list[np.ndarray] = []
    dual_outputs: list[np.ndarray] = []
    for vector, record, start in zip(proposal_np, target_np, initial_np):
        def objective_and_gradient(dual_vector: np.ndarray):
            fixed_point = np.clip(vector - dual_vector @ q_np, 0.0, 1.0)
            residual = fixed_point @ q_np.T - record
            dual_value = 0.5 * np.sum((fixed_point - vector) ** 2) + np.dot(
                dual_vector, residual
            )
            return -float(dual_value), -residual

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        starts = [np.asarray(start, dtype=np.float64), np.zeros(geometry.rank, dtype=np.float64)]
        for start_vector in starts:
            result = minimize(
                objective_and_gradient,
                start_vector,
                jac=True,
                method="L-BFGS-B",
                options={
                    "gtol": 1e-14,
                    "ftol": 0.0,
                    "maxiter": 3000,
                    "maxfun": 5000,
                    "maxls": 200,
                    "maxcor": 100,
                },
            )
            fixed_point = np.clip(vector - np.asarray(result.x) @ q_np, 0.0, 1.0)
            relative = float(
                np.linalg.norm(fixed_point @ q_np.T - record)
                / max(np.linalg.norm(record), 1e-12)
            )
            candidates.append((relative, fixed_point, np.asarray(result.x, dtype=np.float64)))
            if relative <= 1e-9:
                break
        best_relative, best_point, best_dual = min(candidates, key=lambda pair: pair[0])
        if best_relative > 1e-9:
            def root_residual(dual_vector: np.ndarray) -> np.ndarray:
                return np.clip(vector - dual_vector @ q_np, 0.0, 1.0) @ q_np.T - record

            def root_jacobian(dual_vector: np.ndarray) -> np.ndarray:
                preclip = vector - dual_vector @ q_np
                free = ((preclip > 0.0) & (preclip < 1.0)).astype(np.float64)
                return -np.einsum("rn,n,sn->rs", q_np, free, q_np, optimize=True)

            polished = least_squares(
                root_residual,
                best_dual,
                jac=root_jacobian,
                method="trf",
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14,
                max_nfev=5000,
            )
            polished_point = np.clip(vector - polished.x @ q_np, 0.0, 1.0)
            polished_relative = float(
                np.linalg.norm(polished_point @ q_np.T - record)
                / max(np.linalg.norm(record), 1e-12)
            )
            if polished_relative < best_relative:
                best_point = polished_point
                best_dual = np.asarray(polished.x, dtype=np.float64)
        outputs.append(best_point)
        dual_outputs.append(best_dual)
    tensor = torch.from_numpy(np.stack(outputs)).to(
        device=proposal.device, dtype=torch.float64
    )
    dual_tensor = torch.from_numpy(np.stack(dual_outputs)).to(
        device=proposal.device, dtype=torch.float64
    )
    return tensor, dual_tensor


def _piqp_box_fiber_projection(
    proposal: torch.Tensor,
    target: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rare certified interior-point fallback for degenerate box corners."""
    try:
        import piqp  # type: ignore
        from scipy import sparse
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - environment guard
        raise GaugeGeometryError(f"PIQP_FALLBACK_UNAVAILABLE:{exc!r}") from exc

    q_np = geometry.Q.detach().cpu().numpy().astype(np.float64, copy=False)
    key = geometry.info.rows_sha256
    cached = _PIQP_MATRIX_CACHE.get(key)
    if cached is None:
        cached = (
            sparse.eye(geometry.n, format="csc", dtype=np.float64),
            sparse.csc_matrix(q_np),
            np.zeros(geometry.n, dtype=np.float64),
            np.ones(geometry.n, dtype=np.float64),
        )
        _PIQP_MATRIX_CACHE[key] = cached
    p_matrix, equality_matrix, lower_box, upper_box = cached
    proposal_np = proposal.detach().cpu().numpy().astype(np.float64, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    outputs: list[np.ndarray] = []
    dual_outputs: list[np.ndarray] = []
    for vector, record in zip(proposal_np, target_np):
        solver = piqp.SparseSolver()
        solver.settings.eps_abs = 1e-9
        solver.settings.eps_rel = 1e-9
        solver.settings.check_duality_gap = False
        solver.settings.max_iter = 250
        solver.settings.verbose = False
        solver.setup(
            p_matrix,
            -vector,
            equality_matrix,
            record,
            None,
            None,
            None,
            lower_box,
            upper_box,
        )
        solver.solve()
        dual_start = np.asarray(solver.result.y, dtype=np.float64)

        def root_residual(dual_vector: np.ndarray) -> np.ndarray:
            return np.clip(vector - dual_vector @ q_np, 0.0, 1.0) @ q_np.T - record

        def root_jacobian(dual_vector: np.ndarray) -> np.ndarray:
            preclip = vector - dual_vector @ q_np
            free = ((preclip > 0.0) & (preclip < 1.0)).astype(np.float64)
            return -np.einsum("rn,n,sn->rs", q_np, free, q_np, optimize=True)

        polished = least_squares(
            root_residual,
            dual_start,
            jac=root_jacobian,
            method="trf",
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=1000,
        )
        dual_value = np.asarray(polished.x, dtype=np.float64)
        outputs.append(np.clip(vector - dual_value @ q_np, 0.0, 1.0))
        dual_outputs.append(dual_value)
    return (
        torch.from_numpy(np.stack(outputs)).to(proposal.device, torch.float64),
        torch.from_numpy(np.stack(dual_outputs)).to(proposal.device, torch.float64),
    )


def project_box_fiber_exact_dual(
    proposal_flat: torch.Tensor,
    z: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    max_iterations: int = 64,
    record_tolerance: float = 1e-10,
    step_tolerance: float = 1e-10,
    jacobian_ridge: float = 1e-12,
    max_backtracking: int = 24,
) -> GaugeDykstraResult:
    """Exact Euclidean box-fiber projection via its low-dimensional dual.

    For a dual vector ``lambda``, the primal minimizer is
    ``clip(proposal - lambda @ Q, 0, 1)``.  A damped semismooth Newton
    iteration solves its equality residual in the intrinsic row coordinates.
    The returned point is the same convex projection targeted by converged
    Dykstra, but avoids Dykstra's very slow asymptotic feasibility tail when
    only a small fraction of pixels is active at a box boundary.
    """
    if proposal_flat.ndim != 2 or proposal_flat.shape[1] != geometry.n:
        raise GaugeGeometryError(
            f"PROPOSAL_SHAPE_MISMATCH:{tuple(proposal_flat.shape)}:{geometry.n}"
        )
    if z.shape != (proposal_flat.shape[0], geometry.rank):
        raise GaugeGeometryError(
            f"INTRINSIC_SHAPE_MISMATCH:{tuple(z.shape)}:{proposal_flat.shape[0]}:{geometry.rank}"
        )
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    proposal = proposal_flat.to(device=geometry.Q.device, dtype=torch.float64)
    target = z.to(device=geometry.Q.device, dtype=torch.float64)
    q = geometry.Q.to(dtype=torch.float64)
    batch = proposal.shape[0]
    dual = torch.zeros(batch, geometry.rank, device=proposal.device, dtype=torch.float64)
    eye = torch.eye(geometry.rank, device=proposal.device, dtype=torch.float64)
    previous: torch.Tensor | None = None
    current = proposal.clamp(0.0, 1.0)
    converged = False
    completed = 0
    max_step = float("inf")
    for iteration in range(int(max_iterations)):
        preclip = proposal - dual @ q
        current = preclip.clamp(0.0, 1.0)
        residual = current @ q.T - target
        relative = torch.linalg.norm(residual, dim=1) / torch.linalg.norm(
            target, dim=1
        ).clamp_min(1e-12)
        if previous is None:
            step_change = torch.full_like(relative, float("inf"))
        else:
            step_change = (current - previous).abs().amax(dim=1)
        completed = iteration + 1
        max_step = float(step_change.max().detach().cpu())
        converged_mask = relative <= float(record_tolerance)
        if bool(converged_mask.all()):
            converged = True
            break

        free = ((preclip > 0.0) & (preclip < 1.0)).to(torch.float64)
        generalized_hessian = torch.einsum("rn,bn,sn->brs", q, free, q)
        system = generalized_hessian + float(jacobian_ridge) * eye.unsqueeze(0)
        solution, info = torch.linalg.solve_ex(system, residual.unsqueeze(-1))
        delta = solution.squeeze(-1)
        if bool((info != 0).any()) or not bool(torch.isfinite(delta).all()):
            delta = torch.linalg.lstsq(system, residual.unsqueeze(-1)).solution.squeeze(-1)
        delta = torch.where(converged_mask[:, None], torch.zeros_like(delta), delta)

        base_dual_value = 0.5 * (current - proposal).square().sum(dim=1) + (
            dual * residual
        ).sum(dim=1)
        directional_gain = (residual * delta).sum(dim=1)
        bad_direction = (~converged_mask) & (directional_gain <= 0.0)
        if bool(bad_direction.any()):
            delta[bad_direction] = residual[bad_direction]
            directional_gain = (residual * delta).sum(dim=1)
        step_size = 1.0
        accepted_dual: torch.Tensor | None = None
        for _ in range(int(max_backtracking)):
            candidate_dual = dual + step_size * delta
            candidate = (proposal - candidate_dual @ q).clamp(0.0, 1.0)
            candidate_residual = candidate @ q.T - target
            candidate_dual_value = 0.5 * (candidate - proposal).square().sum(dim=1) + (
                candidate_dual * candidate_residual
            ).sum(dim=1)
            sufficient = converged_mask | (
                candidate_dual_value
                >= base_dual_value + 1e-4 * step_size * directional_gain - 1e-12
            )
            if bool(sufficient.all()):
                accepted_dual = candidate_dual
                break
            step_size *= 0.5
        if accepted_dual is None:
            break
        previous = current
        dual = accepted_dual

    final_preclip = proposal - dual @ q
    current = final_preclip.clamp(0.0, 1.0)
    if previous is not None:
        max_step = float((current - previous).abs().max().detach().cpu())

    def certificate(active_dual: torch.Tensor):
        preclip_value = proposal - active_dual @ q
        image = preclip_value.clamp(0.0, 1.0)
        residual_value = image @ q.T - target
        relative_value = torch.linalg.norm(residual_value, dim=1) / torch.linalg.norm(
            target, dim=1
        ).clamp_min(1e-12)
        infinity_value = residual_value.abs().amax(dim=1) / (
            1.0 + target.abs().amax(dim=1)
        )
        box_residual = torch.maximum(
            torch.relu(-image).amax(dim=1), torch.relu(image - 1.0).amax(dim=1)
        )
        dual_image = active_dual @ q
        prox_residual = (image - preclip_value.clamp(0.0, 1.0)).abs().amax(dim=1) / (
            1.0 + proposal.abs().amax(dim=1) + dual_image.abs().amax(dim=1)
        )
        stationarity = image - proposal + dual_image
        lower_multiplier = torch.relu(stationarity)
        upper_multiplier = torch.relu(-stationarity)
        complementarity = torch.maximum(
            (lower_multiplier * image).abs().amax(dim=1),
            (upper_multiplier * (1.0 - image)).abs().amax(dim=1),
        ) / (1.0 + stationarity.abs().amax(dim=1))
        finite = (
            torch.isfinite(image).all(dim=1)
            & torch.isfinite(active_dual).all(dim=1)
            & torch.isfinite(residual_value).all(dim=1)
        )
        passed_mask = (
            finite
            & (relative_value <= float(record_tolerance))
            & (infinity_value <= 1e-11)
            & (box_residual <= 1e-12)
            & (prox_residual <= 1e-11)
            & (complementarity <= 1e-10)
        )
        return image, passed_mask, {
            "relative": float(relative_value.max().detach().cpu()),
            "infinity": float(infinity_value.max().detach().cpu()),
            "box": float(box_residual.max().detach().cpu()),
            "proximal": float(prox_residual.max().detach().cpu()),
            "complementarity": float(complementarity.max().detach().cpu()),
        }

    current, passed_mask, certificate_values = certificate(dual)
    if not bool(passed_mask.all()):
        failed = ~passed_mask
        _fallback_image, fallback_dual = _lbfgs_box_fiber_projection(
            proposal[failed], target[failed], geometry, dual[failed]
        )
        dual = dual.clone()
        dual[failed] = fallback_dual
        current, passed_mask, certificate_values = certificate(dual)
    if not bool(passed_mask.all()):
        failed = ~passed_mask
        _reference_image, reference_dual = _piqp_box_fiber_projection(
            proposal[failed], target[failed], geometry
        )
        dual = dual.clone()
        dual[failed] = reference_dual
        current, passed_mask, certificate_values = certificate(dual)
    converged = bool(passed_mask.all())
    return GaugeDykstraResult(
        image_flat=current,
        iterations=completed,
        converged=converged,
        max_relative_record_error=certificate_values["relative"],
        max_step_change=max_step,
        max_box_violation=certificate_values["box"],
        max_stationarity_residual=certificate_values["proximal"],
        max_intrinsic_infinity_residual=certificate_values["infinity"],
        max_proximal_residual=certificate_values["proximal"],
        max_complementarity_residual=certificate_values["complementarity"],
    )


def project_box_fiber_q(
    proposal_flat: torch.Tensor,
    z: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    iterations: int = 12,
    exact: bool = False,
    record_tolerance: float = 1e-7,
    step_tolerance: float = 1e-8,
) -> GaugeDykstraResult:
    if proposal_flat.ndim != 2 or proposal_flat.shape[1] != geometry.n:
        raise GaugeGeometryError(
            f"PROPOSAL_SHAPE_MISMATCH:{tuple(proposal_flat.shape)}:{geometry.n}"
        )
    if z.shape != (proposal_flat.shape[0], geometry.rank):
        raise GaugeGeometryError(
            f"INTRINSIC_SHAPE_MISMATCH:{tuple(z.shape)}:{proposal_flat.shape[0]}:{geometry.rank}"
        )
    max_iterations = 256 if exact else int(iterations)
    if max_iterations < 1:
        raise ValueError("iterations must be positive")
    dtype = torch.float64 if exact else torch.float32
    current = proposal_flat.to(device=geometry.Q.device, dtype=dtype)
    z_work = z.to(device=geometry.Q.device, dtype=dtype)
    affine_correction = torch.zeros_like(current)
    box_correction = torch.zeros_like(current)
    converged = False
    max_step_change = float("inf")
    completed = 0
    for step in range(max_iterations):
        previous = current
        affine_input = current + affine_correction
        on_fiber = geometry.affine_project_flat(affine_input, z_work)
        affine_correction = affine_input - on_fiber
        box_input = on_fiber + box_correction
        current = box_input.clamp(0.0, 1.0)
        box_correction = box_input - current
        completed = step + 1
        if exact:
            with torch.no_grad():
                record = geometry.relative_record_error(current, z_work).max()
                change = (current - previous).abs().max()
                converged = bool(
                    record <= float(record_tolerance)
                    and change <= float(step_tolerance)
                )
                max_step_change = float(change.detach().cpu())
            if converged:
                break
    record_value = float(
        geometry.relative_record_error(current, z_work).max().detach().cpu()
    )
    box_value = float(
        torch.maximum(torch.relu(-current).amax(), torch.relu(current - 1.0).amax())
        .detach()
        .cpu()
    )
    if not exact:
        max_step_change = float((current - previous).abs().max().detach().cpu())
        converged = True
    return GaugeDykstraResult(
        image_flat=current,
        iterations=completed,
        converged=bool(converged),
        max_relative_record_error=record_value,
        max_step_change=max_step_change,
        max_box_violation=box_value,
    )
