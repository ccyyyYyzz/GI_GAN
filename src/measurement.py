import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class StaleSolverCacheError(RuntimeError):
    """Raised when the cached K = A A^T + lambda I / Cholesky no longer matches A."""


@dataclass(frozen=True)
class MeasurementShape:
    img_size: int
    n: int
    m: int


def _sylvester_hadamard(n: int) -> torch.Tensor:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard size must be a positive power of two.")
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H


def _hadamard_sign_changes(H: torch.Tensor) -> torch.Tensor:
    return (H[:, 1:] != H[:, :-1]).sum(dim=1)


HADAMARD_PATTERN_TYPES = {
    "hadamard",
    "scrambled_hadamard",
    "lowfreq_hadamard",
    "hybrid_hadamard_random",
}


def _bool_config(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _clean_hadamard_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if not str(k).startswith("_")}


def _gram_stats(A: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        row_norm = A.norm(dim=1)
        gram = A @ A.T
        diag = torch.diagonal(gram)
        if gram.shape[0] > 1:
            offdiag = gram - torch.diag_embed(diag)
            offdiag_abs_mean = float(offdiag.abs().sum().detach().cpu() / (gram.shape[0] * (gram.shape[0] - 1)))
            offdiag_abs_max = float(offdiag.abs().max().detach().cpu())
        else:
            offdiag_abs_mean = 0.0
            offdiag_abs_max = 0.0
        return {
            "row_norm_mean": float(row_norm.mean().detach().cpu()),
            "row_norm_std": float(row_norm.std(unbiased=False).detach().cpu()),
            "aa_t_diag_mean": float(diag.mean().detach().cpu()),
            "aa_t_diag_std": float(diag.std(unbiased=False).detach().cpu()),
            "aa_t_offdiag_abs_mean": offdiag_abs_mean,
            "aa_t_offdiag_abs_max": offdiag_abs_max,
        }


def _hadamard_metadata(
    n: int,
    m: int,
    pattern_type: str,
    seed: int,
    hadamard_include_dc: bool = True,
    hadamard_row_order: str = "sequency",
    hadamard_skip_dc: bool = False,
    hadamard_random_column_permutation: bool = False,
    hadamard_random_row_permutation: bool = False,
    hybrid_lowfreq_fraction: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    H = _sylvester_hadamard(n)
    rows_all = torch.arange(n, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    col_perm = None
    pattern_type = str(pattern_type).lower()
    hadamard_row_order = str(hadamard_row_order or "sequency").lower()
    include_dc = _bool_config(hadamard_include_dc) and not _bool_config(hadamard_skip_dc)
    candidate_rows = rows_all if include_dc else rows_all[1:]

    changes = _hadamard_sign_changes(H)

    def natural_order(rows: torch.Tensor) -> torch.Tensor:
        return rows

    def sequency_order(rows: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(changes[rows], stable=True)
        return rows[order]

    if hadamard_row_order in {"sequency", "sign_changes", "lowfreq"}:
        ordered_rows = sequency_order(candidate_rows)
    elif hadamard_row_order in {"natural", "sequential", "index"}:
        ordered_rows = natural_order(candidate_rows)
    else:
        raise ValueError(
            "hadamard_row_order must be one of: sequency, sign_changes, lowfreq, natural, sequential, index."
        )

    if pattern_type == "hadamard":
        row_indices = ordered_rows[:m]
    elif pattern_type == "scrambled_hadamard":
        row_perm = candidate_rows[torch.randperm(len(candidate_rows), generator=generator)]
        if include_dc and row_perm.numel() > 0:
            # Keep DC available and deterministic while still scrambling the remaining rows.
            row_perm = torch.cat([torch.zeros(1, dtype=torch.long), row_perm[row_perm != 0]], dim=0)
        col_perm = torch.randperm(n, generator=generator)
        row_indices = row_perm[:m]
    elif pattern_type == "lowfreq_hadamard":
        row_indices = sequency_order(candidate_rows)[:m]
    elif pattern_type == "hybrid_hadamard_random":
        lowfreq_rows = sequency_order(candidate_rows)
        k_low = max(0, min(m, int(round(float(hybrid_lowfreq_fraction) * m))))
        selected_low = lowfreq_rows[:k_low]
        used = torch.zeros(n, dtype=torch.bool)
        used[selected_low] = True
        remaining = candidate_rows[~used[candidate_rows]]
        random_rows = remaining[torch.randperm(len(remaining), generator=generator)]
        row_indices = torch.cat([selected_low, random_rows[: max(0, m - k_low)]], dim=0)
    else:
        raise ValueError(f"Unsupported Hadamard pattern_type: {pattern_type}")

    if _bool_config(hadamard_random_row_permutation) and pattern_type != "scrambled_hadamard":
        row_indices = row_indices[torch.randperm(row_indices.numel(), generator=generator)]
    if _bool_config(hadamard_random_column_permutation) and col_perm is None:
        col_perm = torch.randperm(n, generator=generator)

    if row_indices.numel() < m:
        raise ValueError(f"Requested m={m} Hadamard rows but only found {row_indices.numel()}.")
    return H, col_perm, row_indices[:m].contiguous(), changes


def create_hadamard_measurement_matrix(
    img_size: int,
    sampling_ratio: float,
    pattern_type: str,
    device: str | torch.device = "cpu",
    seed: int = 42,
    normalization: str | None = None,
    matrix_normalization: str | None = None,
    hadamard_include_dc: bool = True,
    hadamard_row_order: str = "sequency",
    hadamard_skip_dc: bool = False,
    hadamard_random_column_permutation: bool = False,
    hadamard_random_row_permutation: bool = False,
    hybrid_lowfreq_fraction: float = 0.7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str | int | float | None]]:
    n = int(img_size) * int(img_size)
    m = max(1, int(round(float(sampling_ratio) * n)))
    matrix_normalization = str(
        matrix_normalization
        or ("legacy_sqrt_m" if normalization == "row_norm_sqrt_n_over_m" else normalization)
        or "orthonormal_rows"
    ).lower()
    H, col_perm, row_indices, changes = _hadamard_metadata(
        n=n,
        m=m,
        pattern_type=pattern_type,
        seed=seed,
        hadamard_include_dc=hadamard_include_dc,
        hadamard_row_order=hadamard_row_order,
        hadamard_skip_dc=hadamard_skip_dc,
        hadamard_random_column_permutation=hadamard_random_column_permutation,
        hadamard_random_row_permutation=hadamard_random_row_permutation,
        hybrid_lowfreq_fraction=hybrid_lowfreq_fraction,
    )
    H_for_basis = H if col_perm is None else H[:, col_perm]
    if matrix_normalization == "orthonormal_rows":
        A = H_for_basis[row_indices].clone() / math.sqrt(n)
        coeff_scale = 1.0
    elif matrix_normalization == "legacy_sqrt_m":
        A = H_for_basis[row_indices].clone() / math.sqrt(m)
        coeff_scale = math.sqrt(m / n)
    elif matrix_normalization in {"none", ""}:
        A = H_for_basis[row_indices].clone()
        coeff_scale = 1.0 / math.sqrt(n)
    else:
        raise ValueError(
            "matrix_normalization must be one of: orthonormal_rows, legacy_sqrt_m, none."
        )
    stats = _gram_stats(A)
    selected_rows = row_indices[:m].contiguous()
    include_dc = _bool_config(hadamard_include_dc) and not _bool_config(hadamard_skip_dc)
    first_changes = changes[selected_rows[: min(16, selected_rows.numel())]].detach().cpu().tolist()
    metadata = {
        "_basis": (H_for_basis / math.sqrt(n)).to(device=torch.device(device), dtype=torch.float32),
        "_row_indices": selected_rows.to(device=torch.device(device)),
        "col_perm": None if col_perm is None else col_perm.to(device=torch.device(device)),
        "_coeff_scale": float(coeff_scale),
        "pattern_type": pattern_type,
        "matrix_normalization": matrix_normalization,
        "normalization": matrix_normalization,
        "selected_rows": selected_rows.detach().cpu().tolist(),
        "selected_rows_preview": selected_rows[: min(32, selected_rows.numel())].detach().cpu().tolist(),
        "selected_sign_changes_preview": first_changes,
        "hadamard_include_dc": bool(include_dc),
        "hadamard_row_order": hadamard_row_order,
        "hadamard_skip_dc": bool(hadamard_skip_dc),
        "hadamard_random_column_permutation": bool(col_perm is not None),
        "hadamard_random_row_permutation": bool(
            pattern_type == "scrambled_hadamard" or _bool_config(hadamard_random_row_permutation)
        ),
        "dc_row_selected": bool((selected_rows == 0).any().item()),
        "n": n,
        "m": m,
        **stats,
    }
    return A.to(device=torch.device(device), dtype=torch.float32), metadata


def create_fixed_measurement_matrix(
    img_size: int,
    sampling_ratio: float,
    pattern_type: str = "rademacher",
    device: str | torch.device = "cpu",
    seed: int = 42,
    normalization: str = "row_norm_sqrt_n_over_m",
    matrix_normalization: str | None = None,
    hadamard_include_dc: bool = True,
    hadamard_row_order: str = "sequency",
    hadamard_skip_dc: bool = False,
    hadamard_random_column_permutation: bool = False,
    hadamard_random_row_permutation: bool = False,
    hybrid_lowfreq_fraction: float = 0.7,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Create the fixed measurement matrix used by GhostMeasurementOperator."""
    if img_size <= 0:
        raise ValueError("img_size must be positive.")
    if sampling_ratio <= 0:
        raise ValueError("sampling_ratio must be positive.")

    n = int(img_size) * int(img_size)
    m = max(1, int(round(float(sampling_ratio) * n)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    pattern_type = str(pattern_type).lower()

    if pattern_type in HADAMARD_PATTERN_TYPES:
        A, metadata = create_hadamard_measurement_matrix(
            img_size=img_size,
            sampling_ratio=sampling_ratio,
            pattern_type=pattern_type,
            device=device,
            seed=seed,
            normalization=normalization,
            matrix_normalization=matrix_normalization,
            hadamard_include_dc=hadamard_include_dc,
            hadamard_row_order=hadamard_row_order,
            hadamard_skip_dc=hadamard_skip_dc,
            hadamard_random_column_permutation=hadamard_random_column_permutation,
            hadamard_random_row_permutation=hadamard_random_row_permutation,
            hybrid_lowfreq_fraction=hybrid_lowfreq_fraction,
        )
        return (A, metadata) if return_metadata else A
    if pattern_type == "rademacher":
        A = torch.empty(m, n).bernoulli_(0.5, generator=generator)
        A = A.mul_(2.0).sub_(1.0)
    elif pattern_type == "bernoulli":
        A = torch.empty(m, n).bernoulli_(0.5, generator=generator)
        A = A.sub_(0.5)
    elif pattern_type == "gaussian":
        A = torch.randn(m, n, generator=generator)
    else:
        raise ValueError(
            "pattern_type must be one of: rademacher, bernoulli, gaussian, "
            "hadamard, scrambled_hadamard, lowfreq_hadamard, hybrid_hadamard_random."
        )

    if matrix_normalization is not None:
        matrix_normalization = str(matrix_normalization).lower()
        if matrix_normalization in {"legacy_sqrt_m", "orthonormal_rows"}:
            normalization = "row_norm_sqrt_n_over_m"
        elif matrix_normalization in {"none", ""}:
            normalization = "none"
    normalization = str(normalization)
    if normalization == "row_norm_sqrt_n_over_m":
        row_norm = A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        A = A / row_norm * math.sqrt(n / m)
    elif normalization in {"none", ""}:
        pass
    else:
        raise ValueError(
            "normalization must be one of: row_norm_sqrt_n_over_m, none."
        )
    A = A.to(device=torch.device(device), dtype=torch.float32)
    if return_metadata:
        stats = _gram_stats(A)
        metadata = {
            "pattern_type": pattern_type,
            "matrix_normalization": matrix_normalization or normalization,
            "selected_rows": None,
            "hadamard_include_dc": None,
            "hadamard_row_order": None,
            "n": n,
            "m": m,
            **stats,
        }
        return A, metadata
    return A


class GhostMeasurementOperator:
    """Fixed ghost-imaging operator with data and null-space projections.

    The operator stores A in shape [m, n] and never materializes the n x n
    null-space projection matrix. All projections are evaluated through solves
    with K = A A^T + lambda I.
    """

    def __init__(
        self,
        img_size: int = 64,
        sampling_ratio: float = 0.05,
        pattern_type: str = "rademacher",
        noise_std: float = 0.01,
        lambda_dc: float = 1e-3,
        backprojection_mode: str = "ridge_pinv",
        matrix_normalization: str = "orthonormal_rows",
        hadamard_include_dc: bool = True,
        hadamard_row_order: str = "sequency",
        hadamard_skip_dc: bool = False,
        hadamard_random_column_permutation: bool = False,
        hadamard_random_row_permutation: bool = False,
        hybrid_lowfreq_fraction: float = 0.7,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        if img_size <= 0:
            raise ValueError("img_size must be positive.")
        if sampling_ratio <= 0:
            raise ValueError("sampling_ratio must be positive.")

        self.img_size = int(img_size)
        self.n = self.img_size * self.img_size
        self.m = max(1, int(round(float(sampling_ratio) * self.n)))
        self.sampling_ratio = float(sampling_ratio)
        self.pattern_type = pattern_type.lower()
        self.noise_std = float(noise_std)
        self.lambda_dc = float(lambda_dc)
        self.backprojection_mode = str(backprojection_mode)
        self.matrix_normalization = str(matrix_normalization or "orthonormal_rows").lower()
        if self.pattern_type not in HADAMARD_PATTERN_TYPES and self.matrix_normalization == "orthonormal_rows":
            self.matrix_normalization = "legacy_sqrt_m"
        self.hadamard_include_dc = _bool_config(hadamard_include_dc)
        self.hadamard_row_order = str(hadamard_row_order or "sequency")
        self.hadamard_skip_dc = _bool_config(hadamard_skip_dc)
        self.hadamard_random_column_permutation = _bool_config(hadamard_random_column_permutation)
        self.hadamard_random_row_permutation = _bool_config(hadamard_random_row_permutation)
        self.hybrid_lowfreq_fraction = float(hybrid_lowfreq_fraction)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.shape = MeasurementShape(self.img_size, self.n, self.m)

        self.hadamard_metadata = None
        self.measurement_metadata = None
        if self.pattern_type in HADAMARD_PATTERN_TYPES:
            self.A, self.hadamard_metadata = create_hadamard_measurement_matrix(
                img_size=self.img_size,
                sampling_ratio=self.sampling_ratio,
                pattern_type=self.pattern_type,
                device=self.device,
                seed=self.seed,
                matrix_normalization=self.matrix_normalization,
                hadamard_include_dc=self.hadamard_include_dc,
                hadamard_row_order=self.hadamard_row_order,
                hadamard_skip_dc=self.hadamard_skip_dc,
                hadamard_random_column_permutation=self.hadamard_random_column_permutation,
                hadamard_random_row_permutation=self.hadamard_random_row_permutation,
                hybrid_lowfreq_fraction=self.hybrid_lowfreq_fraction,
            )
            self.measurement_metadata = _clean_hadamard_metadata(self.hadamard_metadata)
        else:
            fixed_normalization = (
                "none"
                if self.matrix_normalization in {"none", ""}
                else "row_norm_sqrt_n_over_m"
            )
            self.A, self.measurement_metadata = create_fixed_measurement_matrix(
                img_size=self.img_size,
                sampling_ratio=self.sampling_ratio,
                pattern_type=self.pattern_type,
                device=self.device,
                seed=self.seed,
                normalization=fixed_normalization,
                matrix_normalization=self.matrix_normalization,
                hybrid_lowfreq_fraction=self.hybrid_lowfreq_fraction,
                return_metadata=True,
            )
        self._rebuild_solver_cache()
        self.assert_solver_cache_fresh()

    def _rebuild_solver_cache(self) -> None:
        eye = torch.eye(self.m, device=self.device, dtype=self.A.dtype)
        self.K = self.A @ self.A.T + self.lambda_dc * eye
        self._chol = None
        self._use_cholesky = True
        try:
            self._chol = torch.linalg.cholesky(self.K)
        except RuntimeError:
            self._use_cholesky = False
        # Float64 shadow of the solver cache, used only by
        # assert_solver_cache_fresh to detect stale caches.
        A64 = self.A.detach().to(torch.float64)
        eye64 = torch.eye(self.m, device=self.device, dtype=torch.float64)
        self._K64 = A64 @ A64.T + self.lambda_dc * eye64
        try:
            self._chol64 = torch.linalg.cholesky(self._K64)
        except RuntimeError:
            self._chol64 = None

    def assert_solver_cache_fresh(self, tol: float = 1e-10, seed: int = 20260612) -> dict[str, float]:
        """Assert the cached K/Cholesky matches the CURRENT A.

        A fresh float64 solve of (A A^T + lambda I) z = b is compared against a
        solve using the cached float64 Cholesky factor on a random vector.
        Cache freshness is decided from (i) the cached-vs-current K mismatch
        and (ii) the cached solution's backward residual.  We report the
        forward difference between two solvers for diagnostics, but do not use
        it as a freshness gate: for an ill-conditioned but fresh K, two stable
        solvers may legitimately differ by more than 1e-10 in forward error.
        The float32 runtime cache is checked at float32 rounding accuracy.
        Raises StaleSolverCacheError on violation. Guards against the
        historical exact-A override incident that left a stale cache.
        """
        if self.K is None or getattr(self, "_K64", None) is None:
            raise StaleSolverCacheError(
                "Solver cache is missing (K or _K64 is None); call _rebuild_solver_cache()."
            )
        if self.K.shape[0] != self.A.shape[0] or self._K64.shape[0] != self.A.shape[0]:
            raise StaleSolverCacheError(
                f"Solver cache shape {tuple(self.K.shape)} does not match current A "
                f"with m={self.A.shape[0]}; the cache is stale."
            )
        A64 = self.A.detach().to(torch.float64)
        eye64 = torch.eye(self.m, device=self.device, dtype=torch.float64)
        K_fresh = A64 @ A64.T + self.lambda_dc * eye64
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        b = torch.randn(self.m, 1, dtype=torch.float64, generator=gen).to(self.device)
        z_fresh = torch.linalg.solve(K_fresh, b)
        if self._chol64 is not None:
            z_cached = torch.cholesky_solve(b, self._chol64)
        else:
            z_cached = torch.linalg.solve(self._K64, b)
        rel_solve = float(
            (torch.linalg.norm(z_fresh - z_cached) / torch.linalg.norm(z_fresh).clamp_min(1e-300)).item()
        )
        rel_K64 = float(
            (
                torch.linalg.norm(self._K64 - K_fresh)
                / torch.linalg.norm(K_fresh).clamp_min(1e-300)
            ).item()
        )
        rel_cached_residual = float(
            (
                torch.linalg.norm(K_fresh @ z_cached - b)
                / torch.linalg.norm(b).clamp_min(1e-300)
            ).item()
        )
        rel_K32 = float(
            (
                torch.linalg.norm(self.K.detach().to(torch.float64) - K_fresh)
                / torch.linalg.norm(K_fresh).clamp_min(1e-300)
            ).item()
        )
        if rel_K64 > float(tol):
            raise StaleSolverCacheError(
                f"Stale solver cache: cached K disagrees with current A "
                f"(rel={rel_K64:.3e} > {tol:.1e}). "
                "A was changed without rebuilding K = A A^T + lambda I and its Cholesky cache."
            )
        if rel_cached_residual > float(tol):
            raise StaleSolverCacheError(
                f"Stale solver cache: cached solve has excessive backward residual "
                f"(rel={rel_cached_residual:.3e} > {tol:.1e}). "
                "A was changed without rebuilding K = A A^T + lambda I and its Cholesky cache."
            )
        # float32 cache only needs to match to float32 rounding, but a stale
        # cache produces O(1) mismatch, far above this threshold.
        if rel_K32 > 1e-4:
            raise StaleSolverCacheError(
                f"Stale float32 solver cache: ||K_cached - K_fresh||/||K_fresh|| = {rel_K32:.3e} > 1e-4."
            )
        return {
            "rel_solve_fresh_vs_cached": rel_solve,
            "rel_K64_vs_fresh": rel_K64,
            "rel_cached_backward_residual": rel_cached_residual,
            "rel_K32_vs_fresh": rel_K32,
        }

    def set_A_override(
        self,
        A: torch.Tensor,
        metadata: dict[str, Any] | None = None,
        rebuild_cache: bool = True,
    ) -> dict[str, float | int | str | bool]:
        """Replace the fixed measurement matrix and rebuild dependent solves."""
        if not torch.is_tensor(A):
            raise TypeError("A override must be a torch.Tensor.")
        if A.ndim != 2:
            raise ValueError(f"A override must have shape [m, n], got {tuple(A.shape)}.")
        m, n = int(A.shape[0]), int(A.shape[1])
        expected_n = self.img_size * self.img_size
        if n != expected_n:
            raise ValueError(f"A override n={n} does not match img_size-derived n={expected_n}.")
        A = A.to(device=self.device, dtype=torch.float32).contiguous()
        self.A = A
        self.m = m
        self.n = n
        self.sampling_ratio = float(m / n)
        self.shape = MeasurementShape(self.img_size, self.n, self.m)
        stats = _gram_stats(self.A)
        override_metadata = {
            "override_active": True,
            "override_source": "set_A_override",
            "n": self.n,
            "m": self.m,
            "sampling_ratio": self.sampling_ratio,
            **stats,
        }
        if metadata:
            override_metadata.update(metadata)
        self.measurement_metadata = override_metadata
        if self.pattern_type in HADAMARD_PATTERN_TYPES:
            self.hadamard_metadata = None
        if not rebuild_cache:
            raise StaleSolverCacheError(
                "set_A_override(rebuild_cache=False) is forbidden: any A override "
                "must rebuild K = A A^T + lambda I and its Cholesky cache "
                "(historical stale-cache incident)."
            )
        self._rebuild_solver_cache()
        cache_check = self.assert_solver_cache_fresh()
        return {
            "m": self.m,
            "n": self.n,
            "sampling_ratio": self.sampling_ratio,
            "row_norm_mean": stats["row_norm_mean"],
            "row_norm_std": stats["row_norm_std"],
            "cache_rebuilt": True,
            "uses_cholesky": bool(self._use_cholesky),
            "solver_cache_rel_solve": cache_check["rel_solve_fresh_vs_cached"],
        }

    def flatten_img(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected image tensor with shape [B, 1, H, W].")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected spatial size {self.img_size}x{self.img_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        return x.reshape(x.shape[0], self.n)

    def unflatten_img(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected flat tensor with shape [B, {self.n}].")
        return v.reshape(v.shape[0], 1, self.img_size, self.img_size)

    def A_forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected vector tensor with shape [B, {self.n}].")
        return v @ self.A.T

    def AT_forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2 or y.shape[1] != self.m:
            raise ValueError(f"Expected measurement tensor with shape [B, {self.m}].")
        return y @ self.A

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A_forward(self.flatten_img(x))
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return y

    def solve_K(self, b: torch.Tensor) -> torch.Tensor:
        if b.ndim != 2 or b.shape[1] != self.m:
            raise ValueError(f"Expected right-hand side with shape [B, {self.m}].")
        rhs = b.T.contiguous()
        if self._use_cholesky and self._chol is not None:
            z = torch.cholesky_solve(rhs, self._chol)
        else:
            z = torch.linalg.solve(self.K, rhs)
        return z.T.contiguous()

    def data_solution(self, y: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        mode = str(mode or self.backprojection_mode or "ridge_pinv").lower()
        if mode == "ridge_pinv":
            return self.AT_forward(self.solve_K(y))
        if mode == "adjoint":
            return self.AT_forward(y)
        if mode == "hadamard_zero_filled":
            return self.hadamard_zero_filled_solution(y)
        if mode == "learned_backprojection":
            return self.AT_forward(self.solve_K(y))
        raise ValueError(
            "backprojection_mode must be one of: ridge_pinv, adjoint, "
            "hadamard_zero_filled, learned_backprojection."
        )

    def hadamard_zero_filled_solution(self, y: torch.Tensor) -> torch.Tensor:
        if self.hadamard_metadata is None:
            raise ValueError(
                "hadamard_zero_filled backprojection is only supported for "
                "hadamard, lowfreq_hadamard, scrambled_hadamard, and hybrid_hadamard_random."
            )
        basis = self.hadamard_metadata["_basis"].to(device=y.device, dtype=y.dtype)
        row_indices = self.hadamard_metadata["_row_indices"].to(device=y.device)
        coeff_scale = float(self.hadamard_metadata.get("_coeff_scale", 1.0))
        coeff = torch.zeros(y.shape[0], self.n, device=y.device, dtype=y.dtype)
        coeff[:, row_indices] = y * coeff_scale
        return coeff @ basis

    def null_project(self, v: torch.Tensor) -> torch.Tensor:
        return v - self.AT_forward(self.solve_K(self.A_forward(v)))

    def dc_project(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return v - self.AT_forward(self.solve_K(self.A_forward(v) - y))

    def get_current_A(self) -> torch.Tensor:
        return self.A

    def get_pattern_stats(self) -> dict[str, float | str]:
        A_norm = self.A / self.A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gram = A_norm @ A_norm.T
        offdiag = gram - torch.diag_embed(torch.diagonal(gram))
        denom = max(1, self.m * (self.m - 1))
        return {
            "pattern_mode": "fixed",
            "pattern_type": self.pattern_type,
            "backprojection_mode": self.backprojection_mode,
            "matrix_normalization": self.matrix_normalization,
            "hadamard_include_dc": self.hadamard_include_dc if self.hadamard_metadata is not None else float("nan"),
            "hadamard_row_order": self.hadamard_row_order if self.hadamard_metadata is not None else "n/a",
            "mean": float(self.A.mean().detach().cpu()),
            "std": float(self.A.std(unbiased=False).detach().cpu()),
            "min": float(self.A.min().detach().cpu()),
            "max": float(self.A.max().detach().cpu()),
            "target_transmission": float("nan"),
            "binary_fraction_005_095": float("nan"),
            "row_mean_mean": float(self.A.mean(dim=1).mean().detach().cpu()),
            "row_mean_std": float(self.A.mean(dim=1).std(unbiased=False).detach().cpu()),
            "row_std_mean": float(self.A.std(dim=1, unbiased=False).mean().detach().cpu()),
            "row_std_std": float(self.A.std(dim=1, unbiased=False).std(unbiased=False).detach().cpu()),
            "mean_abs_offdiag_corr": float(offdiag.abs().sum().detach().cpu() / denom),
            **{
                f"measurement_{key}": value
                for key, value in (self.measurement_metadata or {}).items()
                if key not in {"selected_rows"}
            },
        }


class LearnablePatternBank(nn.Module):
    """Physical non-negative speckle patterns with differentiable A_eff.

    The raw physical pattern P is constrained to [0, 1]. The effective ghost
    imaging measurement matrix is centered and row-standardized from P, so the
    learned sensor remains interpretable as projected illumination patterns.
    """

    def __init__(
        self,
        img_size: int = 64,
        sampling_ratio: float = 0.05,
        pattern_mode: str = "learned_binary_ste",
        init_type: str = "bernoulli",
        tau: float = 1.0,
        target_transmission: float = 0.5,
        pattern_logit_abs_init: float = 2.0,
        balanced_target_transmission: float | None = None,
        effective_A_mode: str = "centered_standardized",
        fixed_reference_pattern_type: str = "rademacher",
        fixed_reference_normalization: str = "row_norm_sqrt_n_over_m",
        flip_threshold: float = 0.5,
        flip_noise_std: float = 0.0,
        flip_balance_rows: bool = True,
        continuous_A_normalization: str = "row_standardized",
        continuous_min_contrast: float = 0.05,
        continuous_target_contrast: float = 0.25,
        continuous_max_contrast: float = 0.5,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if img_size <= 0:
            raise ValueError("img_size must be positive.")
        if sampling_ratio <= 0:
            raise ValueError("sampling_ratio must be positive.")
        self.img_size = int(img_size)
        self.n = self.img_size * self.img_size
        self.m = max(1, int(round(float(sampling_ratio) * self.n)))
        self.sampling_ratio = float(sampling_ratio)
        self.pattern_mode = str(pattern_mode)
        self.init_type = str(init_type)
        self.tau = float(tau)
        self.target_transmission = float(target_transmission)
        self.pattern_logit_abs_init = float(pattern_logit_abs_init)
        self.effective_A_mode = str(effective_A_mode)
        self.fixed_reference_pattern_type = str(fixed_reference_pattern_type)
        self.fixed_reference_normalization = str(fixed_reference_normalization)
        self.flip_threshold = float(flip_threshold)
        self.flip_noise_std = float(flip_noise_std)
        self.flip_balance_rows = bool(flip_balance_rows)
        self.continuous_A_normalization = str(continuous_A_normalization)
        self.continuous_min_contrast = float(continuous_min_contrast)
        self.continuous_target_contrast = float(continuous_target_contrast)
        self.continuous_max_contrast = float(continuous_max_contrast)
        self.balanced_target_transmission = (
            float(target_transmission)
            if balanced_target_transmission is None
            else float(balanced_target_transmission)
        )
        self.seed = int(seed)

        fixed_reference_A = create_fixed_measurement_matrix(
            img_size=self.img_size,
            sampling_ratio=self.sampling_ratio,
            pattern_type=self.fixed_reference_pattern_type,
            device=device,
            seed=self.seed,
            normalization=self.fixed_reference_normalization,
        )
        abs_values = fixed_reference_A.abs()
        self.fixed_reference_unique_abs_values_min = float(abs_values.min().detach().cpu())
        self.fixed_reference_unique_abs_values_max = float(abs_values.max().detach().cpu())
        self.fixed_reference_scale = float(abs_values.mean().detach().cpu())
        self.exact_match_possible = (
            self.fixed_reference_unique_abs_values_max
            - self.fixed_reference_unique_abs_values_min
        ) < 1e-7
        self.register_buffer("fixed_reference_A", fixed_reference_A, persistent=False)

        logits = self._init_logits(device=torch.device(device))
        self.logits = nn.Parameter(logits)

    def _init_logits(self, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        shape = (self.m, self.n)
        target = min(max(self.target_transmission, 1e-4), 1.0 - 1e-4)
        base_logit = math.log(target / (1.0 - target))
        init_type = self.init_type.lower()

        if init_type == "bernoulli":
            bits = torch.empty(shape).bernoulli_(target, generator=generator)
            logits = base_logit + (bits * 2.0 - 1.0) * 0.10
        elif init_type == "rademacher_like":
            bits = torch.empty(shape).bernoulli_(0.5, generator=generator)
            logits = (bits * 2.0 - 1.0) * 0.25
        elif init_type == "gaussian_logits":
            logits = base_logit + 0.05 * torch.randn(shape, generator=generator)
        elif init_type in {"fixed_rademacher_match", "fixed_bernoulli_match"}:
            pattern_type = "bernoulli" if init_type == "fixed_bernoulli_match" else "rademacher"
            fixed_A = create_fixed_measurement_matrix(
                img_size=self.img_size,
                sampling_ratio=self.sampling_ratio,
                pattern_type=pattern_type,
                device="cpu",
                seed=self.seed,
                normalization=self.fixed_reference_normalization,
            )
            bits = (fixed_A > 0).float()
            logits = (bits * 2.0 - 1.0) * abs(self.pattern_logit_abs_init)
        elif init_type == "converted_from_continuous":
            bits = torch.empty(shape).bernoulli_(0.5, generator=generator)
            logits = (bits * 2.0 - 1.0) * abs(self.pattern_logit_abs_init)
        else:
            raise ValueError(
                "init_type must be one of: bernoulli, rademacher_like, gaussian_logits, "
                "fixed_rademacher_match, fixed_bernoulli_match, converted_from_continuous."
            )
        return logits.to(device=device, dtype=torch.float32)

    def _balanced_topk_binary(self, p_soft: torch.Tensor) -> torch.Tensor:
        k = int(round(self.balanced_target_transmission * self.n))
        k = max(0, min(self.n, k))
        if k == 0:
            return torch.zeros_like(p_soft)
        if k == self.n:
            return torch.ones_like(p_soft)
        topk = torch.topk(p_soft, k=k, dim=1).indices
        p_hard = torch.zeros_like(p_soft)
        return p_hard.scatter(1, topk, 1.0)

    def set_tau(self, tau: float) -> None:
        self.tau = float(max(tau, 1e-6))

    def get_soft_patterns(self) -> torch.Tensor:
        return torch.sigmoid(self.logits / max(self.tau, 1e-6))

    def _soft_patterns_for_hard(self) -> torch.Tensor:
        logits = self.logits
        if (
            self.training
            and self.pattern_mode == "learned_flip_aware_binary_ste"
            and self.flip_noise_std > 0.0
        ):
            logits = logits + float(self.flip_noise_std) * torch.randn_like(logits)
        return torch.sigmoid(logits / max(self.tau, 1e-6))

    def get_hard_patterns(self, *, noisy: bool = False) -> torch.Tensor:
        p_soft = self._soft_patterns_for_hard() if noisy else self.get_soft_patterns()
        if self.pattern_mode == "learned_balanced_binary_ste":
            return self._balanced_topk_binary(p_soft)
        if self.pattern_mode == "learned_flip_aware_binary_ste":
            if self.flip_balance_rows:
                return self._balanced_topk_binary(p_soft)
            return (p_soft > self.flip_threshold).float()
        return (p_soft > 0.5).float()

    def get_physical_patterns(self) -> torch.Tensor:
        p_soft = self.get_soft_patterns()
        if self.pattern_mode == "learned_continuous":
            return p_soft
        if self.pattern_mode == "learned_binary_ste":
            p_hard = (p_soft > 0.5).float()
            return p_hard.detach() - p_soft.detach() + p_soft
        if self.pattern_mode == "learned_balanced_binary_ste":
            p_hard = self._balanced_topk_binary(p_soft)
            return p_hard.detach() - p_soft.detach() + p_soft
        if self.pattern_mode == "learned_flip_aware_binary_ste":
            p_soft_for_hard = self._soft_patterns_for_hard()
            if self.flip_balance_rows:
                p_hard = self._balanced_topk_binary(p_soft_for_hard)
            else:
                p_hard = (p_soft_for_hard > self.flip_threshold).float()
            return p_hard.detach() - p_soft_for_hard.detach() + p_soft_for_hard
        raise ValueError(
            "pattern_mode must be one of: learned_continuous, learned_binary_ste, "
            "learned_balanced_binary_ste, learned_flip_aware_binary_ste."
        )

    def get_effective_A(self, eps: float = 1e-6) -> torch.Tensor:
        if self.effective_A_mode == "signed_soft_train_hard_eval":
            if self.training:
                p_signed = self.get_soft_patterns()
            else:
                p_signed = self.get_hard_patterns()
            scale = torch.as_tensor(
                self.fixed_reference_scale,
                dtype=p_signed.dtype,
                device=p_signed.device,
            )
            return scale * (2.0 * p_signed - 1.0)
        p = self.get_physical_patterns()
        if self.effective_A_mode == "signed_from_physical":
            return (2.0 * p - 1.0) / math.sqrt(self.m)
        if self.effective_A_mode == "signed_exact_fixed":
            scale = torch.as_tensor(
                self.fixed_reference_scale,
                dtype=p.dtype,
                device=p.device,
            )
            return scale * (2.0 * p - 1.0)
        if self.effective_A_mode == "continuous_differential":
            row_mean = p.mean(dim=1, keepdim=True)
            p_centered = p - row_mean
            mode = self.continuous_A_normalization
            if mode == "row_standardized":
                row_std = p_centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
                return p_centered / (row_std * math.sqrt(self.m))
            if mode == "centered_fixed_scale":
                scale = torch.as_tensor(
                    self.fixed_reference_scale / max(self.continuous_target_contrast, eps),
                    dtype=p.dtype,
                    device=p.device,
                )
                return scale * (p - self.target_transmission)
            if mode == "signed_centered":
                return (2.0 * (p - self.target_transmission)) / math.sqrt(self.m)
            raise ValueError(
                "continuous_A_normalization must be one of: row_standardized, "
                "centered_fixed_scale, signed_centered."
            )
        if self.effective_A_mode != "centered_standardized":
            raise ValueError(
                "effective_A_mode must be one of: centered_standardized, "
                "signed_from_physical, signed_exact_fixed, signed_soft_train_hard_eval, "
                "continuous_differential."
            )
        row_mean = p.mean(dim=1, keepdim=True)
        p_centered = p - row_mean
        row_std = p_centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return p_centered / (row_std * math.sqrt(self.m))

    def get_pattern_stats(self) -> dict[str, float | str]:
        was_training = bool(self.training)
        with torch.no_grad():
            self.eval()
            p_soft = self.get_soft_patterns()
            p = self.get_physical_patterns()
            a_eff = self.get_effective_A()
            p_hard = self.get_hard_patterns()
            row_mean = p.mean(dim=1)
            row_std = (p - row_mean[:, None]).std(dim=1, unbiased=False)
            row_norm = a_eff.norm(dim=1)
            a_norm = a_eff / a_eff.norm(dim=1, keepdim=True).clamp_min(1e-12)
            gram = a_norm @ a_norm.T
            offdiag = gram - torch.diag_embed(torch.diagonal(gram))
            denom = max(1, self.m * (self.m - 1))
            binary_fraction = ((p < 0.05) | (p > 0.95)).float().mean()
            soft_binary_fraction = ((p_soft < 0.05) | (p_soft > 0.95)).float().mean()
            margin = (p_soft - self.flip_threshold).abs()
            physical_type = "binary"
            if self.pattern_mode == "learned_continuous" or self.effective_A_mode == "continuous_differential":
                physical_type = "continuous"
            if self.effective_A_mode == "signed_soft_train_hard_eval":
                physical_type = "soft_train_hard_eval"
            stats = {
                "pattern_mode": self.pattern_mode,
                "effective_A_mode": self.effective_A_mode,
                "pattern_physical_type": physical_type,
                "continuous_A_normalization": self.continuous_A_normalization,
                "mean": float(p.mean().detach().cpu()),
                "std": float(p.std(unbiased=False).detach().cpu()),
                "min": float(p.min().detach().cpu()),
                "max": float(p.max().detach().cpu()),
                "target_transmission": self.target_transmission,
                "binary_fraction_005_095": float(binary_fraction.detach().cpu()),
                "soft_binary_fraction_005_095": float(soft_binary_fraction.detach().cpu()),
                "row_mean_mean": float(row_mean.mean().detach().cpu()),
                "row_mean_std": float(row_mean.std(unbiased=False).detach().cpu()),
                "row_transmission_min": float(row_mean.min().detach().cpu()),
                "row_transmission_max": float(row_mean.max().detach().cpu()),
                "row_transmission_std": float(row_mean.std(unbiased=False).detach().cpu()),
                "row_std_min": float(row_std.min().detach().cpu()),
                "row_std_mean": float(row_std.mean().detach().cpu()),
                "row_std_max": float(row_std.max().detach().cpu()),
                "row_std_std": float(row_std.std(unbiased=False).detach().cpu()),
                "continuous_contrast": float(row_std.mean().detach().cpu()),
                "hard_mean": float(p_hard.mean().detach().cpu()),
                "hard_std": float(p_hard.std(unbiased=False).detach().cpu()),
                "near_threshold_fraction_0p05": float((margin < 0.05).float().mean().detach().cpu()),
                "near_threshold_fraction_0p10": float((margin < 0.10).float().mean().detach().cpu()),
                "effective_A_fro_norm": float(a_eff.norm().detach().cpu()),
                "effective_A_row_norm_mean": float(row_norm.mean().detach().cpu()),
                "effective_A_row_norm_std": float(row_norm.std(unbiased=False).detach().cpu()),
                "mean_abs_offdiag_corr": float(offdiag.abs().sum().detach().cpu() / denom),
                "exact_match_possible": bool(self.exact_match_possible),
                "fixed_reference_scale": float(self.fixed_reference_scale),
                "fixed_reference_unique_abs_values_min": float(
                    self.fixed_reference_unique_abs_values_min
                ),
                "fixed_reference_unique_abs_values_max": float(
                    self.fixed_reference_unique_abs_values_max
                ),
            }
        if was_training:
            self.train()
        return stats


class LearnableGhostMeasurementOperator:
    """Ghost-imaging operator backed by a learnable physical pattern bank."""

    def __init__(
        self,
        pattern_bank: LearnablePatternBank,
        noise_std: float = 0.01,
        lambda_dc: float = 1e-3,
        backprojection_mode: str = "ridge_pinv",
        device: str | torch.device = "cpu",
    ) -> None:
        self.pattern_bank = pattern_bank
        self.img_size = pattern_bank.img_size
        self.n = pattern_bank.n
        self.m = pattern_bank.m
        self.sampling_ratio = pattern_bank.sampling_ratio
        self.pattern_type = "learned"
        self.noise_std = float(noise_std)
        self.lambda_dc = float(lambda_dc)
        self.backprojection_mode = str(backprojection_mode)
        self.device = torch.device(device)
        self.shape = MeasurementShape(self.img_size, self.n, self.m)

    def flatten_img(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected image tensor with shape [B, 1, H, W].")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected spatial size {self.img_size}x{self.img_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        return x.reshape(x.shape[0], self.n)

    def unflatten_img(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected flat tensor with shape [B, {self.n}].")
        return v.reshape(v.shape[0], 1, self.img_size, self.img_size)

    def get_current_A(self) -> torch.Tensor:
        return self.pattern_bank.get_effective_A()

    def _solve_for_A(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if b.ndim != 2 or b.shape[1] != self.m:
            raise ValueError(f"Expected right-hand side with shape [B, {self.m}].")
        eye = torch.eye(self.m, device=A.device, dtype=A.dtype)
        K = A @ A.T + self.lambda_dc * eye
        rhs = b.T.contiguous()
        try:
            chol = torch.linalg.cholesky(K)
            z = torch.cholesky_solve(rhs, chol)
        except RuntimeError:
            z = torch.linalg.solve(K, rhs)
        return z.T.contiguous()

    def A_forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected vector tensor with shape [B, {self.n}].")
        A = self.get_current_A()
        return v @ A.T

    def AT_forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2 or y.shape[1] != self.m:
            raise ValueError(f"Expected measurement tensor with shape [B, {self.m}].")
        A = self.get_current_A()
        return y @ A

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A_forward(self.flatten_img(x))
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return y

    def solve_K(self, b: torch.Tensor) -> torch.Tensor:
        return self._solve_for_A(self.get_current_A(), b)

    def data_solution(self, y: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        mode = str(mode or self.backprojection_mode or "ridge_pinv").lower()
        if mode == "adjoint":
            return self.AT_forward(y)
        if mode in {"ridge_pinv", "hadamard_zero_filled", "learned_backprojection"}:
            if mode == "hadamard_zero_filled":
                # Learned physical operators are not Hadamard bases; keep the
                # data path measurement-consistent by falling back to ridge.
                pass
            A = self.get_current_A()
            return self._solve_for_A(A, y) @ A
        raise ValueError(
            "backprojection_mode must be one of: ridge_pinv, adjoint, "
            "hadamard_zero_filled, learned_backprojection."
        )

    def ridge_data_solution(self, y: torch.Tensor) -> torch.Tensor:
        A = self.get_current_A()
        return self._solve_for_A(A, y) @ A

    def null_project(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected vector tensor with shape [B, {self.n}].")
        A = self.get_current_A()
        return v - self._solve_for_A(A, v @ A.T) @ A

    def dc_project(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"Expected vector tensor with shape [B, {self.n}].")
        A = self.get_current_A()
        return v - self._solve_for_A(A, v @ A.T - y) @ A

    def get_pattern_stats(self) -> dict[str, float | str]:
        return self.pattern_bank.get_pattern_stats()
