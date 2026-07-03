# 02 Theory Core: Range/Null Decomposition and the Measurement Audit

This document derives the forward model and projection algebra that underpin every
experiment in this repo. All claims are grounded in `src/projections.py`,
`src/measurement.py`, and `tests/test_exact_projections.py`.

---

## 1. Forward model

Ghost imaging and single-pixel imaging are both instances of a linear bucket-measurement
system:

```
y = A x + eps,   A in R^{m x n},   m << n
```

- `x in R^n` — the vectorized scene (here `n = 4096` for `64x64` grayscale).
- `y in R^m` — the bucket readings (`m = 205` at 5% sampling rate).
- `A` — the fixed measurement operator, built once per run and never updated.
  Each row is one illumination pattern correlated with one bucket scalar.
- `eps` — additive noise with standard deviation `noise_std` (see
  `GhostMeasurementOperator.__init__` in `src/measurement.py`).

The central fact is `m << n`: the system is severely underdetermined.

---

## 2. Orthonormal-row assumption

For Hadamard and scrambled-Hadamard patterns the rows of `A` are set to be
orthonormal (`matrix_normalization = "orthonormal_rows"` in
`src/measurement.py`), so:

```
A A^T = I_m   (Gram matrix = identity)
```

The `ExactRangeNullProjector` in `src/projections.py` detects this:

```python
# src/projections.py lines 74-83
gram_delta = self.G - eye          # G = A A^T
row_orth_max = float(gram_delta.abs().max())
self._row_orthonormal = row_orth_max <= float(row_orthonormal_tol)
if self._row_orthonormal:
    self.solver = "row_orthonormal"
```

When `row_orthonormal_tol = 1e-7` is satisfied, the Gram solve becomes trivial
(the identity solve) and the pseudoinverse simplifies to `A^dagger = A^T`.
For general (non-orthonormal) operators a Cholesky solve of the `m x m` Gram
matrix `K = A A^T (+ lambda I for soft variants)` is used.

---

## 3. Range/null decomposition

With `A^dagger = A^T` (orthonormal-row case, or more generally the Moore-Penrose
pseudoinverse), define the two complementary orthogonal projectors:

```
P_R = A^dagger A = A^T A     (projects onto the row space of A)
P_0 = I - A^T A              (projects onto the null space / kernel of A)
```

These satisfy:

| Identity | Meaning |
|---|---|
| `P_R + P_0 = I` | partition of unity |
| `P_R^2 = P_R`, `P_0^2 = P_0` | idempotent (each is a true projector) |
| `P_R P_0 = 0` | orthogonal (subspaces are perpendicular) |
| `A P_0 = 0` | the null-space projector annihilates A |

The last identity follows by direct substitution:

```
A P_0 = A (I - A^T A) = A - (A A^T) A = A - I_m A = 0
```

using `A A^T = I_m` in the orthonormal-row case.

**Consequence.** For any vector `x`:

```
A x = A (P_R x + P_0 x) = A P_R x + 0
```

The measurement `y = A x` depends **only** on the row-space component `P_R x`.
The null-space component `P_0 x` (`n - m = 3891` dimensions at 5% sampling) is
entirely invisible to the bucket.

In code, these projectors are applied without materializing the `n x n` matrix
(which at `n = 4096` would be 64 MB per instance). Instead,
`ExactRangeNullProjector.row_project_flat` and `null_project_flat` compose an
`A`-forward pass, a Gram solve, and an `A^T`-backward pass:

```python
# src/projections.py lines 139-145
def row_project_flat(self, v):
    y = self.A_forward(v)            # [B, m]
    return self.AT_forward(self.solve_gram(y))   # [B, n]

def null_project_flat(self, v):
    v_exact = v.to(device=self.device, dtype=self.dtype)
    return v_exact - self.row_project_flat(v_exact)
```

The test suite verifies four numerical properties for both Rademacher and
scrambled-Hadamard patterns at `img_size=16`, `sampling_ratio=0.25`, `seed=123`
(`tests/test_exact_projections.py` lines 31-40):

- `|| A P_0 v || / || A v || < 1e-9` — null-space projector annihilates A
- `|| P_R v + P_0 v - v || / || v || < 1e-12` — partition of unity
- `| P_R v · P_0 v | / (|| P_R v || || P_0 v ||) < 1e-8` — orthogonality
- `|| P_0 (P_0 v) - P_0 v || / || v || < 1e-9` — idempotence of P_0

---

## 4. Data anchor

The minimum-norm measurement-consistent estimate is:

```
x0 = A^dagger y = A^T y   (orthonormal-row case)
```

More precisely, in the presence of regularisation (ridge parameter `lambda_dc`)
the operator uses:

```
x0 = A^T (A A^T + lambda I)^{-1} y
```

This is the `data_solution` / `dc_project` path in `src/measurement.py`
(`null_project` and `dc_project` at lines 601-605) and the
`data_anchor_flat` method in `src/projections.py` (line 147-148):

```python
def data_anchor_flat(self, y):
    return self.AT_forward(self.solve_gram(y))
```

The anchor satisfies `A x0 = y` exactly (up to float64 numerical precision,
tested at `rel < 1e-9` in `tests/test_exact_projections.py` line 45).

For the VQGAN fusion experiments (`vqgan_detail_fusion.py`) the LMMSE anchor
is the empirical version: a ridge-regularized linear map
(`lmmse_lambda = 1e-3`) fit on 20 000 hash-clean training images. It is
measurement-audited by the same projector before use.

---

## 5. Exact measurement audit

Given any candidate reconstruction `x_hat` (e.g. the output of a GAN), the
**exact audit** pulls it back onto the measurement manifold `{v : A v = y}`
by correcting only the row-space component:

```
audit(x_hat, y) = x_hat - A^T (A A^T)^{-1} (A x_hat - y)
                = x_hat - P_R x_hat + A^T (A A^T)^{-1} y
                = P_0 x_hat + x0
```

The second form shows the audit decomposes `x_hat` into its null-space
component (unchanged) plus the unique row-space vector consistent with `y`.
In code (`src/projections.py` lines 150-153):

```python
def audit_flat(self, v, y):
    v_exact = v.to(device=self.device, dtype=self.dtype)
    residual = self.A_forward(v_exact) - y.to(...)
    return v_exact - self.AT_forward(self.solve_gram(residual))
```

The function `exact_audit` (lines 238-250) wraps this with image-shape handling.

**Soft audit.** During training a differentiable variant uses the ridge-regularised
inverse to avoid the exact boundary and keep gradients healthy:

```
soft_audit(x_hat, y; lambda) = x_hat - A^T (A A^T + lambda I)^{-1} (A x_hat - y)
```

This contracts each measured mode by `lambda / (sigma_i^2 + lambda)` where
`sigma_i` are the singular values of `A` (all equal to 1 for orthonormal rows,
giving contraction factor `lambda / (1 + lambda)`). The exact audit is the limit
`lambda -> 0`. Implemented in `src/projections.py` `soft_audit` (lines 253-272)
and `src/measurement.py` `dc_project` (line 604).

---

## 6. Separability: image quality vs bucket accountability

The decomposition `x = P_R x + P_0 x` makes two independent quality axes
explicit:

- **Bucket accountability** — whether `A x_hat = y` (row-space component is
  correct). Controlled entirely by the audit projection. A reconstructor can
  be held accountable to any target `y` by overwriting its row-space component.
- **Perceptual/spectral quality** — determined by the null-space component
  `P_0 x_hat` (`3891` unconstrained dimensions at 5% sampling). LPIPS, RAPSD,
  KID, and high-frequency sharpness live here.

The two axes are orthogonal: improving the null-space content (e.g. by
substituting a better prior) does not change `A x_hat`, and correcting the
row-space component (auditing) does not change the null-space content. This
separability is what makes test-time audit projections possible: they fix
accountability without touching perceptual quality.

Relative measurement error is the practical scalar for bucket accountability:

```python
# src/projections.py lines 275-278
def relative_measurement_error(x, y, operator):
    flat, _ = _as_flat(x, operator)
    pred = operator.A_forward(flat)
    return torch.linalg.norm(pred - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
```

---

## 7. Consistency != correctness: the null-space counterfactual

The key limitation of the measurement framework follows directly from the above.
For any null-space vector `z in N(A)` (i.e. `A z = 0`):

```
A (x_true + z) = A x_true + A z = y + 0 = y
```

The image `x_true + z` and `x_true` are **measurement-identical** — no bucket
experiment can distinguish them — yet they may differ arbitrarily in perceptual
content. Such images are called **feasible wrong images** (or null-space
counterfactuals): they satisfy the measurement constraint exactly but are not
the true scene.

Concretely, in the Rad-5 / Scr-5 gauge experiments (`src/phase77_final_auditable_gan_paper_assembly.py`,
canonical results table), the GAN checkpoint with RelMeasErr controlled to
`~0.005` still cannot certify null-space content as real. The certificate
invariance recheck at
`results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json`
records that the source checkpoint and the mean reconstructor achieve
**identical RelMeasErr** (`source_minus_mean_relmeas = 0.0`), confirming
that measurement fidelity does not distinguish which null-space prior was used.

The practical implication: PSNR and SSIM (which penalise deviation from a
specific `x_true`) and LPIPS / RAPSD (which reward natural-image structure)
can diverge precisely because both priors are measurement-consistent but
fill the null space differently. An audit certificate says `A x_hat = y`; it
does not say `x_hat` is correct.

---

## 8. VQGAN null-space fusion identity

The measurement-conditioned VQGAN detail-fusion (`vqgan_detail_fusion.py`)
adds an adversarial prior's null-space content to a conservative baseline
without ever violating the measurement. The fusion identity is (from the
`fuse` function, lines 290-302):

```
x_hat_B = x0 + P0( d_A + B (d_G - d_A) )
```

where:

- `x0` — the LMMSE anchor with `A x0 = y`
- `d_A = P0(x_A - x0)` — null-space contribution of the VQAE structure branch
- `d_G = P0(x_G - x0)` — null-space contribution of the VQGAN detail branch
- `B` — a global scalar interpolation weight (B=0 recovers x_A; B=1 recovers x_G)

**Proof that `A x_hat_B = y` for all B.**

Apply A and use linearity:

```
A x_hat_B = A x0 + A P0( d_A + B(d_G - d_A) )
           = y   + 0
           = y
```

The first term is `y` by the audited anchor condition. The second term is zero
because `A P0 = 0` (Section 3), regardless of the bracketed vector and
regardless of B.

In code the fusion applies an exact audit after the null-space blend to enforce
this to float64 numerical precision (`vqgan_detail_fusion.py` line 301):

```python
def fuse(spec, x0f, d_A, d_G, y, measurement, projector, masks):
    ...
    d_F = projector.null_project_flat(d_A + dd)
    xhat = projector.audit_flat(x0f + d_F, y)   # enforces A xhat = y exactly
    return measurement.unflatten_img(xhat).to(torch.float32)
```

The locked paper-draft evidence (`outputs/compatibility/measurement_conditioned_vqgan/
detail_fusion_paper/PAPER_DRAFT.md`) reports that at the balanced operating point
`B ~ 0.55`:

- LPIPS improves by `-0.0977` (32.6% relative) over the VQAE branch
- PSNR cost is bounded to `-0.45 dB`
- KID improves from `0.119` to `0.043`
- All 3/3 seeds agree in direction; 8/8 pre-registered gate tests pass

These gains live entirely in the null space; the row-space component and
bucket accountability are unchanged by construction.

---

## 9. Source-file index

| Claim | Source |
|---|---|
| Gram detect + projector solvers | `src/projections.py` `ExactRangeNullProjector` |
| Image-shape API wrappers | `src/projections.py` `exact_row_project`, `exact_null_project`, `exact_data_anchor`, `exact_audit`, `soft_audit` |
| Measurement operator + soft projections | `src/measurement.py` `GhostMeasurementOperator` |
| Projection numerical tests (4 identities) | `tests/test_exact_projections.py` |
| Null-space fusion loop | `vqgan_detail_fusion.py` `fuse`, `prep_residuals` |
| Certificate invariance record | `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json` |
| VQGAN fusion locked results | `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` |

---

*Cross-reference: `HANDOFF/03_CODE_MAP.md` for directory layout; `HANDOFF/05_EXPERIMENTS_AND_EVIDENCE.md`
for which scripts exercise each stage of this theory.*
