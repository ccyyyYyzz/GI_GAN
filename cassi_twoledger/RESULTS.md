# Two-ledger accountability on real CASSI (snapshot spectral imaging)

A second real-operator port of the range–null accountability framework, chosen because the
CASSI forward operator is the airtight complement to masked-Fourier MRI: since every voxel
lands on exactly one detector pixel, `A Aᵀ = diag(Φ_s)` is **exactly diagonal**, so `A†`,
`P_R`, `P_0` and the **entire singular spectrum** are closed-form and O(N) — no SVD, no
iterative solver, no unmatched-adjoint risk. And unlike MRI (all `σ_i = 1`), the CASSI
singular values are **genuinely non-uniform**, which is what makes the modal-contraction
theorem `λ/(λ+σ_i²)` visible.

Operator: SD-CASSI (MST/TSA-Net convention) — a 28-band cube `[28,256,256]` is spectrally
sheared (step-2 per band), modulated by the **real** TSA-Net coded aperture, and summed onto
a `[256,310]` detector = **23.1× undersampling**. All data is directly downloadable from
GitHub (TSA-Net repo: `mask.mat` + 10 KAIST scenes) — no Drive/Baidu gate.

## What runs

| script | what it does |
|---|---|
| `cassi_operator.py` | shift-mask-sum `A`, `At`, `A†`, `P_R`, `P_0`, audit `Π_λ`, witness; `Φ_s`=diag(AAᵀ) |
| `test_identities.py` | float64 certification of every projector identity on the real mask |
| `spectrum_figure.py` | **the headline**: non-uniform singular spectrum + per-mode contraction |
| `cassi_witness.py` | feasible-but-wrong witness: two scenes, one coded snapshot + box-legal POCS |

## Results

**1. The operator is exact (float64).** `‖A P_0‖=2.8e-13`, `‖P_R²−P_R‖=7.7e-14`,
`‖P_R+P_0−I‖=1.7e-14`, `‖A A†−I‖=2.9e-13` on support, `P_R` self-adjoint to 0. Confirms the
diagonal `A Aᵀ = diag(Φ_s)` structure MST's own GAP/DAUHST data-consistency step relies on.

**2. The non-uniform spectrum (the deliverable MRI cannot show).**
`σ_j = √Φ_s` over the **79 360** measured detector modes spans **[0.009, 4.907]**, median
3.51 — a **528× ratio**. The per-mode audit contraction `λ/(λ+σ_j²)` at λ=1e-3 therefore
ranges over `[4.15e-5, 9.21e-1]`, a **2×10⁴×** spread — where masked-Fourier MRI collapses to
a single value `9.99e-4` for every mode. → `CASSI_SPECTRUM.*`

**3. Feasible-but-wrong witness.** Two different spectral scenes, one identical 23× coded
snapshot: `u = A†y_t + P_0 x_d` matches the target snapshot to a **median 1.7e-16** while
carrying the donor's spectral null content (match 1.7e-16). A **box-legal ([0,1]) POCS
witness** reaches median 1.9e-3 (below any physical noise floor). The witness renders as the
donor's scene yet reproduces the target's record exactly; per-pixel spectra differ visibly
under the identical snapshot. → `CASSI_WITNESS.*`, `witness_certificate.json`

## Why CASSI complements the MRI leg

| | masked-Fourier MRI | CASSI |
|---|---|---|
| operator | `A = M∘F` | shift-mask-sum |
| `A Aᵀ` | identity on samples | diagonal `Φ_s` |
| singular spectrum | flat, `σ=1` | **non-uniform, 528× span** |
| audit contraction | one value | **per-mode, 2×10⁴× span** |
| undersampling | 4–8× | 23× |

The two operators bracket the framework: MRI shows it on the most-sampled clinical regime;
CASSI shows the modal-contraction theorem with a genuinely structured spectrum.

## Scope / honesty

Simulation-level cubes with the real released coded aperture; certification is exact w.r.t.
the declared integer-shift operator (real captures carry PSF/dispersion calibration error,
noted). The structural range–null decomposition is prior art *inside* CASSI (RND-SCI,
arXiv:2305.09746); the contribution here is the GT-free audit, the witness, and (pending) the
third-party attribution ledger over the MST model zoo — cite RND-SCI / Bhadra / DDNM defensively.
