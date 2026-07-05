# Two-ledger accountability on real fastMRI single-coil knee

A zero-training port of the range–null accountability framework (from the ghost-imaging
paper) to accelerated MRI, on **real measured k-space** from the NYU fastMRI single-coil
knee validation set. The measurement operator is the exact masked-Fourier map
`A = M∘F` (F = centered orthonormal 2D DFT, M = Cartesian undersampling mask), so the
row/null projectors, the feasible-but-wrong witness, the GT-free audit, and the governed
dial all transfer with closed-form exactness.

## What runs

| script | what it does |
|---|---|
| `mf_operator.py` | masked-Fourier `A`, `A†`, `P_R`, `P_0`, audit `Π_λ`, witness |
| `test_identities.py` | float64 certification of every projector identity (synthetic) |
| `witness_demo.py` | feasible-but-wrong witness on real knee k-space + box-legal POCS variant |
| `run_forensics.py` | range-ceiling forensics + GT-free audit over 19 volumes at 4×/8× |
| `decision_demo.py` | certified-vs-null decomposition of fastMRI+ pathology boxes |
| `data.py`, `reconstructors.py` | val loader; zero-filled / wavelet-CS / official U-Net |

Data + weights live outside the repo at `E:/GAN_FCC_WORK/data_warehouse/fastmri_knee_sc/`
(single-coil val + test, official `knee_sc_leaderboard_state_dict.pt`, fastMRI+ `knee.csv`).

## Results

**1. The operator is exact.** Every identity holds in float64 — on synthetic data
`‖A P_0‖, ‖P_R²−P_R‖, ‖P_R+P_0−I‖, ‖A A†−I‖` all ≈1e-14, and on **real** fastMRI k-space
to ≈1e-17. The ESC ground truth equals `|crop(ifft2c(kspace))|` to corr 1.00000.

**2. Feasible-but-wrong witness (7.8× random).** Two different knees, one identical
undersampled record: `u = A†y_t + P_0 x_d` matches the target k-space to a **median
2.9e-16** while carrying the donor's null content (match 1.3e-15) and sitting 20.3 dB (PSNR)
from the target. A **box-legal POCS witness** matches to **2.5e-16** — the impossibility is
not a complex-vector-space artifact. → `WITNESS_MRI_8x.*`, `witness_certificate_8x.json`

**3. Forensics: the headline gain is null-supplied.** Zero-filled reconstruction equals
`P_R x_gt` exactly, so its magnitude PSNR is the *range ceiling*. Decomposing the official
single-coil U-Net's gain over that ceiling:

(accel is the nominal fastMRI R used to generate the random mask; effective sampling is
higher because of the ACS block — R=4→3.7×, R=8→6.9× effective. Numbers match `forensics_Rx.json`.)

| accel (nominal) | range ceiling | U-Net (raw) | gain | **null share** | U-Net record drift |
|---|---|---|---|---|---|
| 4× (eff 3.7×) | 26.70 dB | 29.68 dB | +2.98 dB | **91 %** | 5.9 % |
| 8× (eff 6.9×) | 24.32 dB | 25.42 dB | +1.09 dB | **92 %** | 3.3 % |

91–92 % of the "SOTA" PSNR gain is prior-invented null content the k-space never certified
— extending the paper's 39–95 % ghost-imaging finding to real clinical MRI. The raw U-Net
is also **3–6 % inconsistent with its own k-space record**; a **governed data-consistent
variant** (`A†y + P_0 x_UNet`) keeps nearly all the perceptual gain (29.42 vs 29.68 dB) while
being **exactly** record-consistent (drift 3.5e-16). → `FORENSICS_MRI_4x.*`, `_8x.*`

**4. GT-free audit contracts every mode exactly.** Applied to the (data-inconsistent) U-Net
output, `Π_λ` contracts the record drift by exactly `λ/(λ+1)` — median 5.86e-2 → 5.85e-5 at
λ=1e-3, per-mode deviation ≤1e-15 — using only `(A, y, λ)`, never ground truth or the network.

**5. Certified-decision on fastMRI+ pathology (429 real boxes).** The ACS lines certify the
**coarse** lesion (low-frequency energy: median null share only 13.5 % at 8×), but the
**fine detail** a radiologist reads — margins, internal texture — is **88.4 % null-supplied
at 8× (93.8 % at 16×)**. A governed null-space edit dials a real "Soft Tissue Lesion" toward
a donor's null content, changing the box fine-detail contrast by −43 % **while `A x = y`
holds to 3.6e-16**. The measurement does not certify the finding's detailed appearance.
→ `DECISION_MRI.*`, `decision_demo.json`

## Honesty / scope

- Single-coil (emulated) keeps `A = M∘F` exact — multi-coil would require estimated ESPIRiT
  maps and break the closed-form certification; deliberately avoided.
- Certification is stated in the complex/linear domain; PSNR/detectors are on magnitude (a
  nonlinear map), noted throughout.
- The **structural** ideas (range–null decomposition, null-space-only injection) are prior
  art in MRI (Bhadra TMI 2021 — GT-based; AAAI 2024; Schwab/Haltmeier null-space nets). The
  contribution here is the **GT-free** record-consistency audit, the feasible-but-wrong
  witness, and the quantitative **third-party attribution ledger** — cite those defensively.
