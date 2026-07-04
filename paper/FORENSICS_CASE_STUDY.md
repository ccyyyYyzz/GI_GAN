# Projector Forensics of Published DL-GI Claims — Case Study Log

Pre-registered forensic case study (framing per `GPT_CROSSCHECK_VERDICT.md`: 2–3 clean targets, papers'
OWN released operators, exact MSE orthogonal attribution only; "case study", not "literature audit").

## Target 1 — Physics-enhanced deep learning SPI (Wang et al., Photonics Research 10, 104 (2022))
Repo: `FeiWang0824/physics-driven-fine-tuning` (cloned to `external_audit/`, gitignored; weights from the
README's live Google-Drive link, 423 MB). Script: `forensics_pedl_stl10.py` → `forensics_pedl_stl10.json`.

### Stage 1 — released-data-only audit (2026-07-04, DONE)
**Own-operator condition: satisfied exactly.** The learned sampling patterns ship as plain data
(`trained_stl10_patterns_1024_Unet_wDGI_64.mat`, 64×64×1024) — the exact sensing matrix A, no TF needed.
Convention verified: `standardize(A · GT_shipped)` reproduces their shipped `y` to **1.18e-7** (float32
precision). Their sim is noiseless; the whole-vector standardization is affine and leaves row(A), hence
P_R/P_0, untouched.

**Their operator, characterized (nominal m=1024 = 25% of n=4096):**
| property | value |
|---|---|
| machine rank / rank@1e-3 | 1024 / 1024 (full — learned patterns non-degenerate) |
| condition number | 64.2 (benign) |
| certificate profile @σ=0.01 | **1024/1024 modes gain ≥ 0.9** |
| null dimension | 3072 |
| true-scene energy in THEIR null space | **10.1%** |

The last row is the learned-acquisition payoff, precisely metered for the first time on their released
operator: training the patterns concentrates ~90% of scene energy into the measured subspace at 25%
sampling. (Co-designed operator+prior — reported as its own category per the pre-registration.)

**Decompositions on their shipped single-scene record (`stl10_sim.mat`):**
| reconstruction | PSNR | row-MSE | null-MSE | null share of error | align w/ true null | range consistency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| min-norm `P_R·GT` (ceiling) | **16.99 dB** | 1.9e-26 | 2.0e-2 | 100% | — | 3.6e-13 |
| their shipped DGI (learned patterns) | 12.92 dB | **3.1e-2** | 2.0e-2 | **39.3%** | +0.002 | **2.9e-1** |

Two findings already:
1. **The 16.99 dB range ceiling.** On this scene, any range-only (measurement-faithful, null-empty)
   reconstruction tops out at 16.99 dB. Every dB their pipeline reports above ~17 dB is necessarily
   **prior-correct null injection** — content the measurement cannot certify. This single number converts
   their reported PSNRs into a two-ledger statement with zero reimplementation.
2. **Their physics baseline is majority row-space error.** The shipped DGI violates measurement consistency
   at 2.9e-1 relative and 61% of its error is in the MEASURED subspace — the correlation estimator does not
   even reproduce the certified component (its null content is uncorrelated junk: alignment +0.002,
   hallucinated-null norm 0.68). So the first job of their DNN is legitimate row-space repair (certifiable,
   up to the 16.99 ceiling); everything beyond is prior.

### Stage 2 — their pipeline reproduced UNTOUCHED and decomposed (2026-07-04, DONE)
Built a minimal TF1.15/py37 env (`D:/tf1_audit_env`); ran their `finetune.py` **without any code
modification** (`MPLBACKEND=Agg` only) at their operating point: loads their pretrained ckpt, 300
fine-tuning steps minimizing only the measurement loss (0.041 → 6.2e-5). Every step's reconstruction
decomposed through the exact projectors on their operator:

| step | PSNR | row-MSE | null-MSE | null share of error | align | consistency |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 (informed DNN) | 20.76 | 3.4e-3 | 5.0e-3 | 60.0% | +0.834 | 5.8e-2 |
| 10 | 22.37 | 1.7e-3 | 4.1e-3 | 70.1% | +0.855 | 8.6e-3 |
| 50 | 24.50 | 5.2e-4 | 3.0e-3 | 85.3% | +0.834 | 1.0e-2 |
| 200 | 25.27 | 9.1e-5 | 2.9e-3 | **96.9%** | +0.847 | 1.6e-2 |
| 299 (final) | 25.12 | 9.9e-5 | 3.0e-3 | 96.8% | +0.845 | 2.3e-2 |

**Findings (the case-study headline set):**
1. **Two-ledger read of their headline number.** Final 25.12 dB vs the 16.99 dB range ceiling: **≥ 8.1 dB
   of the reported quality rests on prior-supplied null content** — truth-aligned on this scene
   (align ≈ 0.85) but unverifiable from the record by the converse. The pretrained DNN *starts* 3.8 dB
   above the ceiling: the prior arrives before any fine-tuning.
2. **The fine-tuning gain decomposes 61.4% / 38.6%.** Of the +4.36 dB bought by 300 steps of
   measurement-loss-only fine-tuning, 61.4% of the MSE improvement is row-space repair (the certifiable
   ledger — exactly what a measurement loss is entitled to fix) and 38.6% is **null-space spillover**:
   the U-Net's coupling moved null content toward the truth as a side effect (null-MSE 5.0e-3 → 3.0e-3,
   hallucinated-null norm 4.29 → 3.20). The spillover helped here — but nothing in the measurement
   certifies it, and that is precisely the paper's point stated on *their* pipeline.
3. **The saturation shape, observed.** By step ≈ 200 the row ledger is exhausted (96.9% of the remaining
   error is null) and PSNR plateaus — fine-tuning ends where the only error left is the kind the
   measurement cannot see. This is the range-contribution saturation curve, per-step, on a published
   pipeline's own operator.
4. **Exact consistency is left on the table.** Their fine-tuned output still violates the record at
   2.3e-2 relative (they minimize the measurement loss but never project). The exact audit step
   `x̂ ← x̂ − A†(Ax̂ − y)` would pin it to ~1e-13 and remove the residual row-MSE for free.

Artifacts: `forensics_pedl_stl10.json` (stages 1+2), `forensics_pedl_trajectory.npy` (full 300-step
trajectory for the figure). Reproduction: their repo untouched at `external_audit/physics-driven-fine-tuning/`
(TF1 env `D:/tf1_audit_env`), audit script `forensics_pedl_stl10.py` (stage 1) / `stage2`.

## Target 2 — GIDC: far-field super-resolution GI (Wang et al., Light Sci Appl 11, 1 (2022))
Repo: `FeiWang0824/GIDC`. **Real experimental data**: 1200 physically measured speckle patterns + integer
photon-count buckets (`data.mat`). Their demo (SR=0.1, m=410, 201 steps) run **untouched** in the TF1 env;
outputs are their released 8-bit BMPs (quantization ~0.002, disclosed). No GT for the scene → reported
gains NOT MSE-decomposable (pre-registered exclusion); everything below is **GT-free**.
Script: `forensics_gidc.py` → `forensics_gidc.json`.

**The real operator, characterized (first time on this released data):**
| property | value |
|---|---|
| m, sampling | 410 physical patterns, 10.0% |
| rank / condition | 410 (full) / 52 |
| row sums (nonnegative speckle) | 82929 ± 8.7% — **DGI's proper regime** (closes our C6 signed-operator caveat with real data) |
| photon counts → shot noise | ~1.8e7 → rel σ = 2.4e-4 |
| certificate | **410/410 modes gain ≥ 0.9**; null dim 3686 |

**GT-free ledgers (their record, their convention — affine-fit consistency, orientation-resolved):**
| reconstruction | consistency rel-err | null fraction (centered) |
|---|:---:|:---:|
| DGI (recomputed, float) | 0.948 | 0.525 |
| GIDC step 0 | 0.715 | 0.743 |
| GIDC step 200 (their result) | **0.134** | **0.438** |

**Findings:**
1. **The super-resolution claim, metered.** "Resolution beyond the diffraction limit" content lives, by
   definition, in null(A) of the patterns actually measured. **43.8% of the published reconstruction's
   (centered) structure is in that null space** — unverifiable from their own record. Not an accusation of
   error: a precise statement of *which ledger* the headline claim is paid from.
2. GIDC improves record-faithfulness 7× over DGI (0.948 → 0.134) but remains far from exact — the
   TV+DIP-style optimization stops at loss ≈ measurement noise, and no exact projection is applied.
3. The audit closes our own C6 caveat with hardware data: real speckle row-sums are large and uniform
   (+8.7% spread), exactly the regime where DGI's reference subtraction is meaningful.

## Target 3 (queued)
- `Noise2Ghost` (arXiv 2504.10288) — pip-installable PyTorch; its natively noisy regime is where the
  range-saturation claim is non-tautological. Also queued: GIDC Part B (their code, our known-GT scene on
  their real patterns + Poisson — labeled extension).
