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

### Stage 2 — reproduce their DNN / fine-tuned numbers (TODO)
Their pretrained U-Net + fine-tuning requires their TF1/py36 env (`environment.yml` ships). Plan: create the
conda env, run `finetune.py` untouched at their operating point (stl10_sim, 1024 patterns, 300 steps),
decompose `DLDC_r[:, :, step]` through the same exact projectors: row-repair vs prior-correct null vs
hallucinated null, as a function of fine-tuning step. Expected shape: rapid row-space repair to ≈ the
ceiling, then null-ledger gains only. The per-step trajectory decomposition (physics fine-tuning as
row-repair → null-injection phases) would be the headline figure of the case study.

### Targets 2–3 (queued)
- `FeiWang0824/GIDC` — real measured patterns+buckets in `data.mat` (exact own operator); only sim numbers
  GT-auditable; experimental claims listed as not-auditable.
- `Noise2Ghost` (arXiv 2504.10288) — pip-installable PyTorch; its natively noisy regime is where the
  range-saturation claim is non-tautological.
