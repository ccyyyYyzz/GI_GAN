# Round 29: VQGAN-secant balanced binary code geometry screen

Date: 2026-07-18

## Question

Does a fixed equal-flux binary DMD code learned to preserve VQGAN reconstruction
secants have better held-out measurement geometry than random or Hadamard codes,
and is any gain specifically caused by adversarial VQGAN pretraining rather than
the matched VQAE representation?

## Protocol

- Frozen seed-0 VQGAN and matched VQAE priors, identical architecture and data.
- Decode the first 1024 cached STL-10 training images through each prior.
- Learn 205 balanced binary rows for 200 updates using a soft minimum of
  normalized secant measurement energy plus row-coherence control.
- Evaluate 4096 unseen pairs from 128 validation images in three domains: real
  images, VQGAN reconstructions, and VQAE reconstructions.
- Test split unopened; no reconstruction model trained.

An initial weak-coherence run produced highly redundant rows and was rejected as
an invalid conditioning surrogate.  The reported screen uses coherence weight
50 and zero mean-energy reward, so the objective emphasizes lower-tail secant
energy while penalizing row repetition.

## Result

Held-out 5th-percentile normalized secant energy:

| Code | Real images | VQGAN reconstructions | VQAE reconstructions | mean absolute row coherence |
|---|---:|---:|---:|---:|
| random balanced binary | 0.021125 | 0.019616 | 0.017745 | 0.01251 |
| low-sequency Hadamard | 0.216451 | 0.221231 | 0.223338 | 0.00000 |
| learned on VQGAN secants | 0.390784 | 0.402099 | 0.417090 | 0.02878 |
| learned on VQAE secants | **0.413891** | **0.416787** | **0.450606** | 0.02873 |

The learned codes generalize well beyond their own decoded training images and
show a large data-adaptive geometry signal relative to random and Hadamard rows.
However, the matched non-adversarial VQAE code is better in every target domain,
including the VQGAN domain.  The screen therefore provides no GAN-specific
causal support.

Minimum energies are noisier but tell the same practical story: on real-image
pairs, VQGAN-code minimum is 0.08266, VQAE-code minimum is 0.12134, and Hadamard
minimum is 0.05933.

## Decision

`DATA_ADAPTIVE_CODE_GEOMETRY_POSITIVE_GAN_CAUSAL_NO_GO`

Do not train an end-to-end reconstructor for the VQGAN-secant code unless the
independent Round-29 theory adjudication identifies a new GAN-specific quantity
that the matched VQAE cannot supply.  The present result may justify a generic
learned-GI-code project, but it does not satisfy the requested GAN+GI innovation.

## Artifacts

- Script: `diagnose_vqgan_secant_binary_codes.py`
- Primary summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round29/vqgan_secant_binary_geometry_coh50/summary.json`
- Rejected weak-coherence diagnostic:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round29/vqgan_secant_binary_geometry/summary.json`

