# GAN + GI method freeze registry

This registry separates the frozen scientific method from later evidence and
exploratory successors.  Later commits must not silently change the identity of
the Round-31 positive method.

## F1 — frozen positive engineering baseline

**Name:** one-bucket physical adjudication of VQGAN--VQAE disagreement  
**Status:** `FROZEN_POSITIVE_FORMAL_NO_GO_FOR_JOURNAL_NOVELTY`  
**First immutable commit:** `385656e` on
`codex/gan-gi-journal-poc-20260718`  
**Script:** `diagnose_vqgan_vqae_disagreement_bucket.py`  
**Report:** `docs/gan_gi_round31_vqgan_vqae_disagreement_bucket_pilot_20260718.md`

Frozen identity:

1. original 205-row operator SHA-256
   `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`;
2. seed-0 measurement-conditioned VQAE anchor and VQGAN alternative;
3. old-fiber difference projected into the null space;
4. exact half-on/half-off top-half balanced binary compiler;
5. one additional signed bucket, selected without truth access;
6. bucket coordinate on the VQAE--VQGAN segment;
7. projection onto the image box and augmented old/new measurement fiber;
8. 128-image validation prefix and 8192 Dykstra iterations;
9. held-out test unopened.

Frozen result: PSNR `23.430561`, SSIM `0.673880`, LPIPS `0.283278`.
Result SHA-256:
`ec341d10a68c13dffca360e0e12e02465f58fc4451f927380d29b8a0ad3c8c27`.

The method and result remain immutable, but F1 is no longer the journal-method
candidate.  GPT Pro Round 31 returned `FORMAL_NO_GO`: the balanced row is the
two-member query-by-committee/T-optimal/rank-one posterior-variance objective,
and the segment update is a clipped scalar likelihood/least-squares estimate.
The perception--distortion-opposed VQAE/VQGAN pair is empirically useful but
does not change that acquisition mathematics.

**Theory response commit:** `6f9dccd` in `ccyyyYyzz/GAN_FCC` branch
`codex/gan-gi-quality-round31-20260718`  
**Response SHA-256:**
`929ce107552ec6b53632775d3dee3777fa1b4d9973961f2165c4325938324776`

Round 31 proposed a discriminator-gradient tangent as the minimum changed
premise.  That premise is already experimentally closed by Round 25
(`KILL_FIBER_CRITIC_SCORE_REFINEMENT`): every tested null-space critic step
increased the discriminator score while worsening LPIPS.  It must not be
re-run or presented as the successor.

## E1 — causal evidence, not a method change

**Commit:** `99280ec`  
**Script:** `diagnose_vqgan_causal_disagreement_controls.py`  
**Report:** `docs/gan_gi_round32_vqgan_causal_disagreement_controls_20260718.md`

This evidence adds two VQAE-seed and three VQGAN-seed alternatives.  It does not
change F1.  VQAE diversity improves distortion but not LPIPS; every VQGAN seed
improves LPIPS while retaining positive distortion changes.

## E2 — incremental photon-noise evidence, not a method change

**Commit:** `6b49414`  
**Script:** `diagnose_vqgan_disagreement_poisson_bucket.py`  
**Report:** `docs/gan_gi_round33_incremental_poisson_bucket_screen_20260718.md`

This evidence replaces only the added floating-point bucket by exact
complementary Poisson counts with background.  F1's query and reconstruction
mechanism are unchanged.  The original 205-row record remains noiseless in this
screen.

## X1 — two-row distortion/perception combination

**Status:** `EXPLORATORY_NOT_PROMOTED`.

One VQAE-diversity row and one VQGAN-disagreement row were tested as a two-row
upper-level variant.  It is not part of F1, is not frozen, and must not be cited
as the primary method unless it independently passes novelty and equal-budget
gates.

## X2 — self-verifying adversarial residual

**Status:** `THEORY_REVIEW_ONLY_NOT_IMPLEMENTED`.

This is a proposed trainable generalization in which a GAN emits one residual
atom whose amplitude is measured by its own complementary bucket.  It is under
GPT Pro prior-art and theory review.  It has not replaced F1 and has no claimed
experimental result.

## Successor promotion rule

F1 remains the immutable positive benchmark, not the journal claim.  A
successor becomes the journal method only if it simultaneously:

1. beats F1 and the strongest same-photon non-GAN adaptive control on PSNR,
   SSIM, and LPIPS;
2. shows a matched zero-adversarial causal advantage;
3. survives mechanism-level prior-art review;
4. passes full-measurement Poisson/background evaluation across seeds; and
5. is frozen in a separate named commit without rewriting F1 evidence.
