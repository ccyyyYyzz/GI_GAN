# Round 34: novelty collision map after the frozen one-bucket positive result

Date: 2026-07-18

## Decision

The frozen one-bucket VQGAN--VQAE method is retained as a positive engineering
baseline, but it is not the journal mechanism.  GPT Pro Round 31 returned
`FORMAL_NO_GO`, and an independent primary-literature audit found a second
strong 2026 collision in generative-prior measurement design.  The held-out
test split remains closed.

## Exact claim subtraction

| Candidate claim | Closest established mechanism | Consequence |
|---|---|---|
| Choose the feasible row that maximally separates VQAE and VQGAN | Two-member query-by-committee and one-point T-optimal model discrimination | Exact mathematical collapse; not novel |
| Interpret the two endpoints as an equal two-atom posterior | Rank-one posterior-variance acquisition | Exact same objective under common variance |
| Adapt a GI pattern from the current posterior | Sun et al., adaptive information-maximization encoding (AIME), *Photonics Research* (2026), doi:10.1364/PRJ.603132 | Broad adaptive-GI claim is occupied |
| Use a generative posterior cloud to acquire measurements that remove hallucination | Jin, Li, and Li, *Measurement Geometry and Design for Trustworthy Generative Inverse Problems*, arXiv:2606.02309 (2026) | Broad generative-hallucination measurement-design claim is occupied |
| Use posterior samples to design the next compressed measurement | Elata, Michaeli, and Elad, *Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling*, arXiv:2407.08256 | Broad generative active-sensing claim is occupied |
| Estimate SPI reconstruction uncertainty and react to it | Shang et al., *Communications Engineering* 2 (2023), doi:10.1038/s44172-023-00103-1 | Uncertainty-triggered acquisition is not enough |
| Put measurements into a GAN discriminator for SPI | Dai et al., *Optics Communications* 560 (2024), doi:10.1016/j.optcom.2024.130485 | Measurement-conditioned adversarial reconstruction is not enough |
| Exploit the GAN perception--distortion trade-off | Blau and Michaeli, CVPR 2018; Esser et al., CVPR 2021 | The endpoint bias is established, although its physical use can still be causal evidence |

The 2026 Jin--Li--Li paper is especially close.  It uses a current generative
posterior cloud, estimates unresolved local directions, and adds structured
measurements without test truth.  It does not use a GAN--VQAE pair, a single
balanced complementary-DMD row, or a scalar amplitude gate.  Those differences
are implementation-level unless they create a different identifiable object or
guarantee.

## Facts that remain scientifically useful

1. The VQGAN--VQAE pair is not generic diversity empirically.  Across seeds,
   VQAE alternatives improve distortion but not LPIPS; VQGAN alternatives
   improve LPIPS while retaining small positive distortion changes after the
   bucket update.
2. The frozen one-bucket update improves PSNR, SSIM, and LPIPS on the 128-image
   validation prefix and beats fixed one-row controls, with paired intervals
   excluding zero.
3. Exact complementary Poisson simulation of the added bucket remains positive
   at `1.00e4` and `1.00e5` expected signal photons per pair.  This is only an
   incremental-noise screen; the old 205-row record is still noiseless.
4. These facts justify the adversarial prior as a useful *proposal source*.
   They do not establish a new acquisition principle.

## Closed successor from Round 31

Round 31 proposed using the old-fiber input gradient of the trained critic as
the candidate direction.  That successor was already tested in Round 25.  The
critic score increased at every tested step, but LPIPS worsened monotonically;
no step improved all three metrics.  It is therefore marked
`KILL_FIBER_CRITIC_SCORE_REFINEMENT` and must not be repeated.

## Remaining live premise

The only live premise in this family is stronger than choosing a row from two
pretrained endpoints: constrain the GAN to emit a single residual atom, allow
the optical system to measure its only coefficient, and prevent any other GAN
degree of freedom from entering the reconstruction.  This self-verifying
adversarial residual is under Round-32 adjudication.  It survives only if it is
not merely a learned one-step sensing policy and if the measured scalar supports
a non-vacuous bound on admitted unsupported detail.

## Provenance

- Frozen method commit: `385656e`
- Causal-control commit: `99280ec`
- Incremental-Poisson commit: `6b49414`
- Round-31 theory response commit: `6f9dccd`
- Round-31 response SHA-256:
  `929ce107552ec6b53632775d3dee3777fa1b4d9973961f2165c4325938324776`
- Test split opened: `false`

