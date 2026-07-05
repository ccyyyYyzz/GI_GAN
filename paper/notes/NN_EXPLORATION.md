# NN Architecture Exploration for Ghost-Imaging Reconstruction (Beyond GAN)

Research-director synthesis of a six-architecture scouting round. Goal: pick what to
prototype *now* on our exact-null-space GI testbed (64x64 STL10 grayscale, DCT+Hadamard+random
operators at 2/5/10%, one RTX 4060 / 12 GB), and be honest about what is trendy vs. useful.

Repo assets every plan below plugs into (verified present):
- `src/measurement.py` -> `GhostMeasurementOperator` (`A_forward`, `flatten_img`, `null_project`)
- `src/projections.py` -> `get_exact_projector`, `exact_audit`, `row_project`, `null_project`
- `src/operator_conditioned_nullspace.py` -> `MatrixFreeNullProjector` (`data_anchor`, `null_project`, `relmeaserr`)
- `gan_high_quality_gi.py` -> `EmpiricalLMMSE` (audited LMMSE anchor `x0`)
- `vqgan_detail_fusion.py` -> one-step fusion `x_hat = x0 + P0 d_F` (the fixed-point seed)
- `src/datasets.py`, `src/metrics.py`, `tests/test_operator_conditioned_nullspace.py` (exact-consistency assertions)

The common wrapper for **all** candidates (this is the thesis, not the architecture):
> Any network produces a raw estimate; we wrap it so `x_hat = x0 + P0 * net_output`, where
> `x0` is the audited LMMSE anchor and `P0` is the exact null-space projector (`A P0 = 0`).
> Then `A x_hat = y` holds **by construction** and the network only ever sculpts the
> (n-m)-dim null space. Measurement consistency is free and exact for every candidate.

---

## (1) Ranked table by promise

| Rank | Architecture | Score | One-line why |
|------|--------------|:---:|--------------|
| 1 | **Deep Equilibrium (DEQ) null-space refiner** | 7 | Best *structural* fit: its core object (a fixed point of an operator-aware map) is exactly what our projector already manufactures; measurement-exact + constant-memory + a clean convergence story; **no prior DEQ-for-GI exists**. |
| 2 | **Implicit Neural Representation (SIREN / Fourier-features)** | 7 | Cheapest, training-free, resolution-free; coordinate-INR + exact null-space fitting for structured-operator GI is under-occupied; ideal fast probe that *stress-tests the projector-wrapping idea* before heavier bets. |
| 3 | **Structured State-Space Models (Mamba / S5)** | 6 | Only candidate that uses the *sequence* facet the physics literally emits (bucket stream); but CMSPI (Opt. Express 2024) already claims "Mamba for SPI", so novelty rides entirely on sequence-native + exact-projector framing; Windows/CUDA tooling friction. |
| 4 | **Liquid Neural Nets (LTC / CfC)** | 4 | Genuine but thin: "swap LSTM cell for a liquid cell" in known GI-RNN pipelines; only defensible via irregular/adaptive timing + our projector wrap. |
| 4 | **Kolmogorov-Arnold Networks (KAN)** | 4 | A possibly-better MLP *head*, not a physics-matched architecture; U-KAN-SPI (Opt. Lett. 2026) already occupies low-rate SPI; needs a real win over a matched-param MLP head. |
| 6 | **Neural Operators (FNO / DeepONet)** | 3 | Marquee properties (resolution independence, smooth PDE kernels) do **not** transfer to GI's global, non-convolutional inverse map; novelty gap is real precisely because the fit is poor. |

---

## (2) TOP 2 to prototype now

### PROTOTYPE A -- Exact-null-space DEQ refiner  (`deq_nullspace_refiner.py`, ~200 lines)

**Why first:** it is the sharpest structural match to assets we already own, and it comes with
a free rigor result even if it "fails" (see Milestone-0).

- **Network.** A small spectral-norm-constrained conv denoiser `g_theta` (3-5 conv layers,
  ~50k params, bounded output). Fixed-point map on the null-space residual `d`:
  `T(d) = P0 * ( g_theta(x0 + P0 d) - x0 )`, solved with **TorchDEQ** Anderson acceleration
  (max ~20 iters, tol 1e-3), backprop via implicit/phantom gradient (constant memory -- this
  is *designed* for a 12 GB card where a 20-stage unrolled net would not fit).
- **Data.** Existing 64x64 STL10 grayscale; synthesize `(A, y, x)` on the fly at 2/5/10% via
  `GhostMeasurementOperator`. Reuse the frozen seed-0 5% task + audited `x0` from
  `gan_high_quality_gi.py::EmpiricalLMMSE`. No new data pipeline.
- **Plug into projector.** `P0 = get_exact_projector(A)`; output `x_hat = x0 + P0 d*`.
  Because `A P0 = 0`, `exact_audit`/`relmeaserr(x_hat, y) < 1e-5` holds at the fixed point --
  assert it as the headline guarantee (reuse `tests/test_operator_conditioned_nullspace.py`).
- **Measure.** RMSE / PSNR / SSIM / LPIPS + `relmeaserr` (the existing `METRIC_COLS`), vs the
  LMMSE anchor and the current one-step VQGAN fusion, same seeds/operators. Monitor solver
  residual to *verify* (not assert) convergence.
- **Milestone-0 (no training, ~1 hr):** set `g_theta` = identity-plus-frozen-VQGAN-denoiser
  and just run the solver. This directly tests "does more-than-one null-space iteration beat
  our one-step `x_hat = x0 + P0 d_F`?" If not, that is itself a *publishable rigor result*
  (our one-step fusion is already near the fixed point).
- **Wall-clock (4060).** Milestone-0: ~1 hr. Full supervised training at one rate/seed:
  a few hours (tiny `g_theta`, short Anderson solves on a 4096-dim null-space vector).
- **Risk to watch.** Early fixed-point instability -> use phantom/Jacobian-free gradients and
  keep `g_theta` nonexpansive (`P0` already helps). Marginal-gain risk at 2-5% -> Milestone-0
  de-risks it before any training spend.

### PROTOTYPE B -- Coordinate-INR / SIREN with exact null-space fitting  (`inr_nullspace_gi.py`, ~120 lines)

**Why second (run in parallel):** lightest possible experiment, no dataset, no exotic libs;
it is the cheapest way to validate the projector-wrapping mechanic that *both* top picks share.

- **Network.** SIREN (5x256, sine, omega_0~30) OR fixed random Fourier-feature map ->
  small ReLU MLP; input = normalized 64x64 `(u,v)` grid `[4096,2]`, output = grayscale intensity.
  ~0.3-0.5 M params. Per-scene test-time optimization (DIP-style), no training set.
- **Data.** None. One scene: `op = GhostMeasurementOperator(img_size=64, sampling_ratio=0.05)`,
  `y = op.A_forward(op.flatten_img(x_gt))`. Repeat over a handful of STL10 test images for a curve.
- **Plug into projector.** Two variants: (a) measurement loss only
  `||op.A_forward(flatten(f_theta)) - y||^2 + lambda_tv*TV`; (b) **projector-wrapped**:
  each step `x_final = exact_audit(f_theta, y, op)` (or `x0 + P0 f_theta`) so `A x_final = y`
  exactly and Adam works *only in the null space*. Variant (b) is the novel coupling.
- **Measure.** PSNR/SSIM vs LMMSE anchor and VQGAN recon at 2/5/10%; render at 128x128 to show
  resolution-free super-res (state honestly: prior-driven interpolation, no sub-sampling-limit info).
  Key ablation: SIREN omega_0 / Fourier bandwidth sweep (spectral bias controls null-space overfit).
- **Wall-clock (4060).** ~seconds to ~2 min *per scene* (1-3k Adam steps). A full 2/5/10% sweep
  over a dozen images is well under an hour.
- **Risk to watch.** Spectral bias caps very-low-rate (2%) high-freq recovery; too-high bandwidth
  hallucinates texture that still satisfies `A x = y` -> needs TV / early-stop. Ignores the bucket
  *sequence* entirely (that is Prototype C territory) -- do not claim otherwise.

**Sequencing:** run B's variant-(b) sanity first (hours) to confirm null-space-only optimization
behaves; then run A's Milestone-0; then commit training to whichever shows a null-space-quality edge.

---

## (3) The single most NOVEL angle

**Treat GI as a sequence-to-image problem: read the ordered bucket time-series with an SSM/liquid
net, and wrap the output in the exact null-space projector.**

Assessment of "truly under-explored": **partially, and the open sliver is specific.**
- The physical datum genuinely *is* a sequence -- the bucket detector emits one scalar `y_t` per
  projected pattern `P_t`; almost all learned GI flattens `y` and destroys this order. That framing
  is correct and under-used.
- BUT the bare idea is **taken**: GI-RNN (arXiv:2112.00736), GI-BRNN (ICCMLDS 2024), semantic-GI-RNN
  (Opt. Lett. 2022), LSTM-SPI (Electronics 2023), and for Mamba specifically **CMSPI**
  (Opt. Express 32(20), 2024) already do "sequence/state-space network for SPI". A naive
  "we used Mamba/liquid for SPI" gets desk-rejected.
- **The genuinely unclaimed intersection** (three conditions, all required): (i) feed the *raw 1-D
  causal bucket stream* to the SSM as a true temporal sequence with each `y_t` tagged by an embedding
  of its operator row `a_t` (CMSPI and all VSS-SPI work scan 2-D image *features*, never the raw
  stream); (ii) support *irregular / adaptive / event-triggered pattern timing* (continuous-time CfC
  or timestamp-channel SSM -- the one thing an LSTM cannot do); (iii) couple it to the **exact
  range/null projector + LMMSE anchor** so `A x_hat = y` holds at every step (all prior sequence-GI
  is a soft, unconstrained regressor).

Verdict: the *sequence facet* is a real, defensible wedge **only** when carried by (i)+(iii) and
ideally (ii). Absent those, it collapses to a cell-swap. It is the highest-*novelty* angle but a
*moderate*-payoff one on raw PSNR -- so it is the paper's differentiator, not necessarily its
quality champion. Pursue it as **Prototype C** (Mamba/SSM sequence-encoder, pure-PyTorch scan
fallback to dodge Windows CUDA-kernel friction) once A and B validate the projector wrap.

---

## (4) Honest hype-flags

- **Neural Operators (FNO/DeepONet) -- trendy, poor fit.** FNO's inductive bias is spectral-local,
  smooth, translation-equivariant *convolution* kernels; GI's inverse is a global, non-convolutional
  linear mixing map with no PDE and no smooth kernel. "Resolution independence" is meaningless on the
  measurement side (a length-m vector on no grid). The novelty gap exists *because* the fit is poor.
  Run at most a 1-day canary with a hard kill-gate vs. the LMMSE anchor; expect a null.
- **KAN -- interpretability oversold at width.** Advantages are on low-D smooth/symbolic functions;
  image detail is high-D and non-smooth, and studies find KAN merely competitive-when-hybridized.
  U-KAN-SPI (Opt. Lett. 2026) already owns low-rate SPI. Allowed only as a strictly matched-param
  A/B head swap with a hard kill criterion; do **not** build a full-KAN reconstructor.
- **Liquid nets -- "brain-inspired / robust" marketing.** On regularly-sampled fixed-length GI
  sequences the continuous-time advantage largely evaporates and CfC often ties a tuned GRU. Only
  worth it if the story is genuinely irregular/adaptive acquisition timing.
- **General hype check for the sequence bets.** At 64x64, m <= 410: "linear attention / long-context"
  selling points are irrelevant. The bet is *order + selectivity + operator-conditioning at 2-5%*,
  which is unproven and may come out flat. Mandatory ablation: liquid/SSM cell vs. a plain GRU/LSTM
  under the *same* projector wrapper, or the projector is silently doing all the work.

---

## (5) Why this is a real research contribution, not just swapping architectures

The contribution is **not** the architecture; it is a physics-coupling *framework* that any modern
reconstructor can be dropped into. Every candidate here is wrapped so that `x_hat = x0 + P0 * net`,
which makes measurement consistency (`A x_hat = y`) exact *by construction* and confines the network
to the unobserved `(n-m)`-dim null space -- turning "reconstruct the image" into the sharper, better-posed
"learn a prior over exactly the degrees of freedom the operator cannot see." That reframing yields
things no prior GI network delivers: a *provable, checkable* consistency certificate (`relmeaserr < 1e-5`)
instead of soft data-fit; an audited LMMSE anchor that separates the guaranteed part of the answer from
the learned part; and an apples-to-apples testbed where a DEQ fixed point, a coordinate-INR, and an SSM
sequence-encoder can be compared *on null-space prior quality alone*, all satisfying the same physics.
The DEQ instantiation additionally recasts our one-step fusion as a genuine operator-fixed-point with a
convergence story, and the sequence instantiation is the first to feed the *raw bucket time-series* to a
causal state-space model under exact consistency. The publishable claim is the coupling and what it
guarantees/reveals -- the architecture is just the interchangeable prior inside it.
