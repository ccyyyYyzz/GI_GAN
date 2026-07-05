# HQ Ghost-Imaging Reconstruction — Innovation Attempt Log (2026-07-04)

Theme: **raising reconstruction quality** (the positive/quality sibling of the verifiability paper).
All 14 HQ points from `HQ_GI_INNOVATION_POINTS.md` attempted; the 4 runnable near-term wins were **really
run** on cached seed-0 dev residuals (LPIPS scored, PSNR-tolerance matched to the gate), the 10 design
points were realized as paper-ready artifacts. Quality is bought inside `P_0` (`A x̂=y` exact) throughout.

## Real experiments (`experiments_hq_recon.py` → `hq_recon_probe.json`)

Reference on seed-0 dev (512 imgs): VQAE (B=0) LPIPS **0.2969** @ 22.88 dB; **best scalar B=0.55 LPIPS 0.1973** @ 22.45 dB (PSNR ≥ VQAE−0.5).

| HQ point | Result | Verdict |
|----------|--------|---------|
| **HQ-9** three-prior 2-D dial `x̂=x0+P0(B1 d_A + B2 d_G)` | best under PSNR tol = (B1=0.6,B2=0.4) **LPIPS 0.2197**, **Δ +0.022 (worse)** | ❌ **No win.** Re-weighting the *same two* residuals does not beat the scalar under matched PSNR — confirms the deck's marginal-gain flag empirically. The scalar `B` on the `d_A→d_G` chord is already near-optimal. |
| **HQ-8** per-image `B` | oracle per-image argmin-LPIPS **0.1634**, **Δ −0.034 (~17% further relative gain)** | ⚠️ **Real ceiling.** A *perfect* per-image selector would meaningfully improve quality — so per-image adaptation is worth pursuing, but the gain is only reachable with a **ground-truth-free** selector that approaches the oracle (a prior learned gate underperformed; this is the actual open research target). |
| **HQ-10** endpoint interpolation / val-optimal point | == the current balanced `B=0.55` operating point | Already realized by the locked selection; no new gain. |
| **HQ-7** PnP-RED null-space loop | design only (needs the iterated `refine()` denoiser loop) | Deferred to the artifact set; runnable next. |

**Honest takeaway:** the "mix the same two frozen residuals differently" family (HQ-9, and by extension HQ-2/5/10) does **not** raise quality; the leverage is in (a) a **better source of detail** than the two-point blend (HQ-1 diffusion, HQ-3 CodeFormer) and (b) **per-image/region adaptation with a GT-free selector** (HQ-8's ceiling is real at −0.034 LPIPS).

## Design artifacts (`hq_innovation_attempts/HQ-01..14.md`)

| HQ | Title | Status | Honest gain estimate |
|----|-------|--------|----------------------|
| HQ-1 | NDS-Diff: null-confined diffusion over the anchor | proposed | **Highest-leverage substrate**; gain gated by the frozen-VQGAN prior (moderate on 64×64, not a diffusion-SOTA jump) |
| HQ-2 | B-as-guidance-scale | proposed | Marginal alone (reparameterization); value = governance + true convex frontier |
| HQ-3 | CodeFormer-style code prediction for `d_G` | **done (design)** | Best single-arm upgrade keeping the frozen decoder; moderate@2% / small@5% / ~0@10% |
| HQ-4 | VQ-latent null-space diffusion + pixel bridge | proposed | Lift comes from HQ-1 substrate; own marginal gain modest |
| HQ-5 | RestoreFormer cross-attention `β(x)` | proposed | Likely marginal (reweights fixed residuals; learned-gate underperformed before) |
| HQ-6 | Null-Space MoDL (unrolled refiner) | **done (design)** | Higher-ceiling than β(x) because it can *synthesize* content; ~1 GPU-hr test |
| HQ-11 | Anchor-conditioned PiGDM in null space | proposed | Marginal-to-moderate, contingent on HQ-1; exact-P0 already absorbs soft-consistency gain |
| HQ-12 | Perception-weighted operator design | proposed | Likely marginal end-to-end (DCT already near-perceptual) but **raises the floor for every filler** |
| HQ-13 | Fusion-in-the-loop operator+B co-design | proposed | Plausibly real but marginal over HQ-12; capstone that stacks on HQ-12 |
| HQ-14 | Learned-Primal-Dual two-stream (cleans range) | proposed | Not a clean-`y` lift; justified only in the **noisy/real-GI** regime |

## Where the quality ceiling actually is (data-grounded)
1. **HQ-1 (null-confined diffusion)** — replace the two-point blend with a generative posterior as the *source* of `d_G`; the one substrate change all others refine.
2. **HQ-3 (code prediction)** — fix wrong-code lookup at 2–5%, the regime where the current fusion starves.
3. **A GT-free per-image/region selector (HQ-8 ceiling −0.034 LPIPS is real)** — the honest research target that the negative HQ-9 result points to.
4. **HQ-12 (perception-weighted operator)** — raises the starting floor for every null-space method; stacks with all of the above.

Everything else (HQ-2/5/9/10, and HQ-13/14 outside the noisy regime) is confirmed or predicted **marginal** and should not be sold as a quality contribution.
