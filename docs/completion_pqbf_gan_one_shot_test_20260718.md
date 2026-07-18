# PQBF-GAN frozen one-shot completion test (2026-07-18)

## Decision

`COMPLETION_TEST_GO`.

The one-shot 128-image test passes every gate frozen before opening the test split. PQBF-GAN improves all three reconstruction metrics relative to the exact bounded-fiber LMMSE anchor. Relative to the Stage-A content model, it produces a large perceptual improvement within the fixed PSNR/SSIM guardrails. Relative to an equal-length zero-adversarial continuation, the GAN-specific LPIPS gain is small but statistically resolved.

## Frozen protocol

- Theory adjudication: `GAN_FCC/theory_exchange/responses/20260718_completion_pqbf_positive_pilot_round20_gptpro.md`, content commit `a8b0937`, provenance commit `65325d0`, file SHA-256 `d8f352bf57fdb84258603e09aeefd0bbf9bf24aed677488c80ece727b7082cf1`.
- Code and test configuration frozen at GI_GAN commit `ff7d2df` before test scoring.
- Allocation: 2048 train, 128 validation, and 128 test images from STL-10; all source hashes are disjoint.
- Stage-A checkpoint SHA-256: `7c50fc5c732e9c2d590cfd47f96f9082bd4ca1698c359b7cefb834fba416c9f3`.
- Validation-selected Stage-B step-750 checkpoint SHA-256: `1aa9272117ca7230552dd96880c14a1d6704533c60d904b8fff90d4b0803c847`.
- Matched zero-adversarial 750-step continuation checkpoint SHA-256: `ca9b0913e97efa7d165f896850d7cfda6a860db631379b04cef2c63c9bd65710`.
- Test source-index SHA-256: `60df529697ce43117b953fce5ec46a48dcd18e8af063afc2a381a45178153da8`.
- Paired bootstrap: 1000 resamples, seed `20260721`.
- No checkpoint reselection, threshold change, post-evaluation clamp, or second test look was used.

## Test metrics

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Exact bounded-fiber LMMSE anchor | 22.8097 | 0.60801 | 0.32443 |
| Stage-A PQBF content model | 23.4680 | 0.67077 | 0.18219 |
| Matched 750-step zero-adversarial continuation | 23.3848 | 0.67178 | 0.15910 |
| **PQBF-GAN, validation-selected step 750** | **23.3800** | **0.67172** | **0.15856** |

### PQBF-GAN versus the physical anchor

- PSNR: `+0.57025 dB`, paired 95% interval `[+0.45822, +0.69844] dB`.
- SSIM: `+0.06371`, paired 95% interval `[+0.05307, +0.07361]`.
- LPIPS: `-0.16586`, paired 95% interval `[-0.17909, -0.15249]`, a 51.12% relative reduction.

### PQBF-GAN versus Stage A

- PSNR: `-0.08796 dB`, within the frozen `-0.10 dB` guardrail.
- SSIM: `+0.000947`, within the frozen `-0.002` guardrail.
- LPIPS: `-0.023628`, paired 95% interval `[-0.026045, -0.021258]`, a 12.97% relative reduction.

### GAN-specific causal contrast

The equal-length supervised continuation explains most of the Stage-A-to-Stage-B gain. Against that stricter control, PQBF-GAN changes:

- PSNR by `-0.00484 dB`;
- SSIM by `-0.0000596`;
- LPIPS by `-0.0005386`, paired 95% interval `[-0.0006598, -0.0004161]`, a 0.3385% relative reduction.

Thus the alternating GAN is genuine and gives a reproducible incremental perceptual benefit, but the report must not attribute the full 12.97% Stage-A-to-GAN LPIPS reduction to adversarial learning alone.

## Constraint audit

- Maximum original-measurement relative residual across the four scored methods: `1.38e-8`, below the frozen `1.00e-6` threshold.
- Every scored pixel lies in `[0,1]`.
- The scored tensor is the certified box-fiber projection; there is no later clamp.

## Auxiliary initialization check

An independent second training seed, evaluated only on validation and not used to select the frozen test method, also produced eligible GAN checkpoints. It selected step 500 with PSNR delta `-0.07463 dB`, SSIM delta `+0.0000656`, and lower LPIPS than its Stage-A endpoint. This supports directional robustness but is not part of the one-shot test claim.

## Artifact provenance

- Test run summary: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_selected_test_once_seed0/reports/run_summary.json`.
- Run-summary SHA-256: `1462bc92674db61d595e270b2515060c17dd64ed6d0ad6fe1d295b257f2a76ed`.
- Frozen test manifest: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_selected_test_once_seed0/reports/frozen_test_manifest.json`.
- Manifest SHA-256: `9b40d74c6fe7ca1cde0ed790fb9469e9c0a086961f751d2262877748856516d8`.
- Per-image metrics SHA-256: `1e06483d01b1dc102a91bdec5b06341c72201b0540cda517004d32c27827be0f`.
- Qualitative grid: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_selected_test_once_seed0/reports/test/qualitative_grid.png`.
- Second-seed validation summary SHA-256: `5bafbcf26b37e71b64ec7e3eed5dcb036007dd97ca3fe6c908f2a607f17b2511`.

## Completion-report conclusion

PQBF-GAN combines a compact projector-query reconstruction network, a genuine conditional GAN, and a certified bounded measurement-fiber output. On an independent STL-10 simulation test, it improves PSNR, SSIM, and LPIPS relative to the physics-derived anchor while preserving the recorded measurements. The adversarial component gives a statistically resolved but modest additional perceptual gain beyond an equal-length supervised continuation. Unmeasured details remain prior-supported plausible estimates rather than uniquely measurement-verified truth.
