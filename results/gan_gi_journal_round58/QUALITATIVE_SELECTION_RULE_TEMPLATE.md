# Frozen qualitative-gallery selection record

Purpose: record the pre-specified visualisation rule separately from any image-quality result.

Selection rule: from lane 0, rate 05, read `cache/test_cache_manifest.json` and its `test_samples` list. For each STL-10 label 2 (car), 3 (cat), 5 (dog), and 6 (horse), select the row with the smallest `source_index`. Do not inspect truth, reconstruction, LPIPS, PSNR, SSIM, loss, or any visual quality criterion before selection. Match the same `source_index`, label, and raw SHA-256 in rate 10.

Required record:

| Class | STL-10 label | Source index | Raw SHA-256 | Local index at 05 | Local index at 10 |
| --- | ---: | ---: | --- | ---: | ---: |
| car | 2 |  |  |  |  |
| cat | 3 |  |  |  |  |
| dog | 5 |  |  |  |  |
| horse | 6 |  |  |  |  |

Required controls:

- Confirm the lane-0 held-out completion receipt, each cache manifest, `metric_vectors.npz`, and `summary.json` against their recorded SHA-256 values.
- Reconstruct solely from the existing cache and frozen lane-0 weights. Do not train, fine-tune, change a hyperparameter, or reconstruct a different candidate.
- Use high-pass cutoff 0.12, transition 0.03, alpha 0.5, and 4096 exact-projection iterations. For all three final outputs, the equality target is `geometry.intrinsic_record(raw cached y)`. The clipped anchor `x0` is a model input only and must never be used as the terminal projection target.
- Save the unmodified arrays and the projection certificates alongside the rendered plate. Render the `|FOHI − truth|` panel from its saved array. Do not attach old clipped-anchor metric vectors to raw-fiber outputs; a full fixed raw-fiber evaluation is required for matched metrics.
