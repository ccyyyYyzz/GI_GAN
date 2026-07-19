# Reproducing the frozen FOHI held-out release

Each `lane*_frozen_fohi_release.tar.gz` is an independent, SHA-verified record
of one already completed held-out lane. It includes frozen source files,
weights, configurations, receipts, metric vectors, logs, the environment
snapshot, and the lane dispatch archive. The 450 MB `test_cache.pt` is absent:
it is regenerated from STL-10 and the included frozen assets.

## Verify and aggregate locally

Extract the three archives and verify each directory from its own root:

```bash
sha256sum -c SHA256SUMS
```

Obtain the public aggregation program before running the decision:

```bash
git clone https://github.com/ccyyyYyzz/GI_GAN.git
cd GI_GAN
git checkout codex/gan-gi-journal-poc-20260718
```

The decision is reproducible without GPU or `/content`:

```bash
python aggregate_frozen_fohi_heldout.py \
  --input-dirs lane0_frozen_fohi_release/results lane1_frozen_fohi_release/results lane2_frozen_fohi_release/results \
  --freeze-manifest lane0_frozen_fohi_release/freeze/heldout_freeze.json \
  --bootstrap-reps 20000 --bootstrap-seed 20260719 \
  --output-dir reproduced_decision
```

The three lane manifests and split hashes must agree. This operation reads only
the released arrays and regenerates the hierarchical bootstrap decision.

## Recreate the Colab `/content` layout on another machine

The original one-shot driver was frozen with Colab absolute paths. Run it inside
a container or VM whose `/content` is a bind mount; do not edit frozen source.
Mount an empty host directory at `/content`, then run:

```bash
python lane0_frozen_fohi_release/tools/materialize_content_layout.py \
  --release-root lane0_frozen_fohi_release --content-root /content
```

This maps bundled code to `/content/GI_GAN`, frozen artifacts to their recorded
`/content/...` locations, and the dispatch zip to `/content/gan_rate_bundle.zip`.
Supply STL-10 under `/content/datasets`, then invoke the copied frozen driver
with a new output directory. This re-execution is a computational audit only;
it must not alter the closed held-out inference or tune the method.
