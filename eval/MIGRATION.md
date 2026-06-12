# Migration Notes

Copy the entire `eval/` directory into the repository root on branch `g2r-eval`. The toolkit is self-contained and does not import training loops or assume training package names.

Install the pinned evaluation environment from `eval/requirements-eval.txt`. CPU execution is the default; GPU is optional through the `--device` argument where supported by PyTorch.

Result dumps may be `.npz`, `.pt`, or `.pth` and must contain ground truth `x`, `samples`, `sample_mean`, deterministic `baseline`, measurements `y`, and paths to `A` and `P0` artifacts. Relative `A_path` and `P0_path` values are resolved relative to the result dump directory.

The checker accepts flattened images `(N, 4096)`, image tensors `(N, 64, 64)`, and samples shaped `(N, K, 4096)` or `(N, K, 64, 64)`. `G-CERT` uses `samples_unclipped` when present; otherwise it falls back to `samples`.

Five-line checker usage:

```bash
python -m pip install -r eval/requirements-eval.txt
python -m eval.checker results/seed0_dump.npz --json-out results/seed0_eval.json
python -m eval.visualize results/seed0_dump.npz --out-dir results/seed0_viz
python -m eval.seed_variance results/seed0_dump.npz results/seed1_dump.npz results/seed2_dump.npz --json-out results/seed_variance.json
RUN_SLOW_EVAL_METRICS=1 python -m unittest eval.tests.test_metrics_slow -v
```
