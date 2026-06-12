# Follow-Up Gap Items

No training was launched. This follow-up only inspected code, loaded checkpoints for certificate diagnostics, and unit-tested the local infrastructure utilities.

## 1. Dataloader Check

### Main pipeline

Source: `src/datasets.py`

Verbatim dataset constructor:

```python
def _make_dataset(dataset_root: str, dataset_name: str, split: str, transform, download: bool = True):
    name = str(dataset_name).lower()
    if name in {"stl10", "stl10_train_only", "stl10_unlabeled_only"}:
        if name == "stl10_train_only" and split == "train+unlabeled":
            split = "train"
        if name == "stl10_unlabeled_only" and split == "train+unlabeled":
            split = "unlabeled"
        return datasets.STL10(root=dataset_root, split=split, transform=transform, download=download)
```

Verbatim main split defaults:

```python
def get_dataloaders(
    ...
    train_split: str = "train+unlabeled",
    val_split: str = "test",
    ...
):
    ...
    train_set = _make_dataset(dataset_root, dataset_name, train_split, train_transform)
    val_set = _make_dataset(dataset_root, dataset_name, val_split, val_transform)
```

Verbatim subsampling logic and seed source:

```python
def _limit_dataset(dataset, limit: int | None, seed: int):
    if limit is None:
        return dataset
    limit = min(int(limit), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)
...
train_set = _limit_dataset(train_set, limit_train_samples, seed)
val_set = _limit_dataset(val_set, limit_val_samples, seed + 1)
```

Main training call site, source `src/train.py`:

```python
train_loader, val_loader = get_dataloaders(
    dataset_root=config["dataset_root"],
    img_size=config["img_size"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    limit_train_samples=config["limit_train_samples"],
    limit_val_samples=config["limit_val_samples"],
    seed=config["seed"],
    pin_memory=pin_memory,
    dataset_name=config.get("dataset_name", "stl10"),
    class_filter=config.get("class_filter"),
    use_augmentation=bool(config.get("use_augmentation", False)),
)
```

Main eval call site, source `src/eval.py`:

```python
val_loader = get_val_dataloader(
    dataset_root=config["dataset_root"],
    img_size=config["img_size"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    limit_val_samples=config["limit_val_samples"],
    seed=config["seed"],
    pin_memory=device.type == "cuda",
    dataset_name=config.get("dataset_name", "stl10"),
    class_filter=config.get("class_filter"),
)
```

Answer: for the main Scr-5 pipeline, train/test disjointness is guaranteed by the canonical `torchvision.datasets.STL10` partition if the canonical dataset files are trusted: training uses `split="train+unlabeled"` by default and eval uses `split="test"`. The subsampled subsets are independently drawn from those canonical partitions with `seed` for train and `seed + 1` for val/test. However, this is a code-level guarantee; the exported result bundle still lacks saved train/test index hashes.

### G1 pilot loaders

G1 pilot loader source: `src/phase53B_common.py`

```python
def make_loader(config: dict[str, Any], device: torch.device):
    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 2)),
        limit_val_samples=int(config.get("limit_val_samples", 512)),
        seed=int(config.get("seed", 123)),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
```

G1 optional GAN source: `src/phase53C_optional_gan_posterior.py`

```python
while steps < args.max_steps:
    for batch in make_loader(config, device):
```

```python
for bidx, batch in enumerate(make_loader(config, device)):
```

```python
pr, grid = _posterior_rows(generator, config, measurement, make_loader(config, device), device, task, info["metadata"]["display"], args.num_samples_per_y, args.eval_batches)
```

Answer: G1 uses `get_val_dataloader`, whose default `val_split` is `test`. The same val/test loader is used both for the optional GAN update loop and later pilot evaluation/posterior diagnostics. This is a direct protocol-drift/test-set-adaptation risk for the G1 pilot and is sufficient to keep G1 out of any claim-ready supplement evidence.

## 2. Certificate-Invariance Number

Script: `certificate_invariance_recheck.py`

Output: `CERTIFICATE_INVARIANCE_RECHECK.json`

Result on 256 STL-10 test samples, same subset seed `43`, device `cuda`:

| checkpoint | loaded key | RelMeasErr mean | RelMeasErr std | n |
|---|---|---:|---:|---:|
| published mean `last.pt` | `generator_ema` | 0.005484469700604677 | 0.002216092310845852 | 256 |
| G1 `scr5/source_checkpoint.pt` | `generator_ema` | 0.005484469700604677 | 0.002216092310845852 | 256 |

`source_minus_mean_relmeas = 0.0`.

The post-GAN pilot checkpoint does not exist. Search under `E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling` found no Scr-5 `.pt` candidate other than `source_checkpoint.pt`, `_source_bundle_leaf/source_checkpoint_last.pt`, and `Q_exact_null.pt`. Exact failure recorded in JSON:

```text
The optional GAN pilot updated the generator in memory but did not save a post-GAN/fine-tuned checkpoint; only source_checkpoint.pt is present for Scr-5.
```

Therefore the old post-GAN G1 value remains reportable only as an aggregate CSV value from `optional_gan_results.csv`: average Scr-5 pilot RelMeasErr `0.005365385441109538`, compared with published mean-mode full eval `0.005452804392203688`. It cannot be recomputed from a saved post-GAN checkpoint.

## 3. Blockers Verbatim

From `G2_READY.md`:

```text
READY TO LAUNCH: no - blockers: ['No saved main no-leak train/val/test split hashes are available.', 'Pilot split/eval index hashes are not available.', 'Old G1 code path appears deterministic with no explicit stochastic z.', 'Controlled G2 smoke was not run because provenance is unsafe and stochastic branch implementation has not been reviewed.']
```

The <=200-iteration smoke test was skipped because the provenance safety precondition failed: no saved main no-leak train/val/test split hashes were available. The follow-up dataloader inspection additionally found that G1's optional GAN update loop used the val/test loader, so running a smoke test that inherits this line would be unsafe until the loader/split provenance is repaired.

## 4. Infrastructure Status

Created files:

- `eval_sampling.py`
- `tools/split_hash.py`
- `sampling_metrics.py`

Unit-test script:

- `test_infra_utilities.py`

Unit-test output:

- `INFRA_UNIT_TEST_RESULTS.json`

Unit-test status: `pass`.

Functions/paths exercised:

- `eval_sampling.save_stochastic_samples_npz`
- `eval_sampling.save_batch_stochastic_samples`
- `tools/split_hash.py` CLI
- `sampling_metrics.summarize_saved_samples`
- `sampling_metrics.optional_perceptual_availability`

Perceptual package availability remains unavailable locally: `lpips=false`, `cleanfid=false`, `torchmetrics=false`.
