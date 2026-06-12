"""Runtime split guards for the g2r_ run series.

Every training-time loop (including any adversarial/GAN update and any
critic-data collection that gradients flow from) may consume the TRAIN split
only.  This module identifies, at dataloader creation time, exactly which
underlying dataset split and base indices a loader draws from, and raises
``SplitViolationError`` if any of them intersect the test split.

Background: a previous adversarial pilot consumed the default TEST split via
``get_val_dataloader`` (whose ``val_split`` defaults to ``"test"``).  The
guard below makes that class of mistake impossible to reintroduce silently.
"""

from __future__ import annotations

from typing import Any, Iterable

from torch.utils.data import DataLoader, Subset

from .datasets import _filter_dataset, _limit_dataset, _make_dataset, build_transform

# Split tags that a training-time loader is allowed to draw from.
TRAIN_ALLOWED_SPLIT_TAGS = {"train", "unlabeled", "train+unlabeled"}
TEST_SPLIT_TAG = "test"


class SplitViolationError(RuntimeError):
    """Raised when a training-time dataloader can reach test-split samples."""


def _base_dataset_and_indices(dataset: Any) -> tuple[Any, list[int] | None]:
    """Walk a (possibly nested) Subset chain back to the base dataset.

    Returns the base dataset and the resolved base-dataset indices the input
    can reach, or ``None`` for "all indices".
    """
    # Subset(Subset(base, a), b) reaches base index a[b[k]]: each outer layer's
    # indices select into the next-inner layer, so compose outside-in.
    indices: list[int] | None = None
    current = dataset
    while isinstance(current, Subset):
        layer = [int(i) for i in current.indices]
        indices = layer if indices is None else [layer[i] for i in indices]
        current = current.dataset
    return current, indices


def split_tag_of(dataset: Any) -> str:
    """Canonical split tag of a base torchvision dataset."""
    base, _ = _base_dataset_and_indices(dataset)
    split = getattr(base, "split", None)
    if split is not None:
        return str(split)
    train_flag = getattr(base, "train", None)
    if train_flag is not None:
        return "train" if bool(train_flag) else TEST_SPLIT_TAG
    raise SplitViolationError(
        f"Cannot determine the split of dataset type {type(base).__name__}; "
        "refusing to use it in a training-time loop."
    )


def _stl10_labeled_count(base: Any) -> int | None:
    """Number of labeled samples in an STL10 'train+unlabeled' dataset.

    torchvision concatenates the labeled train images first, then the
    unlabeled ones (label -1). Returns None when labels are unavailable.
    """
    labels = getattr(base, "labels", None)
    if labels is None:
        return None
    try:
        return sum(1 for label in labels if int(label) != -1)
    except (TypeError, ValueError):
        return None


def _canonical_identity(tag: str, idx: int, n_labeled: int | None) -> tuple[str, int]:
    """Map ('train+unlabeled', i) onto the physical ('train'/'unlabeled', j)
    identity so the same image gets the same key regardless of which split
    string a loader was built with."""
    if tag == "train+unlabeled" and n_labeled is not None:
        if idx < n_labeled:
            return ("train", idx)
        return ("unlabeled", idx - n_labeled)
    return (tag, idx)


def collect_sample_identities(loader_or_dataset: Any) -> set[tuple[str, int]]:
    """Return the set of canonical (split_tag, base_index) pairs the loader can reach."""
    # Unwrap a DataLoader only; a bare Subset also has a .dataset attribute
    # and must keep its outer index layer.
    if isinstance(loader_or_dataset, DataLoader):
        dataset = loader_or_dataset.dataset
    else:
        dataset = loader_or_dataset
    base, indices = _base_dataset_and_indices(dataset)
    tag = split_tag_of(base)
    if indices is None:
        indices = list(range(len(base)))
    n_labeled = _stl10_labeled_count(base) if tag == "train+unlabeled" else None
    return {_canonical_identity(tag, int(i), n_labeled) for i in indices}


def protected_test_identities(
    test_loader_or_dataset: Any = None,
    *,
    test_indices: Iterable[int] | None = None,
) -> set[tuple[str, int]]:
    """Identity set of the protected test data.

    With no arguments the whole test split is protected (any sample whose
    base split is "test" counts as a test sample).  Pass an explicit loader
    or index list to protect a specific evaluation subset instead.
    """
    if test_loader_or_dataset is not None:
        return collect_sample_identities(test_loader_or_dataset)
    if test_indices is not None:
        return {(TEST_SPLIT_TAG, int(i)) for i in test_indices}
    return set()  # sentinel: whole-test-split rule applies


def assert_train_loader_disjoint_from_test(
    loader: Any,
    *,
    test: set[tuple[str, int]] | None = None,
    context: str = "training loop",
) -> dict[str, Any]:
    """Assert a training-time loader cannot reach test-split samples.

    Collects the loader's (split, index) identities and (1) requires the
    underlying split to be a train split, (2) requires an empty intersection
    with the protected test identity set.  Raises SplitViolationError on any
    violation; returns a small audit dict on success.
    """
    identities = collect_sample_identities(loader)
    tags = {tag for tag, _ in identities}
    forbidden_tags = tags - TRAIN_ALLOWED_SPLIT_TAGS
    if forbidden_tags:
        raise SplitViolationError(
            f"Split guard violation in {context}: training-time loader draws from "
            f"split(s) {sorted(forbidden_tags)}; only {sorted(TRAIN_ALLOWED_SPLIT_TAGS)} "
            "are allowed for training. The previous pilot's test-set adaptation "
            "incident is exactly this code path."
        )
    protected = test if test else set()
    overlap = identities & protected
    if overlap:
        sample = sorted(overlap)[:8]
        raise SplitViolationError(
            f"Split guard violation in {context}: {len(overlap)} training sample(s) "
            f"intersect the test set (first few: {sample})."
        )
    return {
        "context": context,
        "n_samples": len(identities),
        "splits": sorted(tags),
        "test_overlap": 0,
        "split_guard_active": True,
    }


def assert_val_loader_held_out(
    val_loader: Any,
    train_loader: Any,
    *,
    context: str = "validation loader",
) -> dict[str, Any]:
    """g2r protocol: validation must be genuinely held out.

    (1) The val loader's RESOLVED split must not be the test split — a
    name-based config check is not enough because _make_dataset maps the
    string 'val' onto the test set for mnist/fashion_mnist/cifar10_gray.
    (2) The val samples must be disjoint from the training samples under
    canonical identities (so 'train+unlabeled' vs 'train' overlaps are
    caught), otherwise best-checkpoint selection happens on trained data.
    """
    val_ids = collect_sample_identities(val_loader)
    val_tags = {tag for tag, _ in val_ids}
    if TEST_SPLIT_TAG in val_tags:
        raise SplitViolationError(
            f"Split guard violation in {context}: the validation loader resolves to the "
            "TEST split. g2r runs touch the test split exactly once, at the end; "
            "model selection on test metrics is test-set adaptation."
        )
    train_ids = collect_sample_identities(train_loader)
    overlap = val_ids & train_ids
    if overlap:
        sample = sorted(overlap)[:8]
        raise SplitViolationError(
            f"Split guard violation in {context}: {len(overlap)} validation sample(s) "
            f"are also in the training set (first few: {sample}); best-checkpoint "
            "selection on trained samples is model-selection leakage. Use sample-"
            "disjoint splits, e.g. train_split='unlabeled' with val_split='train'."
        )
    return {
        "context": context,
        "n_val": len(val_ids),
        "val_splits": sorted(val_tags),
        "train_overlap": 0,
        "heldout_val_verified": True,
    }


def get_train_dataloader_guarded(
    dataset_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    limit_train_samples: int | None = None,
    seed: int = 42,
    train_split: str = "train+unlabeled",
    pin_memory: bool = False,
    dataset_name: str = "stl10",
    class_filter=None,
    use_augmentation: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    context: str = "guarded train loader",
) -> DataLoader:
    """Build a TRAIN-split dataloader and run the split guard on it.

    This is the only loader factory training-time loops (GAN updates, critic
    data collection) are allowed to use.  There is deliberately no
    ``val_split``/``test`` parameter.
    """
    if str(train_split) not in TRAIN_ALLOWED_SPLIT_TAGS:
        raise SplitViolationError(
            f"get_train_dataloader_guarded refuses train_split={train_split!r}; "
            f"allowed: {sorted(TRAIN_ALLOWED_SPLIT_TAGS)}."
        )
    transform = build_transform(
        img_size, dataset_name=dataset_name, train=True, use_augmentation=use_augmentation
    )
    train_set = _make_dataset(dataset_root, dataset_name, train_split, transform)
    train_set = _filter_dataset(train_set, dataset_name, class_filter)
    train_set = _limit_dataset(train_set, limit_train_samples, seed)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if len(loader) == 0:
        raise ValueError(
            f"{context}: train loader yields zero batches "
            f"({len(train_set)} samples, batch_size={batch_size}, drop_last={drop_last}); "
            "a step-counted training loop would hang forever."
        )
    assert_train_loader_disjoint_from_test(loader, context=context)
    return loader
