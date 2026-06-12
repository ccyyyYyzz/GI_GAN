"""Split-guard tests. Each guard has a deliberate-violation test that must be
caught by the guard (the test passes because the guard raises)."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.split_guard import (
    SplitViolationError,
    assert_train_loader_disjoint_from_test,
    assert_val_loader_held_out,
    collect_sample_identities,
    get_train_dataloader_guarded,
    protected_test_identities,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeSTL10:
    """Mimics torchvision STL10: exposes .split."""

    def __init__(self, split: str, n: int = 24):
        self.split = split
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return torch.zeros(1, 8, 8), 0


class FakeMNIST:
    """Mimics torchvision MNIST: exposes .train (bool)."""

    def __init__(self, train: bool, n: int = 24):
        self.train = train
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return torch.zeros(1, 8, 8), 0


def loader_of(dataset):
    return DataLoader(dataset, batch_size=4)


def test_train_split_loader_passes():
    audit = assert_train_loader_disjoint_from_test(loader_of(FakeSTL10("train+unlabeled")))
    assert audit["split_guard_active"] is True
    assert audit["test_overlap"] == 0
    assert audit["splits"] == ["train+unlabeled"]


def test_violation_test_split_loader_raises():
    # Deliberate violation: a training-time loader built on the test split —
    # exactly the previous pilot's get_val_dataloader(val_split="test") incident.
    with pytest.raises(SplitViolationError):
        assert_train_loader_disjoint_from_test(loader_of(FakeSTL10("test")))


def test_violation_nested_subset_of_test_raises():
    nested = Subset(Subset(FakeSTL10("test"), [0, 2, 4, 6]), [1, 3])
    with pytest.raises(SplitViolationError):
        assert_train_loader_disjoint_from_test(loader_of(nested))


def test_nested_subset_indices_resolve_to_base():
    nested = Subset(Subset(FakeSTL10("train"), [0, 2, 4, 6]), [1, 3])
    ids = collect_sample_identities(nested)
    assert ids == {("train", 2), ("train", 6)}


def test_violation_index_intersection_with_protected_set_raises():
    # Same-split index overlap: a "val" subset carved from train must stay
    # disjoint from a protected evaluation subset.
    train_loader = loader_of(Subset(FakeSTL10("train"), [0, 1, 2, 3]))
    protected = collect_sample_identities(Subset(FakeSTL10("train"), [3, 4, 5]))
    with pytest.raises(SplitViolationError):
        assert_train_loader_disjoint_from_test(train_loader, test=protected)


def test_disjoint_indices_pass_against_protected_set():
    train_loader = loader_of(Subset(FakeSTL10("train"), [0, 1, 2]))
    protected = protected_test_identities(test_indices=[0, 1, 2])  # ("test", i) identities
    audit = assert_train_loader_disjoint_from_test(train_loader, test=protected)
    assert audit["test_overlap"] == 0


def test_violation_mnist_test_flag_raises():
    with pytest.raises(SplitViolationError):
        assert_train_loader_disjoint_from_test(loader_of(FakeMNIST(train=False)))
    assert_train_loader_disjoint_from_test(loader_of(FakeMNIST(train=True)))


def test_violation_unknown_dataset_type_raises():
    class Opaque:
        def __len__(self):
            return 4

    with pytest.raises(SplitViolationError):
        assert_train_loader_disjoint_from_test(loader_of(Opaque()))


def test_violation_guarded_factory_refuses_test_split():
    # The guarded factory must reject a test split before touching any data.
    with pytest.raises(SplitViolationError):
        get_train_dataloader_guarded(
            dataset_root="unused",
            img_size=64,
            batch_size=4,
            num_workers=0,
            train_split="test",
        )


class FakeSTL10TrainUnlabeled(FakeSTL10):
    """train+unlabeled with torchvision label layout: labeled first, then -1."""

    def __init__(self, n_labeled: int, n_unlabeled: int):
        super().__init__("train+unlabeled", n_labeled + n_unlabeled)
        self.labels = [0] * n_labeled + [-1] * n_unlabeled


def test_canonical_identities_for_train_plus_unlabeled():
    base = FakeSTL10TrainUnlabeled(n_labeled=4, n_unlabeled=6)
    ids = collect_sample_identities(Subset(base, [2, 5]))
    assert ids == {("train", 2), ("unlabeled", 1)}


def test_violation_val_overlapping_training_pool_detected():
    # Deliberate violation: val_split='train' while training on
    # 'train+unlabeled' — the same physical images under two split strings.
    train_loader = loader_of(FakeSTL10TrainUnlabeled(n_labeled=4, n_unlabeled=6))
    val_loader = loader_of(Subset(FakeSTL10("train"), [2, 3]))
    with pytest.raises(SplitViolationError, match="model-selection leakage"):
        assert_val_loader_held_out(val_loader, train_loader)


def test_violation_val_resolving_to_test_split_detected():
    # mnist/fashion_mnist/cifar10_gray map the string 'val' onto train=False,
    # i.e. the TEST set; the resolved-loader check must catch it.
    train_loader = loader_of(FakeMNIST(train=True))
    val_loader = loader_of(FakeMNIST(train=False))
    with pytest.raises(SplitViolationError, match="TEST split"):
        assert_val_loader_held_out(val_loader, train_loader)


def test_heldout_val_configuration_passes():
    # train on the unlabeled images, validate on the labeled train images.
    train_loader = loader_of(FakeSTL10("unlabeled"))
    val_loader = loader_of(FakeSTL10("train"))
    audit = assert_val_loader_held_out(val_loader, train_loader)
    assert audit["heldout_val_verified"] is True
    assert audit["train_overlap"] == 0


# ---------------------------------------------------------------------------
# Physical-removal regression tests: the GAN update loops must no longer be
# able to pull get_val_dataloader / the default test split.
# ---------------------------------------------------------------------------

def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _training_region(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_phase53c_gan_loop_cannot_reach_test_loader():
    src = _read("src/phase53C_optional_gan_posterior.py")
    region = _training_region(src, "while steps < args.max_steps:", "generator.eval()")
    assert "make_loader(" not in region, "GAN update loop must not call the eval/test loader"
    assert "get_val_dataloader" not in src
    assert "make_train_loader(config, device)" in src


def test_phase53b_gan_loop_cannot_reach_test_loader():
    src = _read("src/phase53B_blind_gan_pilot.py")
    region = _training_region(src, "while step < args.max_steps:", "generator.eval()")
    assert "make_loader(" not in region, "GAN update loop must not call the eval/test loader"
    assert "loader = make_train_loader(config, device)" in src


def test_critic_data_collectors_use_train_split():
    b = _read("src/phase53B_common.py")
    body = b[b.index("def collect_pair_dataset(") :]
    body = body[: body.index("\ndef ", 10)]
    assert "make_train_loader(config, device)" in body
    assert "loader = make_loader(" not in body

    c = _read("src/phase53C_common.py")
    body = c[c.index("def collect_exact_null_pair_dataset(") :]
    body = body[: body.index("\ndef ", 10)]
    assert "make_train_loader(config, device)" in body
    assert "loader = make_loader(" not in body


def test_train_py_split_guard_wired():
    src = _read("src/train.py")
    assert "assert_train_loader_disjoint_from_test(" in src
    assert "enforce_run_protocol(config[\"output_dir\"], config)" in src
    assert "assert_val_loader_held_out(" in src
    assert 'train_split=str(config.get("train_split", "train+unlabeled"))' in src


def test_pilots_compute_relmeaserr_on_unclipped_vector():
    # Binding convention from gates.yaml: RelMeasErr is ALWAYS computed on the
    # UNCLIPPED reconstruction vector.
    for rel_path in ["src/phase53B_blind_gan_pilot.py", "src/phase53C_optional_gan_posterior.py"]:
        src = _read(rel_path)
        assert "x_hat_unclamped" in src, rel_path
        for line in src.splitlines():
            if "relmeas_tensor(" in line and "def " not in line:
                assert "x_hat_unclamped" in line or "flatten_img" not in line, (
                    f"{rel_path}: RelMeasErr computed on a possibly clipped tensor: {line.strip()}"
                )


def test_configure_task_caps_training_loader_samples():
    # Bundle resolved configs carry limit_train_samples: 50000; the session
    # limit must override it for phase53 training-time loaders.
    src = _read("src/phase53B_common.py")
    assert 'config["limit_train_samples"] = int(args.limit_samples)' in src
