from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class PILToTensor01:
    """Convert a grayscale PIL image to [1, H, W] float tensor without NumPy."""

    def __call__(self, img):
        if img.mode != "L":
            img = img.convert("L")
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        return data.view(img.height, img.width).float().div(255.0).unsqueeze(0)


def build_transform(img_size: int, *, dataset_name: str = "stl10", train: bool = False, use_augmentation: bool = False):
    ops = [transforms.Resize((img_size, img_size))]
    if train and use_augmentation and dataset_name in {"stl10", "stl10_train_only", "stl10_unlabeled_only", "cifar10_gray"}:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((img_size, img_size), padding=max(2, img_size // 16)),
            ]
        )
    if train and use_augmentation and dataset_name in {"mnist", "fashion_mnist"}:
        ops.append(transforms.RandomRotation(5))
    ops.extend([transforms.Grayscale(num_output_channels=1), PILToTensor01()])
    return transforms.Compose(ops)


def _limit_dataset(dataset, limit: int | None, seed: int):
    if limit is None:
        return dataset
    limit = min(int(limit), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _labels_for_dataset(dataset):
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        return None
    return torch.as_tensor(labels)


def _class_ids(dataset_name: str, class_filter) -> set[int] | None:
    if class_filter is None or class_filter == "" or class_filter == "null":
        return None
    if not isinstance(class_filter, (list, tuple, set)):
        class_filter = [class_filter]
    name_maps = {
        "stl10": {
            "airplane": 0,
            "bird": 1,
            "car": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "horse": 6,
            "monkey": 7,
            "ship": 8,
            "truck": 9,
        },
        "cifar10_gray": {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        },
    }
    base_name = "stl10" if str(dataset_name).startswith("stl10") else str(dataset_name)
    mapping = name_maps.get(base_name, {})
    ids = set()
    for item in class_filter:
        if isinstance(item, str) and not item.isdigit():
            if item not in mapping:
                raise ValueError(f"Unknown class_filter item for {dataset_name}: {item}")
            ids.add(mapping[item])
        else:
            ids.add(int(item))
    return ids


def _filter_dataset(dataset, dataset_name: str, class_filter):
    ids = _class_ids(dataset_name, class_filter)
    if ids is None:
        return dataset
    labels = _labels_for_dataset(dataset)
    if labels is None:
        return dataset
    indices = [idx for idx, label in enumerate(labels.tolist()) if int(label) in ids]
    return Subset(dataset, indices)


def _make_dataset(dataset_root: str, dataset_name: str, split: str, transform, download: bool = True):
    name = str(dataset_name).lower()
    if name in {"stl10", "stl10_train_only", "stl10_unlabeled_only"}:
        if name == "stl10_train_only" and split == "train+unlabeled":
            split = "train"
        if name == "stl10_unlabeled_only" and split == "train+unlabeled":
            split = "unlabeled"
        return datasets.STL10(root=dataset_root, split=split, transform=transform, download=download)
    if name == "mnist":
        train = split not in {"test", "val"}
        return datasets.MNIST(root=dataset_root, train=train, transform=transform, download=download)
    if name == "fashion_mnist":
        train = split not in {"test", "val"}
        return datasets.FashionMNIST(root=dataset_root, train=train, transform=transform, download=download)
    if name == "cifar10_gray":
        train = split not in {"test", "val"}
        return datasets.CIFAR10(root=dataset_root, train=train, transform=transform, download=download)
    raise ValueError(
        "dataset_name must be one of: stl10, stl10_train_only, stl10_unlabeled_only, "
        "cifar10_gray, mnist, fashion_mnist."
    )


def get_dataloaders(
    dataset_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
    seed: int = 42,
    train_split: str = "train+unlabeled",
    val_split: str = "test",
    pin_memory: bool = False,
    dataset_name: str = "stl10",
    class_filter=None,
    use_augmentation: bool = False,
):
    train_transform = build_transform(
        img_size, dataset_name=dataset_name, train=True, use_augmentation=use_augmentation
    )
    val_transform = build_transform(
        img_size, dataset_name=dataset_name, train=False, use_augmentation=False
    )
    train_set = _make_dataset(dataset_root, dataset_name, train_split, train_transform)
    val_set = _make_dataset(dataset_root, dataset_name, val_split, val_transform)
    train_set = _filter_dataset(train_set, dataset_name, class_filter)
    val_set = _filter_dataset(val_set, dataset_name, class_filter)

    train_set = _limit_dataset(train_set, limit_train_samples, seed)
    val_set = _limit_dataset(val_set, limit_val_samples, seed + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def get_val_dataloader(
    dataset_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    limit_val_samples: int | None = None,
    seed: int = 42,
    val_split: str = "test",
    pin_memory: bool = False,
    dataset_name: str = "stl10",
    class_filter=None,
):
    transform = build_transform(img_size, dataset_name=dataset_name, train=False, use_augmentation=False)
    val_set = _make_dataset(dataset_root, dataset_name, val_split, transform)
    val_set = _filter_dataset(val_set, dataset_name, class_filter)
    val_set = _limit_dataset(val_set, limit_val_samples, seed + 1)
    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
