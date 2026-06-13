from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_CONFIG_DIR = REPO_ROOT / "configs" / "pub_baselines"
COLAB_CONFIG_DIR = WINDOWS_CONFIG_DIR / "colab"

WINDOWS_PREFIX = r"E:\ns_mc_gan_gi"
COLAB_PREFIX = "/content/drive/MyDrive/ns_mc_gan_gi"

BASELINE_CONFIGS = [
    "unet_rad5_pub.yaml",
    "unet_scr5_pub.yaml",
    "unrolled_ista_rad5_pub.yaml",
    "unrolled_ista_scr5_pub.yaml",
    "resunet_rad5_pub.yaml",
    "resunet_scr5_pub.yaml",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a mapping.")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def rewrite_path_string(value: str) -> str:
    if not value.startswith(WINDOWS_PREFIX):
        return value
    suffix = value[len(WINDOWS_PREFIX) :].replace("\\", "/")
    return f"{COLAB_PREFIX}{suffix}"


def rewrite_paths(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[tuple[str, str, str]]]:
    rewrites: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        rewritten: dict[Any, Any] = {}
        for key, child in value.items():
            new_child, child_rewrites = rewrite_paths(child, (*path, str(key)))
            rewritten[key] = new_child
            rewrites.extend(child_rewrites)
        return rewritten, rewrites
    if isinstance(value, list):
        rewritten_list = []
        for index, child in enumerate(value):
            new_child, child_rewrites = rewrite_paths(child, (*path, str(index)))
            rewritten_list.append(new_child)
            rewrites.extend(child_rewrites)
        return rewritten_list, rewrites
    if isinstance(value, str):
        new_value = rewrite_path_string(value)
        if new_value != value:
            rewrites.append((".".join(path), value, new_value))
        return new_value, rewrites
    return value, rewrites


def colab_name(windows_name: str) -> str:
    stem = Path(windows_name).stem
    return f"{stem}_colab.yaml"


def main() -> int:
    COLAB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_CONFIGS:
        src = WINDOWS_CONFIG_DIR / name
        dst = COLAB_CONFIG_DIR / colab_name(name)
        data = load_yaml(src)
        rewritten, rewrites = rewrite_paths(data)
        if not isinstance(rewritten, dict):
            raise TypeError(f"{src} did not rewrite to a mapping.")
        dump_yaml(dst, rewritten)
        load_yaml(dst)

        print(f"{src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
        if rewrites:
            for key_path, old, new in rewrites:
                print(f"  {key_path}: {old} -> {new}")
        else:
            print("  no path rewrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
