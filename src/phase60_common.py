from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .phase48_49_common import write_csv, write_markdown_table, write_sha256s
from .utils import ensure_dir, load_config, save_config, save_json


OUT_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase60_gan_sampling_mode_g2")
G1_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase59_gan_sampling_mode_g1")
G1_PILOT_ROOT = Path(
    "E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling"
)
MEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/scrambled_hadamard5_hq_noise001_colab")
MEAN_CONFIG = MEAN_ROOT / "resolved_config.yaml"
MEAN_CHECKPOINT = MEAN_ROOT / "last.pt"
MEAN_METRICS = MEAN_ROOT / "eval_metrics.json"
G1_SCR5_ROOT = G1_PILOT_ROOT / "scr5"
G1_SOURCE_CHECKPOINT = G1_SCR5_ROOT / "source_checkpoint.pt"
G1_CONFIG = G1_SCR5_ROOT / "config_used.yaml"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def maybe_sha256(path: Path) -> str:
    return file_sha256(path) if path.exists() else ""


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[Any]) -> float:
    xs = [to_float(v) for v in values]
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    v = to_float(value)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_config(child, child_prefix))
        return out
    return {prefix: value}


def load_config_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_config(path)
    return data if isinstance(data, dict) else {}


def find_individual_stochastic_samples(root: Path) -> list[Path]:
    if not root.exists():
        return []
    patterns = ["*sample*.pt", "*samples*.pt", "*stochastic*.pt", "*sample*.npy", "*sample*.npz"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    excluded = {"source_checkpoint.pt", "Q_exact_null.pt", "measurement_operator_exact.pt"}
    return sorted(p for p in set(found) if p.name not in excluded)


def looks_like_data_split_manifest(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    if '"parts"' in text or '"zip"' in text or "chunk_mb" in text:
        return False
    has_train = "train" in text
    has_eval = "val" in text or "test" in text
    has_hash = "hash" in text or "sha256" in text or "indices" in text or "ids" in text
    return has_train and has_eval and has_hash


def find_data_split_hash_files(*roots: Path) -> tuple[list[Path], list[Path]]:
    candidates: list[Path] = []
    ignored_transfer_manifests: list[Path] = []
    names = ["*split*.json", "*split*.yaml", "*split*.yml", "*indices*.json", "*hash*.json"]
    for root in roots:
        if not root.exists():
            continue
        for pattern in names:
            for path in root.rglob(pattern):
                if not path.is_file():
                    continue
                if path.suffix.lower() == ".json" and looks_like_data_split_manifest(path):
                    candidates.append(path)
                elif "split_manifest" in path.name.lower() or "manifest" in path.name.lower():
                    ignored_transfer_manifests.append(path)
    return sorted(set(candidates)), sorted(set(ignored_transfer_manifests))


def save_placeholder_figure(path: Path, title: str, body: str, size: tuple[int, int] = (1200, 760)) -> None:
    ensure_dir(path.parent)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, size[0] - 20, size[1] - 20), outline=(40, 40, 40), width=2)
    draw.text((50, 60), title, fill=(0, 0, 0))
    y = 120
    for line in body.splitlines():
        draw.text((50, y), line, fill=(30, 30, 30))
        y += 34
    image.save(path)


def save_pdf_from_png(png: Path, pdf: Path) -> None:
    if not png.exists():
        return
    try:
        image = Image.open(png).convert("RGB")
        image.save(pdf, "PDF", resolution=120.0)
    except Exception:
        return


def write_rows(path_prefix: Path, rows: list[dict[str, Any]], title: str) -> None:
    write_csv(path_prefix.with_suffix(".csv"), rows)
    write_markdown_table(path_prefix.with_suffix(".md"), rows, title)


__all__ = [
    "OUT_ROOT",
    "G1_ROOT",
    "G1_PILOT_ROOT",
    "MEAN_ROOT",
    "MEAN_CONFIG",
    "MEAN_CHECKPOINT",
    "MEAN_METRICS",
    "G1_SCR5_ROOT",
    "G1_SOURCE_CHECKPOINT",
    "G1_CONFIG",
    "ensure_dir",
    "save_json",
    "save_config",
    "write_sha256s",
    "read_json",
    "read_csv_rows",
    "maybe_sha256",
    "to_float",
    "mean",
    "fmt",
    "flatten_config",
    "load_config_or_empty",
    "find_individual_stochastic_samples",
    "find_data_split_hash_files",
    "save_placeholder_figure",
    "save_pdf_from_png",
    "write_rows",
]
