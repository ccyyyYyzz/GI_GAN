from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_sorted_ints(values: list[int]) -> str:
    arr = np.array(sorted(int(v) for v in values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _flatten_json(obj: Any) -> list[int]:
    if isinstance(obj, dict):
        vals: list[int] = []
        for value in obj.values():
            vals.extend(_flatten_json(value))
        return vals
    if isinstance(obj, list):
        vals = []
        for value in obj:
            vals.extend(_flatten_json(value))
        return vals
    if isinstance(obj, (int, np.integer)):
        return [int(obj)]
    return []


def load_indices(path: Path, column: str | None = None) -> list[int]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _flatten_json(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return []
        key = column or ("image_id" if "image_id" in rows[0] else next(iter(rows[0].keys())))
        return [int(row[key]) for row in rows if row.get(key, "") != ""]
    if suffix == ".npy":
        return [int(x) for x in np.load(path).reshape(-1).tolist()]
    if suffix == ".npz":
        data = np.load(path)
        vals: list[int] = []
        for key in sorted(data.files):
            vals.extend(int(x) for x in data[key].reshape(-1).tolist())
        return vals
    raise ValueError(f"Unsupported split file type: {path}")


def summarize(path: Path, column: str | None = None) -> dict[str, Any]:
    values = load_indices(path, column=column)
    return {
        "path": str(path),
        "count": len(values),
        "unique_count": len(set(values)),
        "sha256_sorted_int64": sha256_sorted_ints(values) if values else "",
        "min": min(values) if values else "",
        "max": max(values) if values else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute split-index hashes for provenance.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--column", default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rows = [summarize(Path(p), args.column) for p in args.paths]
    payload = {"splits": rows}
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
