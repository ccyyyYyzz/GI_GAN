from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MAX_HASH_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a lightweight JSON/CSV manifest for Colab or local run artifacts."
    )
    parser.add_argument("output_dir", help="Directory to walk.")
    parser.add_argument("--json-name", default="artifact_manifest.json")
    parser.add_argument("--csv-name", default="artifact_manifest.csv")
    parser.add_argument(
        "--max-sha256-bytes",
        type=int,
        default=DEFAULT_MAX_HASH_BYTES,
        help="Skip sha256 for files larger than this many bytes. Default: 256 MiB.",
    )
    return parser.parse_args()


def file_sha256(path: Path, max_bytes: int) -> str | None:
    size = path.stat().st_size
    if size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_for(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if suffix in {".yaml", ".yml", ".json"} and ("config" in name or "configs" in parts):
        return "config"
    if suffix in {".log", ".txt"} or "log" in name or "logs" in parts:
        return "log"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        return "figure"
    if suffix in {".csv", ".tsv", ".xlsx"}:
        return "table"
    if suffix in {".json", ".yaml", ".yml"}:
        return "metadata"
    if suffix in {".zip", ".tar", ".gz"}:
        return "archive"
    if suffix in {".py", ".ipynb", ".sh", ".ps1"}:
        return "code"
    if suffix in {".md", ".rst"}:
        return "report"
    return "other"


def iter_artifacts(root: Path, json_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in {json_path.resolve(), csv_path.resolve()}:
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "sha256": None,
                "category": category_for(path.relative_to(root)),
            }
        )
    return rows


def attach_hashes(root: Path, rows: list[dict[str, Any]], max_bytes: int) -> None:
    for row in rows:
        path = root / str(row["relative_path"])
        row["sha256"] = file_sha256(path, max_bytes)


def write_json(path: Path, root: Path, rows: list[dict[str, Any]], max_hash_bytes: int) -> None:
    payload = {
        "root": str(root),
        "generated_time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "max_sha256_bytes": max_hash_bytes,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["relative_path", "size_bytes", "modified_time_utc", "sha256", "category"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Output directory does not exist or is not a directory: {root}")

    json_path = root / args.json_name
    csv_path = root / args.csv_name
    rows = iter_artifacts(root, json_path, csv_path)
    attach_hashes(root, rows, int(args.max_sha256_bytes))
    write_json(json_path, root, rows, int(args.max_sha256_bytes))
    write_csv(csv_path, rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Collected {len(rows)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
