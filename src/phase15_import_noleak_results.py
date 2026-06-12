from __future__ import annotations

import json
import shutil
from pathlib import Path

from .phase15_common import (
    IMPORTED_NOLEAK,
    METHODS,
    PHASE15,
    backup_existing,
    copy_if_exists,
    ensure_dir,
    find_large_zip,
    generated_sha_manifest,
    method_import_dir,
    method_source_dir,
    unzip_to,
    write_csv,
    write_json,
    write_md_table,
)


SUMMARY_FIELDS = [
    "method_id",
    "source_dir",
    "import_dir",
    "zip_path",
    "zip_size_mb",
    "files_imported",
    "backup_dir",
    "status",
    "notes",
]


def import_one(method: dict) -> dict:
    source_dir = method_source_dir(method)
    import_dir = method_import_dir(method)
    backup_dir = backup_existing(import_dir)
    ensure_dir(import_dir)
    zip_path = find_large_zip(source_dir)
    if zip_path is None:
        return {
            "method_id": method["method_id"],
            "source_dir": str(source_dir),
            "import_dir": str(import_dir),
            "zip_path": "",
            "zip_size_mb": "",
            "files_imported": 0,
            "backup_dir": str(backup_dir) if backup_dir else "",
            "status": "missing_zip",
            "notes": "No >100MB zip found in no-leak download directory.",
        }
    unzip_to(zip_path, import_dir)
    for sidecar in source_dir.glob("*.json"):
        copy_if_exists(sidecar, import_dir / sidecar.name)
    for sidecar in source_dir.glob("*_recon_grid.png"):
        copy_if_exists(sidecar, import_dir / sidecar.name)
    copy_if_exists(source_dir / "measurement_operator_exact.pt", import_dir / "measurement_operator_exact.pt")
    copy_if_exists(
        source_dir / "measurement_operator_exact_manifest.json",
        import_dir / "measurement_operator_exact_manifest.json",
    )
    manifest = generated_sha_manifest(import_dir)
    write_json(import_dir / "sha256_manifest.json", manifest)
    return {
        "method_id": method["method_id"],
        "source_dir": str(source_dir),
        "import_dir": str(import_dir),
        "zip_path": str(zip_path),
        "zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 3),
        "files_imported": manifest["file_count"],
        "backup_dir": str(backup_dir) if backup_dir else "",
        "status": "imported",
        "notes": "Copied/extracted only into outputs_phase15; source archive untouched.",
    }


def main() -> None:
    ensure_dir(IMPORTED_NOLEAK)
    rows = [import_one(method) for method in METHODS]
    out_dir = ensure_dir(PHASE15)
    write_csv(out_dir / "import_noleak_summary.csv", rows, SUMMARY_FIELDS)
    write_md_table(out_dir / "import_noleak_summary.md", rows, SUMMARY_FIELDS)
    write_json(out_dir / "import_noleak_summary.json", rows)
    print(json.dumps({"imported": len(rows), "output_dir": str(IMPORTED_NOLEAK)}, indent=2))


if __name__ == "__main__":
    main()
