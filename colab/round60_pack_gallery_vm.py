from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path("/content/gan_r60_qualitative_gallery/lane0")
provenance_path = root / "provenance.json"
provenance_sidecar = root / "provenance.json.sha256"
if not provenance_path.is_file() or not provenance_sidecar.is_file():
    raise RuntimeError("ROUND60_GALLERY_NOT_COMPLETE")

provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if provenance.get("status") != "FROZEN_FOHI_HELDOUT_QUALITATIVE_GALLERY_COMPLETE":
    raise RuntimeError(f"ROUND60_GALLERY_BAD_STATUS:{provenance.get('status')!r}")
terminal_projection = provenance.get("terminal_projection")
if not isinstance(terminal_projection, dict):
    raise RuntimeError("ROUND60_TERMINAL_PROJECTION_RECEIPT_MISSING")
if terminal_projection.get("target") != "geometry.intrinsic_record(raw cached y)":
    raise RuntimeError(
        f"ROUND60_TERMINAL_TARGET_NOT_RAW_Y:{terminal_projection.get('target')!r}"
    )
metric_source = str(terminal_projection.get("metric_source", ""))
if "Round59 raw-y metric vectors" not in metric_source or "No Round56 clipped-anchor vectors" not in metric_source:
    raise RuntimeError("ROUND60_METRIC_SOURCE_NOT_RAW_Y_ONLY")
if provenance.get("no_post_test_tuning") is not True:
    raise RuntimeError("ROUND60_NO_POST_TEST_TUNING_RECEIPT_MISSING")
expected_provenance_sha256 = provenance_sidecar.read_text(encoding="ascii").split()[0]
actual_provenance_sha256 = sha256(provenance_path)
if actual_provenance_sha256 != expected_provenance_sha256:
    raise RuntimeError(
        "ROUND60_PROVENANCE_HASH_MISMATCH:"
        f"{actual_provenance_sha256}!={expected_provenance_sha256}"
    )

declared_outputs = provenance.get("output_sha256_excluding_provenance")
if not isinstance(declared_outputs, dict) or not declared_outputs:
    raise RuntimeError("ROUND60_OUTPUT_HASH_RECEIPT_MISSING")
for relative_name, expected_sha256 in sorted(declared_outputs.items()):
    path = root / relative_name
    if not path.is_file():
        raise RuntimeError(f"ROUND60_DECLARED_OUTPUT_MISSING:{relative_name}")
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"ROUND60_DECLARED_OUTPUT_HASH_MISMATCH:{relative_name}:"
            f"{actual_sha256}!={expected_sha256}"
        )

archive_path = Path("/content/round60_raw_y_qualitative_gallery_lane0.zip")
if archive_path.exists():
    archive_path.unlink()
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        archive.write(path, path.relative_to(root.parent))

archive_sha256 = sha256(archive_path)
archive_sidecar = archive_path.with_suffix(".zip.sha256")
archive_sidecar.write_text(
    f"{archive_sha256}  {archive_path.name}\n", encoding="ascii"
)
print(
    json.dumps(
        {
            "status": "ROUND60_GALLERY_PACKED",
            "archive": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
            "files": len([item for item in root.rglob("*") if item.is_file()]),
        },
        indent=2,
        sort_keys=True,
    )
)
