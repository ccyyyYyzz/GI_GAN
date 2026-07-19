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


with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
root = Path(f"/content/gan_r59_raw_fiber/lane{lane}")
complete = root / "ROUND59_COMPLETE.json"
if not complete.is_file():
    raise RuntimeError(f"ROUND59_NOT_COMPLETE:{lane}")
receipt = json.loads(complete.read_text(encoding="utf-8"))
if receipt.get("status") != "ROUND59_RAW_FIBER_LANE_COMPLETE":
    raise RuntimeError(f"ROUND59_BAD_COMPLETE_STATUS:{lane}")
archive_path = Path(f"/content/round59_raw_fiber_lane{lane}.zip")
if archive_path.exists():
    archive_path.unlink()
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "reused_cache_manifests" in path.relative_to(root).parts:
            continue
        output.write(path, path.relative_to(root.parent))
    reused_caches = receipt.get("reused_caches")
    if not isinstance(reused_caches, dict):
        raise RuntimeError(f"ROUND59_REUSED_CACHE_RECEIPT_MISSING:{lane}")
    for rate in ("05", "10"):
        cache = reused_caches.get(rate)
        if not isinstance(cache, dict):
            raise RuntimeError(f"ROUND59_REUSED_CACHE_RATE_MISSING:{lane}:{rate}")
        manifest = Path(str(cache.get("cache_manifest", "")))
        expected_sha256 = cache.get("cache_manifest_sha256")
        if not manifest.is_file():
            raise RuntimeError(f"ROUND59_REUSED_CACHE_MANIFEST_MISSING:{lane}:{rate}:{manifest}")
        actual_sha256 = sha256(manifest)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"ROUND59_REUSED_CACHE_MANIFEST_HASH_MISMATCH:{lane}:{rate}:"
                f"{actual_sha256}!={expected_sha256}"
            )
        portable_name = (
            Path(root.name)
            / "reused_cache_manifests"
            / f"rate{rate}"
            / "test_cache_manifest.json"
        )
        output.write(manifest, portable_name)
digest = sha256(archive_path)
sidecar = archive_path.with_suffix(".zip.sha256")
sidecar.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
print(json.dumps({"lane": lane, "archive": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": digest}, sort_keys=True))
