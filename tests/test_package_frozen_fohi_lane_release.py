import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from package_frozen_fohi_lane_release import package


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    inventory = write(tmp_path / "inventory.json", '{"lane_index": 0}')
    artifact = write(tmp_path / "assets" / "model.pt", "weights")
    lane = tmp_path / "results" / "lane0"
    write(lane / "preflight_receipt.json", "{}")
    write(lane / "HELDOUT_ONCE_STARTED.json", "{}")
    write(lane / "driver.log", "driver log")
    write(lane / "launch_receipt.txt", "launch receipt")
    rates = {}
    for rate in ("05", "10"):
        summary = write(lane / f"rate{rate}/fohi/summary.json", "{}")
        vectors = write(lane / f"rate{rate}/fohi/metric_vectors.npz", "vectors")
        cache_manifest = write(lane / f"rate{rate}/cache/test_cache_manifest.json", "{}")
        rates[rate] = {"summary": str(summary), "summary_sha256": digest(summary), "metric_vectors": str(vectors), "metric_vectors_sha256": digest(vectors), "test_cache_manifest": str(cache_manifest), "test_cache_manifest_sha256": digest(cache_manifest)}
        write(lane / f"rate{rate}/cache.log", "cache log")
        write(lane / f"rate{rate}/fohi.log", "fohi log")
        write(lane / f"rate{rate}/cache/test_cache.pt", "do not package")
    write(lane / "HELDOUT_ONCE_COMPLETE.json", json.dumps({"status": "VQGAN_GUIDED_FOHI_HELDOUT_LANE_COMPLETE", "lane_index": 0, "test_split_opened": True, "rates": rates}))
    code = write(tmp_path / "frozen_code.py", "print('frozen')\n")
    rate_archive = tmp_path / "gan_rate_bundle.zip"
    with zipfile.ZipFile(rate_archive, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"seed": 0}))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"status": "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN", "inventory_sha256": {"E:/remote/lane0.json": digest(inventory)}, "code_sha256": {"/content/GI_GAN/frozen_code.py": digest(code)}, "lanes": {"0": {"artifact_sha256": {str(artifact): digest(artifact)}}}}), encoding="utf-8")
    return freeze, lane.parent, tmp_path / "release", inventory, rate_archive


def test_package_success_and_excludes_test_cache(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    artifact = tmp_path / "assets/model.pt"
    second = tmp_path / "duplicate/model.pt"
    write(second, "weights")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["lanes"]["0"]["artifact_sha256"] = {
        str(artifact): digest(artifact), str(second): digest(second)
    }
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    archive = package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)
    release = output / "lane0_frozen_fohi_release"
    manifest = json.loads((release / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    paths = {item["relative_path"] for item in manifest["files"]}
    assert archive.is_file()
    assert archive.with_suffix(".gz.sha256").is_file()
    assert (release / "environment/environment.json").is_file()
    assert (release / "environment/pip_freeze.txt").is_file()
    assert (release / "code/frozen/frozen_code.py").read_text(encoding="utf-8") == "print('frozen')\n"
    assert (release / "runtime/gan_rate_bundle.zip").is_file()
    artifact_entries = [item for item in manifest["files"] if item["category"] == "artifact"]
    assert len(artifact_entries) == 2
    assert len({item["relative_path"] for item in artifact_entries}) == 2
    assert (release / "tools/materialize_content_layout.py").is_file()
    assert (release / "SHA256SUMS").is_file()
    content_root = tmp_path / "portable_content"
    subprocess.run(
        ["python", str(release / "tools/materialize_content_layout.py"), "--release-root", str(release), "--content-root", str(content_root)],
        check=True,
    )
    assert (content_root / "GI_GAN/frozen_code.py").is_file()
    assert (content_root / "gan_rate_bundle.zip").is_file()
    inventory_entries = [item for item in manifest["files"] if item["category"] == "inventory"]
    assert inventory_entries == [next(item for item in manifest["files"] if item["relative_path"] == "inventory/lane0_inventory.json")]
    assert inventory_entries[0]["source_path"] == str(inventory)
    assert "results/rate05/cache/test_cache.pt" not in paths
    assert not (release / "results/rate05/cache/test_cache.pt").exists()
    with tarfile.open(archive) as bundled:
        assert all(not name.endswith("test_cache.pt") for name in bundled.getnames())


def test_package_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    key = next(iter(payload["lanes"]["0"]["artifact_sha256"]))
    payload["lanes"]["0"]["artifact_sha256"][key] = "0" * 64
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)


def test_package_requires_complete_marker(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    (result_root / "lane0" / "HELDOUT_ONCE_COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError, match="MISSING_FILE:COMPLETE"):
        package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)


def test_package_rejects_result_receipt_hash_mismatch(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    write(result_root / "lane0/rate05/fohi/summary.json", "changed")
    with pytest.raises(RuntimeError, match="RESULT_RECEIPT_HASH_MISMATCH"):
        package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)


def test_package_rejects_code_hash_drift(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    write(tmp_path / "frozen_code.py", "changed\n")
    with pytest.raises(RuntimeError, match="HASH_MISMATCH:frozen_code"):
        package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)


def test_package_rejects_inventory_from_other_lane(tmp_path: Path) -> None:
    freeze, result_root, output, inventory, rate_archive = fixture(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["inventory_sha256"] = {"E:/remote/lane1.json": digest(inventory)}
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN_INVENTORY_LANE_NOT_UNIQUE"):
        package(freeze_manifest=freeze, result_root=result_root, output_dir=output, lane_index=0, inventory=inventory, repo_root=tmp_path, rate_archive=rate_archive)
