"""Package one completed frozen-FOHI held-out lane for portable release.

The source result tree is treated as immutable: this command only reads it and
writes a self-describing release directory plus a ``tar.gz`` under output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any


COMPLETE_STATUS = "VQGAN_GUIDED_FOHI_HELDOUT_LANE_COMPLETE"
FREEZE_STATUS = "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def regular_file(path: Path, *, label: str) -> Path:
    """Reject missing files and symlinks before reading/copying them."""
    if path.is_symlink():
        raise RuntimeError(f"SYMLINK_REJECTED:{label}:{path}")
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_FILE:{label}:{path}")
    return path


def lane_index_from_archive(path: Path) -> int:
    regular_file(path, label="RATE_ARCHIVE")
    with zipfile.ZipFile(path) as archive:
        value = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
    if value not in (0, 1, 2):
        raise RuntimeError(f"INVALID_LANE_INDEX:{value}")
    return value


def lane_root(result_root: Path, lane_index: int) -> Path:
    """Accept either a lane directory itself or its parent directory."""
    direct = result_root / "HELDOUT_ONCE_COMPLETE.json"
    candidate = result_root if direct.exists() else result_root / f"lane{lane_index}"
    if candidate.is_symlink():
        raise RuntimeError(f"SYMLINK_REJECTED:RESULT_ROOT:{candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"MISSING_LANE_RESULT_ROOT:{candidate}")
    return candidate


def copy_entry(
    source: Path,
    relative: Path,
    category: str,
    destination_root: Path,
    entries: list[dict[str, Any]],
    expected_sha256: str | None = None,
    source_identity: str | None = None,
) -> None:
    source = regular_file(source, label=category)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"PATH_TRAVERSAL_REJECTED:{relative}")
    actual = sha256(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"HASH_MISMATCH:{category}:{source}:{actual}:{expected_sha256}")
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"DUPLICATE_RELEASE_PATH:{relative}")
    shutil.copyfile(source, target, follow_symlinks=False)
    entries.append(
        {
            "source_path": source_identity or str(source),
            "relative_path": relative.as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": actual,
            "category": category,
        }
    )


def generated_entry(path: Path, relative: Path, category: str, entries: list[dict[str, Any]]) -> None:
    entries.append({"source_path": "generated:" + relative.as_posix(), "relative_path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path), "category": category})


def frozen_inventory_hash(freeze: dict[str, Any], lane_index: int) -> str:
    """Return the lane-specific inventory digest, rejecting ambiguous manifests."""
    matches = [
        digest
        for raw_path, digest in freeze.get("inventory_sha256", {}).items()
        if PureWindowsPath(raw_path).stem.lower() == f"lane{lane_index}"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"FROZEN_INVENTORY_LANE_NOT_UNIQUE:{lane_index}:{len(matches)}")
    return matches[0]


def frozen_code_paths(freeze: dict[str, Any], repo_root: Path) -> list[tuple[Path, Path, str]]:
    """Map the frozen /content/GI_GAN code contract onto a checkout safely."""
    result: list[tuple[Path, Path, str]] = []
    prefix = "/content/GI_GAN/"
    for raw_path, expected in freeze.get("code_sha256", {}).items():
        if not raw_path.startswith(prefix):
            raise RuntimeError(f"UNRELOCATABLE_FROZEN_CODE_PATH:{raw_path}")
        relative = Path(raw_path.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"PATH_TRAVERSAL_REJECTED:{relative}")
        result.append((repo_root / relative, relative, expected))
    if not result:
        raise RuntimeError("FREEZE_MANIFEST_HAS_NO_CODE_HASHES")
    return result


def content_relative_path(raw_path: str, *, label: str) -> Path:
    """Return a safe path below ``/content`` while preserving source identity."""
    prefix = "/content/"
    if not raw_path.startswith(prefix):
        raise RuntimeError(f"UNRELOCATABLE_{label}_PATH:{raw_path}")
    relative = Path(raw_path.removeprefix(prefix))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"PATH_TRAVERSAL_REJECTED:{relative}")
    return relative


def write_release_readme(staging: Path, entries: list[dict[str, Any]]) -> None:
    """Write a lane-local README; it is intentionally usable without /content."""
    readme = staging / "README.md"
    readme.write_text(
        "# Frozen FOHI held-out lane release\n\n"
        "This archive is a byte-addressed record of one completed held-out lane. "
        "`RELEASE_MANIFEST.json` lists every copied file; `SHA256SUMS` verifies the "
        "release directory and the adjacent `.tar.gz.sha256` verifies the archive.\n\n"
        "The 450 MB `test_cache.pt` is deliberately excluded because it is deterministically "
        "regenerated from the included frozen weights, configurations, and the STL-10 test data. "
        "The included cache manifest records the selected raw-hash-disjoint test samples.\n\n"
        "## Local result aggregation\n\n"
        "Extract all three lane archives, then run the checked-out `aggregate_frozen_fohi_heldout.py` "
        "against their `results/` trees. The aggregate consumes only `summary.json`, "
        "`metric_vectors.npz`, and `test_cache_manifest.json`; it does not require `/content`, GPU, "
        "or `test_cache.pt`.\n\n"
        "## Re-running the held-out lane\n\n"
        "The original one-shot driver uses Colab's `/content` layout. To rerun it locally, create a "
        "container or VM whose `/content` is a bind mount of a fresh runtime directory, then materialize "
        "the archived `artifacts/` and `code/frozen/` paths there with `tools/materialize_content_layout.py`. "
        "This keeps the frozen source bytes unchanged while making the host path arbitrary. Supply an "
        "unseen STL-10 test dataset at `/content/datasets`, restore the lane's original `gan_rate_bundle.zip`, "
        "and invoke `code/frozen/run_frozen_fohi_heldout_once.py` once. Do not rerun on the already opened "
        "test split for scientific confirmation; this is a computational reproducibility check.\n",
        encoding="utf-8",
    )
    generated_entry(readme, Path("README.md"), "documentation", entries)


def write_materializer(staging: Path, entries: list[dict[str, Any]]) -> None:
    """Add a transparent tool for rebuilding the /content layout inside a container."""
    tool = staging / "tools" / "materialize_content_layout.py"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(
        "\"\"\"Materialize this release under a chosen container /content mount.\"\"\"\n"
        "from __future__ import annotations\n"
        "import argparse, json, shutil\nfrom pathlib import Path\n\n"
        "p=argparse.ArgumentParser()\np.add_argument('--release-root', type=Path, required=True)\n"
        "p.add_argument('--content-root', type=Path, required=True)\na=p.parse_args()\n"
        "root=a.release_root.resolve(); manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())\n"
        "for item in manifest['files']:\n"
        "    source_path=item['source_path']\n"
        "    if not source_path.startswith('/content/'):\n"
        "        continue\n"
        "    source=root/item['relative_path']; target=a.content_root/source_path.removeprefix('/content/')\n"
        "    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)\n"
        "print(a.content_root)\n",
        encoding="utf-8",
    )
    generated_entry(tool, Path("tools/materialize_content_layout.py"), "reproduction_tool", entries)


def environment_payload() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            from importlib.metadata import version as installed_version
            return installed_version(name)
        except Exception:
            return None

    torch_info: dict[str, Any] = {"version": version("torch"), "cuda_available": None, "cuda_version": None, "cudnn_version": None, "gpu": None}
    try:
        import torch
        torch_info.update({"cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version()})
        if torch.cuda.is_available():
            torch_info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    repo = Path(__file__).resolve().parent
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"python": sys.version, "platform": platform.platform(), "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"), "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "scipy", "lpips")}, "torch": torch_info}


def write_environment(staging: Path, entries: list[dict[str, Any]]) -> None:
    environment = staging / "environment"
    environment.mkdir()
    environment_json = environment / "environment.json"
    environment_json.write_text(json.dumps(environment_payload(), indent=2, sort_keys=True), encoding="utf-8")
    generated_entry(environment_json, Path("environment/environment.json"), "environment", entries)
    pip_freeze = environment / "pip_freeze.txt"
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
        pip_freeze.write_text(result.stdout, encoding="utf-8")
    except Exception as error:
        pip_freeze.write_text(f"pip freeze unavailable: {error}\n", encoding="utf-8")
    generated_entry(pip_freeze, Path("environment/pip_freeze.txt"), "environment", entries)


def package(
    *, freeze_manifest: Path, result_root: Path, output_dir: Path, lane_index: int,
    inventory: Path | None = None, repo_root: Path | None = None, rate_archive: Path | None = None,
) -> Path:
    freeze_manifest = regular_file(freeze_manifest, label="FREEZE_MANIFEST")
    freeze = read_json(freeze_manifest)
    if freeze.get("status") != FREEZE_STATUS:
        raise RuntimeError("INVALID_FREEZE_MANIFEST_STATUS")
    lane = freeze.get("lanes", {}).get(str(lane_index))
    if not isinstance(lane, dict):
        raise RuntimeError(f"UNFROZEN_LANE:{lane_index}")
    source_lane = lane_root(result_root, lane_index)
    complete_path = regular_file(source_lane / "HELDOUT_ONCE_COMPLETE.json", label="COMPLETE")
    complete = read_json(complete_path)
    if complete.get("status") != COMPLETE_STATUS:
        raise RuntimeError("INVALID_OR_INCOMPLETE_LANE_STATUS")
    if complete.get("lane_index") != lane_index or complete.get("test_split_opened") is not True:
        raise RuntimeError("COMPLETE_RECEIPT_LANE_OR_SCOPE_MISMATCH")
    rate_files = {}
    for rate in ("05", "10"):
        receipt = complete.get("rates", {}).get(rate)
        if not isinstance(receipt, dict):
            raise RuntimeError(f"MISSING_RATE_RECEIPT:{rate}")
        rate_files[rate] = {
            "summary": source_lane / f"rate{rate}/fohi/summary.json",
            "metric_vectors": source_lane / f"rate{rate}/fohi/metric_vectors.npz",
            "test_cache_manifest": source_lane / f"rate{rate}/cache/test_cache_manifest.json",
        }
        for field, path in rate_files[rate].items():
            expected_path = str(path)
            if receipt.get(field) != expected_path:
                raise RuntimeError(f"RESULT_RECEIPT_PATH_MISMATCH:{rate}:{field}")
            actual = sha256(regular_file(path, label=field))
            if actual != receipt.get(f"{field}_sha256"):
                raise RuntimeError(f"RESULT_RECEIPT_HASH_MISMATCH:{rate}:{field}")
    inventory = inventory or Path(f"/content/gan_r54_heldout_inventory_lane{lane_index}.json")
    inventory = regular_file(inventory, label="inventory")
    inventory_payload = read_json(inventory)
    if inventory_payload.get("lane_index") != lane_index:
        raise RuntimeError("INVENTORY_LANE_MISMATCH")
    inventory_hash = sha256(inventory)
    if inventory_hash != frozen_inventory_hash(freeze, lane_index):
        raise RuntimeError("INVENTORY_NOT_FROZEN_FOR_THIS_LANE")
    repo_root = (repo_root or Path(__file__).resolve().parent).resolve()
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise RuntimeError(f"INVALID_REPO_ROOT:{repo_root}")
    frozen_code = frozen_code_paths(freeze, repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise RuntimeError(f"SYMLINK_REJECTED:OUTPUT_DIR:{output_dir}")
    release_name = f"lane{lane_index}_frozen_fohi_release"
    release_dir = output_dir / release_name
    archive_path = output_dir / f"{release_name}.tar.gz"
    if release_dir.exists() or archive_path.exists():
        raise FileExistsError(f"RELEASE_OUTPUT_ALREADY_EXISTS:{release_dir}")

    with tempfile.TemporaryDirectory(prefix="fohi_release_", dir=output_dir) as temporary:
        staging = Path(temporary) / release_name
        staging.mkdir()
        entries: list[dict[str, Any]] = []
        copy_entry(freeze_manifest, Path("freeze") / "heldout_freeze.json", "freeze_manifest", staging, entries)
        copy_entry(inventory, Path("inventory") / f"lane{lane_index}_inventory.json", "inventory", staging, entries, inventory_hash)
        for source, relative, expected in frozen_code:
            copy_entry(source, Path("code") / "frozen" / relative, "frozen_code", staging, entries, expected, f"/content/GI_GAN/{relative.as_posix()}")
        rate_archive = rate_archive or Path("/content/gan_rate_bundle.zip")
        if lane_index_from_archive(rate_archive) != lane_index:
            raise RuntimeError("RATE_ARCHIVE_LANE_MISMATCH")
        copy_entry(rate_archive, Path("runtime") / "gan_rate_bundle.zip", "lane_dispatch_archive", staging, entries, source_identity="/content/gan_rate_bundle.zip")
        for source_text, expected in lane.get("artifact_sha256", {}).items():
            source = regular_file(Path(source_text), label="artifact")
            # Preserve the full path below /content.  Two frozen locations may
            # intentionally contain identical bytes and identical basenames;
            # both locations must survive packaging and later materialization.
            if source_text.startswith("/content/"):
                relative = content_relative_path(source_text, label="ARTIFACT")
            else:
                # Unit fixtures and non-Colab archival audits may use native
                # absolute paths.  Keep them collision-free without claiming
                # that the materializer can restore a non-/content identity.
                identity = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]
                relative = Path("external") / identity / source.name
            copy_entry(
                source,
                Path("artifacts") / relative,
                "artifact",
                staging,
                entries,
                expected,
                source_identity=source_text,
            )
        for name in ("preflight_receipt.json", "HELDOUT_ONCE_STARTED.json", "HELDOUT_ONCE_COMPLETE.json", "driver.log", "launch_receipt.txt"):
            copy_entry(source_lane / name, Path("receipts") / name, "receipt", staging, entries)
        for rate in ("05", "10"):
            for relative, category in (
                (Path(f"rate{rate}/fohi/summary.json"), "summary"),
                (Path(f"rate{rate}/fohi/metric_vectors.npz"), "metric_vectors"),
                (Path(f"rate{rate}/cache/test_cache_manifest.json"), "cache_manifest"),
                (Path(f"rate{rate}/cache.log"), "log"),
                (Path(f"rate{rate}/fohi.log"), "log"),
            ):
                copy_entry(source_lane / relative, Path("results") / relative, category, staging, entries)
        write_environment(staging, entries)
        write_release_readme(staging, entries)
        write_materializer(staging, entries)
        manifest = {
            "status": "FROZEN_FOHI_LANE_RELEASE_COMPLETE",
            "lane_index": lane_index,
            "source_result_root": str(source_lane),
            "excluded": ["rate05/cache/test_cache.pt", "rate10/cache/test_cache.pt"],
            "files": sorted(entries, key=lambda item: item["relative_path"]),
        }
        (staging / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        sums = staging / "SHA256SUMS"
        sum_rows = []
        for file_path in sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS"):
            sum_rows.append(f"{sha256(file_path)}  {file_path.relative_to(staging).as_posix()}")
        sums.write_text("\n".join(sum_rows) + "\n", encoding="ascii")
        shutil.move(str(staging), str(release_dir))
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        archive.add(release_dir, arcname=release_name, recursive=True)
    archive_sha = sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{archive_sha}  {archive_path.name}\n", encoding="ascii"
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lane-index", type=int, choices=(0, 1, 2))
    parser.add_argument("--rate-archive", type=Path, default=Path("/content/gan_rate_bundle.zip"))
    parser.add_argument("--inventory", type=Path, help="Current lane's Round54 inventory JSON; defaults to /content/gan_r54_heldout_inventory_lane{lane}.json")
    parser.add_argument("--repo-root", type=Path, help="Checkout containing the frozen code; defaults to this script's directory")
    args = parser.parse_args()
    lane_index = args.lane_index if args.lane_index is not None else lane_index_from_archive(args.rate_archive)
    archive = package(
        freeze_manifest=args.freeze_manifest,
        result_root=args.result_root,
        output_dir=args.output_dir,
        lane_index=lane_index,
        inventory=args.inventory,
        repo_root=args.repo_root,
        rate_archive=args.rate_archive,
    )
    print(archive)


if __name__ == "__main__":
    main()
