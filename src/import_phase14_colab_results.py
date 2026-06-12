from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from .phase14_common import COLAB_DOWNLOADS, PHASE14, PHASE14_IMPORTS, ensure_dir, row_from_output, write_json, write_md_table


def find_zip(download_dir: Path) -> Path:
    candidates = sorted(download_dir.glob("phase14_colab_outputs*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No phase14_colab_outputs*.zip found in {download_dir}")
    return candidates[0]


def locate_source(root: Path, stem: str) -> Path | None:
    candidates = [root / stem]
    candidates.extend(root.rglob(stem))
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def copy_import(src: Path, dst: Path) -> None:
    if dst.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = dst.with_name(f"{dst.name}_backup_{stamp}")
        shutil.move(str(dst), str(backup))
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Phase 14 Colab zip into E:/ outputs.")
    parser.add_argument("--download_dir", default=str(COLAB_DOWNLOADS))
    parser.add_argument("--target_dir", default=str(PHASE14))
    parser.add_argument("--zip_path", default="")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    target_dir = ensure_dir(Path(args.target_dir))
    zip_path = Path(args.zip_path) if args.zip_path else find_zip(download_dir)
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase14_colab_import_") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        for exp in PHASE14_IMPORTS:
            src = locate_source(tmpdir, exp["config"].stem)
            dst = target_dir / f"{exp['config'].stem.replace('_colab', '')}_colab_import"
            if src is None:
                rows.append({"method_id": exp["method_id"], "status": "missing_in_zip", "target": str(dst)})
                continue
            copy_import(src, dst)
            rows.append(row_from_output({**exp, "path": dst}))

    write_json(target_dir / "phase14_colab_import_manifest.json", {"zip_path": str(zip_path), "rows": rows})
    write_md_table(
        target_dir / "phase14_colab_import_manifest.md",
        rows,
        ["method_id", "status", "psnr", "ssim", "threshold_reached", "best_checkpoint_path", "checkpoint_sha256"],
    )
    print(f"Imported from {zip_path} into {target_dir}")


if __name__ == "__main__":
    main()
