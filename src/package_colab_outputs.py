from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .phase14_common import PHASE14_IMPORTS, ensure_dir, sha256_file, write_json


KEEP_NAMES = {
    "eval_metrics.json",
    "best_hq_metrics.json",
    "best_metrics.json",
    "best_mse_metrics.json",
    "best_psnr_metrics.json",
    "best_score_metrics.json",
    "best_hq.pt",
    "best_mse.pt",
    "best_psnr.pt",
    "best_score.pt",
    "last.pt",
    "resolved_config.yaml",
    "per_epoch_metrics.csv",
    "convergence_summary.md",
}


def iter_files(base_dir: Path) -> list[Path]:
    files: list[Path] = []
    for exp in PHASE14_IMPORTS:
        exp_dir = base_dir / exp["config"].stem
        if not exp_dir.exists():
            continue
        for path in exp_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.name in KEEP_NAMES or "eval_samples" in path.parts or "_summary" in path.parts:
                files.append(path)
    summary_dir = base_dir / "_summary"
    if summary_dir.exists():
        files.extend([p for p in summary_dir.rglob("*") if p.is_file()])
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Phase 14 Colab outputs into a zip.")
    parser.add_argument("--input_dir", default="/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase14_colab")
    parser.add_argument("--output_zip", default="/content/phase14_colab_outputs.zip")
    parser.add_argument("--manifest", default="/content/phase14_colab_outputs_manifest.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_zip = Path(args.output_zip)
    manifest_path = Path(args.manifest)
    ensure_dir(output_zip.parent)
    files = iter_files(input_dir)
    manifest = {
        "input_dir": str(input_dir),
        "output_zip": str(output_zip),
        "file_count": len(files),
        "files": [],
    }
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in files:
            rel = path.relative_to(input_dir)
            zf.write(path, rel.as_posix())
            manifest["files"].append(
                {"path": rel.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    manifest["zip_size_bytes"] = output_zip.stat().st_size
    manifest["zip_sha256"] = sha256_file(output_zip)
    write_json(manifest_path, manifest)
    print(json.dumps({"zip": str(output_zip), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
