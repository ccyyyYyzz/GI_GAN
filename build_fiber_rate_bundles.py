from __future__ import annotations

import argparse
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


def source_files(source_root: Path, seed: int) -> dict[str, Path]:
    base = source_root / "outputs/compatibility/measurement_conditioned_vqgan"
    prior = base / f"prior_multiseed_hashclean_seed{seed}"
    result = {
        "config_rate02.yaml": base / f"anchor_rate02_seed{seed}_local/config_used.yaml",
        "config_rate10.yaml": base / f"anchor_rate10_seed{seed}_local/config_used.yaml",
        "priors/vqae.pt": prior / "vqae_continuation/checkpoints/vqae_continuation_best_by_lpips.pt",
        "priors/vqgan.pt": prior / "vqgan_continuation/checkpoints/vqgan_continuation_best_by_lpips.pt",
    }
    for rate in ("02", "10"):
        experiment = base / f"anchor_rate{rate}_seed{seed}_local/runs/seed{seed}"
        result[f"rate{rate}/vqae_refiner.pt"] = (
            experiment / "vqae_refiner/checkpoints/vqae_refiner_best_by_val_lpips.pt"
        )
        result[f"rate{rate}/vqgan_refiner.pt"] = (
            experiment / "vqgan_refiner/checkpoints/vqgan_refiner_best_by_val_lpips.pt"
        )
    missing = [f"{relative}:{path}" for relative, path in result.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("RATE_BUNDLE_INPUTS_MISSING:" + ",".join(missing))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        files = source_files(args.source_root, seed)
        manifest = {
            "status": "FIBER_RATE_INPUT_BUNDLE",
            "seed": seed,
            "rates": ["02", "10"],
            "development_source": "STL10 train+unlabeled",
            "test_split_included": False,
            "files": {relative: sha256(path) for relative, path in sorted(files.items())},
            "source_paths": {
                relative: str(path.relative_to(args.source_root)).replace("\\", "/")
                for relative, path in sorted(files.items())
            },
        }
        output = args.output_dir / f"gan_rate_seed{seed}.zip"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
            for relative, path in sorted(files.items()):
                archive.write(path, arcname=relative)
        print(f"WROTE {output} {output.stat().st_size} {sha256(output)}")


if __name__ == "__main__":
    main()
