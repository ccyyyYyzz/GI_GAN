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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.source_root / "outputs/compatibility/measurement_conditioned_vqgan"
    anchor = base / "anchor_multiseed_hashclean_seed0"
    prior = base / "prior_multiseed_hashclean_seed0"
    files = {
        "config_base.yaml": anchor / "config_used.yaml",
        "priors/vqae.pt": prior / "vqae_continuation/checkpoints/vqae_continuation_best_by_lpips.pt",
        "priors/vqgan.pt": prior / "vqgan_continuation/checkpoints/vqgan_continuation_best_by_lpips.pt",
        "rate05/vqae_refiner.pt": anchor / "runs/seed0/vqae_refiner/checkpoints/vqae_refiner_best_by_val_lpips.pt",
        "rate05/vqgan_refiner.pt": anchor / "runs/seed0/vqgan_refiner/checkpoints/vqgan_refiner_best_by_val_lpips.pt",
    }
    missing = [f"{name}:{path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("OPERATOR_BUNDLE_INPUTS_MISSING:" + ",".join(missing))
    manifest = {
        "status": "FOHI_OPERATOR_SEED_INPUT_BUNDLE",
        "development_source": "STL10 train+unlabeled seed0 split",
        "test_split_included": False,
        "files": {name: sha256(path) for name, path in sorted(files.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for name, path in sorted(files.items()):
            archive.write(path, arcname=name)
    print(f"WROTE {args.output} {args.output.stat().st_size} {sha256(args.output)}")


if __name__ == "__main__":
    main()
