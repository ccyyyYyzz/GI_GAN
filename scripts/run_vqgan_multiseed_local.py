from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "compatibility" / "measurement_conditioned_vqgan"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def zip_outputs(seed_id: int) -> Path:
    out_zip = OUT_ROOT / f"VQGAN_MULTI_SEED_LOCAL_SEED{seed_id}_ARTIFACT.zip"
    if out_zip.exists():
        out_zip.unlink()
    include_dirs = [
        OUT_ROOT / f"prior_multiseed_hashclean_seed{seed_id}",
        OUT_ROOT / f"anchor_multiseed_hashclean_seed{seed_id}",
    ]
    include_files = [
        ROOT / "configs" / "compatibility" / f"mc_vqgan_prior_multiseed_hashclean_seed{seed_id}_local.yaml",
        ROOT / "configs" / "compatibility" / f"anchor_vqgan_inversion_multiseed_hashclean_seed{seed_id}_local.yaml",
        ROOT / "mc_vqgan_prior_long_canary.py",
        ROOT / "anchor_initialized_vqgan_inversion.py",
    ]
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder in include_dirs:
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(ROOT))
        for path in include_files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    t0 = time.time()
    seed_id = int(args.seed)
    prior_cfg = ROOT / "configs" / "compatibility" / f"mc_vqgan_prior_multiseed_hashclean_seed{seed_id}_local.yaml"
    anchor_cfg = ROOT / "configs" / "compatibility" / f"anchor_vqgan_inversion_multiseed_hashclean_seed{seed_id}_local.yaml"
    if not prior_cfg.exists():
        raise FileNotFoundError(prior_cfg)
    if not anchor_cfg.exists():
        raise FileNotFoundError(anchor_cfg)
    run([sys.executable, "-m", "py_compile", "mc_vqgan_prior_long_canary.py", "anchor_initialized_vqgan_inversion.py"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_anchor_initialized_vqgan_inversion.py", "tests/test_measurement_conditioned_vqgan.py"])
    run([sys.executable, "mc_vqgan_prior_long_canary.py", "--config", str(prior_cfg)])
    run([sys.executable, "anchor_initialized_vqgan_inversion.py", "--config", str(anchor_cfg)])
    artifact = zip_outputs(seed_id)
    status = {
        "seed_id": seed_id,
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "seconds": time.time() - t0,
    }
    status_path = OUT_ROOT / f"VQGAN_MULTI_SEED_LOCAL_SEED{seed_id}_STATUS.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
