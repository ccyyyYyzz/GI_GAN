from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


CONTENT = Path("/content")
BUNDLE = CONTENT / "vqgan_multiseed_repo_bundle.zip"
REPO = CONTENT / "repo"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def tail_text(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace")


def run_with_heartbeat(cmd: list[str], *, cwd: Path, log_path: Path, heartbeat_seconds: int = 60) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("wb") as log_f:
        process = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(heartbeat_seconds)
            elapsed = time.time() - start
            size = log_path.stat().st_size if log_path.exists() else 0
            print(f"[heartbeat] pid={process.pid} elapsed_s={elapsed:.0f} log_bytes={size} log={log_path}", flush=True)
    elapsed = time.time() - start
    print(f"[done] rc={process.returncode} elapsed_s={elapsed:.0f} log={log_path}", flush=True)
    print(tail_text(log_path), flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def unzip_repo() -> None:
    if REPO.exists():
        return
    if not BUNDLE.exists():
        raise FileNotFoundError(f"Missing uploaded bundle: {BUNDLE}")
    with zipfile.ZipFile(BUNDLE, "r") as zf:
        zf.extractall(REPO)


def install_deps() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "lpips",
        "scikit-image",
        "PyYAML",
        "tqdm",
        "tensorboard",
        "pytest",
    ])


def zip_outputs(seed_id: int) -> Path:
    out_zip = CONTENT / f"vqgan_multiseed_seed{seed_id}_artifact.zip"
    if out_zip.exists():
        out_zip.unlink()
    root = REPO / "outputs" / "compatibility" / "measurement_conditioned_vqgan"
    include_dirs = [
        root / f"prior_multiseed_hashclean_seed{seed_id}",
        root / f"anchor_multiseed_hashclean_seed{seed_id}",
    ]
    include_files = [
        root / f"ANCHOR_INITIALIZED_VQGAN_INVERSION_SEED{seed_id}_PACKAGE.zip",
        REPO / "configs" / "compatibility" / f"mc_vqgan_prior_multiseed_hashclean_seed{seed_id}.yaml",
        REPO / "configs" / "compatibility" / f"anchor_vqgan_inversion_multiseed_hashclean_seed{seed_id}.yaml",
        REPO / "mc_vqgan_prior_long_canary.py",
        REPO / "anchor_initialized_vqgan_inversion.py",
    ]
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder in include_dirs:
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(REPO))
        for path in include_files:
            if path.exists():
                zf.write(path, path.relative_to(REPO))
    return out_zip


def main(seed_id: int) -> None:
    t0 = time.time()
    unzip_repo()
    install_deps()
    os.chdir(REPO)
    run([sys.executable, "-m", "py_compile", "mc_vqgan_prior_long_canary.py", "anchor_initialized_vqgan_inversion.py"], cwd=REPO)
    run([sys.executable, "-m", "pytest", "-q", "tests/test_anchor_initialized_vqgan_inversion.py", "tests/test_measurement_conditioned_vqgan.py"], cwd=REPO)
    prior_cfg = REPO / "configs" / "compatibility" / f"mc_vqgan_prior_multiseed_hashclean_seed{seed_id}.yaml"
    anchor_cfg = REPO / "configs" / "compatibility" / f"anchor_vqgan_inversion_multiseed_hashclean_seed{seed_id}.yaml"
    run_with_heartbeat(
        [sys.executable, "mc_vqgan_prior_long_canary.py", "--config", str(prior_cfg)],
        cwd=REPO,
        log_path=CONTENT / f"vqgan_multiseed_seed{seed_id}_prior.log",
    )
    run_with_heartbeat(
        [sys.executable, "anchor_initialized_vqgan_inversion.py", "--config", str(anchor_cfg)],
        cwd=REPO,
        log_path=CONTENT / f"vqgan_multiseed_seed{seed_id}_anchor.log",
    )
    artifact = zip_outputs(seed_id)
    status = {
        "seed_id": int(seed_id),
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "seconds": time.time() - t0,
    }
    (CONTENT / f"vqgan_multiseed_seed{seed_id}_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
