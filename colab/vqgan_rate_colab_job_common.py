"""Colab job for the cross-sampling-rate generalization study (B-tier).

Reuses the existing rate-agnostic VQAE/VQGAN priors (bundled) and retrains ONLY the anchor
refiner at the new operator (sampling rate). One (rate, seed) per session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CONTENT = Path("/content")
BUNDLE = CONTENT / "vqgan_rate_repo_bundle.zip"
REPO = CONTENT / "repo"


def run(cmd, *, cwd=None):
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def tail_text(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def run_with_heartbeat(cmd, *, cwd: Path, log_path: Path, heartbeat_seconds: int = 60):
    print("[cmd]", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(heartbeat_seconds)
            size = log_path.stat().st_size if log_path.exists() else 0
            print(f"[heartbeat] pid={proc.pid} elapsed_s={time.time()-start:.0f} log_bytes={size}", flush=True)
    print(f"[done] rc={proc.returncode} elapsed_s={time.time()-start:.0f}", flush=True)
    print(tail_text(log_path), flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def unzip_repo():
    if REPO.exists():
        return
    if not BUNDLE.exists():
        raise FileNotFoundError(f"Missing bundle: {BUNDLE}")
    with zipfile.ZipFile(BUNDLE, "r") as zf:
        zf.extractall(REPO)


def install_deps():
    run([sys.executable, "-m", "pip", "install", "-q", "lpips", "scikit-image", "PyYAML", "tqdm", "tensorboard", "pytest"])


def zip_outputs(tag: str) -> Path:
    out_zip = CONTENT / f"vqgan_{tag}_artifact.zip"
    if out_zip.exists():
        out_zip.unlink()
    root = REPO / "outputs" / "compatibility" / "measurement_conditioned_vqgan"
    anchor_dir = root / f"anchor_{tag}"
    cfg = REPO / "configs" / "compatibility" / f"anchor_vqgan_inversion_{tag}.yaml"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if anchor_dir.exists():
            for p in anchor_dir.rglob("*"):
                if p.is_file():  # include refiner .pt (needed for the fusion analysis)
                    zf.write(p, p.relative_to(REPO))
        if cfg.exists():
            zf.write(cfg, cfg.relative_to(REPO))
    return out_zip


def main(rate: str, seed: int, smoke: bool = False):
    t0 = time.time()
    unzip_repo()
    install_deps()
    os.chdir(REPO)
    tag = f"rate{rate}{'_smoke' if smoke else ''}_seed{seed}"
    run([sys.executable, "-m", "py_compile", "anchor_initialized_vqgan_inversion.py"], cwd=REPO)
    anchor_cfg = REPO / "configs" / "compatibility" / f"anchor_vqgan_inversion_{tag}.yaml"
    run_with_heartbeat(
        [sys.executable, "anchor_initialized_vqgan_inversion.py", "--config", str(anchor_cfg)],
        cwd=REPO, log_path=CONTENT / f"vqgan_{tag}_anchor.log")
    artifact = zip_outputs(tag)
    status = {"rate": rate, "seed": int(seed), "smoke": bool(smoke), "tag": tag,
              "artifact": str(artifact), "artifact_bytes": artifact.stat().st_size, "seconds": time.time() - t0}
    (CONTENT / f"vqgan_{tag}_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
