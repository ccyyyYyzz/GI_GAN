from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import base64
from pathlib import Path


ROOT_ZIP = Path("/content/gan_hq_repo.zip")
ARGS_JSON = Path("/content/gan_hq_args.json")
WORK = Path("/content/gan_hq_repo")
ARTIFACT = Path("/content/gan_hq_artifact.zip")


def run(cmd, *, cwd=None, check=True):
    print("[runner]", " ".join(map(str, cmd)), flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def main() -> None:
    started = time.time()
    if not ROOT_ZIP.exists():
        raise FileNotFoundError(ROOT_ZIP)
    if not ARGS_JSON.exists():
        raise FileNotFoundError(ARGS_JSON)
    args = json.loads(ARGS_JSON.read_text(encoding="utf-8-sig"))
    variant = str(args["variant"])
    seeds = str(args["seeds"])
    run_name = str(args.get("name", variant))
    config_rel = str(args.get("config_rel", "configs/compatibility/gan_high_quality_gi_canary.yaml"))
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT_ZIP, "r") as zf:
        zf.extractall(WORK)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numpy<2",
            "tqdm",
            "matplotlib",
            "scikit-image",
            "PyYAML",
            "scipy",
            "lpips",
        ]
    )
    import yaml

    cfg_path = WORK / config_rel
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["data"]["dataset_root"] = "/content/data"
    cfg["data"]["num_workers"] = 2
    cfg["output_dir"] = f"/content/gan_hq_outputs/{run_name}"
    tmp_cfg = WORK / f"configs/compatibility/gan_high_quality_gi_{run_name}.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    command_log = {
        "variant": variant,
        "seeds": seeds,
        "run_name": run_name,
        "config_rel": config_rel,
        "python": sys.version,
        "cwd": str(WORK),
        "config": str(tmp_cfg),
    }
    (Path("/content") / f"gan_hq_command_{variant}.json").write_text(
        json.dumps(command_log, indent=2), encoding="utf-8"
    )
    run(
        [
            sys.executable,
            "gan_high_quality_gi.py",
            "--config",
            str(tmp_cfg),
            "--variants",
            variant,
            "--train-seeds",
            seeds,
        ],
        cwd=WORK,
    )
    out_dir = Path(cfg["output_dir"])
    manifest = {
        "status": "PASS",
        "variant": variant,
        "seeds": seeds,
        "run_name": run_name,
        "config_rel": config_rel,
        "output_dir": str(out_dir),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "COLAB_RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ARTIFACT.exists():
        ARTIFACT.unlink()
    shutil.make_archive(str(ARTIFACT.with_suffix("")), "zip", root_dir=out_dir)
    mini = Path("/content/gan_hq_reports_mini.zip")
    if mini.exists():
        mini.unlink()
    reports_dir = out_dir / "reports"
    with zipfile.ZipFile(mini, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in reports_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(out_dir).as_posix())
        complete = out_dir / "GAN_HIGH_QUALITY_GI_CANARY_COMPLETE.json"
        if complete.exists():
            zf.write(complete, complete.name)
        manifest_path = out_dir / "COLAB_RUN_MANIFEST.json"
        if manifest_path.exists():
            zf.write(manifest_path, manifest_path.name)
        for file in (out_dir / "runs").rglob("figures/*.png"):
            if file.is_file():
                zf.write(file, file.relative_to(out_dir).as_posix())
    encoded = base64.b64encode(mini.read_bytes()).decode("ascii")
    print("GAN_HQ_MINI_ARTIFACT_BASE64_BEGIN", flush=True)
    for i in range(0, len(encoded), 760):
        print(encoded[i : i + 760], flush=True)
    print("GAN_HQ_MINI_ARTIFACT_BASE64_END", flush=True)
    print("GAN_HQ_COLAB_COMPLETE", json.dumps(manifest, sort_keys=True), flush=True)
    print(f"ARTIFACT={ARTIFACT}", flush=True)
    print(f"MINI_ARTIFACT={mini}", flush=True)


if __name__ == "__main__":
    main()
