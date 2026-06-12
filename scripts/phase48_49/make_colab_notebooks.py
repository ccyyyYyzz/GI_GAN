from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "colab" / "phase48_49"


SESSIONS = {
    "session_01_eval_probes": {
        "title": "Session 01 eval-only mechanistic probes",
        "trains": False,
        "command": [
            "python",
            "-m",
            "src.phase48_49_mechanistic_probes",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
            "--limit_samples",
            "{LIMIT_SAMPLES}",
        ],
        "notes": "Loads Rad-5/Scr-5/Rad-10/Scr-10 strict no-leak checkpoints and runs diagnostic probes only.",
    },
    "session_02_rad5_no_gate": {
        "title": "Session 02 Rad-5 train-time no_gate ablation",
        "trains": True,
        "command": [
            "python",
            "-m",
            "src.phase48_49_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_02_rad5_no_gate",
            "--task",
            "rad5",
            "--variant",
            "no_gate",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 Rad-5 with P_N disabled and final Pi_y audit kept active.",
    },
    "session_03_rad5_no_final_audit": {
        "title": "Session 03 Rad-5 train-time no_final_audit ablation",
        "trains": True,
        "command": [
            "python",
            "-m",
            "src.phase48_49_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_03_rad5_no_final_audit",
            "--task",
            "rad5",
            "--variant",
            "no_final_audit",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 Rad-5 with P_N active, stage-1 audit active, final/refiner Pi_y disabled.",
    },
    "session_04_scr5_no_gate": {
        "title": "Session 04 Scr-5 train-time no_gate ablation",
        "trains": True,
        "command": [
            "python",
            "-m",
            "src.phase48_49_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_04_scr5_no_gate",
            "--task",
            "scr5",
            "--variant",
            "no_gate",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 scrambled Hadamard 5% with P_N disabled and final Pi_y audit kept active.",
    },
    "session_05_scr5_no_final_audit": {
        "title": "Session 05 Scr-5 train-time no_final_audit ablation",
        "trains": True,
        "command": [
            "python",
            "-m",
            "src.phase48_49_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_05_scr5_no_final_audit",
            "--task",
            "scr5",
            "--variant",
            "no_final_audit",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 scrambled Hadamard 5% with P_N active, stage-1 audit active, final/refiner Pi_y disabled.",
    },
}


SETUP_CODE = r'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

SESSION_NAME = "{session_name}"
SESSION_TITLE = "{title}"
SESSION_TRAINS = {trains}
PROJECT_ROOT = Path("/content/ns_mc_gan_gi")
OUT_ROOT = Path("/content/outputs_phase48_49")
SESSION_OUT = OUT_ROOT / SESSION_NAME
DATASET_ROOT = Path("/content/ns_mc_gan_gi_data")
BUNDLE_ROOT = Path("/content/noleak_bundle")
DEVICE = "cuda"
LIMIT_SAMPLES = 1000
CHUNK_BYTES = int(1.8 * 1024**3)

PROJECT_ZIP_NAME = "ns_mc_gan_gi_project_phase48_49.zip"
BUNDLE_ZIP_NAME = "noleak_bundle_phase48_49.zip"
DRIVE_UPLOAD_DIR = Path("/content/drive/MyDrive/ns_mc_gan_gi/colab_upload")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase48_49")
USE_DRIVE_IF_AVAILABLE = True

print(SESSION_TITLE)
print("trains:", SESSION_TRAINS)
print("project root:", PROJECT_ROOT)
print("session output:", SESSION_OUT)
'''.strip()


UPLOAD_CODE = r'''
def maybe_mount_drive():
    if not USE_DRIVE_IF_AVAILABLE:
        return False
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return Path("/content/drive").exists()
    except Exception as exc:
        print("Drive mount skipped/failed:", repr(exc))
        return False

def upload_or_copy_zip(zip_name, drive_path):
    local = Path("/content") / zip_name
    if local.exists():
        print("Using existing", local)
        return local
    if drive_path.exists():
        shutil.copy2(drive_path, local)
        print("Copied from Drive:", drive_path)
        return local
    part_paths = sorted(Path("/content").glob(f"{zip_name}.part_*"))
    if part_paths:
        print("Merging existing input parts:", [p.name for p in part_paths])
        with local.open("wb") as out:
            for part in part_paths:
                with part.open("rb") as f:
                    shutil.copyfileobj(f, out)
        return local
    print(f"Upload required: {zip_name}")
    from google.colab import files
    uploaded = files.upload()
    for name in uploaded:
        p = Path("/content") / name
        if name == zip_name:
            return p
    part_paths = sorted(Path("/content").glob(f"{zip_name}.part_*"))
    if part_paths:
        print("Merging uploaded input parts:", [p.name for p in part_paths])
        with local.open("wb") as out:
            for part in part_paths:
                with part.open("rb") as f:
                    shutil.copyfileobj(f, out)
        return local
    matches = list(Path("/content").glob("*.zip"))
    if matches:
        print("Using uploaded zip:", matches[-1])
        return matches[-1]
    raise FileNotFoundError(f"No zip uploaded for {zip_name}")

def safe_extract_zip(zip_path, dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            normalized = info.filename.replace("\\", "/").lstrip("/")
            if not normalized or normalized.endswith("/"):
                continue
            target = dest / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)

def unzip_project(project_zip):
    tmp = Path("/content/_phase48_49_project_unzip")
    safe_extract_zip(project_zip, tmp)
    roots = [p for p in tmp.rglob("*") if p.is_dir() and (p / "src").exists() and (p / "configs").exists() and (p / "scripts").exists()]
    if not roots and (tmp / "src").exists():
        roots = [tmp]
    if not roots:
        raise AssertionError("No project root found: expected src/configs/scripts inside uploaded project zip.")
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    shutil.copytree(roots[0], PROJECT_ROOT)
    print("Project extracted to:", PROJECT_ROOT)

def unzip_bundle(bundle_zip):
    safe_extract_zip(bundle_zip, BUNDLE_ROOT)
    nested = [p for p in BUNDLE_ROOT.iterdir() if p.is_dir() and (p / "rademacher5_hq_noise001_colab").exists()]
    if nested:
        nested_name = nested[0].name
        tmp = BUNDLE_ROOT.with_name("_noleak_bundle_nested")
        if tmp.exists():
            shutil.rmtree(tmp)
        BUNDLE_ROOT.rename(tmp)
        shutil.move(str(tmp / nested_name), str(BUNDLE_ROOT))
        shutil.rmtree(tmp)
    print("Bundle extracted to:", BUNDLE_ROOT)

drive_ok = maybe_mount_drive()
project_zip = upload_or_copy_zip(PROJECT_ZIP_NAME, DRIVE_UPLOAD_DIR / PROJECT_ZIP_NAME)
bundle_zip = upload_or_copy_zip(BUNDLE_ZIP_NAME, DRIVE_UPLOAD_DIR / BUNDLE_ZIP_NAME)
unzip_project(project_zip)
unzip_bundle(bundle_zip)

print("Bundle top-level:")
for p in sorted(BUNDLE_ROOT.iterdir()):
    print(" -", p.name)

sys.path.insert(0, str(PROJECT_ROOT))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "tqdm", "scikit-image", "matplotlib"], check=False)
'''.strip()


VERIFY_CODE = r'''
required = {
    "rademacher5_hq_noise001_colab": ["resolved_config.yaml", "last.pt", "measurement_operator_exact.pt"],
    "scrambled_hadamard5_hq_noise001_colab": ["resolved_config.yaml", "last.pt"],
    "rademacher10_full_noise001_colab": ["resolved_config.yaml", "last.pt", "measurement_operator_exact.pt"],
    "scrambled_hadamard10_full_noise001_colab": ["resolved_config.yaml", "last.pt"],
}
print("Input bundle verification:")
for task, names in required.items():
    base = BUNDLE_ROOT / task
    print("\n", task, "exists:", base.exists())
    for name in names:
        p = base / name
        print("  ", name, "OK" if p.exists() else "MISSING", p)
'''.strip()


def command_code(command: list[str]) -> str:
    rendered = []
    for part in command:
        rendered.append(
            part.replace("{BUNDLE_ROOT}", "{BUNDLE_ROOT}")
            .replace("{SESSION_OUT}", "{SESSION_OUT}")
            .replace("{DATASET_ROOT}", "{DATASET_ROOT}")
            .replace("{DEVICE}", "{DEVICE}")
            .replace("{LIMIT_SAMPLES}", "{LIMIT_SAMPLES}")
        )
    template = r'''
def run_logged(cmd, cwd):
    SESSION_OUT.mkdir(parents=True, exist_ok=True)
    log_path = SESSION_OUT / "command_log.txt"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\\n$ " + " ".join(map(str, cmd)) + "\\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {cmd}")

cmd = __CMD_JSON__
cmd = [str(x).format(BUNDLE_ROOT=BUNDLE_ROOT, SESSION_OUT=SESSION_OUT, DATASET_ROOT=DATASET_ROOT, DEVICE=DEVICE, LIMIT_SAMPLES=LIMIT_SAMPLES) for x in cmd]
run_logged(cmd, PROJECT_ROOT)
'''.strip()
    return template.replace("__CMD_JSON__", json.dumps(rendered))


EXPORT_CODE = r'''
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def split_file(path, chunk_bytes):
    path = Path(path)
    parts_dir = Path("/content") / f"{SESSION_NAME}_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    parts = []
    with path.open("rb") as f:
        i = 0
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            part = parts_dir / f"{path.name}.part_{i:03d}"
            part.write_bytes(data)
            parts.append(part)
            i += 1
    return parts_dir, parts

SESSION_OUT.mkdir(parents=True, exist_ok=True)
if not (SESSION_OUT / "SESSION_STATUS.json").exists():
    (SESSION_OUT / "SESSION_STATUS.json").write_text(json.dumps({"ok": False, "note": "runner finished without session script status"}, indent=2), encoding="utf-8")
drive_ok = bool(globals().get("drive_ok", Path("/content/drive").exists()))
DRIVE_OUTPUT_DIR = Path(globals().get("DRIVE_OUTPUT_DIR", "/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase48_49"))

zip_base = Path("/content") / f"{SESSION_NAME}_outputs"
zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=str(OUT_ROOT), base_dir=SESSION_NAME))
manifest = {
    "session": SESSION_NAME,
    "zip": str(zip_path),
    "size_bytes": zip_path.stat().st_size,
    "sha256": sha256_file(zip_path),
    "chunk_bytes": CHUNK_BYTES,
}
(Path("/content") / f"{SESSION_NAME}_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))

if drive_ok:
    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, DRIVE_OUTPUT_DIR / zip_path.name)
    shutil.copy2(Path("/content") / f"{SESSION_NAME}_download_manifest.json", DRIVE_OUTPUT_DIR / f"{SESSION_NAME}_download_manifest.json")
    print("Copied zip/manifest to Drive:", DRIVE_OUTPUT_DIR)

from google.colab import files
files.download(str(Path("/content") / f"{SESSION_NAME}_download_manifest.json"))
if zip_path.stat().st_size <= CHUNK_BYTES:
    files.download(str(zip_path))
else:
    parts_dir, parts = split_file(zip_path, CHUNK_BYTES)
    split_manifest = {
        **manifest,
        "parts": [p.name for p in parts],
        "parts_dir": str(parts_dir),
        "part_sha256": {p.name: sha256_file(p) for p in parts},
    }
    split_manifest_path = parts_dir / f"{SESSION_NAME}_split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    files.download(str(split_manifest_path))
    for part in parts:
        print("Downloading", part.name, part.stat().st_size / 1024**2, "MB")
        files.download(str(part))
print("FINAL STATUS: exported", SESSION_NAME)
'''.strip()


def make_notebook(session_name: str, meta: dict) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# Phase 48/49 - {meta['title']}\n", "\n", f"{meta['notes']}\n"],
        },
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": SETUP_CODE.format(session_name=session_name, title=meta["title"], trains=str(meta["trains"])) .splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": UPLOAD_CODE.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": VERIFY_CODE.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": command_code(meta["command"]).splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": EXPORT_CODE.splitlines(True)},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for session_name, meta in SESSIONS.items():
        path = OUT / f"{session_name}.ipynb"
        path.write_text(json.dumps(make_notebook(session_name, meta), ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
