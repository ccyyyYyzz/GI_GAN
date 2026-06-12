from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "colab" / "phase51A"


SESSIONS = {
    "session_06_rad5_no_gate_no_final_audit": {
        "title": "Session 06 Rad-5 combined no_gate + no_final_audit ablation",
        "trains": True,
        "task": "rad5",
        "variant": "no_gate_no_final_audit",
        "command": [
            "python",
            "-m",
            "src.phase51A_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_06_rad5_no_gate_no_final_audit",
            "--task",
            "rad5",
            "--variant",
            "no_gate_no_final_audit",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 Rad-5 with P_N disabled and final Pi_y audit disabled; measurement-domain loss stays active.",
    },
    "session_07_scr5_no_gate_no_final_audit": {
        "title": "Session 07 Scr-5 combined no_gate + no_final_audit ablation",
        "trains": True,
        "task": "scr5",
        "variant": "no_gate_no_final_audit",
        "command": [
            "python",
            "-m",
            "src.phase51A_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_07_scr5_no_gate_no_final_audit",
            "--task",
            "scr5",
            "--variant",
            "no_gate_no_final_audit",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 scrambled Hadamard 5% with P_N disabled and final Pi_y audit disabled; measurement-domain loss stays active.",
    },
    "session_08_rad5_no_final_audit_no_meas_loss": {
        "title": "Session 08 Rad-5 no_final_audit + no_meas_loss ablation",
        "trains": True,
        "task": "rad5",
        "variant": "no_final_audit_no_meas_loss",
        "command": [
            "python",
            "-m",
            "src.phase51A_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_08_rad5_no_final_audit_no_meas_loss",
            "--task",
            "rad5",
            "--variant",
            "no_final_audit_no_meas_loss",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 Rad-5 with P_N and stage-1 audit active, final Pi_y audit disabled, and measurement-domain training losses set to zero.",
    },
    "session_09_scr5_no_final_audit_no_meas_loss": {
        "title": "Session 09 Scr-5 no_final_audit + no_meas_loss ablation",
        "trains": True,
        "task": "scr5",
        "variant": "no_final_audit_no_meas_loss",
        "command": [
            "python",
            "-m",
            "src.phase51A_train_ablation",
            "--bundle_root",
            "{BUNDLE_ROOT}",
            "--output_dir",
            "{SESSION_OUT}",
            "--session_name",
            "session_09_scr5_no_final_audit_no_meas_loss",
            "--task",
            "scr5",
            "--variant",
            "no_final_audit_no_meas_loss",
            "--dataset_root",
            "{DATASET_ROOT}",
            "--device",
            "{DEVICE}",
        ],
        "notes": "Trains STL-10 scrambled Hadamard 5% with P_N and stage-1 audit active, final Pi_y audit disabled, and measurement-domain training losses set to zero.",
    },
}


SETUP_CODE = r'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

SESSION_NAME = "{session_name}"
SESSION_TITLE = "{title}"
SESSION_TRAINS = {trains}
SESSION_TASK = "{task}"
SESSION_VARIANT = "{variant}"
PROJECT_ROOT = Path("/content/ns_mc_gan_gi")
OUT_ROOT = Path("/content/outputs_phase51A")
SESSION_OUT = OUT_ROOT / SESSION_NAME
DATASET_ROOT = Path("/content/ns_mc_gan_gi_data")
BUNDLE_ROOT = Path("/content/noleak_bundle")
DEVICE = "cuda"
CHUNK_BYTES = int(1.8 * 1024**3)

PROJECT_ZIP_NAME = "ns_mc_gan_gi_project_phase51A.zip"
BUNDLE_ZIP_CANDIDATES = [
    "noleak_bundle_phase51A.zip",
    "noleak_bundle_phase48_49.zip",
]
DRIVE_UPLOAD_DIR = Path("/content/drive/MyDrive/ns_mc_gan_gi/colab_upload")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase51A")
USE_DRIVE_IF_AVAILABLE = True

print(SESSION_TITLE)
print("trains:", SESSION_TRAINS)
print("task:", SESSION_TASK, "variant:", SESSION_VARIANT)
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

def merge_parts_to_local(zip_name, part_paths, local):
    part_paths = sorted(Path(p) for p in part_paths)
    if not part_paths:
        return None
    print("Merging parts for", zip_name, ":", [p.name for p in part_paths])
    with local.open("wb") as out:
        for part in part_paths:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out)
    return local

def upload_or_copy_zip(zip_name, drive_path):
    local = Path("/content") / zip_name
    if local.exists():
        print("Using existing", local)
        return local
    if drive_path.exists():
        shutil.copy2(drive_path, local)
        print("Copied from Drive:", drive_path)
        return local
    local_parts = sorted(Path("/content").glob(f"{zip_name}.part_*"))
    if local_parts:
        return merge_parts_to_local(zip_name, local_parts, local)
    drive_parts = sorted(drive_path.parent.glob(f"{zip_name}.part_*")) if drive_path.parent.exists() else []
    if drive_parts:
        return merge_parts_to_local(zip_name, drive_parts, local)
    return None

def require_zip(zip_name, drive_path):
    path = upload_or_copy_zip(zip_name, drive_path)
    if path is not None:
        return path
    print(f"Upload required: {zip_name}")
    from google.colab import files
    uploaded = files.upload()
    for name in uploaded:
        p = Path("/content") / name
        if name == zip_name:
            return p
    local_parts = sorted(Path("/content").glob(f"{zip_name}.part_*"))
    if local_parts:
        return merge_parts_to_local(zip_name, local_parts, Path("/content") / zip_name)
    matches = list(Path("/content").glob("*.zip"))
    if matches:
        print("Using uploaded zip:", matches[-1])
        return matches[-1]
    raise FileNotFoundError(f"No zip uploaded for {zip_name}")

def find_bundle_zip():
    for zip_name in BUNDLE_ZIP_CANDIDATES:
        path = upload_or_copy_zip(zip_name, DRIVE_UPLOAD_DIR / zip_name)
        if path is not None:
            print("Using no-leak bundle:", path.name)
            return path
    print("No no-leak bundle found in /content or Drive.")
    print("Upload one of:", BUNDLE_ZIP_CANDIDATES)
    from google.colab import files
    files.upload()
    for zip_name in BUNDLE_ZIP_CANDIDATES:
        path = upload_or_copy_zip(zip_name, DRIVE_UPLOAD_DIR / zip_name)
        if path is not None:
            print("Using no-leak bundle:", path.name)
            return path
    matches = list(Path("/content").glob("noleak_bundle*.zip"))
    if matches:
        print("Using uploaded bundle:", matches[-1])
        return matches[-1]
    raise FileNotFoundError("No no-leak bundle zip found after upload.")

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
    tmp = Path("/content/_phase51A_project_unzip")
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
project_zip = require_zip(PROJECT_ZIP_NAME, DRIVE_UPLOAD_DIR / PROJECT_ZIP_NAME)
bundle_zip = find_bundle_zip()
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
}
print("Input bundle verification:")
ok = True
for task, names in required.items():
    if SESSION_TASK == "rad5" and not task.startswith("rademacher5"):
        continue
    if SESSION_TASK == "scr5" and not task.startswith("scrambled_hadamard5"):
        continue
    base = BUNDLE_ROOT / task
    print("\n", task, "exists:", base.exists())
    ok = ok and base.exists()
    for name in names:
        p = base / name
        present = p.exists()
        ok = ok and present
        print("  ", name, "OK" if present else "MISSING", p)
assert ok, "Required no-leak input files are missing. Stop this runtime and fix bundle/upload before training."
'''.strip()


def command_code(command: list[str]) -> str:
    rendered = []
    for part in command:
        rendered.append(
            part.replace("{BUNDLE_ROOT}", "{BUNDLE_ROOT}")
            .replace("{SESSION_OUT}", "{SESSION_OUT}")
            .replace("{DATASET_ROOT}", "{DATASET_ROOT}")
            .replace("{DEVICE}", "{DEVICE}")
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
cmd = [str(x).format(BUNDLE_ROOT=BUNDLE_ROOT, SESSION_OUT=SESSION_OUT, DATASET_ROOT=DATASET_ROOT, DEVICE=DEVICE) for x in cmd]
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
DRIVE_OUTPUT_DIR = Path(globals().get("DRIVE_OUTPUT_DIR", "/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase51A"))

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
            "source": [f"# Phase 51A - {meta['title']}\n", "\n", f"{meta['notes']}\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": SETUP_CODE.format(
                session_name=session_name,
                title=meta["title"],
                trains=str(meta["trains"]),
                task=meta["task"],
                variant=meta["variant"],
            ).splitlines(True),
        },
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
