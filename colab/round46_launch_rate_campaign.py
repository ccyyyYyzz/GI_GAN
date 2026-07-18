from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


archive = Path("/content/gan_rate_bundle.zip")
if not archive.is_file():
    raise FileNotFoundError(archive)
with zipfile.ZipFile(archive) as bundle:
    manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
    seed = int(manifest["seed"])
    bundle_root = Path(f"/content/gan_rate_bundle_seed{seed}")
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle.extractall(bundle_root)
for relative, expected in manifest["files"].items():
    path = bundle_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"BUNDLE_FILE_MISSING:{relative}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"BUNDLE_HASH_MISMATCH:{relative}:{actual}:{expected}")

repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
output = Path(f"/content/gan_r46_results/seed{seed}")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "run_fiber_rate_campaign.py"),
    "--bundle-root",
    str(bundle_root),
    "--dataset-root",
    "/content/datasets",
    "--rates",
    "02,10",
    "--seed",
    str(seed),
    "--training-seed",
    "20260718",
    "--steps",
    "1500",
    "--output-dir",
    str(output),
]
with (output / "driver.log").open("w") as log, open(os.devnull, "r") as null:
    subprocess.run(
        ["setsid", "-f", *command],
        cwd=repo,
        stdin=null,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        check=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
(output / "launch_receipt.txt").write_text(
    datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n" + " ".join(command) + "\n"
)
print("ROUND46_RATE_CAMPAIGN_BACKGROUND_LAUNCHED", seed, output)

