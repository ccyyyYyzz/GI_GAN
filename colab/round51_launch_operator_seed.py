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


archive_path = Path("/content/gan_operator_assets.zip")
if not archive_path.is_file():
    raise FileNotFoundError(archive_path)
asset_root = Path("/content/gan_operator_assets")
asset_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path) as archive:
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    archive.extractall(asset_root)
for name, expected in manifest["files"].items():
    path = asset_root / name
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"OPERATOR_ASSET_HASH_MISMATCH:{name}")

with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane_index = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
operator_seed = 772101 + lane_index
repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
output = Path(f"/content/gan_r51_results/operator_seed_{operator_seed}")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, "-u", str(repo / "run_operator_seed_campaign.py"),
    "--asset-root", str(asset_root),
    "--dataset-root", "/content/datasets",
    "--operator-seed", str(operator_seed),
    "--lane-index", str(lane_index),
    "--steps", "1500",
    "--output-dir", str(output),
]
with (output / "driver.log").open("w", encoding="utf-8") as log, open(os.devnull, "r") as null:
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
    datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n" + " ".join(command) + "\n",
    encoding="utf-8",
)
print("ROUND51_OPERATOR_SEED_BACKGROUND_LAUNCHED", lane_index, operator_seed, output)
