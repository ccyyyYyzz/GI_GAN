from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import zipfile


with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    seed = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
rate_root = Path(f"/content/gan_r46_results/seed{seed}/rate10")
cache = Path(f"/content/gan_r46_results/seed{seed}/cache/rate10/seed{seed}_val.pt")
config = Path(f"/content/gan_rate_bundle_seed{seed}/config_rate10.yaml")
output = rate_root / "fohi_reprojected_4096"
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fiber_orthogonal_highpass_innovation.py"),
    "--primary-val",
    str(cache),
    "--control-val",
    str(cache),
    "--config",
    str(config),
    "--control-checkpoint",
    str(rate_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt"),
    "--proposal-checkpoint",
    str(rate_root / "gan/checkpoint_gan_rot0.5_adv0.0015.pt"),
    "--cutoff",
    "0.12",
    "--transition",
    "0.03",
    "--alpha",
    "0.5",
    "--batch-size",
    "32",
    "--exact-iterations",
    "4096",
    "--bootstrap-reps",
    "10000",
    "--seed",
    str(20260728 + 1000 * seed),
    "--output-dir",
    str(output),
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
print("ROUND46_RATE10_REPROJECT_BACKGROUND_LAUNCHED", seed, output)
