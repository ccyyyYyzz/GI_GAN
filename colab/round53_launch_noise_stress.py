from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import zipfile


with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane_index = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
operator_seed = 772101 + lane_index
repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
output = Path(f"/content/gan_r53_results/operator_seed_{operator_seed}")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "run_noisy_fohi_stress.py"),
    "--asset-root",
    "/content/gan_operator_assets",
    "--dataset-root",
    "/content/datasets",
    "--clean-checkpoint-root",
    f"/content/gan_r51_results/operator_seed_{operator_seed}",
    "--operator-seed",
    str(operator_seed),
    "--lane-index",
    str(lane_index),
    "--snr-db",
    "30",
    "20",
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
print("ROUND53_NOISE_STRESS_BACKGROUND_LAUNCHED", lane_index, operator_seed, output)
