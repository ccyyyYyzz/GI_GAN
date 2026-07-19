from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import zipfile


repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane_index = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
output = Path(f"/content/gan_r55_heldout/lane{lane_index}")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "run_frozen_fohi_heldout_once.py"),
    "--freeze-manifest",
    str(repo / "results/gan_gi_journal_round52/heldout_freeze.json"),
    "--dataset-root",
    "/content/datasets",
    "--split",
    "test",
    "--bootstrap-reps",
    "20000",
    "--bootstrap-seed",
    "20260719",
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
print("ROUND55_HELDOUT_BACKGROUND_LAUNCHED", lane_index, output)
