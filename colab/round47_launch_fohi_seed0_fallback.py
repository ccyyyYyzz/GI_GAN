from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
output = Path("/content/gan_r47_results/seed0_primary_seed1_control")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fiber_orthogonal_highpass_innovation.py"),
    "--primary-val",
    "/content/data_primary/seed0_val.pt",
    "--control-val",
    "/content/data_control/seed1_val.pt",
    "--config",
    "/content/data_primary/config_used.yaml",
    "--control-checkpoint",
    "/content/checkpoint_vqae_control_rot0.5_adv0.pt",
    "--gan-checkpoint",
    "/content/checkpoint_gan_rot0.5_adv0.0015.pt",
    "--cutoff",
    "0.12",
    "--transition",
    "0.03",
    "--alpha",
    "0.5",
    "--batch-size",
    "32",
    "--bootstrap-reps",
    "10000",
    "--seed",
    "20260719",
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
print("ROUND47_FOHI_SEED0_FALLBACK_BACKGROUND_LAUNCHED", output)
