from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
output = Path("/content/gan_r44_results/fused_physical_readout")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fused_residual_physical_readout.py"),
    "--primary-val",
    "/content/data_primary/seed0_val.pt",
    "--control-val",
    "/content/data_control/seed1_val.pt",
    "--config",
    "/content/data_primary/config_used.yaml",
    "--control-checkpoint",
    "/content/gan_r40_results/gan_r38_vqae/checkpoint_vqae_control_rot0.5_adv0.pt",
    "--gan-checkpoint",
    "/content/gan_r41_inputs/checkpoint_gan_rot0.5_adv0.0015.pt",
    "--cutoff",
    "0.12",
    "--alpha",
    "0.5",
    "--limit",
    "512",
    "--pairs",
    "16",
    "--photon-levels",
    "1e4,1e5,1e6",
    "--poisson-replicates",
    "8",
    "--bootstrap-reps",
    "5000",
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
print("ROUND44_PHYSICAL_BACKGROUND_LAUNCHED", output)
