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
seed1_root = Path("/content/gan_r43_results/seed1_primary_seed2_control")
seed2_root = Path("/content/gan_r43_results/seed2_primary_seed0_control")
if seed1_root.exists():
    label = "seed1_primary_seed2_control"
    primary_val = "/content/data_control/seed1_val.pt"
    control_val = "/content/data_seed2/seed2_val.pt"
    control_checkpoint = str(seed1_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
    gan_checkpoint = str(seed1_root / "gan/checkpoint_gan_rot0.5_adv0.0015.pt")
elif seed2_root.exists():
    label = "seed2_primary_seed0_control"
    primary_val = "/content/data_seed2/seed2_val.pt"
    control_val = "/content/data_primary/seed0_val.pt"
    control_checkpoint = str(seed2_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
    gan_checkpoint = str(seed2_root / "gan/checkpoint_gan_rot0.5_adv0.0015.pt")
else:
    label = "seed0_primary_seed1_control"
    primary_val = "/content/data_primary/seed0_val.pt"
    control_val = "/content/data_control/seed1_val.pt"
    control_checkpoint = "/content/gan_r40_results/gan_r38_vqae/checkpoint_vqae_control_rot0.5_adv0.pt"
    gan_checkpoint = "/content/gan_r41_inputs/checkpoint_gan_rot0.5_adv0.0015.pt"

output = Path("/content/gan_r47_results") / label
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fiber_orthogonal_highpass_innovation.py"),
    "--primary-val",
    primary_val,
    "--control-val",
    control_val,
    "--config",
    "/content/data_primary/config_used.yaml",
    "--control-checkpoint",
    control_checkpoint,
    "--gan-checkpoint",
    gan_checkpoint,
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
    "--final-target",
    "legacy_clipped_anchor",
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
print("ROUND47_FOHI_BACKGROUND_LAUNCHED", label, output)
