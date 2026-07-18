from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
output = Path("/content/gan_r41_results/frequency_fusion")
output.mkdir(parents=True, exist_ok=True)
gan_checkpoints = ",".join(
    [
        "/content/gan_r41_inputs/checkpoint_gan_rot0.25_adv0.pt",
        "/content/gan_r41_inputs/checkpoint_gan_rot0.25_adv0.0015.pt",
        "/content/gan_r41_inputs/checkpoint_gan_rot0.5_adv0.0015.pt",
    ]
)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fiber_residual_frequency_fusion.py"),
    "--primary-val",
    "/content/data_primary/seed0_val.pt",
    "--control-val",
    "/content/data_control/seed1_val.pt",
    "--config",
    "/content/data_primary/config_used.yaml",
    "--control-checkpoint",
    "/content/gan_r40_results/gan_r38_vqae/checkpoint_vqae_control_rot0.5_adv0.pt",
    "--gan-checkpoints",
    gan_checkpoints,
    "--top-exact",
    "12",
    "--batch-size",
    "32",
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
    datetime.datetime.now(datetime.timezone.utc).isoformat()
    + "\n"
    + " ".join(command)
    + "\n"
)
print("ROUND41_BACKGROUND_LAUNCHED", output)
