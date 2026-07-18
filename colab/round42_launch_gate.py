from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
output = Path("/content/gan_r42_results/spectral_gate")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "train_fiber_residual_spectral_fusion.py"),
    "--primary-dev",
    "/content/data_primary/seed0_dev.pt",
    "--primary-val",
    "/content/data_primary/seed0_val.pt",
    "--control-dev",
    "/content/data_control/seed1_dev.pt",
    "--control-val",
    "/content/data_control/seed1_val.pt",
    "--config",
    "/content/data_primary/config_used.yaml",
    "--reference-checkpoint",
    "/content/gan_r42_inputs/reference.pt",
    "--proposal-checkpoint",
    "/content/gan_r42_inputs/proposal.pt",
    "--proposal-arm",
    "gan",
    "--lpips-weights",
    "0,0.001,0.003,0.006",
    "--steps",
    "1200",
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
print("ROUND42_GATE_BACKGROUND_LAUNCHED", output)
