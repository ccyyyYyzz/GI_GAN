from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
inputs = Path("/content")
required = {
    "primary_val": Path("/content/data_primary/seed0_val.pt"),
    "control_val": Path("/content/data_control/seed1_val.pt"),
    "config": Path("/content/data_primary/config_used.yaml"),
    "control_checkpoint": inputs / "checkpoint_vqae_control_rot0.5_adv0.pt",
    "proposal_checkpoint": inputs / "checkpoint_gan_rot0.5_adv0.0015.pt",
}
missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
if missing:
    raise FileNotFoundError("ROUND45_SEED0_INPUTS_MISSING:" + ",".join(missing))

output = Path("/content/gan_r45_results/seed0_primary_seed1_control")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_fiber_residual_frequency_fusion.py"),
    "--primary-val",
    str(required["primary_val"]),
    "--control-val",
    str(required["control_val"]),
    "--config",
    str(required["config"]),
    "--control-checkpoint",
    str(required["control_checkpoint"]),
    "--proposal-checkpoints",
    str(required["proposal_checkpoint"]),
    "--cutoffs",
    "0.15,0.18,0.21",
    "--alphas",
    "0.55,0.58,0.6,0.62,0.65",
    "--top-exact",
    "15",
    "--batch-size",
    "32",
    "--bootstrap-reps",
    "5000",
    "--seed",
    "20260718",
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
print("ROUND45_SEED0_FALLBACK_BACKGROUND_LAUNCHED", output)
