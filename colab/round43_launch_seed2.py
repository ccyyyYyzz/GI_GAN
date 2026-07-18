from pathlib import Path
import datetime
import os
import subprocess
import sys


repo = Path("/content/GI_GAN")
output = Path("/content/gan_r43_results/seed2_primary_seed0_control")
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-u",
    str(repo / "run_fiber_fusion_multiseed_pipeline.py"),
    "--primary-dev",
    "/content/data_seed2/seed2_dev.pt",
    "--primary-val",
    "/content/data_seed2/seed2_val.pt",
    "--control-dev",
    "/content/data_primary/seed0_dev.pt",
    "--control-val",
    "/content/data_primary/seed0_val.pt",
    "--config",
    "/content/data_primary/config_used.yaml",
    "--label",
    "seed2_primary_seed0_control",
    "--seed",
    "20260718",
    "--output-dir",
    str(output),
]
with (output / "pipeline_driver.log").open("w") as log, open(os.devnull, "r") as null:
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
print("ROUND43_SEED2_BACKGROUND_LAUNCHED", output)
