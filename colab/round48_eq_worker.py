from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


repo = Path("/content/GI_GAN")
seed1_root = Path("/content/gan_r43_results/seed1_primary_seed2_control")
seed2_root = Path("/content/gan_r43_results/seed2_primary_seed0_control")
if seed1_root.exists():
    rate_seed = 1
    label = "seed1_primary_seed2_control"
    primary_val = "/content/data_control/seed1_val.pt"
    control_val = "/content/data_seed2/seed2_val.pt"
    control_checkpoint = str(seed1_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
    proposal_checkpoint = str(seed1_root / "gan/checkpoint_gan_rot0.5_adv0.0015.pt")
elif seed2_root.exists():
    rate_seed = 2
    label = "seed2_primary_seed0_control"
    primary_val = "/content/data_seed2/seed2_val.pt"
    control_val = "/content/data_primary/seed0_val.pt"
    control_checkpoint = str(seed2_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
    proposal_checkpoint = str(seed2_root / "gan/checkpoint_gan_rot0.5_adv0.0015.pt")
else:
    rate_seed = 0
    label = "seed0_primary_seed1_control"
    primary_val = "/content/data_primary/seed0_val.pt"
    control_val = "/content/data_control/seed1_val.pt"
    control_checkpoint = "/content/gan_r40_results/gan_r38_vqae/checkpoint_vqae_control_rot0.5_adv0.pt"
    proposal_checkpoint = "/content/gan_r41_inputs/checkpoint_gan_rot0.5_adv0.0015.pt"

output = Path("/content/gan_r48_results/eq_fohi") / label
output.mkdir(parents=True, exist_ok=True)
queue_log = output / "queue.log"
rate_summary = Path(f"/content/gan_r46_results/seed{rate_seed}/campaign_summary.json")
deadline = time.time() + 3 * 60 * 60
with queue_log.open("a", encoding="utf-8") as log:
    log.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} WAIT_RATE seed={rate_seed}\n")
    log.flush()
    while not rate_summary.is_file() and time.time() < deadline:
        running = subprocess.run(
            ["pgrep", "-f", "[r]un_fiber_rate_campaign.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not running:
            log.write("RATE_CAMPAIGN_ENDED_WITHOUT_SUMMARY\n")
            raise RuntimeError("RATE_CAMPAIGN_ENDED_WITHOUT_SUMMARY")
        time.sleep(30)
    if not rate_summary.is_file():
        log.write("RATE_CAMPAIGN_WAIT_TIMEOUT\n")
        raise TimeoutError("RATE_CAMPAIGN_WAIT_TIMEOUT")
    log.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} RATE_COMPLETE\n")
    log.flush()

branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
command = [
    sys.executable,
    "-u",
    str(repo / "diagnose_box_fiber_endpoint_fohi.py"),
    "--primary-val",
    primary_val,
    "--control-val",
    control_val,
    "--config",
    "/content/data_primary/config_used.yaml",
    "--control-checkpoint",
    control_checkpoint,
    "--proposal-checkpoint",
    proposal_checkpoint,
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
    str(20260719 + rate_seed),
    "--output-dir",
    str(output),
]
with (output / "driver.log").open("w", encoding="utf-8") as log, open(os.devnull, "r") as null:
    subprocess.run(
        command,
        cwd=repo,
        stdin=null,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        check=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
(output / "completion_receipt.txt").write_text(
    datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n" + " ".join(command) + "\n",
    encoding="utf-8",
)

