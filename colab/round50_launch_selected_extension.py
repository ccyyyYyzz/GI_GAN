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
    rate_seed = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])

seed1_root = Path("/content/gan_r43_results/seed1_primary_seed2_control")
seed2_root = Path("/content/gan_r43_results/seed2_primary_seed0_control")
if seed1_root.exists():
    label = "seed1_primary_seed2_control"
    primary_val = "/content/data_control/seed1_val.pt"
    control_val = "/content/data_seed2/seed2_val.pt"
    structural_checkpoint = str(seed1_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
elif seed2_root.exists():
    label = "seed2_primary_seed0_control"
    primary_val = "/content/data_seed2/seed2_val.pt"
    control_val = "/content/data_primary/seed0_val.pt"
    structural_checkpoint = str(seed2_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt")
else:
    label = "seed0_primary_seed1_control"
    primary_val = "/content/data_primary/seed0_val.pt"
    control_val = "/content/data_control/seed1_val.pt"
    structural_checkpoint = "/content/gan_r40_results/gan_r38_vqae/checkpoint_vqae_control_rot0.5_adv0.pt"

output = Path("/content/gan_r50_results") / label
output.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, "-u", str(repo / "run_selected_fohi_extension.py"),
    "--five-primary-val", primary_val,
    "--five-control-val", control_val,
    "--five-config", "/content/data_primary/config_used.yaml",
    "--five-structural-checkpoint", structural_checkpoint,
    "--five-adv0-checkpoint", str(Path("/content/gan_r49_results") / label / "train_gan_adv0/checkpoint_gan_rot0.5_adv0.pt"),
    "--rate-seed", str(rate_seed),
    "--rate-cache-root", str(Path(f"/content/gan_r46_results/seed{rate_seed}/cache")),
    "--rate-bundle-root", str(Path(f"/content/gan_rate_bundle_seed{rate_seed}")),
    "--rate-result-root", str(Path(f"/content/gan_r46_results/seed{rate_seed}")),
    "--steps", "1500",
    "--output-dir", str(output),
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
print("ROUND50_SELECTED_EXTENSION_BACKGROUND_LAUNCHED", label, rate_seed, output)
