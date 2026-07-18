from pathlib import Path
import subprocess
import tarfile


repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
archive = Path("/content/gan_r38_control.tar.gz")
if archive.exists() and not Path("/content/data_control/seed1_val.pt").exists():
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall("/content", filter="data")
for required in (
    Path("/content/data_primary/seed0_dev.pt"),
    Path("/content/data_primary/seed0_val.pt"),
    Path("/content/data_control/seed1_dev.pt"),
    Path("/content/data_control/seed1_val.pt"),
):
    if not required.exists():
        raise FileNotFoundError(required)
Path("/content/gan_r42_inputs").mkdir(parents=True, exist_ok=True)
Path("/content/gan_r42_results").mkdir(parents=True, exist_ok=True)
print("ROUND42_PREPARED", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
