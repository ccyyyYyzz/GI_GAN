from pathlib import Path
import subprocess
import tarfile


repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
target = Path("/content/data_seed2")
target.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/gan_r43_seed2.tar.gz", "r:gz") as handle:
    handle.extractall(target, filter="data")
for required in (target / "seed2_dev.pt", target / "seed2_val.pt"):
    if not required.exists():
        raise FileNotFoundError(required)
print("ROUND43_PREPARED", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
