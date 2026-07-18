from pathlib import Path
import subprocess


repo = Path("/content/GI_GAN")
branch = "codex/gan-gi-journal-poc-20260718"
subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
Path("/content/gan_r41_inputs").mkdir(parents=True, exist_ok=True)
Path("/content/gan_r41_results").mkdir(parents=True, exist_ok=True)
print("ROUND41_PREPARED", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
