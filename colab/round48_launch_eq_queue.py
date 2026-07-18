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
worker = repo / "colab/round48_eq_worker.py"
subprocess.run(
    ["pkill", "-f", "/content/GI_GAN/colab/[r]ound48_eq_worker.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
with open(os.devnull, "r") as null, open("/content/gan_r48_queue_launcher.log", "a") as log:
    subprocess.run(
        ["setsid", "-f", sys.executable, "-u", str(worker)],
        cwd=repo,
        stdin=null,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        check=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
Path("/content/gan_r48_queue_launch_receipt.txt").write_text(
    datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n" + str(worker) + "\n",
    encoding="utf-8",
)
print("ROUND48_EQ_QUEUE_LAUNCHED", worker)
