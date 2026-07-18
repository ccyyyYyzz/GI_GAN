from pathlib import Path
import datetime
import os
import subprocess
import sys


worker = Path("/content/round48_eq_worker_fixed.py")
if not worker.is_file():
    raise FileNotFoundError(worker)
subprocess.run(
    ["pkill", "-f", "[r]ound48_eq_worker"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
with open(os.devnull, "r") as null, open("/content/gan_r48_queue_launcher.log", "a") as log:
    subprocess.run(
        ["setsid", "-f", sys.executable, "-u", str(worker)],
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
print("ROUND48_EQ_UPLOADED_QUEUE_LAUNCHED", worker)
