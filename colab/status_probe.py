"""Short status probe: run via `colab exec -f status_probe.py --timeout 40`.
Prints which job files exist (+sizes), refiner train-log progress, and the job_out.log tail,
so the local monitor can poll cheaply WITHOUT keeping a long exec connection open.
"""
import glob
import os
from pathlib import Path

print("PROBE_BEGIN", flush=True)
for pat in ("/content/job_out.log", "/content/vqgan_*_status.json", "/content/vqgan_*_artifact.zip"):
    for p in sorted(glob.glob(pat)):
        print(f"FILE {p} size={os.path.getsize(p)}", flush=True)

for tl in sorted(glob.glob("/content/repo/outputs/**/runs/seed*/*_refiner/train_log.csv", recursive=True)):
    try:
        lines = Path(tl).read_text().strip().splitlines()
        step = lines[-1].split(",")[1] if len(lines) > 1 else "header"
        short = tl.split("/runs/")[-1]
        print(f"TRAINLOG {short} step={step} nlines={len(lines)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"TRAINLOG {tl} ERR {e}", flush=True)

jo = Path("/content/job_out.log")
if jo.exists():
    tail = jo.read_text(errors="replace")[-500:].replace("\n", " | ")
    print(f"JOBOUT_TAIL {tail}", flush=True)
print("PROBE_END", flush=True)
