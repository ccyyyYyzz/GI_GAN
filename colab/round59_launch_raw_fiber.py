"""Launch the isolated Round59 raw-y replay on one existing Pro2 lane.

Run this exact file through ``colab exec --file`` on each of the three current
sessions.  It only fast-forwards the declared branch and launches the runner
under ``setsid``.  The runner reads the completed Round56 cache and writes a
new ``/content/gan_r59_raw_fiber/laneN`` tree; it does not train, download, or
modify any Round56 file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPO = Path("/content/GI_GAN")
BRANCH = "codex/gan-gi-journal-poc-20260718"
FREEZE = REPO / "results/gan_gi_journal_round52/heldout_freeze_v2.json"
ARCHIVE = Path("/content/gan_rate_bundle.zip")
SOURCE_ROOT = Path("/content/gan_r56_heldout_recovery")
OUTPUT_ROOT = Path("/content/gan_r59_raw_fiber")


def lane_from_archive(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])


def require_clean_repo() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if status.strip():
        raise RuntimeError("ROUND59_REQUIRES_CLEAN_REPO")


def main() -> None:
    if not REPO.is_dir() or not FREEZE.is_file() or not ARCHIVE.is_file():
        raise RuntimeError("ROUND59_REQUIRED_COLAB_INPUT_MISSING")
    lane = lane_from_archive(ARCHIVE)
    if lane not in (0, 1, 2):
        raise RuntimeError(f"ROUND59_INVALID_LANE:{lane}")
    source_lane = SOURCE_ROOT / f"lane{lane}"
    if not (source_lane / "HELDOUT_ONCE_COMPLETE.json").is_file():
        raise RuntimeError(f"ROUND59_ROUND56_COMPLETE_RECEIPT_MISSING:{source_lane}")
    for rate in ("05", "10"):
        if not (source_lane / f"rate{rate}/cache/test_cache.pt").is_file():
            raise RuntimeError(f"ROUND59_ROUND56_CACHE_MISSING:{lane}:{rate}")
    output = OUTPUT_ROOT / f"lane{lane}"
    if output.exists():
        raise RuntimeError(f"ROUND59_ONE_SHOT_OUTPUT_ALREADY_EXISTS:{output}")

    require_clean_repo()
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO, check=True)
    require_clean_repo()
    if not (REPO / "run_frozen_fohi_raw_fiber_once.py").is_file():
        raise RuntimeError("ROUND59_RUNNER_NOT_ON_CURRENT_BRANCH")

    command = [
        sys.executable,
        "-u",
        str(REPO / "run_frozen_fohi_raw_fiber_once.py"),
        "--freeze-manifest",
        str(FREEZE),
        "--repo-root",
        str(REPO),
        "--rate-archive",
        str(ARCHIVE),
        "--source-root",
        str(SOURCE_ROOT),
        "--output-root",
        str(OUTPUT_ROOT),
    ]
    launcher_log = Path(f"/content/round59_raw_fiber_lane{lane}_launcher.log")
    with launcher_log.open("w", encoding="utf-8") as log, open(os.devnull, "r") as null:
        subprocess.run(
            ["setsid", "-f", *command],
            cwd=REPO,
            stdin=null,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            check=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
    receipt = Path(f"/content/round59_raw_fiber_lane{lane}_launch_receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "status": "ROUND59_RAW_FIBER_BACKGROUND_LAUNCHED",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "lane_index": lane,
                "command": command,
                "launcher_log": str(launcher_log),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
