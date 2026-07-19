"""Run the Round60 raw-y qualitative-gallery renderer on lane-0 Colab.

This launcher only pulls the declared branch and invokes the post-decision
renderer against the read-only Round56 cache and the completed Round59 raw-y
receipt.  It neither opens a new dataset split nor trains or tunes a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


REPO = Path("/content/GI_GAN")
BRANCH = "codex/gan-gi-journal-poc-20260718"
FREEZE = REPO / "results/gan_gi_journal_round52/heldout_freeze_v2.json"
LANE_ROOT = Path("/content/gan_r56_heldout_recovery/lane0")
ROUND59_ROOT = Path("/content/gan_r59_raw_fiber/lane0")
OUTPUT = Path("/content/gan_r60_qualitative_gallery/lane0")


def lane_from_archive(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])


def main() -> None:
    lane = lane_from_archive(Path("/content/gan_rate_bundle.zip"))
    if lane != 0:
        raise RuntimeError(f"ROUND58_REQUIRES_EXISTING_LANE0_SESSION:{lane}")
    if not (LANE_ROOT / "HELDOUT_ONCE_COMPLETE.json").is_file():
        raise FileNotFoundError(f"ROUND56_HELDOUT_CACHE_MISSING:{LANE_ROOT}")
    if not (ROUND59_ROOT / "ROUND59_COMPLETE.json").is_file():
        raise FileNotFoundError(f"ROUND59_RAW_Y_RESULT_MISSING:{ROUND59_ROOT}")
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO, check=True)
    command = [
        sys.executable,
        "-u",
        str(REPO / "make_frozen_fohi_qualitative_gallery.py"),
        "--freeze-manifest",
        str(FREEZE),
        "--repo-root",
        str(REPO),
        "--source-round56-lane",
        str(LANE_ROOT),
        "--round59-lane",
        str(ROUND59_ROOT),
        "--output-dir",
        str(OUTPUT),
    ]
    subprocess.run(command, cwd=REPO, check=True)
    print("ROUND60_QUALITATIVE_GALLERY_COMPLETE", OUTPUT)


if __name__ == "__main__":
    main()
