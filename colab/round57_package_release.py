"""Package the already-complete held-out FOHI lane on an existing Colab VM.

This script is deliberately read-only with respect to the held-out result tree.
It is sent through ``colab exec --file`` after a token rebind; it neither creates
sessions nor invokes the one-shot evaluator.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/content/GI_GAN")
BRANCH = "codex/gan-gi-journal-poc-20260718"

def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)

def main() -> None:
    run(["git", "fetch", "origin", BRANCH])
    run(["git", "checkout", BRANCH])
    run(["git", "pull", "--ff-only", "origin", BRANCH])
    archive = Path("/content/gan_rate_bundle.zip")
    with zipfile.ZipFile(archive) as bundle:
        lane_index = int(json.loads(bundle.read("manifest.json").decode("utf-8"))["seed"])
    if lane_index not in (0, 1, 2):
        raise RuntimeError(f"INVALID_LANE_INDEX:{lane_index}")
    output = Path("/content/gan_r57_release")
    command = [
        sys.executable, "-u", str(REPO / "package_frozen_fohi_lane_release.py"),
        "--freeze-manifest", str(REPO / "results/gan_gi_journal_round52/heldout_freeze_v2.json"),
        "--result-root", f"/content/gan_r56_heldout_recovery/lane{lane_index}",
        "--output-dir", str(output), "--lane-index", str(lane_index),
        "--rate-archive", str(archive),
        "--inventory", f"/content/gan_r54_heldout_inventory_lane{lane_index}.json",
        "--repo-root", str(REPO),
    ]
    run(command)
    receipt = output / f"lane{lane_index}_packaging_receipt.json"
    receipt.write_text(json.dumps({
        "status": "FROZEN_FOHI_LANE_PACKAGED", "lane_index": lane_index,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "archive": str(output / f"lane{lane_index}_frozen_fohi_release.tar.gz"),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(receipt, flush=True)

if __name__ == "__main__":
    main()
