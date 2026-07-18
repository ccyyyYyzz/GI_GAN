"""Dependency-free layout check for a fresh GI_GAN checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "HANDOFF/00_START_HERE.md",
    "HANDOFF/04_REPRODUCIBILITY_GUIDE.md",
    "HANDOFF/07_RED_LINES_AND_WORKING_RULES.md",
    "docs/core_experiments/claim_evidence_matrix.csv",
    "repro/DATA_AND_ARTIFACTS.md",
    "repro/PAPER_EVIDENCE_MAP.md",
    "repro/RESULTS_STATUS.md",
    "paper/README.md",
    "paper/OPTICS_DRAFT.tex",
    "src/projections.py",
    "eval/audit.py",
    "tests/test_exact_projections.py",
    "vqgan_detail_fusion.py",
    "inspect_gate.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    payload = {
        "repository": "ccyyyYyzz/GI_GAN",
        "root": str(ROOT),
        "python": sys.version.split()[0],
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "required_files": len(REQUIRED),
        "missing_files": missing,
        "layout_pass": not missing,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
        print("PASS" if not missing else "FAIL: restore the missing repository files")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())

