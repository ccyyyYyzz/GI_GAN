from __future__ import annotations

import argparse

from src.phase1_4v4b0r_complete_runner import (
    CONFIRM_TOKEN_V2,
    EXPECTED_PARENT_HASH,
    score_final_once_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1.4V4-B0R complete UID-safe final-v4 scorer. "
            "The final scope is guarded by READY V2, confirm token, exact hashes, "
            "and PHASE1_4V4_B0R_ALLOW_FINAL=1."
        )
    )
    parser.add_argument("--dataset-scope", choices=["dev", "final"], required=True)
    parser.add_argument("--confirm", default="", help=f"Required for final scope: {CONFIRM_TOKEN_V2}")
    parser.add_argument("--parent-protocol-hash", default="", help=f"Required parent hash: {EXPECTED_PARENT_HASH}")
    parser.add_argument("--runner-freeze-hash", default="", help="Required FINAL_V4_SCORING_RUNNER_FROZEN_V2 hash.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    return score_final_once_v2(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
