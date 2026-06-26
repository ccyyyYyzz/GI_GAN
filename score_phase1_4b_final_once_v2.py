from __future__ import annotations

import argparse

from src.phase1_4b_scoring import FINAL_CONFIRM_TOKEN, score_final_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4B one-shot final scorer v2.")
    parser.add_argument("--dataset-scope", choices=["dev", "final"], required=True)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--protocol-freeze-hash", default="")
    parser.add_argument("--incident-override", default="")
    parser.add_argument("--device", default="cuda")
    parser.epilog = (
        "Final scoring requires --dataset-scope final --confirm "
        f"{FINAL_CONFIRM_TOKEN} --protocol-freeze-hash <exact hash>. "
        "Phase 1.4B0 only uses --dataset-scope dev for guard validation."
    )
    return parser.parse_args()


def main() -> int:
    return score_final_once(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
