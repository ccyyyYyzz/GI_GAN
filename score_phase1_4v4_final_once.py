from __future__ import annotations

import argparse

from src.phase1_4v4b0_scoring import FINAL_CONFIRM_TOKEN, score_final_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4V4 UID-safe one-shot final scorer.")
    parser.add_argument("--dataset-scope", choices=["dev", "final"], required=True)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--scoring-protocol-hash", default="")
    parser.add_argument("--incident-override", default="")
    parser.add_argument("--device", default="cuda")
    parser.epilog = (
        "Final scoring requires --dataset-scope final --confirm "
        f"{FINAL_CONFIRM_TOKEN} --scoring-protocol-hash <exact hash>. "
        "Phase 1.4V4-B0 only exercises --dataset-scope dev and refusal guards."
    )
    return parser.parse_args()


def main() -> int:
    return score_final_once(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
