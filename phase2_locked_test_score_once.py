from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase2_locked_protocol import CONFIRM_TOKEN, DEFAULT_CONFIG, score_locked_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 2 add-on witnessed-selection locked test exactly once. "
            "Requires a READY preflight marker, matching protocol hash, and confirm token."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Locked-test YAML config.")
    parser.add_argument("--protocol-hash", required=True, help="Protocol hash from READY marker.")
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"Must equal {CONFIRM_TOKEN}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = score_locked_once(
        Path(args.config),
        confirm=args.confirm,
        protocol_hash=args.protocol_hash,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

