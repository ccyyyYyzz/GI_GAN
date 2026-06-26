from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase2_locked_protocol import DEFAULT_CONFIG, run_locked_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Phase 2 add-on witnessed-selection locked-test protocol metadata "
            "without loading truth, generating candidates, or computing locked metrics."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config to freeze and preflight.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_locked_preflight(Path(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

