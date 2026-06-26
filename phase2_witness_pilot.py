from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase2_witness import create_brief_package, run_witness_pilot, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 add-on witnessed candidate-selection development pilot."
    )
    parser.add_argument(
        "--config",
        default="configs/compatibility/phase2_witness_pilot.yaml",
        help="YAML config path. Defaults to configs/compatibility/phase2_witness_pilot.yaml.",
    )
    parser.add_argument(
        "--make-brief-package",
        action="store_true",
        help="Create a small GPT-facing zip with only the decision/audit/gate artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_witness_pilot(Path(args.config))
    package_info = None
    if args.make_brief_package:
        package = create_brief_package(Path(summary["output_dir"]))
        package_info = {"path": str(package), "sha256": sha256_file(package)}
    payload = {
        "status": summary["status"],
        "output_dir": summary["output_dir"],
        "gate_decision": summary["gate"]["decision"],
        "package": package_info,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
