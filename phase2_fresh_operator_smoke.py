from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase2_fresh_operator import (
    create_fixed_total_brief_package,
    create_fresh_brief_package,
    run_fixed_total_smoke,
    run_fresh_operator_smoke,
)
from src.phase2_witness import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Phase 2 fresh-context-operator witnessed-selection smoke test."
    )
    parser.add_argument(
        "--config",
        default="configs/compatibility/phase2_fresh_operator_smoke.yaml",
        help="YAML config path for the selected smoke mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["add-on", "fixed-total"],
        default="add-on",
        help="Run the add-on witness smoke or fixed-total context/witness split smoke.",
    )
    parser.add_argument(
        "--make-brief-package",
        action="store_true",
        help="Create a concise GPT-facing zip with protocol, audit, gate, and summary artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "fixed-total":
        summary = run_fixed_total_smoke(Path(args.config))
    else:
        summary = run_fresh_operator_smoke(Path(args.config))
    package_info = None
    if args.make_brief_package:
        package = (
            create_fixed_total_brief_package(Path(summary["output_dir"]))
            if args.mode == "fixed-total"
            else create_fresh_brief_package(Path(summary["output_dir"]))
        )
        package_info = {"path": str(package), "sha256": sha256_file(package)}
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": summary["output_dir"],
                "gate_decision": summary["gate"]["decision"],
                "mode": args.mode,
                "package": package_info,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
