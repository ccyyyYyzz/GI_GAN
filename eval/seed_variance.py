"""Aggregate gate quantities across three or more seed result dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checker import check_results


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else key, child, out)
    elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        out[prefix] = float(value)


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in reports:
        flat: dict[str, float] = {}
        _flatten_numeric("metrics", report.get("metrics", {}), flat)
        for gate, gate_report in report["gates"].items():
            _flatten_numeric(f"gates.{gate}", gate_report.get("values", {}), flat)
            flat[f"gates.{gate}.passed"] = 1.0 if gate_report["passed"] else 0.0
        rows.append(flat)
    keys = sorted({key for row in rows for key in row})
    summary = {}
    for key in keys:
        vals = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": int(len(vals))}
    return {"n_seeds": len(reports), "summary": summary, "overall_pass_rate": float(np.mean([r["overall_passed"] for r in reports]))}


def format_summary(agg: dict[str, Any]) -> str:
    lines = ["quantity                                      mean +/- std"]
    for key, val in agg["summary"].items():
        lines.append(f"{key:<45} {val['mean']:.6g} +/- {val['std']:.6g}")
    lines.append(f"overall_pass_rate                            {agg['overall_pass_rate']:.6g}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", help="Seed result dumps (.npz/.pt)")
    parser.add_argument("--perceptual-backend", choices=("lpips", "mse", "edge_mse"), default="lpips")
    parser.add_argument("--no-fid-kid", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    reports = [
        check_results(path, perceptual_backend=args.perceptual_backend, compute_distributional=not args.no_fid_kid)
        for path in args.dumps
    ]
    agg = aggregate_reports(reports)
    print(format_summary(agg))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
