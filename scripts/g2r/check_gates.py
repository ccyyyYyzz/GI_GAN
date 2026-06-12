"""Stub checker for the pre-registered g2r_ admissibility gates.

Loads gates.yaml and (optionally) a metrics JSON produced by a g2r_ eval run,
evaluates each gate, and prints PASS / FAIL / NOT_EVALUATED. Exit code is 1 if
any evaluated gate fails.

Usage:
    python scripts/g2r/check_gates.py                       # list gates (stub mode)
    python scripts/g2r/check_gates.py --metrics metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT_EVALUATED"


def _get(metrics: dict, key: str):
    return metrics.get(key) if metrics else None


def evaluate_gate(name: str, gate: dict, metrics: dict | None) -> tuple[str, str]:
    """Return (status, detail) for one gate against a metrics dict."""
    if metrics is None:
        return NOT_EVALUATED, "no metrics provided"

    if name == "G-CAL":
        offsets = _get(metrics, gate["metric"])
        if offsets is None:
            return NOT_EVALUATED, f"missing {gate['metric']}"
        lo, hi = float(gate["band"][0]), float(gate["band"][1])
        in_band = [lo <= float(v) <= hi for v in offsets]
        frac = sum(in_band) / max(1, len(in_band))
        ok = frac >= float(gate["min_fraction_in_band"])
        return (PASS if ok else FAIL), f"{frac:.3f} of samples in [{lo}, {hi}] dB (need >= {gate['min_fraction_in_band']})"

    if name == "G-DIV":
        med = _get(metrics, gate["metric"])
        corr = _get(metrics, gate["edge_correlation_metric"])
        if med is None or corr is None:
            return NOT_EVALUATED, "missing median std or edge correlation"
        ok = float(med) >= float(gate["min"]) and float(corr) >= float(gate["edge_correlation_min"])
        return (PASS if ok else FAIL), (
            f"median std {float(med):.3e} (min {gate['min']}), edge corr {float(corr):.3f} "
            f"(min {gate['edge_correlation_min']})"
        )

    if name == "G-NVR":
        val = _get(metrics, gate["metric"])
        if val is None:
            return NOT_EVALUATED, f"missing {gate['metric']}"
        ok = float(val) >= float(gate["min"])
        return (PASS if ok else FAIL), f"null variance ratio {float(val):.4f} (min {gate['min']})"

    if name == "G-MEAN":
        val = _get(metrics, gate["metric"])
        base = _get(metrics, gate["baseline_metric"])
        if val is None or base is None:
            return NOT_EVALUATED, "missing mean-of-samples or baseline PSNR"
        drop = float(base) - float(val)
        ok = drop <= float(gate["max_drop_db"])
        return (PASS if ok else FAIL), f"PSNR drop {drop:+.3f} dB vs baseline (max {gate['max_drop_db']} dB)"

    if name == "G-CERT":
        val = _get(metrics, gate["metric"])
        if val is None:
            return NOT_EVALUATED, f"missing {gate['metric']} (UNCLIPPED, float64 recompute)"
        ok = float(val) <= float(gate["max"])
        return (PASS if ok else FAIL), f"max unclipped RelMeasErr (float64) {float(val):.3e} (max {gate['max']:.0e})"

    if name == "G-PERC":
        val = _get(metrics, gate["metric"])
        base = _get(metrics, gate["baseline_metric"])
        if val is None or base is None:
            return NOT_EVALUATED, "missing LPIPS values"
        ok = float(val) < float(base)
        reported = {k: _get(metrics, k) for k in gate.get("report_only_metrics", [])}
        return (PASS if ok else FAIL), (
            f"LPIPS sample {float(val):.4f} vs baseline {float(base):.4f} (strictly lower required); "
            f"report-only: {reported}"
        )

    if name == "G-PROTO":
        required = gate["require"]
        missing = [k for k in required if _get(metrics, k) is None]
        if missing:
            return NOT_EVALUATED, f"missing {missing}"
        bad = {k: metrics[k] for k, v in required.items() if metrics[k] != v}
        ok = not bad
        return (PASS if ok else FAIL), ("all protocol flags as required" if ok else f"violations: {bad}")

    return NOT_EVALUATED, "unknown gate"


def evaluate_gates(gates_config: dict, metrics: dict | None) -> list[dict]:
    results = []
    for name, gate in gates_config["gates"].items():
        status, detail = evaluate_gate(name, gate, metrics)
        results.append({"gate": name, "status": status, "detail": detail})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check g2r_ admissibility gates.")
    parser.add_argument("--gates", default=str(REPO_ROOT / "gates.yaml"))
    parser.add_argument("--metrics", default=None, help="JSON file with evaluated run metrics.")
    args = parser.parse_args()

    gates_config = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8"))
    metrics = None
    if args.metrics:
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    print(f"g2r_ admissibility gates (preregistered {gates_config.get('preregistered_at')}):")
    results = evaluate_gates(gates_config, metrics)
    width = max(len(r["gate"]) for r in results)
    for r in results:
        print(f"  {r['gate']:<{width}}  {r['status']:<13}  {r['detail']}")
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_pass = sum(1 for r in results if r["status"] == PASS)
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {len(results) - n_pass - n_fail} NOT_EVALUATED.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
