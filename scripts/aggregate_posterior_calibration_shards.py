from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_coverage(shards: list[Path]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, float], list[int]] = {}
    for shard in shards:
        summary = load_json(shard / "session_01_coverage" / "coverage_summary.json")
        for row in summary["coverage_curve"]:
            key = (str(row["space"]), float(row["level"]))
            acc = counts.setdefault(key, [0, 0])
            acc[0] += int(row["covered_count"])
            acc[1] += int(row["total_count"])
    rows = []
    for (space, level), (covered, total) in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        coverage = covered / max(total, 1)
        rows.append(
            {
                "space": space,
                "level": level,
                "covered_count": covered,
                "total_count": total,
                "empirical_coverage": coverage,
                "coverage_minus_nominal": coverage - level,
            }
        )
    return rows


def aggregate_weighted_means(shards: list[Path], summary_rel: Path, fields: list[str]) -> dict[str, float]:
    totals = {field: 0.0 for field in fields}
    n_total = 0
    for shard in shards:
        summary = load_json(shard / summary_rel)
        n = int(summary["sample_bank"]["N"])
        n_total += n
        for field in fields:
            totals[field] += float(summary[field]) * n
    return {field: totals[field] / max(n_total, 1) for field in fields}


def aggregate_mean_shift(shards: list[Path]) -> dict[str, float]:
    totals: dict[str, float] = {}
    n_total = 0
    for shard in shards:
        summary = load_json(shard / "session_03_mean_shift" / "mean_shift_summary.json")
        n = int(summary["sample_bank"]["N"])
        n_total += n
        for field, value in summary["means"].items():
            totals[field] = totals.get(field, 0.0) + float(value) * n
    return {field: value / max(n_total, 1) for field, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate posterior calibration shard summaries.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard", action="append", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    shards = [Path(p) for p in args.shard]
    manifests = [load_json(shard / "sample_bank_manifest.json") for shard in shards]
    n_total = sum(int(m["N"]) for m in manifests)
    k_values = sorted({int(m["K"]) for m in manifests})
    if len(k_values) != 1:
        raise ValueError(f"Mixed K values across shards: {k_values}")

    coverage = aggregate_coverage(shards)
    kappa_fields = [
        "null_mse_sample_to_gt",
        "null_mse_sample_mean_to_gt",
        "null_mse_deterministic_to_gt",
        "null_sample_width",
    ]
    kappa_means = aggregate_weighted_means(shards, Path("session_02_kappa") / "kappa_summary.json", kappa_fields)
    kappa = {
        **kappa_means,
        "kappa_vs_sample_mean": kappa_means["null_mse_sample_to_gt"] / max(kappa_means["null_mse_sample_mean_to_gt"], 1e-30),
        "kappa_vs_deterministic": kappa_means["null_mse_sample_to_gt"] / max(kappa_means["null_mse_deterministic_to_gt"], 1e-30),
        "admissible_interval": [1.0, 2.0],
    }
    mean_shift = aggregate_mean_shift(shards)
    payload = {
        "mode": "aggregate_shards",
        "N": n_total,
        "K": k_values[0],
        "shards": [
            {
                "path": str(shard),
                "sample_offset": int(manifest.get("sample_offset", 0)),
                "sample_stop": int(manifest.get("sample_stop", int(manifest.get("N", 0)))),
                "N": int(manifest["N"]),
                "samples_path": manifest["samples_path"],
                "samples_sha256": manifest["samples_sha256"],
                "sample_mean_sha256": manifest["sample_mean_sha256"],
                "relmeas_max_generated": float(manifest["relmeas_max_generated"]),
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            }
            for shard, manifest in zip(shards, manifests)
        ],
        "coverage_curve": coverage,
        "kappa": kappa,
        "mean_shift": mean_shift,
    }
    save_json(out_dir / "aggregate_calibration_summary.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
