from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("psnr", "ssim", "lpips")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crossed_bootstrap(
    deltas: dict[str, list[np.ndarray]], *, reps: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    seed_count = len(next(iter(deltas.values())))
    draws = {metric: np.empty(int(reps), dtype=np.float64) for metric in METRICS}
    for replicate in range(int(reps)):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        for metric in METRICS:
            seed_means = []
            for selected in selected_seeds:
                values = deltas[metric][int(selected)]
                indices = rng.integers(0, len(values), size=len(values))
                seed_means.append(float(values[indices].mean()))
            draws[metric][replicate] = float(np.mean(seed_means))
    result: dict[str, Any] = {}
    for metric in METRICS:
        values = np.concatenate(deltas[metric])
        result[metric] = {
            "mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(draws[metric], 0.025)),
            "ci95_high": float(np.quantile(draws[metric], 0.975)),
        }
    return result


def direct_ci_favorable(paired: dict[str, Any]) -> bool:
    return bool(
        paired["psnr"]["ci95_low"] > 0.0
        and paired["ssim"]["ci95_low"] > 0.0
        and paired["lpips"]["ci95_high"] < 0.0
    )


def favorable_means(paired: dict[str, Any]) -> bool:
    return bool(
        paired["psnr"]["mean_delta"] > 0.0
        and paired["ssim"]["mean_delta"] > 0.0
        and paired["lpips"]["mean_delta"] < 0.0
    )


def paired_image_bootstrap(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = len(candidate["psnr"])
    indices = rng.integers(0, n, size=(int(reps), n))
    result = {}
    for metric in METRICS:
        delta = candidate[metric] - reference[metric]
        bootstrap = delta[indices].mean(axis=1)
        result[metric] = {
            "mean_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != 3:
        raise ValueError("EXACTLY_THREE_FROZEN_PAIRINGS_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    seed_vectors = []
    input_hashes = {}
    deltas = {metric: [] for metric in METRICS}
    for directory in args.input_dirs:
        summary_path = directory / "summary.json"
        vectors_path = directory / "metric_vectors.npz"
        if not summary_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(f"MISSING_EQ_INPUT:{directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("validation_only") is not True or summary.get("test_split_opened") is not False:
            raise RuntimeError(f"SPLIT_PROTOCOL_VIOLATION:{directory}")
        vectors = np.load(vectors_path)
        arm_vectors = {
            arm: {
                metric: np.asarray(vectors[f"{arm}_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            for arm in ("structural", "raw_fohi", "eq_fohi")
        }
        for metric in METRICS:
            deltas[metric].append(
                arm_vectors["eq_fohi"][metric] - arm_vectors["raw_fohi"][metric]
            )
        summaries.append(summary)
        seed_vectors.append(arm_vectors)
        input_hashes[directory.name] = {
            "summary_sha256": sha256(summary_path),
            "metric_vectors_sha256": sha256(vectors_path),
        }

    direct = crossed_bootstrap(
        deltas, reps=int(args.bootstrap_reps), seed=int(args.seed)
    )
    per_seed_results = []
    for index, (directory, summary, vectors) in enumerate(
        zip(args.input_dirs, summaries, seed_vectors)
    ):
        eq_vs_structural = paired_image_bootstrap(
            vectors["eq_fohi"],
            vectors["structural"],
            reps=10000,
            seed=int(args.seed) + index,
        )
        eq_vs_raw = paired_image_bootstrap(
            vectors["eq_fohi"],
            vectors["raw_fohi"],
            reps=10000,
            seed=int(args.seed) + 100 + index,
        )
        gate = bool(
            summary["all_projection_certificates_pass"]
            and favorable_means(eq_vs_structural)
            and favorable_means(eq_vs_raw)
            and direct_ci_favorable(eq_vs_structural)
        )
        per_seed_results.append(
            {
                "pairing": directory.name,
                "eq_fohi_vs_structural": eq_vs_structural,
                "eq_fohi_vs_raw_fohi": eq_vs_raw,
                "gate": gate,
            }
        )
    per_seed_gate = all(item["gate"] for item in per_seed_results)
    crossed_gate = direct_ci_favorable(direct)
    replace = bool(per_seed_gate and crossed_gate)
    payload = {
        "status": "ENDPOINT_QUOTIENTED_FOHI_CROSSED_SEED_DECISION",
        "validation_only": True,
        "test_split_opened": False,
        "pairings": [directory.name for directory in args.input_dirs],
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.seed),
        "input_hashes": input_hashes,
        "per_seed_results": per_seed_results,
        "per_seed_gate": per_seed_gate,
        "crossed_eq_fohi_vs_raw_fohi": direct,
        "crossed_direct_triple_ci_favorable": crossed_gate,
        "replace_fohi_with_eq_fohi": replace,
        "decision": "FREEZE_EQ_FOHI" if replace else "KEEP_FOHI_KILL_EQ_FOHI",
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Endpoint-quotiented FOHI decision",
        "",
        f"Decision: **{payload['decision']}**.",
        "",
        "The held-out test remains unopened. The decision uses exactly three frozen validation pairings and no retuning.",
        "",
        "## Crossed seed-by-image direct contrast: EQ-FOHI minus FOHI",
        "",
        "| Metric | Mean delta | 95% interval | Favorable direction |",
        "|---|---:|---:|---|",
    ]
    for metric in METRICS:
        item = direct[metric]
        direction = "positive" if metric != "lpips" else "negative"
        lines.append(
            f"| {metric.upper()} | {item['mean_delta']:.8f} | "
            f"[{item['ci95_low']:.8f}, {item['ci95_high']:.8f}] | {direction} |"
        )
    lines.extend(
        [
            "",
            f"Per-seed gate: **{per_seed_gate}**. Crossed direct gate: **{crossed_gate}**.",
            "",
            "The endpoint quotient is not rescued by coefficient, cutoff, norm, or tangent-cone changes if this gate fails.",
        ]
    )
    (args.output_dir / "FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
