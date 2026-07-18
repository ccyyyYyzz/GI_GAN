from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_split_opened") is not False:
        raise RuntimeError(f"TEST_SPLIT_NOT_CLOSED:{path}")
    if payload.get("validation_only") is not True:
        raise RuntimeError(f"NOT_VALIDATION_ONLY:{path}")
    if payload.get("fohi_triple_ci_favorable_vs_structural") is not True:
        raise RuntimeError(f"FOHI_TRIPLE_CI_GATE_FAILED:{path}")
    if payload.get("parallel_energy_mean_at_least_one_percent") is not True:
        raise RuntimeError(f"FOHI_PRACTICALLY_INERT:{path}")
    for name in ("structural_projection_audit", "fohi_projection_audit"):
        audit = payload[name]
        if audit.get("all_converged") is not True:
            raise RuntimeError(f"PROJECTION_NOT_CONVERGED:{name}:{path}")
        if float(audit["max_relative_record_error"]) > 1.0e-7:
            raise RuntimeError(f"RECORD_ERROR_GATE_FAILED:{name}:{path}")
        if float(audit["max_box_violation"]) > 0.0:
            raise RuntimeError(f"BOX_GATE_FAILED:{name}:{path}")
    return payload


def freeze(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise RuntimeError(f"THREE_SEEDS_REQUIRED:{len(paths)}")
    records = [(path.parent.name, path, load(path)) for path in paths]
    operator_hashes = {payload["operator_sha256"] for _, _, payload in records}
    settings = {
        (payload["cutoff"], payload["transition"], payload["alpha"])
        for _, _, payload in records
    }
    if len(operator_hashes) != 1:
        raise RuntimeError("OPERATOR_MISMATCH")
    if settings != {(0.12, 0.03, 0.5)}:
        raise RuntimeError(f"FOHI_SETTING_MISMATCH:{settings}")
    per_seed = {}
    for label, path, payload in records:
        per_seed[label] = {
            "validation_images": payload["validation_images"],
            "structural_means": payload["structural_means"],
            "fixed_means": payload["fixed_means"],
            "fohi_means": payload["fohi_means"],
            "fohi_vs_structural": payload["fohi_vs_structural"],
            "fohi_vs_fixed": payload["fohi_vs_fixed"],
            "fohi_mse_vs_structural": payload["fohi_mse_vs_structural"],
            "parallel_energy_fraction": payload["parallel_energy_fraction"],
            "frozen_alpha_inside_mse_improvement_interval_fraction": payload[
                "frozen_alpha_inside_mse_improvement_interval_fraction"
            ],
            "relative_orthogonality_residual": payload[
                "relative_orthogonality_residual"
            ],
            "fohi_projection_audit": payload["fohi_projection_audit"],
            "source_sha256": sha256(path),
        }
    return {
        "status": "FOHI_FROZEN_BEFORE_HELD_OUT_TEST",
        "method": "fiber-orthogonal high-pass innovation",
        "formula": "v = u - <u,c_S>/||c_S||^2 c_S; x_hat = Pi_F(x_B + c_S + alpha v)",
        "frozen_cutoff": 0.12,
        "frozen_transition": 0.03,
        "frozen_alpha": 0.5,
        "operator_sha256": next(iter(operator_hashes)),
        "gate": {
            "triple_ci_vs_structural_on_every_seed": True,
            "parallel_energy_mean_at_least_one_percent_on_every_seed": True,
            "exact_projection_converged_on_every_seed": True,
        },
        "per_seed": per_seed,
        "supersedes_validation_choice": {
            "method": "unorthogonalized fixed frequency fusion",
            "cutoff": 0.18,
            "alpha": 0.58,
            "reason": "FOHI removes structural rescaling and passes the preregistered three-seed falsification gate without retuning",
        },
        "validation_only": True,
        "test_split_opened": False,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FOHI three-seed freeze",
        "",
        "Fiber-orthogonal high-pass innovation (FOHI) is frozen before the held-out test. It removes, image by image, the component of the filtered GAN correction that is parallel to the structural correction. No norm restoration or learned gate is used.",
        "",
        "Frozen setting: cutoff = 0.12, transition = 0.03, alpha = 0.50.",
        "",
        "| validation pairing | removed parallel energy | delta PSNR (95% CI) | delta SSIM (95% CI) | delta LPIPS (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in sorted(payload["per_seed"].items()):
        paired = row["fohi_vs_structural"]
        parallel = row["parallel_energy_fraction"]["mean"]
        cells = []
        for metric in ("psnr", "ssim", "lpips"):
            value = paired[metric]
            cells.append(
                f"{value['mean_delta']:+.6f} [{value['ci95_low']:+.6f}, {value['ci95_high']:+.6f}]"
            )
        lines.append(
            f"| {label} | {100.0 * parallel:.1f}% | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    lines.extend(
        [
            "",
            "Every seed passes the paired three-metric confidence-interval gate against the matched non-GAN structural control. Every exact box-fiber projection converges with zero box violation and relative record error below 1.00e-7.",
            "",
            "This freeze supersedes the validation-only unorthogonalized setting (cutoff 0.18, alpha 0.58). The replacement is based solely on the pre-test FOHI falsification rule and introduces no new fitted parameter.",
            "",
            f"Operator SHA-256: `{payload['operator_sha256']}`.",
            "",
            "Source summary hashes:",
            "",
        ]
    )
    for label, row in sorted(payload["per_seed"].items()):
        lines.append(f"- `{label}`: `{row['source_sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "freeze.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "FREEZE.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["gate"], sort_keys=True))
    print(f"WROTE {args.output_dir / 'freeze.json'}")
    print(f"WROTE {args.output_dir / 'FREEZE.md'}")


if __name__ == "__main__":
    main()
