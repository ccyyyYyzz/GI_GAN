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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inventories) != 3:
        raise ValueError("EXACTLY_THREE_FROZEN_LANE_INVENTORIES_REQUIRED")
    loaded = []
    for path in args.inventories:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "FOHI_HELDOUT_LANE_INVENTORY_COMPLETE"
            or payload.get("test_split_opened") is not False
            or payload.get("repo_dirty") is not False
        ):
            raise RuntimeError(f"INVALID_HELDOUT_INVENTORY:{path}")
        loaded.append((path, payload))
    lanes = {int(payload["lane_index"]): payload for _, payload in loaded}
    if set(lanes) != {0, 1, 2}:
        raise RuntimeError("HELDOUT_LANE_SET_MISMATCH")
    heads = {payload["repo_head"] for payload in lanes.values()}
    if len(heads) != 1:
        raise RuntimeError("HELDOUT_CODE_COMMIT_DRIFT")
    canonical_code = lanes[0]["code_sha256"]
    if any(payload["code_sha256"] != canonical_code for payload in lanes.values()):
        raise RuntimeError("HELDOUT_CODE_HASH_DRIFT")
    if len({payload["rates"]["05"]["operator_sha256"] for payload in lanes.values()}) != 3:
        raise RuntimeError("FIVE_PERCENT_OPERATORS_NOT_DISTINCT")

    manifest: dict[str, Any] = {
        "status": "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN",
        "method": "VQGAN-guided fiber-orthogonal high-pass innovation",
        "test_split_opened": False,
        "validation_only": False,
        "repo_head_at_inventory": next(iter(heads)),
        "source_split": "STL10 test",
        "expected_test_images": 8000,
        "rates": ["05", "10"],
        "excluded_rate": "02",
        "method_parameters": {
            "proposal_source": "pretrained VQGAN",
            "adapter_source_arm": "gan",
            "adapter_adversarial_weight": 0.0,
            "adapter_rotation_scale": 0.5,
            "adapter_lpips_weight": 0.003,
            "filter_mode": "highpass",
            "cutoff": 0.12,
            "transition": 0.03,
            "alpha": 0.5,
            "exact_projection_iterations": 4096,
        },
        "bootstrap": {
            "reps": 20000,
            "seed": 20260719,
            "scheme": "lane-by-image hierarchical paired bootstrap",
        },
        "decision_rule": {
            "headline": "six-component intersection-union gate",
            "components": [
                "rate05_psnr_ci_low_gt_0",
                "rate05_ssim_ci_low_gt_0",
                "rate05_lpips_ci_high_lt_0",
                "rate10_psnr_ci_low_gt_0",
                "rate10_ssim_ci_low_gt_0",
                "rate10_lpips_ci_high_lt_0",
            ],
            "additional_requirements": [
                "all_lane_means_favorable",
                "all_projection_and_hash_gates_pass",
                "holm_adjusted_one_sided_p_values_reported",
            ],
        },
        "code_sha256": canonical_code,
        "lanes": {
            str(index): {
                "label": lanes[index]["label"],
                "artifact_sha256": lanes[index]["artifact_sha256"],
                "rates": lanes[index]["rates"],
            }
            for index in (0, 1, 2)
        },
        "inventory_sha256": {
            str(path): sha256(path) for path, _ in loaded
        },
        "provenance": {
            "gpt_pro_round52_commit": "f46c1b65ff7ec42e866f9835d77a06cd0ab6d450",
            "theory_response_sha256": "3ea6f862ddf8e98c34ac08710770ad256ec3683a3b23dfce3a564e74595f13ab",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
