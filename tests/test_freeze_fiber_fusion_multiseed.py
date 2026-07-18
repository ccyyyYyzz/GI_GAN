import json
from pathlib import Path

import pytest

from freeze_fiber_fusion_multiseed import freeze


def candidate(cutoff: float, alpha: float, gain: float, *, ci_ok: bool = True) -> dict:
    low = gain / 2.0 if ci_ok else -gain / 2.0
    return {
        "cutoff": cutoff,
        "alpha": alpha,
        "means": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.2},
        "proposal_arm": "gan",
        "proposal_manifest": {
            "source_arm": "gan",
            "rotation_scale": 0.5,
            "adversarial_weight": 0.0015,
            "lpips_weight": 0.003,
            "step": 100,
            "channels": 32,
            "seed": 7,
        },
        "projection_audit": {"all_converged": True},
        "paired_vs_control": {
            "psnr": {"mean_delta": gain, "ci95_low": low, "ci95_high": gain * 1.5},
            "ssim": {"mean_delta": gain / 40.0, "ci95_low": low / 40.0, "ci95_high": gain / 20.0},
            "lpips": {"mean_delta": -gain / 4.0, "ci95_low": -gain / 2.0, "ci95_high": -low / 4.0},
        },
    }


def write_summary(path: Path, rows: list[dict], *, test_opened: bool = False) -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "validation_only": True,
                "test_split_opened": test_opened,
                "operator_sha256": "operator",
                "validation_images": 8,
                "exact_candidates": rows,
            }
        )
    )
    return path


def test_freeze_prefers_best_worst_seed_ci_robust_candidate(tmp_path: Path) -> None:
    paths = []
    gains = ((0.10, 0.20), (0.30, 0.15), (0.25, 0.18))
    for seed, (gain_a, gain_b) in enumerate(gains):
        paths.append(
            write_summary(
                tmp_path / f"seed{seed}" / "summary.json",
                [candidate(0.1, 0.5, gain_a), candidate(0.2, 0.5, gain_b)],
            )
        )
    payload = freeze(paths)
    assert payload["frozen_method"]["cutoff"] == 0.2
    assert payload["test_split_opened"] is False


def test_freeze_fails_closed_if_test_was_opened(tmp_path: Path) -> None:
    paths = [
        write_summary(
            tmp_path / f"seed{seed}" / "summary.json",
            [candidate(0.1, 0.5, 0.1)],
            test_opened=seed == 2,
        )
        for seed in range(3)
    ]
    with pytest.raises(RuntimeError, match="TEST_SPLIT_NOT_CLOSED"):
        freeze(paths)
