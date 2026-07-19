"""Legacy FOHI launchers must remain explicit after the target became required."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CALLERS = (
    "colab/round46_launch_rate10_reproject.py",
    "colab/round47_launch_fohi_seed0_fallback.py",
    "colab/round47_launch_fohi_common.py",
    "run_frozen_fohi_heldout_once.py",
    "run_noisy_fohi_stress.py",
    "run_fiber_rate_campaign.py",
    "run_operator_seed_campaign.py",
    "run_fohi_causal_campaign.py",
    "run_selected_fohi_extension.py",
)


def test_legacy_fohi_callers_pin_legacy_projection_target() -> None:
    for relative in LEGACY_CALLERS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "diagnose_fiber_orthogonal_highpass_innovation.py" in source
        assert "legacy_clipped_anchor" in source, relative


def test_round59_raw_runner_remains_on_raw_measurements() -> None:
    source = (ROOT / "run_frozen_fohi_raw_fiber_once.py").read_text(encoding="utf-8")
    assert 'FINAL_TARGET = "raw_y"' in source
    assert '"--final-target"' in source
