from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src import phase1_4b_scoring as s


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")


def test_guard_rejects_final_without_confirm() -> None:
    args = argparse.Namespace(dataset_scope="final", confirm="", protocol_freeze_hash="", incident_override="")
    ok, reason = s.guard_stage_b_v2(args)
    assert not ok
    assert reason == "MISSING_OR_INVALID_CONFIRM_TOKEN"


def test_guard_allows_dev_scope_without_final_truth() -> None:
    args = argparse.Namespace(dataset_scope="dev", confirm="", protocol_freeze_hash="", incident_override="")
    ok, reason = s.guard_stage_b_v2(args)
    assert ok
    assert reason == "DEV_SCOPE_ALLOWED"


def test_stage_b_script_no_confirm_refuses_and_does_not_start() -> None:
    started = ROOT / "outputs" / "compatibility" / "phase1_4a_final_freeze_and_blind" / "final_scoring_v2" / "FINAL_SCORING_STARTED.json"
    before = started.exists()
    res = subprocess.run(
        [sys.executable, str(ROOT / "score_phase1_4b_final_once_v2.py"), "--dataset-scope", "final"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode != 0
    assert "REFUSING" in res.stdout
    assert started.exists() == before


def test_stage_b_script_dev_scope_is_guard_only() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "score_phase1_4b_final_once_v2.py"), "--dataset-scope", "dev"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0
    assert "DEV_SCOPE_OK" in res.stdout


def test_b0_mode_never_accepts_final_scope_without_hash() -> None:
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "score_phase1_4b_final_once_v2.py"),
            "--dataset-scope",
            "final",
            "--confirm",
            s.FINAL_CONFIRM_TOKEN,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode != 0
    assert "REFUSING" in res.stdout
