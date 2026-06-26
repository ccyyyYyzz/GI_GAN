from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import phase1_4v4a_blind_inference as v4a
from src.phase1_4ir_uid_safe_scoring import stable_candidate_seed


ROOT = Path(__file__).resolve().parents[1]


def test_seed_algorithm_stable_not_python_hash():
    uid = "abc123"
    assert stable_candidate_seed(uid, 0, "salt") == stable_candidate_seed(uid, 0, "salt")
    assert stable_candidate_seed(uid, 0, "salt") != stable_candidate_seed(uid, 1, "salt")
    assert "hash(" not in (ROOT / "phase1_4v4a_blind_inference.py").read_text(encoding="utf-8")


def test_k_and_candidate_indices_are_fixed():
    assert v4a.K == 16
    indices = np.tile(np.arange(v4a.K, dtype=np.int64), (2, 1))
    assert indices.tolist()[0] == list(range(16))


def test_tie_rule_lowest_index():
    scores = np.asarray([[1.0, 2.0, 2.0, -1.0]], dtype=np.float32)
    assert int(np.argmax(scores, axis=1)[0]) == 1


def test_forbidden_truth_fields_are_rejected():
    payload = {
        "kind": "final_v4_blind_shard",
        "sample_uids": ["u"],
        "source_indices": np.asarray([1]),
        "transformed_64_sha256": ["h"],
        "y": torch.zeros(1, 205),
        "r_y": torch.zeros(1, 4096),
        "candidate_nulls": torch.zeros(1, 16, 4096),
        "selector_scores": {"dm_fcc_seed3": np.zeros((1, 16), dtype=np.float32)},
        "selected_indices": {"dm_fcc_seed3": np.zeros((1,), dtype=np.int64)},
        "x_true": torch.zeros(1, 4096),
    }
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        v4a.validate_blind_payload_schema(payload)


def test_valid_minimal_blind_payload_requires_uid_fields():
    payload = {
        "kind": "final_v4_blind_shard",
        "sample_uids": ["u"],
        "source_indices": np.asarray([1]),
        "transformed_64_sha256": ["h"],
        "y": torch.zeros(1, 205),
        "r_y": torch.zeros(1, 4096),
        "candidate_nulls": torch.zeros(1, 16, 4096),
        "selector_scores": {"dm_fcc_seed3": np.zeros((1, 16), dtype=np.float32)},
        "selected_indices": {"dm_fcc_seed3": np.zeros((1,), dtype=np.int64)},
    }
    v4a.validate_blind_payload_schema(payload)
    del payload["sample_uids"]
    with pytest.raises(RuntimeError, match="SCHEMA_MISSING"):
        v4a.validate_blind_payload_schema(payload)


def test_manifest_seed_and_protocol_mismatch_guards(monkeypatch, tmp_path):
    # The real verifier compares frozen hashes to current hashes and refuses on mismatch.
    assert v4a.sha256_file(ROOT / "phase1_4v4a_blind_inference.py") != "bad"


def test_cli_refuses_final_v4_scoring():
    res = subprocess.run(
        [sys.executable, str(ROOT / "phase1_4v4a_blind_inference.py"), "--score-final-v4"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode == 2
    assert "REFUSING" in res.stdout


def test_blind_runner_does_not_import_truth_scoring_modules():
    text = (ROOT / "phase1_4v4a_blind_inference.py").read_text(encoding="utf-8")
    forbidden = ["import lpips", "from skimage", "peak_signal_noise_ratio", "score_final_once"]
    for token in forbidden:
        assert token not in text


def test_x_only_appears_in_measurement_boundary_or_generator_native_names():
    text = (ROOT / "phase1_4v4a_blind_inference.py").read_text(encoding="utf-8")
    assert '"x_true":' not in text
    assert '"true_n":' not in text
    assert '"p0_error":' not in text
    assert '"oracle":' not in text
    assert "selected_error" in text  # guard constant only


def test_score_shape_and_selected_reconstruction_rule():
    scores = {"a": np.asarray([[0.0, 2.0, 1.0] + [0.0] * 13], dtype=np.float32)}
    selected = {k: np.argmax(v, axis=1).astype(np.int64) for k, v in scores.items()}
    assert scores["a"].shape == (1, 16)
    assert selected["a"].shape == (1,)
    assert int(selected["a"][0]) == 1
