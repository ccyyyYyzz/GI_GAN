from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from src import phase1_4v4b0r_complete_runner as r


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")


def test_b0r_old_runner_gap_is_real() -> None:
    old_source = (ROOT / "src" / "phase1_4v4b0_scoring.py").read_text(encoding="utf-8")
    assert "def score_final_once" in old_source
    assert "FINAL_V4_SCORING_COMPLETE" in old_source
    assert "primary_selected_p0_rmse_mean" in old_source
    assert "h2_report.json" not in old_source
    assert "per_candidate_metrics.csv" not in old_source


def test_b0r_methods_and_scientific_identities_are_frozen() -> None:
    assert r.PRIMARY_SELECTOR == "dm_fcc_seed3"
    assert "raw_fcc_seed1" in r.ALL_SELECTOR_KEYS
    assert r.METHODS == ["deterministic", "random_expectation", "posterior_mean", *r.ALL_SELECTOR_KEYS, "primary_oracle"]
    p0 = np.zeros((2, r.K), dtype=np.float64)
    selected = {key: np.zeros(2, dtype=np.int64) for key in r.ALL_SELECTOR_KEYS}
    h3 = r.h3_report(p0, selected)
    s1 = r.s1_report(p0, selected)
    h5 = r.h5_report(
        {
            "canonical_relmeaserr": np.zeros((2, r.K), dtype=np.float64),
            "native_relmeaserr": np.zeros((2, r.K), dtype=np.float64),
            "exact_row_sharing_residual": np.zeros((2, r.K), dtype=np.float64),
            "exact_null_residual": np.zeros((2, r.K), dtype=np.float64),
        },
        selected,
        {key: np.zeros(2, dtype=np.float64) for key in r.ALL_SELECTOR_KEYS},
    )
    assert h3["identity"] == "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE"
    assert h3["can_create_strong_fcc_class"] is False
    assert s1["identity"] == "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW"
    assert s1["not_H5"] is True
    assert h5["identity"] == "measurement consistency"
    assert h5["clarification"] == "PRE_TRUTH_IMPLEMENTATION_CLARIFICATION"


def test_b0r_final_guard_refuses_without_env_and_dev_is_allowed(monkeypatch) -> None:
    monkeypatch.delenv(r.ALLOW_FINAL_ENV, raising=False)
    final_args = argparse.Namespace(
        dataset_scope="final",
        confirm=r.CONFIRM_TOKEN_V2,
        parent_protocol_hash=r.EXPECTED_PARENT_HASH,
        runner_freeze_hash="bad",
        device="cpu",
    )
    ok, reason = r.guard_final_v2(final_args)
    assert not ok
    assert reason == "B0R_FINAL_SCOPE_DISABLED"
    dev_args = argparse.Namespace(dataset_scope="dev", confirm="", parent_protocol_hash="", runner_freeze_hash="", device="cpu")
    assert r.guard_final_v2(dev_args) == (True, "DEV_SCOPE_ALLOWED")


def test_b0r_cli_dev_scope_is_guard_only() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "score_phase1_4v4_final_once_v2.py"), "--dataset-scope", "dev"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0
    assert "DEV_SCOPE_OK" in res.stdout
    assert not (r.FINAL_RUN / "FINAL_V4_SCORING_STARTED.json").exists()


def test_b0r_output_schema_rejects_missing_fields(tmp_path) -> None:
    with (tmp_path / "per_candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_uid", "candidate_index"])
        writer.writeheader()
        writer.writerow({"sample_uid": "u0", "candidate_index": 0})
    with (tmp_path / "per_image_method_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_uid", "method"])
        writer.writeheader()
        writer.writerow({"sample_uid": "u0", "method": "deterministic"})
    audit = r.validate_complete_outputs(tmp_path, sample_count=1, report_path=None)
    assert audit["status"] == "FAIL"
    assert audit["per_candidate_missing_fields"]
    assert audit["per_method_missing_fields"]


def test_b0r_source_has_no_positional_truth_slice_pattern() -> None:
    source_path = ROOT / "src" / "phase1_4v4b0r_complete_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    bad_names = {"start", "count"}
    for node in ast.walk(tree):
        value_name = node.value.id if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) else ""
        assert not (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and isinstance(node.slice.lower, ast.Name)
            and node.slice.lower.id in bad_names
            and value_name in {"truth", "x_true", "truth_rows", "truth_tensor"}
        )
    text = source_path.read_text(encoding="utf-8")
    assert "x_true[start" not in text
    assert "truth_rows[start" not in text


def test_b0r_complete_requires_full_output_files(tmp_path) -> None:
    for name in [
        "per_candidate_metrics.csv",
        "per_image_method_metrics.csv",
        "all_selector_summary.csv",
        "h1_report.json",
        "h2_report.json",
        "h3_report.json",
        "h4_report.json",
        "h5_report.json",
        "s1_report.json",
        "statistics_report.json",
        "final_classification.json",
        "summary.json",
    ]:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    assert r.validate_staging_schema(tmp_path, sample_count=1)["status"] == "PASS"
    (tmp_path / "h5_report.json").unlink()
    audit = r.validate_staging_schema(tmp_path, sample_count=1)
    assert audit["status"] == "FAIL"
    assert audit["missing"] == ["h5_report.json"]
