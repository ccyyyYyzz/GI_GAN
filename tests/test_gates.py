"""gates.yaml pre-registration and checker tests with deliberate violations."""

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "check_gates", REPO_ROOT / "scripts" / "g2r" / "check_gates.py"
)
check_gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_gates)

GATES = yaml.safe_load((REPO_ROOT / "gates.yaml").read_text(encoding="utf-8"))

PASSING_METRICS = {
    "per_sample_psnr_offset_db": [-2.0, -1.5, -3.2, -1.1],
    "median_per_pixel_sample_std": 0.02,
    "std_edge_correlation": 0.55,
    "null_variance_ratio": 0.35,
    "psnr_mean_of_samples": 22.10,
    "psnr_deterministic_baseline": 22.27,
    "rel_meas_err_unclipped_max_float64": 3.0e-11,
    "lpips_sample_mean": 0.21,
    "lpips_deterministic_baseline": 0.25,
    "fid": 41.0,
    "kid": 0.012,
    "split_guard_active": True,
    "test_eval_count": 1,
}


def test_gates_yaml_preregisters_all_seven_gates():
    names = set(GATES["gates"].keys())
    assert names == {"G-CAL", "G-DIV", "G-NVR", "G-MEAN", "G-CERT", "G-PERC", "G-PROTO"}
    assert GATES["run_series"] == "g2r_"
    g = GATES["gates"]
    assert g["G-CAL"]["band"] == [-3.5, -1.0]
    assert g["G-DIV"]["min"] == 1e-2
    assert g["G-NVR"]["min"] == 0.1
    assert g["G-MEAN"]["max_drop_db"] == 0.3
    assert g["G-CERT"]["max"] == 1e-10
    assert g["G-PERC"]["direction"] == "lower_is_better"
    assert g["G-PROTO"]["require"] == {"split_guard_active": True, "test_eval_count": 1}


def _statuses(metrics):
    return {r["gate"]: r["status"] for r in check_gates.evaluate_gates(GATES, metrics)}


def test_all_gates_pass_on_admissible_metrics():
    statuses = _statuses(PASSING_METRICS)
    assert all(s == "PASS" for s in statuses.values()), statuses


def test_stub_mode_reports_not_evaluated():
    statuses = _statuses(None)
    assert all(s == "NOT_EVALUATED" for s in statuses.values())


def test_violation_each_gate_fails_on_inadmissible_metrics():
    # Deliberate violations, one per gate; every gate must FAIL.
    bad = dict(PASSING_METRICS)
    bad["per_sample_psnr_offset_db"] = [-2.0, -0.5]        # G-CAL: -0.5 dB above band
    bad["median_per_pixel_sample_std"] = 1e-3              # G-DIV: collapsed diversity
    bad["null_variance_ratio"] = 0.01                      # G-NVR: below 0.1
    bad["psnr_mean_of_samples"] = 21.5                     # G-MEAN: 0.77 dB drop > 0.3
    bad["rel_meas_err_unclipped_max_float64"] = 1e-6       # G-CERT: certificate broken
    bad["lpips_sample_mean"] = 0.25                        # G-PERC: not strictly better
    bad["test_eval_count"] = 2                             # G-PROTO: test touched twice
    statuses = _statuses(bad)
    assert all(s == "FAIL" for s in statuses.values()), statuses


def test_violation_split_guard_inactive_fails_g_proto():
    bad = dict(PASSING_METRICS)
    bad["split_guard_active"] = False
    statuses = _statuses(bad)
    assert statuses["G-PROTO"] == "FAIL"


def test_g_cert_unclipped_convention_documented():
    # RelMeasErr must be specified on the UNCLIPPED vector.
    desc = GATES["gates"]["G-CERT"]["description"].lower()
    assert "unclipped" in desc
    assert "float64" in desc
