from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.checker import check_results, pass_fail_table
from eval.seed_variance import aggregate_reports


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def _project(P0: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values @ P0.T


def _scale_to_mse(values: np.ndarray, target_mse: float) -> np.ndarray:
    current = float(np.mean(values**2))
    if current <= 0:
        raise ValueError("Cannot scale a zero-valued array")
    return values * np.sqrt(target_mse / current)


def _checker_kwargs() -> dict:
    return {"perceptual_backend": "edge_mse", "compute_distributional": False}


def _status_row(report: dict) -> dict[str, str]:
    return {gate: result["status"] for gate, result in report["gates"].items()}


def _gate_matrix(reports: dict[str, dict]) -> str:
    gates = ["G-CAL", "G-DIV", "G-NVR", "G-MEAN", "G-CERT", "G-PERC"]
    lines = ["case          " + "  ".join(f"{gate:>6}" for gate in gates)]
    for case, report in reports.items():
        statuses = [report["gates"][gate]["status"] for gate in gates]
        lines.append(f"{case:<13} " + "  ".join(f"{status:>6}" for status in statuses))
    return "\n".join(lines)


def _make_images(rng: np.random.Generator, n_images: int) -> np.ndarray:
    images = []
    yy, xx = np.mgrid[0:8, 0:8]
    for idx in range(n_images):
        img = 0.25 + 0.04 * xx + 0.03 * yy
        img[(xx >= 2 + idx % 3) & (xx <= 5) & (yy >= 1) & (yy <= 5 + idx % 2)] += 0.22
        img[(yy - 3.5) ** 2 + (xx - (2.5 + 0.4 * idx)) ** 2 < 5.0] += 0.12
        img += 0.015 * rng.normal(size=img.shape)
        images.append(np.clip(img, 0.05, 0.95))
    return np.asarray(images, dtype=np.float64).reshape(n_images, -1)


def _make_matrix(rng: np.random.Generator, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    raw = rng.normal(size=(n, m))
    q, _ = np.linalg.qr(raw)
    A = q[:, :m].T
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    # With orthonormal rows A^+ A = A.T A. The SVD call above is intentional:
    # the synthetic setup exercises an exact-SVD construction as requested.
    P0 = np.eye(n) - vt.T @ vt
    return A, P0


def _high_frequency_error(target_mse: float, n_images: int) -> np.ndarray:
    yy, xx = np.mgrid[0:8, 0:8]
    del yy
    base = np.where((xx // 2) % 2 == 0, 1.0, -1.0).reshape(1, -1)
    raw = np.repeat(base, n_images, axis=0)
    for i in range(n_images):
        raw[i] = np.roll(raw[i], i, axis=0)
    return _scale_to_mse(raw, target_mse)


def write_synthetic_case(root: Path, variant: str = "pass", seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n_images, k, n, m = 6, 80, 64, 8
    A, P0 = _make_matrix(rng, m, n)
    x = _make_images(rng, n_images)
    target_mean_mse = 4e-3
    noise_factor = 1.0
    baseline_factor = 5.0
    noise_mode = "edge"
    ref_multiplier = 0.25
    baseline_mode = "random"

    if variant == "gcal_far":
        noise_factor = 5.0
        baseline_factor = 10.0
    elif variant == "gcal_close":
        noise_factor = 0.08
    elif variant == "gdiv":
        noise_mode = "flat"
    elif variant == "gnvr":
        ref_multiplier = 8.0
    elif variant == "gmean":
        baseline_factor = 0.85
        baseline_mode = "high_frequency"
    elif variant == "gperc":
        baseline_factor = 0.98
    elif variant == "gcert":
        pass
    elif variant != "pass":
        raise ValueError(variant)

    mean_raw = _project(P0, rng.normal(size=(n_images, n)))
    mean_err = _scale_to_mse(mean_raw, target_mean_mse)
    sample_mean = x + mean_err

    raw_noise = np.empty((n_images, k, n), dtype=np.float64)
    for i in range(n_images):
        if noise_mode == "edge":
            amp = _sobel_magnitude(x[i].reshape(8, 8))
            amp = 0.15 + amp / max(float(amp.max()), 1e-12)
            raw = rng.normal(size=(k, 8, 8)) * amp[None, :, :]
        else:
            raw = rng.normal(size=(k, 8, 8))
        raw = raw.reshape(k, n)
        raw -= raw.mean(axis=0, keepdims=True)
        raw_noise[i] = _project(P0, raw)
    noise = _scale_to_mse(raw_noise, target_mean_mse * noise_factor)
    samples = sample_mean[:, None, :] + noise
    samples_unclipped = samples.copy()

    if baseline_mode == "high_frequency":
        baseline = x + _high_frequency_error(target_mean_mse * baseline_factor, n_images)
    else:
        baseline_raw = _project(P0, rng.normal(size=(n_images, n)))
        baseline = x + _scale_to_mse(baseline_raw, target_mean_mse * baseline_factor)
    ref_x = 0.5 + ref_multiplier * (x - np.mean(x, axis=0, keepdims=True))

    y = x @ A.T
    if variant == "gcert":
        noise_vec = rng.normal(size=n)
        # A has orthonormal rows, so A_pinv = A.T. This is the range-space
        # perturbation A_pinv @ (A @ noise) requested for the certificate fail.
        rowspace_bump = A.T @ (A @ noise_vec)
        rowspace_bump = rowspace_bump / max(float(np.linalg.norm(rowspace_bump)), 1e-12) * 1e-3
        samples_unclipped[0, 0] = samples_unclipped[0, 0] + rowspace_bump

    A_path = root / f"{variant}_A.npz"
    P0_path = root / f"{variant}_P0.npz"
    dump_path = root / f"{variant}_dump.npz"
    np.savez(A_path, A=A)
    np.savez(P0_path, P0=P0)
    np.savez(
        dump_path,
        x=x,
        samples=samples,
        samples_unclipped=samples_unclipped,
        sample_mean=sample_mean,
        baseline=baseline,
        y=y,
        ref_x=ref_x,
        A_path=str(A_path),
        P0_path=str(P0_path),
    )
    return dump_path


class SyntheticGateTests(unittest.TestCase):
    def test_pass_case_passes_all_gates_and_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = write_synthetic_case(Path(tmp), "pass")
            report = check_results(dump, **_checker_kwargs())
            self.assertTrue(report["overall_passed"], pass_fail_table(report))
            self.assertEqual({gate["status"] for gate in report["gates"].values()}, {"PASS"})
            self.assertIn("G-CAL", pass_fail_table(report))

    def test_each_gate_has_a_targeted_failure_case_and_no_cross_gate_failures(self):
        expected = {
            "gcal_close": "G-CAL",
            "gcal_far": "G-CAL",
            "gdiv": "G-DIV",
            "gnvr": "G-NVR",
            "gmean": "G-MEAN",
            "gcert": "G-CERT",
            "gperc": "G-PERC",
        }
        with tempfile.TemporaryDirectory() as tmp:
            reports = {}
            for variant, gate in expected.items():
                dump = write_synthetic_case(Path(tmp), variant)
                report = check_results(dump, **_checker_kwargs())
                reports[variant] = report
                statuses = _status_row(report)
                expected_statuses = {name: "PASS" for name in statuses}
                expected_statuses[gate] = "FAIL"
                self.assertEqual(
                    expected_statuses,
                    statuses,
                    f"{variant} did not fail only {gate}\n{_gate_matrix(reports)}\n{pass_fail_table(report)}",
                )
            print("\n" + _gate_matrix(reports))

    def test_gcal_failure_directions_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            close = check_results(write_synthetic_case(Path(tmp), "gcal_close"), **_checker_kwargs())
            far = check_results(write_synthetic_case(Path(tmp), "gcal_far"), **_checker_kwargs())
            self.assertLess(close["gates"]["G-CAL"]["values"]["sample_mean_minus_sample_gap_db"], 1.0)
            self.assertGreater(far["gates"]["G-CAL"]["values"]["sample_mean_minus_sample_gap_db"], 3.5)

    def test_seed_variance_aggregates_numeric_gate_quantities(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = [
                check_results(write_synthetic_case(Path(tmp), "pass", seed=seed), **_checker_kwargs())
                for seed in range(3)
            ]
            agg = aggregate_reports(reports)
            self.assertEqual(agg["n_seeds"], 3)
            self.assertEqual(agg["overall_pass_rate"], 1.0)
            self.assertIn("gates.G-CAL.avg_sample_psnr_db", agg["summary"])


if __name__ == "__main__":
    unittest.main()
