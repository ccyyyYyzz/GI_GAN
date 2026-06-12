from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from eval_sampling import save_batch_stochastic_samples, save_stochastic_samples_npz
from sampling_metrics import optional_perceptual_availability, summarize_saved_samples


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        one = save_stochastic_samples_npz(tmp / "one.npz", lambda seed: np.full((1, 4, 4), seed % 7, dtype=np.float32), k=4)
        assert one.exists()
        with np.load(one) as loaded:
            assert loaded["samples"].shape == (4, 1, 4, 4)
        batch_paths = save_batch_stochastic_samples(
            tmp / "batch",
            lambda seed: np.ones((2, 1, 4, 4), dtype=np.float32) * (seed % 5),
            image_ids=[10, 11],
            k=3,
        )
        assert len(batch_paths) == 2
        split_csv = tmp / "ids.csv"
        split_csv.write_text("image_id\n3\n1\n2\n", encoding="utf-8")
        split_json = tmp / "split_hash.json"
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "split_hash.py"),
            str(split_csv),
            "--output",
            str(split_json),
        ]
        subprocess.run(cmd, check=True)
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        assert payload["splits"][0]["count"] == 3
        A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        samples = np.array([[0.0, 0.0, 0.1, 0.2], [0.0, 0.0, 0.2, 0.3]], dtype=np.float64)
        gt = np.zeros(4, dtype=np.float64)
        mean_mode = np.array([0.0, 0.0, 0.05, 0.05], dtype=np.float64)
        y = np.zeros(2, dtype=np.float64)
        metrics = summarize_saved_samples(samples, gt, mean_mode, A, y)
        assert "kappa_proxy" in metrics
        availability = optional_perceptual_availability()
    out = {
        "status": "pass",
        "tested": [
            "eval_sampling.save_stochastic_samples_npz",
            "eval_sampling.save_batch_stochastic_samples",
            "tools/split_hash.py CLI",
            "sampling_metrics.summarize_saved_samples",
            "sampling_metrics.optional_perceptual_availability",
        ],
        "perceptual_availability": availability,
    }
    (ROOT / "INFRA_UNIT_TEST_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
