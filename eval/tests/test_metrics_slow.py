from __future__ import annotations

import os
import unittest

import numpy as np

from eval.metrics import fid_kid, lpips_distance, to_nchw3


try:
    import pytest
except Exception:  # pragma: no cover - pytest is optional for unittest runs
    pytest = None


if pytest is not None:
    pytestmark = pytest.mark.slow


@unittest.skipUnless(os.environ.get("RUN_SLOW_EVAL_METRICS") == "1", "set RUN_SLOW_EVAL_METRICS=1 to run real LPIPS/FID/KID")
class SlowMetricSmokeTests(unittest.TestCase):
    def test_real_lpips_fid_kid_on_grayscale_64x64(self):
        rng = np.random.default_rng(123)
        x = rng.random((8, 64, 64), dtype=np.float32)
        y = np.clip(x + 0.05 * rng.normal(size=(8, 64, 64)).astype(np.float32), 0.0, 1.0)

        replicated = to_nchw3(x)
        self.assertEqual(replicated.shape, (8, 3, 64, 64))

        lpips_vals = lpips_distance(y, x, image_shape=(64, 64), batch_size=4, device="cpu")
        self.assertEqual(lpips_vals.shape, (8,))
        self.assertTrue(np.all(np.isfinite(lpips_vals)))

        dist = fid_kid(y, x, image_shape=(64, 64), batch_size=4, device="cpu")
        self.assertIsNone(dist.warning)
        self.assertTrue(np.isfinite(dist.fid))
        self.assertTrue(np.isfinite(dist.kid_mean))
        self.assertTrue(np.isfinite(dist.kid_std))
        print(f"LPIPS shape={lpips_vals.shape} FID={dist.fid:.6g} KID={dist.kid_mean:.6g}+/-{dist.kid_std:.6g}")


if __name__ == "__main__":
    unittest.main()
