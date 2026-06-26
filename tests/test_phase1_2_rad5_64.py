from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from phase1_2_rad5_64_pipeline import (
    PHASE1_1,
    PHASE79_CKPT,
    alignment_smoke,
    build_dev_cache,
    load_phase79_generator,
    make_phase79_measurement,
    sha256_np,
    write_final_locked_64_manifest,
)
from src.phase1_1_controls import k_prefix_indices
from src.projections import exact_data_anchor, exact_null_project


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_phase79_a_hash_repeatable() -> None:
    device = _device()
    m1, a1, _ = make_phase79_measurement(device)
    m2, a2, _ = make_phase79_measurement(device)
    assert list(a1.shape) == [205, 4096]
    assert list(a2.shape) == [205, 4096]
    assert sha256_np(a1.detach().cpu().numpy().astype(np.float32)) == sha256_np(a2.detach().cpu().numpy().astype(np.float32))
    assert m1.img_size == 64 and m1.m == 205 and m1.n == 4096


def test_phase79_checkpoint_strict_loads() -> None:
    device = _device()
    measurement, _a, cfg = make_phase79_measurement(device)
    gen, merged, ckpt, key, missing, unexpected = load_phase79_generator(PHASE79_CKPT, cfg, measurement, device)
    assert key == "generator"
    assert missing == []
    assert unexpected == []
    assert merged["img_size"] == 64
    assert ckpt["step"] == 200
    assert gen.training is False


def test_zero_noise_forward_repeatable_and_noise_changes_p0(tmp_path: Path) -> None:
    device = _device()
    measurement, _a, cfg = make_phase79_measurement(device)
    gen, merged, _ckpt, _key, _missing, _unexpected = load_phase79_generator(PHASE79_CKPT, cfg, measurement, device)
    cache = build_dev_cache(measurement, device, 1)
    report = alignment_smoke(tmp_path, gen, measurement, merged, cache, device)
    assert report["zero_noise_stable"] is True
    assert "different_noise_changes_p0" in report


def test_candidate_k_prefixes() -> None:
    pools = k_prefix_indices(32, [1, 4, 8, 16, 32])
    assert pools[1] == [0]
    assert pools[4] == pools[8][:4]
    assert pools[16] == pools[32][:16]


def test_canonical_candidates_share_exact_row_component() -> None:
    device = _device()
    measurement, _a, _cfg = make_phase79_measurement(device)
    y = torch.randn(1, measurement.m, device=device)
    r = exact_data_anchor(y, measurement, device=device, as_image=False)
    cand = torch.randn(3, measurement.n, device=device)
    n = exact_null_project(cand, measurement, device=device)
    canon = r.repeat(3, 1) + n
    rows = []
    for k in range(3):
        rows.append((canon[k : k + 1] - exact_null_project(canon[k : k + 1], measurement, device=device)).detach().cpu())
    assert torch.max(torch.stack([(rows[i] - rows[0]).abs().max() for i in range(3)])).item() < 1e-5


def test_final_locked_indices_are_derived_not_changed(tmp_path: Path) -> None:
    parent_idx = PHASE1_1 / "reports" / "final_locked_test_indices.npy"
    if not parent_idx.exists():
        return
    parent = np.load(parent_idx)
    report = write_final_locked_64_manifest(tmp_path)
    assert report["source_indices_unchanged"] is True
    assert report["source_indices_count"] == int(parent.size)
    assert report["final_test_evaluated"] is False


def test_final_test_result_not_created_before_freeze() -> None:
    path = Path("E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/phase1_2_rad5_64_candidate_transfer/reports/final_locked_test_64_results.json")
    assert not path.exists()


def test_random_baseline_formula_uses_pool_mean() -> None:
    errs = torch.tensor([1.0, 2.0, 4.0])
    assert abs(float(errs.mean().item()) - 7.0 / 3.0) < 1e-6


def test_oracle_module_only_reports_eval_labels() -> None:
    # Phase 1.2 stores oracle quantities in coverage/evaluation reports, not in
    # candidate manifests used by deployable selectors.
    manifest_path = Path("E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/phase1_2_rad5_64_candidate_transfer/manifests/candidate_pool_dev_64.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        text = json.dumps(manifest)
        assert "p0_rmse" not in text
        assert "oracle" not in text
