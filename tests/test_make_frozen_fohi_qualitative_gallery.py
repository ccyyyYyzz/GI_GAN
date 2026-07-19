import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualitative_gallery", ROOT / "make_frozen_fohi_qualitative_gallery.py"
)
gallery = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gallery
SPEC.loader.exec_module(gallery)


def samples():
    return [
        {"label": 6, "source_index": 17, "raw_sha256": "horse-later"},
        {"label": 2, "source_index": 8, "raw_sha256": "car-later"},
        {"label": 5, "source_index": 5, "raw_sha256": "dog-earlier"},
        {"label": 3, "source_index": 4, "raw_sha256": "cat-earlier"},
        {"label": 2, "source_index": 1, "raw_sha256": "car-earlier"},
        {"label": 6, "source_index": 3, "raw_sha256": "horse-earlier"},
        {"label": 5, "source_index": 9, "raw_sha256": "dog-later"},
        {"label": 3, "source_index": 6, "raw_sha256": "cat-later"},
    ]


def test_selection_is_class_and_minimum_source_index_only():
    selected = gallery.select_records_rate05(samples())
    assert [(row.label, row.source_index, row.local_index_rate05) for row in selected] == [
        (2, 1, 4),
        (3, 4, 3),
        (5, 5, 2),
        (6, 3, 5),
    ]


def test_same_raw_record_is_required_at_second_rate():
    selected = gallery.select_records_rate05(samples())
    reordered = list(reversed(samples()))
    indices = gallery.local_indices_for_rate(selected, reordered)
    assert [reordered[index]["source_index"] for index in indices] == [1, 4, 5, 3]
    broken = [dict(row) for row in reordered]
    broken[indices[0]]["raw_sha256"] = "not-the-same-image"
    with pytest.raises(RuntimeError, match="RATE_SAMPLE_IDENTITY_MISMATCH"):
        gallery.local_indices_for_rate(selected, broken)


def test_freeze_parameters_are_exact_and_reject_mutation():
    manifest = {
        "status": "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN",
        "method_parameters": dict(gallery.REQUIRED_PARAMETERS),
    }
    gallery.require_frozen_parameters(manifest)
    manifest["method_parameters"]["alpha"] = 0.6
    with pytest.raises(RuntimeError, match="FROZEN_PARAMETER_MISMATCH"):
        gallery.require_frozen_parameters(manifest)


def test_metric_loader_uses_recorded_local_index_without_selection(tmp_path):
    import numpy as np

    selected = gallery.select_records_rate05(samples())
    pairs = [(arm, metric) for arm in ("structural", "fixed", "fohi") for metric in gallery.METRICS]
    vectors = {
        f"{arm}_{metric}": np.arange(8, dtype=np.float64) + offset
        for offset, (arm, metric) in enumerate(pairs)
    }
    path = tmp_path / "metric_vectors.npz"
    np.savez_compressed(path, **vectors)
    rows = gallery.load_metric_rows(path, selected, [4, 3, 2, 5], "05")
    assert [row["local_index"] for row in rows] == [4, 3, 2, 5]
    assert rows[0]["structural_psnr"] == pytest.approx(4.0)


def test_final_fiber_target_comes_from_raw_cached_y_not_clipped_anchor():
    import torch

    geometry = gallery.GaugeGeometry.from_rows_qr(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    )
    pack = {
        "y": torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        "x0": torch.tensor([[[[0.0, 0.0, 0.0]]]], dtype=torch.float32),
    }
    actual = gallery.raw_fiber_intrinsic(pack, geometry)
    expected = geometry.intrinsic_record(pack["y"])
    anchor_target = pack["x0"].flatten(1).to(torch.float64) @ geometry.Q.T
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, anchor_target)


def test_raw_y_code_changes_are_explicit_and_all_other_frozen_code_stays_pinned(
    tmp_path, monkeypatch
):
    marker = "/content/GI_GAN"
    changed = sorted(gallery.RAW_Y_INTENTIONALLY_CHANGED_PATHS)
    unchanged = "src/metrics.py"
    for relative in [*changed, unchanged]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"current {relative}\n", encoding="utf-8")
    manifest = {
        "code_sha256": {
            f"{marker}/{relative}": "old-raw-y-hash" for relative in changed
        }
    }
    manifest["code_sha256"][f"{marker}/{unchanged}"] = gallery.sha256(tmp_path / unchanged)
    monkeypatch.setattr(gallery, "git_head", lambda root: "current-head")
    monkeypatch.setattr(gallery, "git_file_is_clean", lambda root, relative: True)

    receipt = gallery.verify_frozen_code(tmp_path, manifest)

    assert receipt["current_git_head"] == "current-head"
    assert set(receipt["intentional_round59_raw_y_code"]) == set(changed)
    assert receipt["unchanged_round52_code_sha256"][str(tmp_path / unchanged)] == gallery.sha256(
        tmp_path / unchanged
    )


def test_raw_y_code_change_must_be_clean_at_current_head(tmp_path, monkeypatch):
    marker = "/content/GI_GAN"
    changed = sorted(gallery.RAW_Y_INTENTIONALLY_CHANGED_PATHS)
    for relative in changed:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("current\n", encoding="utf-8")
    manifest = {"code_sha256": {f"{marker}/{relative}": "old" for relative in changed}}
    monkeypatch.setattr(gallery, "git_head", lambda root: "current-head")
    monkeypatch.setattr(
        gallery, "git_file_is_clean", lambda root, relative: str(relative).replace("\\", "/") != changed[0]
    )
    with pytest.raises(RuntimeError, match="RAW_Y_CODE_NOT_CLEAN_AT_HEAD"):
        gallery.verify_frozen_code(tmp_path, manifest)


def test_round60_launcher_uses_round59_raw_y_receipt_and_new_output_tree():
    source = (ROOT / "colab" / "round58_make_qualitative_gallery.py").read_text(encoding="utf-8")
    assert 'ROUND59_ROOT = Path("/content/gan_r59_raw_fiber/lane0")' in source
    assert 'OUTPUT = Path("/content/gan_r60_qualitative_gallery/lane0")' in source
    assert '"--source-round56-lane"' in source
    assert '"--round59-lane"' in source


def test_round60_core_reconstruction_code_must_match_round59_receipt(tmp_path):
    expected = {}
    for relative in gallery.ROUND59_RECONSTRUCTION_CORE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"core {relative}\n", encoding="utf-8")
        expected[relative] = gallery.sha256(path)
    receipt = {"code_sha256": expected}

    verified = gallery.verify_round59_reconstruction_core(tmp_path, receipt)

    assert verified == expected
    changed = tmp_path / gallery.ROUND59_RECONSTRUCTION_CORE[0]
    changed.write_text("post-round59 mutation\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ROUND59_CORE_CODE_HASH_MISMATCH"):
        gallery.verify_round59_reconstruction_core(tmp_path, receipt)
