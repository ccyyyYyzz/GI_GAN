"""Run-ID / output-dir / checkpoint-discipline tests with deliberate violations."""

import json

import pytest

from src.run_protocol import (
    CheckpointManager,
    RunProtocolError,
    enforce_run_protocol,
    is_paper1_output_dir,
    validate_output_dir,
    validate_run_id,
)


def test_run_id_valid():
    assert validate_run_id("g2r_pilot01") == "g2r_pilot01"
    assert validate_run_id("g2r_scr5_beta-1e-4") == "g2r_scr5_beta-1e-4"


def test_violation_run_id_wrong_prefix():
    for bad in ["phase60_g2", "g2_pilot", "G2R_pilot", "", None, "g2r_"]:
        with pytest.raises(RunProtocolError):
            validate_run_id(bad)


def test_violation_paper1_output_dirs_refused(tmp_path):
    # Real paper-1 trees (string-level checks; nothing is written).
    for bad in [
        r"E:\ns_mc_gan_gi\outputs_phase15\imported_noleak\g2r_run",
        r"E:\ns_mc_gan_gi\outputs_phase60_gan_sampling_mode_g2\g2r_run",
        r"E:\ns_mc_gan_gi\outputs_clean_phase2\g2r_run",
        r"E:\ns_mc_gan_gi\colab_run_package\g2r_run",
        r"E:\ns_mc_gan_gi\results\cert_package_20260612\g2r_run",
        str(tmp_path / "outputs" / "phase14" / "g2r_run"),
        str(tmp_path / "outputs_phase8_hq" / "g2r_run"),
    ]:
        assert is_paper1_output_dir(bad), bad
        with pytest.raises(RunProtocolError):
            validate_output_dir(bad, "g2r_run")


def test_violation_leaf_without_prefix_refused(tmp_path):
    with pytest.raises(RunProtocolError):
        validate_output_dir(tmp_path / "my_run", "g2r_run1")


def test_violation_foreign_nonempty_dir_refused(tmp_path):
    other = tmp_path / "g2r_other"
    other.mkdir()
    (other / "last.pt").write_text("x", encoding="utf-8")
    with pytest.raises(RunProtocolError):
        validate_output_dir(other, "g2r_run1")
    # Resuming the same run is fine.
    assert validate_output_dir(other, "g2r_other") == other.resolve()


def test_fresh_g2r_dir_accepted(tmp_path):
    target = tmp_path / "g2r_run1"
    assert validate_output_dir(target, "g2r_run1") == target.resolve()


def test_violation_enforce_blocks_val_split_test_for_g2r(tmp_path):
    with pytest.raises(RunProtocolError):
        enforce_run_protocol(tmp_path / "g2r_run1", {"run_id": "g2r_run1", "val_split": "test"})
    # val_split omitted counts as the default "test" -> also refused.
    with pytest.raises(RunProtocolError):
        enforce_run_protocol(tmp_path / "g2r_run1", {"run_id": "g2r_run1"})


def test_enforce_passes_for_g2r_with_heldout_val(tmp_path):
    info = enforce_run_protocol(tmp_path / "g2r_run1", {"run_id": "g2r_run1", "val_split": "train"})
    assert info["g2r_protocol_enforced"] is True
    assert info["run_id"] == "g2r_run1"


def test_enforce_legacy_run_only_blocks_paper1(tmp_path):
    # Non-g2r runs: only the paper-1 refusal applies.
    info = enforce_run_protocol(tmp_path / "scratch", {})
    assert info["g2r_protocol_enforced"] is False
    with pytest.raises(RunProtocolError):
        enforce_run_protocol(tmp_path / "outputs_phase15" / "rerun", {})


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def _json_save_fn(path, context):
    path.write_text(json.dumps(context), encoding="utf-8")


def test_periodic_saves_every_n_steps(tmp_path):
    out = tmp_path / "g2r_run1"
    with CheckpointManager(out, "g2r_run1", _json_save_fn, save_every_steps=3) as ckpt:
        for _ in range(7):
            ckpt.step()
    assert (out / "step_00000003.pt").exists()
    assert (out / "step_00000006.pt").exists()
    assert not (out / "step_00000007.pt").exists()
    final = json.loads((out / "final.pt").read_text(encoding="utf-8"))
    assert final["reason"] == "final"
    assert final["step"] == 7
    assert final["run_id"] == "g2r_run1"


def test_violation_exception_still_forces_final_save(tmp_path):
    # Deliberate violation of a clean exit: the crash that historically lost
    # the post-training checkpoint must still produce a final save.
    out = tmp_path / "g2r_run1"
    with pytest.raises(RuntimeError, match="boom"):
        with CheckpointManager(out, "g2r_run1", _json_save_fn, save_every_steps=100) as ckpt:
            ckpt.step()
            raise RuntimeError("boom")
    final = json.loads((out / "final.pt").read_text(encoding="utf-8"))
    assert final["reason"] == "final_on_exception"
    assert "boom" in final["exception"]


def test_violation_manager_rejects_bad_run_id(tmp_path):
    with pytest.raises(RunProtocolError):
        CheckpointManager(tmp_path / "g2r_x", "phase60_x", _json_save_fn)


def test_violation_manager_rejects_bad_dir(tmp_path):
    with pytest.raises(RunProtocolError):
        CheckpointManager(tmp_path / "not_prefixed", "g2r_run1", _json_save_fn)


def test_violation_manager_rejects_nonpositive_interval(tmp_path):
    with pytest.raises(RunProtocolError):
        CheckpointManager(tmp_path / "g2r_x", "g2r_x", _json_save_fn, save_every_steps=0)
