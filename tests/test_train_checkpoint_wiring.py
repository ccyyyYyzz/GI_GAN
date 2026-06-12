"""Static wiring checks for train.py checkpoint discipline.

The behavioral guarantees are tested through their modules
(CheckpointManager, run_protocol, split_guard); these tests pin the train.py
integration so the wiring cannot silently regress.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SRC = (REPO_ROOT / "src" / "train.py").read_text(encoding="utf-8")


def test_epoch_loop_wrapped_in_try_finally_with_final_save():
    tree = ast.parse(TRAIN_SRC)
    main_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Try) and node.finalbody:
            has_epoch_loop = any(
                isinstance(inner, ast.For)
                and isinstance(inner.target, ast.Name)
                and inner.target.id == "epoch"
                for inner in node.body
            )
            finally_saves = any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "save_checkpoint"
                for stmt in node.finalbody
                for inner in ast.walk(stmt)
            )
            if has_epoch_loop and finally_saves:
                return
    raise AssertionError("epoch loop is not wrapped in try/finally with a forced save_checkpoint")


def test_step_periodic_save_is_configurable():
    assert 'config.get("checkpoint_every_steps"' in TRAIN_SRC
    assert "global_step % checkpoint_every_steps == 0" in TRAIN_SRC


def test_protocol_gate_runs_before_output_dir_creation():
    idx_enforce = TRAIN_SRC.index('enforce_run_protocol(config["output_dir"], config)')
    idx_ensure = TRAIN_SRC.index('output_dir = ensure_dir(config["output_dir"])')
    assert idx_enforce < idx_ensure


def test_val_split_is_config_driven():
    assert 'val_split=str(config.get("val_split", "test"))' in TRAIN_SRC


def test_mid_epoch_saves_record_last_completed_epoch():
    # Mid-epoch last.pt must record the last COMPLETED epoch so resume
    # retrains an interrupted epoch instead of skipping its remainder.
    assert "completed_epoch = start_epoch_index - 1" in TRAIN_SRC
    assert "completed_epoch = epoch" in TRAIN_SRC
    assert '{"periodic_step_save": global_step, "in_flight_epoch": epoch}' in TRAIN_SRC
    assert 'locals().get("epoch"' not in TRAIN_SRC


def test_finally_save_never_masks_inflight_exception():
    assert "exception_in_flight = sys.exc_info()[0] is not None" in TRAIN_SRC
    assert "if not exception_in_flight:" in TRAIN_SRC
