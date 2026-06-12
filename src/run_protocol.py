"""Run-ID, output-dir, and checkpoint discipline for the g2r_ run series.

Hardens against two real incidents:
  (b) a GAN pilot's post-training checkpoint was never saved — fixed by
      ``CheckpointManager``: periodic auto-save every N steps plus a forced
      final save on exit (including exceptions) via try/finally semantics;
  and against accidental clobbering of paper-1 artifacts — fixed by
      ``validate_output_dir``, which refuses to write into output dirs of
      prior paper-1 runs.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any, Callable


RUN_ID_PREFIX = "g2r_"
_RUN_ID_RE = re.compile(r"^g2r_[A-Za-z0-9][A-Za-z0-9_\-]*$")

# Output trees of prior paper-1 runs. Any resolved output path containing one
# of these components (or component pairs) is refused for new runs.
_PAPER1_COMPONENT_RES = (
    re.compile(r"^outputs_phase\d+.*$", re.IGNORECASE),          # outputs_phase15, outputs_phase60_gan_...
    re.compile(r"^outputs_clean_phase\d+$", re.IGNORECASE),      # outputs_clean_phase2
    re.compile(r"^colab_run_package$", re.IGNORECASE),
    re.compile(r"^imported_noleak$", re.IGNORECASE),
    re.compile(r"^cert_package_\d+$", re.IGNORECASE),
)


class RunProtocolError(RuntimeError):
    """Raised on run-ID / output-dir protocol violations."""


def validate_run_id(run_id: str | None) -> str:
    if not run_id or not isinstance(run_id, str):
        raise RunProtocolError(
            f"Run ID is required for the g2r_ series and must start with {RUN_ID_PREFIX!r}; got {run_id!r}."
        )
    if not _RUN_ID_RE.match(run_id):
        raise RunProtocolError(
            f"Run ID {run_id!r} violates the g2r protocol: it must match {_RUN_ID_RE.pattern!r}."
        )
    return run_id


def is_paper1_output_dir(path: str | Path) -> bool:
    parts = [p for p in Path(path).resolve().parts]
    for i, part in enumerate(parts):
        for pattern in _PAPER1_COMPONENT_RES:
            if pattern.match(part):
                return True
        # repo-local paper-1 outputs: <repo>/outputs/phaseNN...
        if part.lower() == "outputs" and i + 1 < len(parts) and parts[i + 1].lower().startswith("phase"):
            return True
    return False


def validate_output_dir(output_dir: str | Path, run_id: str | None = None) -> Path:
    """Refuse paper-1 output trees and foreign non-empty directories.

    A valid g2r output dir (1) is not inside any prior paper-1 output tree,
    (2) has a leaf name starting with ``g2r_``, and (3) if it already exists
    and is non-empty, its leaf must equal the run ID (resume of the same run).
    """
    path = Path(output_dir)
    resolved = path.resolve()
    if is_paper1_output_dir(resolved):
        raise RunProtocolError(
            f"Refusing to write into a prior paper-1 output dir: {resolved}. "
            "g2r_ runs must use fresh output directories."
        )
    leaf = resolved.name
    if not leaf.startswith(RUN_ID_PREFIX):
        raise RunProtocolError(
            f"Output dir leaf {leaf!r} must start with {RUN_ID_PREFIX!r} (got path {resolved})."
        )
    if run_id is not None:
        validate_run_id(run_id)
        if resolved.exists() and any(resolved.iterdir()) and leaf != run_id:
            raise RunProtocolError(
                f"Output dir {resolved} already contains files for a different run "
                f"(leaf {leaf!r} != run_id {run_id!r}); refusing to overwrite."
            )
    return resolved


def enforce_run_protocol(output_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hard protocol gate used at training startup.

    Always refuses paper-1 output dirs.  When the run is part of the g2r_
    series (run_id in config, or output-dir leaf prefixed g2r_), additionally
    enforces the run-ID pattern and forbids ``val_split == "test"`` so the
    test split can only be touched by the single final evaluation.
    """
    resolved = Path(output_dir).resolve()
    if is_paper1_output_dir(resolved):
        raise RunProtocolError(
            f"Refusing to write into a prior paper-1 output dir: {resolved}."
        )
    config = config or {}
    run_id = config.get("run_id")
    is_g2r = bool(run_id) or resolved.name.startswith(RUN_ID_PREFIX)
    if is_g2r:
        run_id = validate_run_id(run_id or resolved.name)
        validate_output_dir(resolved, run_id)
        if str(config.get("val_split", "test")) == "test":
            raise RunProtocolError(
                "g2r protocol: val_split must not be 'test' during training "
                "(use sample-disjoint train-side splits, e.g. train_split: "
                "'unlabeled' with val_split: 'train'; the resolved loaders are "
                "additionally verified at startup, and the test split is "
                "evaluated exactly once at the end)."
            )
    return {"output_dir": str(resolved), "run_id": run_id, "g2r_protocol_enforced": is_g2r}


class CheckpointManager:
    """Periodic + guaranteed-final checkpoint saving.

    Usage::

        def save_fn(path: Path, context: dict) -> None:
            torch.save({"generator": g.state_dict(), **context}, path)

        with CheckpointManager(out_dir, run_id="g2r_pilot01", save_fn=save_fn,
                               save_every_steps=200) as ckpt:
            for batch in loader:
                ...optimizer step...
                ckpt.step()

    ``save_fn`` is called with the target path and a context dict containing
    ``run_id``, ``step``, ``reason`` and (on exception exit) ``exception``.
    The final save happens in ``__exit__`` even when the block raises, and the
    exception is never suppressed.
    """

    FINAL_NAME = "final.pt"

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        save_fn: Callable[[Path, dict[str, Any]], None],
        save_every_steps: int = 200,
        validate_dir: bool = True,
    ) -> None:
        self.run_id = validate_run_id(run_id)
        self.output_dir = (
            validate_output_dir(output_dir, run_id) if validate_dir else Path(output_dir).resolve()
        )
        if int(save_every_steps) <= 0:
            raise RunProtocolError("save_every_steps must be a positive integer.")
        self.save_every_steps = int(save_every_steps)
        self.save_fn = save_fn
        self.global_step = 0
        self._final_saved = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _context(self, reason: str, exception: BaseException | None = None) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "run_id": self.run_id,
            "step": self.global_step,
            "reason": reason,
        }
        if exception is not None:
            ctx["exception"] = "".join(
                traceback.format_exception_only(type(exception), exception)
            ).strip()
        return ctx

    def save(self, name: str, reason: str, exception: BaseException | None = None) -> Path:
        path = self.output_dir / name
        self.save_fn(path, self._context(reason, exception))
        return path

    def step(self, n: int = 1) -> None:
        """Advance the step counter; auto-save every ``save_every_steps``."""
        self.global_step += int(n)
        if self.global_step % self.save_every_steps == 0:
            self.save(f"step_{self.global_step:08d}.pt", reason="periodic")

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        reason = "final" if exc is None else "final_on_exception"
        if exc is None:
            self.save(self.FINAL_NAME, reason=reason)
            self._final_saved = True
        else:
            # An exception is already in flight: attempt the forced final save
            # but never let a save failure mask the original error.
            try:
                self.save(self.FINAL_NAME, reason=reason, exception=exc)
                self._final_saved = True
            except Exception:
                pass
        return False  # never suppress exceptions
