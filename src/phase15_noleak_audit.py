from __future__ import annotations

import json

from .phase15_common import METHODS, PHASE15, ensure_dir, method_import_dir, read_yaml, write_csv, write_json, write_md_table


FIELDS = [
    "method_id",
    "strict_noleak_claimed",
    "evidence_found",
    "train_split",
    "val_split",
    "test_split",
    "checkpoint_selection_split",
    "test_monitored_during_training",
    "final_test_only",
    "risk_level",
    "paper_safe",
    "notes",
]


def audit_one(method: dict) -> dict:
    method_id = method["method_id"]
    out = method_import_dir(method)
    cfg = read_yaml(out / "resolved_config.yaml")
    epochs = int(cfg.get("epochs", 0) or 0)
    eval_before = bool(cfg.get("eval_before_training", True))
    eval_every = int(cfg.get("eval_every", 0) or 0)
    save_every = int(cfg.get("save_every", 0) or 0)
    no_eval_during_training = (not eval_before) and eval_every > epochs
    no_periodic_best = save_every > epochs
    has_last = (out / "last.pt").exists()
    has_metrics = (out / "eval_metrics.json").exists()
    has_config = bool(cfg)
    best_files = list(out.glob("best_*.pt"))
    evidence = []
    if has_config:
        evidence.append("resolved_config.yaml")
    if not eval_before:
        evidence.append("eval_before_training=false")
    if eval_every > epochs:
        evidence.append("eval_every>epochs")
    if save_every > epochs:
        evidence.append("save_every>epochs")
    if has_last:
        evidence.append("last.pt final checkpoint")
    if has_metrics:
        evidence.append("post-training eval_metrics.json")
    if not best_files:
        evidence.append("no best_*.pt checkpoint in imported strict package")
    paper_safe = bool(has_config and has_last and has_metrics and no_eval_during_training and no_periodic_best)
    risk_level = "low" if paper_safe else ("unknown" if not has_config else "medium")
    notes = "Strict no-leak evidence is sufficient." if paper_safe else "Missing or incomplete no-leak evidence."
    if best_files:
        notes += " best_*.pt files present; verify they are not test-selected before paper use."
    return {
        "method_id": method_id,
        "strict_noleak_claimed": True,
        "evidence_found": "; ".join(evidence),
        "train_split": "training split only",
        "val_split": "not monitored during training" if no_eval_during_training else "unknown/possibly monitored",
        "test_split": "final post-training evaluation",
        "checkpoint_selection_split": "none; last.pt final endpoint" if no_periodic_best else "unknown",
        "test_monitored_during_training": False if no_eval_during_training else "unknown",
        "final_test_only": bool(no_eval_during_training and has_metrics),
        "risk_level": risk_level,
        "paper_safe": paper_safe,
        "notes": notes,
    }


def main() -> None:
    out_dir = ensure_dir(PHASE15 / "noleak_audit")
    rows = [audit_one(method) for method in METHODS]
    write_csv(out_dir / "noleak_audit.csv", rows, FIELDS)
    write_md_table(out_dir / "noleak_audit.md", rows, FIELDS)
    write_json(out_dir / "noleak_audit.json", rows)
    print(json.dumps({"audit_rows": len(rows), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
