from __future__ import annotations

from .phase60_common import OUT_ROOT, ensure_dir, read_json, save_json


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    safety = read_json(out / "g2_safety_status.json")
    safe_to_run = bool(safety.get("safe_to_run", False))

    if not safe_to_run:
        status = {
            "phase": 60,
            "script": "phase60_train_g2_null_gan",
            "status": "skipped_unsafe_to_run",
            "trained_any_model": False,
            "trained_gan": False,
            "trained_reconstruction_network": False,
            "checkpoint_modified": False,
            "main_results_unchanged": True,
            "reasons": safety.get("reasons", []),
        }
        save_json(status, out / "g2_training_status.json")
        lines = [
            "# G2 Null-Gauge GAN Training Status",
            "",
            "Status: **skipped_unsafe_to_run**",
            "",
            "No GAN, reconstruction network, or checkpoint was trained or modified.",
            "",
            "## Reasons",
            "",
        ]
        lines.extend([f"- {reason}" for reason in status["reasons"]] or ["- Safety gate returned unsafe_to_run."])
        (out / "g2_training_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    status = {
        "phase": 60,
        "script": "phase60_train_g2_null_gan",
        "status": "blocked_manual_review_required",
        "trained_any_model": False,
        "trained_gan": False,
        "trained_reconstruction_network": False,
        "checkpoint_modified": False,
        "main_results_unchanged": True,
        "reasons": [
            "Safety checks passed unexpectedly in this local package, but the training implementation is intentionally blocked for manual review before running any new GAN optimization."
        ],
    }
    save_json(status, out / "g2_training_status.json")
    (out / "g2_training_status.md").write_text(
        "# G2 Null-Gauge GAN Training Status\n\n"
        "Status: **blocked_manual_review_required**\n\n"
        "No model was trained. This guard prevents accidental GAN optimization without explicit review of the exact split artifacts and stochastic branch implementation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
