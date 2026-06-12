from __future__ import annotations

import argparse
from pathlib import Path

from .phase53B_common import (
    BlindCriticSmall,
    ProjectionConditionedCritic,
    add_common_args,
    collect_pair_dataset,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    make_image_inputs,
    resolve_device,
    save_bar_plot,
    save_score_histogram,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53B Session20 blind critic separability pretest.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    hist_saved = False
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        dataset = collect_pair_dataset(config, measurement, device, "audited_cross_pair")
        labels = dataset["label"]
        for model_name, model in [
            ("blind_critic_small", BlindCriticSmall()),
            ("blind_critic_projection_conditioned", ProjectionConditionedCritic()),
        ]:
            inputs = make_image_inputs(dataset, measurement, "blind", device)
            trained, metrics, val_labels, val_scores = train_binary_critic(
                model,
                inputs,
                labels,
                epochs=args.critic_epochs,
                lr=args.critic_lr,
                device=device,
            )
            row = {
                "task": task,
                "family": info["metadata"]["display"],
                "sampling_pct": info["metadata"]["sampling_pct"],
                "model": model_name,
                "negative_mode": "audited_cross_pair",
                **metrics,
                "n_pairs": int(labels.numel()),
                "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
            }
            rows.append(row)
            save_score_histogram(task_out / f"{model_name}_score_histogram.png", val_labels, val_scores, f"{task} {model_name}")
            if not hist_saved:
                save_score_histogram(out / "score_histograms.png", val_labels, val_scores, f"{task} {model_name}")
                hist_saved = True
            save_json(row, task_out / f"{model_name}_metrics.json")
            trained.cpu()
    write_rows(out, "blind_critic_pretest_results", rows, "Blind Critic Pretest Results")
    save_bar_plot(out / "auc_by_family.png", rows, "task", "auc", "Blind critic AUC by family", "AUC")
    save_bar_plot(out / "condition_ablation.png", rows, "model", "auc", "Condition coupling ablation", "AUC")
    report = [
        "# Blind Critic Separability Pretest Report",
        "",
        "This session trains only discriminators/classifiers. It does not train the generator.",
        "",
        "The blind critic input is restricted to `[P_N u, x_data]`; it never receives `Au-y`, RelMeasErr, delta, or audit displacement.",
        "",
        "Success screen: Scr-5 AUC >= 0.65 or Scr-10 clearly > 0.65. Rad-5 near 0.5 should be interpreted as a weak anchor, not total failure.",
        "",
        "See `blind_critic_pretest_results.csv` for metrics.",
    ]
    (out / "BLIND_CRITIC_PRETEST_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "blind_critic_pretest",
            "trains_generator": False,
            "trains_discriminator": True,
            "results_csv": str(out / "blind_critic_pretest_results.csv"),
        },
    )


if __name__ == "__main__":
    main()
