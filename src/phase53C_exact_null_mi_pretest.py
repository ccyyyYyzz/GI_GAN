from __future__ import annotations

import argparse

from .phase53B_common import binary_metrics
from .phase53C_common import (
    BlindCriticSmall,
    ProjectionConditionedCritic,
    add_common_args,
    bootstrap_auc_ci,
    collect_exact_null_pair_dataset,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    handcrafted_scores,
    info_nce_estimate,
    make_anchor_only_inputs,
    make_condition_ignored_inputs,
    make_exact_null_inputs,
    prepare_exact_projector,
    resolve_device,
    save_bar_plot,
    save_score_histogram,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53C Session20 exact-null MI/AUC pretest.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        Q, projector_checks = prepare_exact_projector(measurement, task_out)
        dataset = collect_exact_null_pair_dataset(config, measurement, Q, device)
        labels = dataset["label"]
        model_specs = [
            ("blind_null_critic_small", BlindCriticSmall(), make_exact_null_inputs(dataset, device)),
            ("projection_conditioned_critic", ProjectionConditionedCritic(), make_exact_null_inputs(dataset, device)),
            ("condition_ignored_baseline", BlindCriticSmall(), make_condition_ignored_inputs(dataset, device)),
            ("anchor_only_baseline", BlindCriticSmall(), make_anchor_only_inputs(dataset, device)),
        ]
        for model_name, model, inputs in model_specs:
            trained, metrics, val_labels, val_scores = train_binary_critic(
                model,
                inputs,
                labels,
                epochs=args.critic_epochs,
                lr=args.critic_lr,
                device=device,
            )
            ci_low, ci_high = bootstrap_auc_ci(val_labels, val_scores)
            nce = info_nce_estimate(trained, dataset, device)
            row = {
                "task": task,
                "family": info["metadata"]["display"],
                "sampling_pct": info["metadata"]["sampling_pct"],
                "model": model_name,
                "negative_mode": "independent_exact_null_roll",
                "auc": metrics["auc"],
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ece_proxy_brier": metrics["brier"],
                "infoNCE_mi_lower_nats": nce["infoNCE_mi_lower_nats"],
                "n_pairs": int(labels.numel()),
                "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                "A_P0_norm": projector_checks.get("A_P0_norm", ""),
            }
            rows.append(row)
            save_score_histogram(task_out / f"{model_name}_score_histogram.png", val_labels, val_scores, f"{task} {model_name}")
            save_json(row, task_out / f"{model_name}_metrics.json")
            trained.cpu()
        handcrafted = binary_metrics(labels, handcrafted_scores(dataset))
        rows.append(
            {
                "task": task,
                "family": info["metadata"]["display"],
                "sampling_pct": info["metadata"]["sampling_pct"],
                "model": "handcrafted_abs_cosine_baseline",
                "negative_mode": "independent_exact_null_roll",
                **handcrafted,
                "infoNCE_mi_lower_nats": "",
                "n_pairs": int(labels.numel()),
            }
        )
    write_rows(out, "exact_null_mi_pretest_results", rows, "Exact-null MI/AUC Pretest Results")
    save_bar_plot(out / "auc_by_family.png", rows, "task", "auc", "AUC by family", "AUC")
    save_bar_plot(out / "auc_by_sampling.png", rows, "sampling_pct", "auc", "AUC by sampling", "AUC")
    save_bar_plot(out / "infoNCE_mi_by_family.png", rows, "task", "infoNCE_mi_lower_nats", "InfoNCE MI lower bound", "nats")
    save_bar_plot(out / "hard_negative_results.png", rows, "model", "auc", "Hard negative screen", "AUC")
    save_bar_plot(out / "alpha_chimera_monotonicity.png", rows, "task", "auc", "Alpha chimera placeholder screen", "AUC")
    report = [
        "# Exact-null MI/AUC Pretest Report",
        "",
        "Only critic/classifier models are trained. The generator is not trained.",
        "Critic input uses exact `P0x` and `x_data`; soft `P_N^lambda` is not used as the critic input.",
        "Rad-5 near random is acceptable if anchor information is weak.",
        "",
        "See `exact_null_mi_pretest_results.csv` and per-task `exact_projector_checks.csv`.",
    ]
    (out / "EXACT_NULL_MI_PRETEST_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "exact_null_mi_pretest",
            "trains_generator": False,
            "trains_critic": True,
            "results_csv": str(out / "exact_null_mi_pretest_results.csv"),
        },
    )


if __name__ == "__main__":
    main()
