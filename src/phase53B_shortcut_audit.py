from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .phase53B_common import (
    BlindCriticSmall,
    FullShortcutCritic,
    add_common_args,
    collect_pair_dataset,
    configure_task,
    copy_checkpoint_for_manifest,
    eval_binary_critic,
    finalize_session,
    make_full_shortcut_inputs,
    make_image_inputs,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    save_score_histogram,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53B Session21 shortcut audit.")
    add_common_args(parser)
    return parser.parse_args()


def _inputs(kind: str, dataset, measurement, device):
    if kind == "D_full":
        return make_full_shortcut_inputs(dataset, measurement, device)
    if kind == "D_image_cond":
        return make_image_inputs(dataset, measurement, "image_cond", device)
    if kind == "D_blind":
        return make_image_inputs(dataset, measurement, "blind", device)
    raise ValueError(kind)


def _model(kind: str, inputs: torch.Tensor):
    if kind == "D_full":
        return FullShortcutCritic(inputs.shape[1])
    return BlindCriticSmall(in_channels=2)


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    kinds = ["D_full", "D_image_cond", "D_blind"]
    tests = [
        ("train_wrong_y_test_audited", "residual_easy_wrong_y", "audited_cross_pair"),
        ("train_audited_test_wrong_y", "audited_cross_pair", "residual_easy_wrong_y"),
        ("equalized_residual_train_audited_test_audited", "audited_cross_pair", "audited_cross_pair"),
    ]
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        datasets = {
            "audited_cross_pair": collect_pair_dataset(config, measurement, device, "audited_cross_pair"),
            "residual_easy_wrong_y": collect_pair_dataset(config, measurement, device, "residual_easy_wrong_y"),
        }
        for test_name, train_mode, eval_mode in tests:
            for kind in kinds:
                train_inputs = _inputs(kind, datasets[train_mode], measurement, device)
                eval_inputs = _inputs(kind, datasets[eval_mode], measurement, device)
                labels_train = datasets[train_mode]["label"]
                labels_eval = datasets[eval_mode]["label"]
                model = _model(kind, train_inputs)
                trained, train_metrics, val_labels, val_scores = train_binary_critic(
                    model,
                    train_inputs,
                    labels_train,
                    epochs=args.critic_epochs,
                    lr=args.critic_lr,
                    device=device,
                )
                eval_metrics = eval_binary_critic(trained, eval_inputs, labels_eval, device)
                row = {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "test": test_name,
                    "model": kind,
                    "train_negative": train_mode,
                    "eval_negative": eval_mode,
                    "train_auc": train_metrics["auc"],
                    "eval_auc": eval_metrics["auc"],
                    "eval_accuracy": eval_metrics["accuracy"],
                    "eval_precision": eval_metrics["precision"],
                    "eval_recall": eval_metrics["recall"],
                    "eval_brier": eval_metrics["brier"],
                }
                rows.append(row)
                save_score_histogram(task_out / f"{test_name}_{kind}_hist.png", val_labels, val_scores, f"{task} {test_name} {kind}")
                trained.cpu()
        with torch.no_grad():
            u_flat = measurement.flatten_img(datasets["residual_easy_wrong_y"]["u"].to(device))
            rel = relmeas_tensor(measurement, u_flat, datasets["residual_easy_wrong_y"]["y"].to(device)).detach().cpu()
        rel_rows = [{"index": i, "rel_meas_err": float(v)} for i, v in enumerate(rel[:128])]
        write_rows(task_out, "D_score_vs_RelMeasErr", rel_rows, "Residual-Easy RelMeasErr Diagnostic")
    write_rows(out, "shortcut_audit_results", rows, "Shortcut Audit Results")
    save_bar_plot(out / "full_vs_blind_auc.png", rows, "model", "eval_auc", "Full vs Blind AUC", "Eval AUC")
    save_bar_plot(out / "residual_shortcut_failure.png", rows, "test", "eval_auc", "Residual Shortcut Failure", "Eval AUC")
    report = [
        "# Shortcut Audit Report",
        "",
        "This is a diagnostic audit, not the proposed main discriminator.",
        "",
        "`D_full` intentionally receives forbidden residual features for shortcut diagnosis only. The proposed blind critic remains restricted to `[P_N u, x_data]`.",
        "",
        "Interpretation target: if `D_full` succeeds on residual-easy wrong-y but does not transfer to audited cross-pairs, full-input discrimination is a residual shortcut.",
        "",
        "See `shortcut_audit_results.csv` for the full table.",
    ]
    (out / "SHORTCUT_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "shortcut_audit",
            "trains_generator": False,
            "trains_discriminator": True,
            "results_csv": str(out / "shortcut_audit_results.csv"),
        },
    )


if __name__ == "__main__":
    main()
