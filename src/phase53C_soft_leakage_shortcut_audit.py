from __future__ import annotations

import argparse

import torch

from .phase53B_common import collect_pair_dataset
from .phase53C_common import (
    BlindCriticSmall,
    FullShortcutCritic,
    add_common_args,
    configure_task,
    copy_checkpoint_for_manifest,
    eval_binary_critic,
    exact_null_component,
    finalize_session,
    leakage_probe,
    make_exact_null_inputs,
    make_full_shortcut_inputs,
    make_image_inputs,
    make_loader,
    prepare_exact_projector,
    resolve_device,
    save_bar_plot,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53C Session21 soft leakage and shortcut audit.")
    add_common_args(parser)
    parser.add_argument("--lambda_grid", nargs="*", type=float, default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    return parser.parse_args()


def _exact_dataset(dataset, measurement, Q, device):
    out = dict(dataset)
    out["u"] = exact_null_component(measurement, dataset["u"].to(device), Q).detach().cpu()
    return out


def _inputs(kind: str, dataset, measurement, Q, device):
    if kind == "D_full":
        return make_full_shortcut_inputs(dataset, measurement, device)
    if kind == "D_image_cond":
        return make_image_inputs(dataset, measurement, "image_cond", device)
    if kind == "D_exact_null":
        return make_exact_null_inputs(_exact_dataset(dataset, measurement, Q, device), device)
    raise ValueError(kind)


def _model(kind: str, inputs: torch.Tensor):
    if kind == "D_full":
        return FullShortcutCritic(inputs.shape[1])
    return BlindCriticSmall(in_channels=2)


def _regression_r2(A: torch.Tensor, projected: torch.Tensor, target: torch.Tensor) -> float:
    X = projected.float()
    Y = target.float()
    X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    sol = torch.linalg.lstsq(X, Y).solution
    pred = X @ sol
    ss_res = torch.sum((Y - pred) ** 2)
    ss_tot = torch.sum((Y - Y.mean(dim=0, keepdim=True)) ** 2).clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    leakage_rows = []
    shortcut_rows = []
    kinds = ["D_full", "D_image_cond", "D_exact_null"]
    tests = [
        ("train_residual_easy_test_feasible", "residual_easy_wrong_y", "audited_cross_pair"),
        ("train_feasible_test_residual_easy", "audited_cross_pair", "residual_easy_wrong_y"),
        ("equalized_residual_train_feasible_test_feasible", "audited_cross_pair", "audited_cross_pair"),
    ]
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        Q, checks = prepare_exact_projector(measurement, task_out)
        first_batch = next(iter(make_loader(config, device)))[0].to(device)
        task_leak = leakage_probe(measurement, Q, first_batch, args.lambda_grid)
        A = measurement.get_current_A().detach().float().to(device)
        flat = measurement.flatten_img(first_batch.float())
        target = (A @ flat.T).T
        for row in task_leak:
            if row["projection"] == "exact_P0":
                projected = torch.flatten(exact_null_component(measurement, first_batch, Q), 1)
            else:
                from .phase53C_common import soft_project_flat

                projected = soft_project_flat(A, flat, float(row["lambda"]))
            row.update(
                {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "recover_Au_R2": _regression_r2(A, projected, target),
                    "A_P0_norm": checks["A_P0_norm"],
                }
            )
            leakage_rows.append(row)
        datasets = {
            "audited_cross_pair": collect_pair_dataset(config, measurement, device, "audited_cross_pair"),
            "residual_easy_wrong_y": collect_pair_dataset(config, measurement, device, "residual_easy_wrong_y"),
        }
        for test_name, train_mode, eval_mode in tests:
            for kind in kinds:
                train_inputs = _inputs(kind, datasets[train_mode], measurement, Q, device)
                eval_inputs = _inputs(kind, datasets[eval_mode], measurement, Q, device)
                labels_train = datasets[train_mode]["label"]
                labels_eval = datasets[eval_mode]["label"]
                model = _model(kind, train_inputs)
                trained, train_metrics, _val_labels, _val_scores = train_binary_critic(
                    model,
                    train_inputs,
                    labels_train,
                    epochs=args.critic_epochs,
                    lr=args.critic_lr,
                    device=device,
                )
                eval_metrics = eval_binary_critic(trained, eval_inputs, labels_eval, device)
                shortcut_rows.append(
                    {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "test": test_name,
                        "model": kind,
                        "train_negative": train_mode,
                        "eval_negative": eval_mode,
                        "train_auc": train_metrics["auc"],
                        "eval_auc": eval_metrics["auc"],
                        "eval_accuracy": eval_metrics["accuracy"],
                    }
                )
                trained.cpu()
    write_rows(out, "soft_leakage_results", leakage_rows, "Soft Leakage Results")
    write_rows(out, "shortcut_audit_results", shortcut_rows, "Shortcut Audit Results")
    save_bar_plot(out / "leakage_factor_by_lambda.png", leakage_rows, "lambda", "mean_leakage_ratio", "Soft projector leakage", "||APu||/||Au||")
    save_bar_plot(out / "recover_Au_from_projected_R2.png", leakage_rows, "lambda", "recover_Au_R2", "Recover Au from projected input", "R2")
    save_bar_plot(out / "D_full_vs_exact_null_auc.png", shortcut_rows, "model", "eval_auc", "D_full vs exact-null AUC", "AUC")
    save_bar_plot(out / "D_score_vs_RelMeasErr.png", shortcut_rows, "test", "eval_auc", "Shortcut diagnostic", "AUC")
    report = [
        "# Soft Leakage and Shortcut Audit Report",
        "",
        "Part A tests exact P0 versus soft P_N^lambda leakage. Soft P_N^lambda is not used as the critic input for exact-blind claims.",
        "Part B diagnoses full-input residual shortcuts. `D_full` is diagnostic only, not a valid null-space critic.",
        "",
        "See `soft_leakage_results.csv` and `shortcut_audit_results.csv`.",
    ]
    (out / "SHORTCUT_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "soft_leakage_and_shortcut_audit",
            "trains_generator": False,
            "trains_critic": True,
            "leakage_csv": str(out / "soft_leakage_results.csv"),
            "shortcut_csv": str(out / "shortcut_audit_results.csv"),
        },
    )


if __name__ == "__main__":
    main()
