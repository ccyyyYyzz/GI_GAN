from __future__ import annotations

import argparse

import torch

from .phase53C_common import (
    ProjectionConditionedCritic,
    add_common_args,
    collect_exact_null_pair_dataset,
    configure_task,
    copy_checkpoint_for_manifest,
    exact_null_component,
    finalize_session,
    hard_audit,
    load_generator,
    make_exact_null_inputs,
    make_loader,
    prepare_exact_projector,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .metrics import batch_metrics
from .phase53B_common import anchor_from_y
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53C Session23 exact-null critic evaluator.")
    add_common_args(parser)
    parser.add_argument("--eval_batches", type=int, default=8)
    return parser.parse_args()


def _score(critic, measurement, Q, u, anchor, device):
    with torch.no_grad():
        p0 = exact_null_component(measurement, u, Q)
        return critic(torch.cat([p0.to(device), anchor.to(device)], dim=1)).detach().cpu()


@torch.no_grad()
def _append_rows(rows, task, family, method, score, pred, target, measurement, y):
    metrics = batch_metrics(torch.clamp(pred, 0, 1), target, measurement, y)
    rel = relmeas_tensor(measurement, measurement.flatten_img(pred.float()), y)
    rows.append(
        {
            "task": task,
            "family": family,
            "method": method,
            "critic_score_mean": float(score.mean()),
            "critic_score_std": float(score.std(unbiased=False)),
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "rel_meas_err": float(rel.mean().cpu()),
        }
    )


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        Q, _checks = prepare_exact_projector(measurement, task_out)
        train_data = collect_exact_null_pair_dataset(config, measurement, Q, device)
        critic = ProjectionConditionedCritic()
        critic, _metrics, _labels, _scores = train_binary_critic(
            critic,
            make_exact_null_inputs(train_data, device),
            train_data["label"],
            epochs=args.critic_epochs,
            lr=args.critic_lr,
            device=device,
        )
        critic.eval()
        generator = load_generator(info, config, measurement, device)
        loader = make_loader(config, device)
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.eval_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            if x.shape[0] < 2:
                continue
            y = measurement.measure(x)
            anchor = anchor_from_y(measurement, y, config)
            xhat, _xdata, _extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config.get("use_null_project", True)),
                use_dc_project=True,
                use_final_dc_project=True,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            xj = torch.roll(x, shifts=1, dims=0)
            cross = torch.clamp(hard_audit(measurement, xj, y), 0.0, 1.0)
            methods = {
                "GT": x,
                "full_model_reconstruction": xhat,
                "BP_anchor": torch.clamp(anchor, 0.0, 1.0),
                "feasible_hallucination": cross,
            }
            for method, pred in methods.items():
                score = _score(critic, measurement, Q, pred, anchor, device)
                _append_rows(rows, task, info["metadata"]["display"], method, score, pred, x, measurement, y)
    write_rows(out, "critic_evaluator_scores", rows, "Critic Evaluator Scores")
    save_bar_plot(out / "critic_score_by_method.png", rows, "method", "critic_score_mean", "Critic score by method", "score")
    save_bar_plot(out / "critic_score_vs_psnr.png", rows, "method", "psnr", "Critic score vs PSNR proxy", "PSNR")
    save_bar_plot(out / "critic_score_vs_relmeaserr.png", rows, "method", "rel_meas_err", "Critic score vs RelMeasErr proxy", "RelMeasErr")
    save_bar_plot(out / "critic_score_for_feasible_hallucinations.png", rows, "method", "critic_score_mean", "Feasible hallucination critic score", "score")
    report = [
        "# Critic Evaluator Report",
        "",
        "This session uses an exact-null critic as an evaluator, not a generator training loss.",
        "If the critic scores distinguish feasible hallucinations from GT while RelMeasErr is similar, the two-track framework is supported.",
        "If the score only tracks PSNR or class, downgrade the claim.",
    ]
    (out / "CRITIC_EVALUATOR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "exact_null_critic_evaluator",
            "trains_generator": False,
            "trains_critic": True,
            "results_csv": str(out / "critic_evaluator_scores.csv"),
        },
    )


if __name__ == "__main__":
    main()
