from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from .phase53B_common import (
    ProjectionConditionedCritic,
    add_common_args,
    anchor_from_y,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    load_generator,
    make_loader,
    make_train_loader,
    null_component,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    save_image_grid,
    write_command_log,
    write_rows,
)
from .metrics import batch_metrics
from .run_protocol import CheckpointManager, enforce_run_protocol
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53B Session23 blind critic GAN pilot.")
    add_common_args(parser)
    parser.set_defaults(tasks=["scr5"])
    parser.add_argument("--beta_grid", nargs="*", type=float, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--alpha_meas", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--g_lr", type=float, default=5e-5)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    return parser.parse_args()


def _blind_input(measurement, u: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    return torch.cat([null_component(measurement, u), anchor], dim=1)


@torch.no_grad()
def _eval_generator(generator, config, measurement, loader, device, max_batches: int):
    rows = []
    grid_rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        xhat, x_data, _extras = reconstruct_from_measurements(
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
        metrics = batch_metrics(xhat, x, measurement, y)
        # RelMeasErr is ALWAYS computed on the UNCLIPPED vector.
        metrics["rel_meas_error"] = float(
            relmeas_tensor(measurement, measurement.flatten_img(_extras["x_hat_unclamped"].float()), y)
            .mean()
            .cpu()
        )
        rows.append(metrics)
        if len(grid_rows) < 6:
            grid_rows.append([x[0].detach().cpu(), x_data[0].detach().cpu(), xhat[0].detach().cpu(), torch.abs(xhat[0] - x[0]).detach().cpu()])
    out = {}
    if rows:
        for key in rows[0]:
            vals = [float(r[key]) for r in rows if key in r]
            out[key] = sum(vals) / max(1, len(vals))
    return out, grid_rows


def main() -> None:
    args = parse_args()
    # g2r protocol: run IDs prefixed g2r_, no paper-1 output dirs; training
    # in this pilot consumes the TRAIN split only ("none" = no val loader).
    enforce_run_protocol(args.output_dir, {"run_id": args.session_name, "val_split": "none"})
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    all_grid_rows = []
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        base_generator = load_generator(info, config, measurement, device)
        base_state = copy.deepcopy(base_generator.state_dict())
        for beta in args.beta_grid:
            beta_tag = f"beta_{beta:.0e}".replace("-", "m")
            beta_out = ensure_dir(task_out / beta_tag)
            generator = load_generator(info, config, measurement, device)
            generator.load_state_dict(base_state)
            generator.train()
            critic = ProjectionConditionedCritic().to(device)
            g_opt = torch.optim.AdamW(generator.parameters(), lr=args.g_lr, weight_decay=1e-5)
            d_opt = torch.optim.AdamW(critic.parameters(), lr=args.d_lr, weight_decay=1e-4)
            # TRAIN split only: the GAN update loop must never see test data.
            loader = make_train_loader(config, device)
            step = 0

            def _save_pilot_checkpoint(path, context, _generator=generator, _critic=critic, _beta=beta):
                torch.save(
                    {
                        "generator": _generator.state_dict(),
                        "critic": _critic.state_dict(),
                        "beta": _beta,
                        "config": config,
                        **context,
                    },
                    path,
                )

            with CheckpointManager(
                beta_out,
                run_id=args.session_name,
                save_fn=_save_pilot_checkpoint,
                save_every_steps=int(args.checkpoint_every),
                validate_dir=False,
            ) as ckpt:
                while step < args.max_steps:
                    for batch in loader:
                        x = batch[0].to(device, non_blocking=True)
                        y = measurement.measure(x)
                        anchor = anchor_from_y(measurement, y, config)
                        with torch.no_grad():
                            xhat_detached, _x_data, _extras = reconstruct_from_measurements(
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
                        d_pos = critic(_blind_input(measurement, x, anchor))
                        d_neg = critic(_blind_input(measurement, xhat_detached, anchor))
                        d_loss = 0.5 * (
                            F.binary_cross_entropy_with_logits(d_pos, torch.ones_like(d_pos))
                            + F.binary_cross_entropy_with_logits(d_neg, torch.zeros_like(d_neg))
                        )
                        d_opt.zero_grad(set_to_none=True)
                        d_loss.backward()
                        d_opt.step()

                        xhat, _x_data, _extras = reconstruct_from_measurements(
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
                        g_score = critic(_blind_input(measurement, xhat, anchor))
                        adv_loss = F.binary_cross_entropy_with_logits(g_score, torch.ones_like(g_score))
                        img_loss = F.l1_loss(xhat, x)
                        # RelMeasErr is ALWAYS computed on the UNCLIPPED vector.
                        rel = relmeas_tensor(
                            measurement, measurement.flatten_img(_extras["x_hat_unclamped"].float()), y
                        ).mean()
                        g_loss = img_loss + float(args.alpha_meas) * rel + float(beta) * adv_loss
                        g_opt.zero_grad(set_to_none=True)
                        g_loss.backward()
                        g_opt.step()
                        step += 1
                        ckpt.step()
                        if step >= args.max_steps:
                            break
            generator.eval()
            eval_loader = make_loader(config, device)
            metrics, grid_rows = _eval_generator(generator, config, measurement, eval_loader, device, args.eval_batches)
            torch.save({"generator": generator.state_dict(), "beta": beta, "config": config}, beta_out / "blind_critic_gan_pilot.pt")
            row = {
                "task": task,
                "family": info["metadata"]["display"],
                "beta": beta,
                "steps": args.max_steps,
                "psnr": metrics.get("psnr", ""),
                "ssim": metrics.get("ssim", ""),
                "rel_meas_error": metrics.get("rel_meas_error", ""),
                "status": "exploratory_pilot",
            }
            rows.append(row)
            write_rows(beta_out, "blind_critic_gan_results", [row], "Blind Critic GAN Pilot Result")
            all_grid_rows.extend(grid_rows)
    write_rows(out, "blind_critic_gan_results", rows, "Blind Critic GAN Pilot Results")
    save_bar_plot(out / "perception_distortion_curve.png", rows, "beta", "psnr", "Pilot PSNR by beta", "PSNR")
    save_image_grid(out / "blind_gan_visual_grid.png", all_grid_rows, ["GT", "Anchor", "GAN pilot", "Abs error"], max_rows=8)
    report = [
        "# Blind Critic GAN Pilot Report",
        "",
        "This is exploratory innovation screening, not a main result.",
        "",
        "The discriminator input is restricted to `[P_N xhat, x_data]`; analytic audit remains active for measurement consistency.",
        "",
        "Success should be judged by perceptual/null-space plausibility at controlled PSNR loss, not by RelMeasErr improvement.",
    ]
    (out / "BLIND_CRITIC_GAN_PILOT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "blind_critic_gan_pilot",
            "trains_generator": True,
            "trains_discriminator": True,
            "results_csv": str(out / "blind_critic_gan_results.csv"),
        },
    )


if __name__ == "__main__":
    main()
