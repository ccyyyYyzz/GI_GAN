from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from . import phase73_overnight_gauge_gan_expansion as p73
from . import phase79_rad5_rowspace_diversity_diagnostic as p79
from .models import build_generator
from .utils import apply_experiment_defaults, set_seed


ROOT = Path("E:/ns_mc_gan_gi")
PHASE79 = ROOT / "outputs_phase79_posterior_anti_collapse" / "rad5_rowspace_diversity_diagnostic"
OUT = ROOT / "outputs_phase80_posterior_calibration" / "rad5_centered_diversity_anchor"
BATCH_SIZE = 8
DEFAULT_STEP_BUDGET = 200
DEFAULT_EVAL_EVERY = 25


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def noise_for_y(measurement, y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    x_data_flat = p69a.data_solution_safe(measurement, y.float(), config.get("backprojection_mode", "ridge_pinv"))
    return torch.randn_like(measurement.unflatten_img(x_data_flat))


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    base_config: dict[str, Any],
    measurement,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    merged = dict(base_config)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        merged.update(ckpt["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(ROOT / "data")
    merged["output_dir"] = str(OUT)
    merged["batch_size"] = BATCH_SIZE
    merged["num_workers"] = 0
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged["output_range_mode"] = "clamp_eval_only"
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    state = ckpt.get("generator_ema") or ckpt.get("generator")
    if state is None:
        raise RuntimeError(f"{checkpoint_path} has no generator/generator_ema state.")
    generator.load_state_dict(state, strict=True)
    generator.train(True)
    return generator, merged, ckpt


def load_deterministic_generator(config: dict[str, Any], measurement, device: torch.device):
    det_config = p73.regime_config("rad5", device)
    det_config["output_dir"] = str(OUT)
    det_config["use_final_dc_project"] = True
    det_config["output_range_mode"] = "clamp_eval_only"
    det_generator, det_config = p73.load_regime_generator("rad5", det_config, measurement, device, train=False)
    det_config["output_dir"] = str(OUT)
    det_config["use_final_dc_project"] = True
    det_config["output_range_mode"] = "clamp_eval_only"
    for param in det_generator.parameters():
        param.requires_grad_(False)
    det_generator.eval()
    return det_generator, det_config


def deterministic_anchor_flat(
    det_generator,
    measurement,
    y: torch.Tensor,
    det_config: dict[str, Any],
) -> torch.Tensor:
    with torch.no_grad():
        x_data_flat = p69a.data_solution_safe(measurement, y.float(), det_config.get("backprojection_mode", "ridge_pinv"))
        zero_noise = torch.zeros_like(measurement.unflatten_img(x_data_flat))
        return p79.forward_with_noise(det_generator, measurement, y, zero_noise, det_config)["x_hat_flat"].detach()


@torch.no_grad()
def smoke_centered(
    generator,
    det_generator,
    measurement,
    cache: p69b.SplitCache,
    config: dict[str, Any],
    det_config: dict[str, Any],
    device: torch.device,
    a64: torch.Tensor,
    gram64: torch.Tensor,
    *,
    k: int,
    seed: int,
) -> dict[str, float]:
    generator.eval()
    det_generator.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    y = cache.y[:1].to(device)
    y_rep = y.repeat(int(k), 1)
    x_data_flat = p69a.data_solution_safe(measurement, y_rep, config.get("backprojection_mode", "ridge_pinv"))
    noise = torch.randn(measurement.unflatten_img(x_data_flat).shape, device=device, dtype=x_data_flat.dtype, generator=gen)
    samples = p79.forward_with_noise(generator, measurement, y_rep, noise, config)["x_hat_flat"]
    det_flat = deterministic_anchor_flat(det_generator, measurement, y, det_config)
    centered = samples - samples.mean(dim=0, keepdim=True)
    p0 = p79.p0_exact(centered, a64, gram64)
    pr = centered.to(torch.float64) - p0
    p0_samples = p79.p0_exact(samples, a64, gram64)
    p0_mean = p0_samples.mean(dim=0, keepdim=True)
    p0_det = p79.p0_exact(det_flat, a64, gram64)
    offset = p0_mean - p0_det
    rel = p79.relmeas_batch(samples, y_rep, a64)
    generator.train()
    return {
        "smoke_K": int(k),
        "smoke_mean_pixel_std": float(samples.std(dim=0, unbiased=False).mean().detach().cpu()),
        "smoke_p0_variance": float((p0 * p0).mean().detach().cpu()),
        "smoke_pr_variance": float((pr * pr).mean().detach().cpu()),
        "smoke_p0_pr_ratio": float((p0 * p0).mean().detach().cpu() / max(float((pr * pr).mean().detach().cpu()), 1e-30)),
        "smoke_p0_mean_anchor_rmse": float(torch.sqrt((offset * offset).mean()).detach().cpu()),
        "smoke_p0_mean_anchor_mae": float(offset.abs().mean().detach().cpu()),
        "smoke_relmeas_max": float(rel.max().detach().cpu()),
    }


def save_checkpoint(
    path: Path,
    generator,
    critic,
    opt_g,
    opt_d,
    config: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    start_checkpoint: Path,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase80",
        "experiment": "rad5_centered_diversity_anchor",
        "step": int(metrics.get("step", -1)),
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "config": config,
        "metrics": metrics,
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": p69a.sha256_file(start_checkpoint),
        "deterministic_anchor_checkpoint": str(p73.REGIMES["rad5"]["checkpoint"]),
        "deterministic_anchor_checkpoint_sha256": p69a.sha256_file(p73.REGIMES["rad5"]["checkpoint"]),
        "predeclared_hyperparameters": {
            "step_budget": int(args.steps),
            "train_k": int(args.train_k),
            "lambda_adv": float(args.lambda_adv),
            "lambda_diversity": float(args.lambda_diversity),
            "lambda_mean_anchor": float(args.lambda_mean_anchor),
            "target_p0_center_l1": float(args.target_p0_center_l1),
            "lr_g": float(args.lr_g),
            "lr_d": float(args.lr_d),
            "batch_size": BATCH_SIZE,
        },
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rad-5 centered P0 diversity + deterministic P0 mean anchor calibration round.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(OUT))
    parser.add_argument("--start_checkpoint", default=str(PHASE79 / "checkpoints" / "final.pt"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET)
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=80001)
    parser.add_argument("--lr_g", type=float, default=1e-5)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lambda_adv", type=float, default=0.05)
    parser.add_argument("--lambda_diversity", type=float, default=1.0)
    parser.add_argument("--lambda_mean_anchor", type=float, default=5.0)
    parser.add_argument("--target_p0_center_l1", type=float, default=0.02)
    parser.add_argument("--train_k", type=int, default=4)
    parser.add_argument("--smoke_k", type=int, default=16)
    parser.add_argument("--fresh_optimizers", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    global OUT
    args = parse_args()
    OUT = Path(args.output_dir)
    ensure_dir(OUT)
    append_log("start")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(int(args.seed))

    start_checkpoint = Path(args.start_checkpoint)
    config = p73.regime_config("rad5", device)
    config["output_dir"] = str(OUT)
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"
    config["phase80_centered_diversity"] = {
        "hypothesis": "Phase79 calibration failure is P0 location drift caused by pairwise diversity hinge",
        "start_checkpoint": str(start_checkpoint),
        "mean_anchor": "Charbonnier mean over P0(sample_mean_k - deterministic_rad5_anchor)",
        "diversity": "hinge on mean absolute centered P0 sample deviations, not pairwise distance",
        "test_split_training": False,
    }
    measurement, _a_tensor = p73.make_regime_measurement("rad5", config, device)
    generator, config, start_payload = load_generator_from_checkpoint(start_checkpoint, config, measurement, device)
    config["output_dir"] = str(OUT)
    config["phase80_centered_diversity"] = {
        "hypothesis": "Phase79 calibration failure is P0 location drift caused by pairwise diversity hinge",
        "start_checkpoint": str(start_checkpoint),
        "mean_anchor": "Charbonnier mean over P0(sample_mean_k - deterministic_rad5_anchor)",
        "diversity": "hinge on mean absolute centered P0 sample deviations, not pairwise distance",
        "test_split_training": False,
    }
    det_generator, det_config = load_deterministic_generator(config, measurement, device)
    train, val, _test, split = p73.build_caches("rad5", config, measurement, device)
    save_json(OUT / "split_manifest.json", split)

    a64, gram64, k64 = p73.exact_projectors(measurement.A, float(config["lambda_solver"]))
    critic = p69a.PatchCritic(1).to(device)
    if isinstance(start_payload, dict) and start_payload.get("critic"):
        critic.load_state_dict(start_payload["critic"], strict=True)
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(args.lr_g), betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(critic.parameters(), lr=float(args.lr_d), betas=(0.5, 0.9))
    loader = cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=int(args.seed) + 11))
    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    protocol = {
        "phase": "Phase80",
        "regime": "rad5",
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": p69a.sha256_file(start_checkpoint),
        "deterministic_anchor_checkpoint": str(p73.REGIMES["rad5"]["checkpoint"]),
        "deterministic_anchor_checkpoint_sha256": p69a.sha256_file(p73.REGIMES["rad5"]["checkpoint"]),
        "source_config": str(p73.REGIMES["rad5"]["config"]),
        "source_A": str(p73.REGIMES["rad5"]["A"]),
        "A_sha256_float32_bytes": p69a.sha256_np(np.load(p73.REGIMES["rad5"]["A"]).astype(np.float32)),
        "steps": int(args.steps),
        "batch_size": BATCH_SIZE,
        "train_k": int(args.train_k),
        "lambda_adv": float(args.lambda_adv),
        "lambda_diversity": float(args.lambda_diversity),
        "lambda_mean_anchor": float(args.lambda_mean_anchor),
        "target_p0_center_l1": float(args.target_p0_center_l1),
        "lr_g": float(args.lr_g),
        "lr_d": float(args.lr_d),
        "train_source": split["train_source"],
        "val_source": split["val_source"],
        "test_source": "not used during training; final calibration eval uses frozen main_rad5 cache",
        "predeclared_before_run": True,
    }
    save_json(OUT / "calibration_repair_protocol.json", protocol)

    initial_smoke = smoke_centered(
        generator,
        det_generator,
        measurement,
        val,
        config,
        det_config,
        device,
        a64,
        gram64,
        k=int(args.smoke_k),
        seed=int(args.seed) + 101,
    )
    initial_smoke.update({"step": 0, "event": "initial_smoke"})
    rows.append(initial_smoke)
    append_log(
        f"initial_smoke std={initial_smoke['smoke_mean_pixel_std']:.6g} "
        f"p0var={initial_smoke['smoke_p0_variance']:.6g} "
        f"anchor_rmse={initial_smoke['smoke_p0_mean_anchor_rmse']:.6g}"
    )

    for step in range(1, int(args.steps) + 1):
        x, y, _labels, _indices = next(loader)
        x = x.to(device)
        y = y.to(device)
        bsz = int(y.shape[0])
        k_train = int(args.train_k)
        x_flat = measurement.flatten_img(x)

        generator.eval()
        critic.train()
        with torch.no_grad():
            noise_d = noise_for_y(measurement, y, config)
            out_d = p79.forward_with_noise(generator, measurement, y, noise_d, config)
            real_gauge = p79.gauge_from_flat(x_flat, y, measurement, a64, gram64, k64)
            fake_gauge = p79.gauge_from_flat(out_d["x_hat_flat"], y, measurement, a64, gram64, k64)
        opt_d.zero_grad(set_to_none=True)
        real_score = critic(real_gauge)
        fake_score = critic(fake_gauge)
        d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
        d_loss.backward()
        opt_d.step()
        d_acc = float(0.5 * ((real_score.detach() > 0).float().mean() + (fake_score.detach() < 0).float().mean()).cpu())

        generator.train()
        critic.eval()
        y_rep = y[:, None, :].repeat(1, k_train, 1).reshape(bsz * k_train, -1)
        x_rep_flat = x_flat[:, None, :].repeat(1, k_train, 1).reshape(bsz * k_train, -1)
        x_data_rep_flat = p69a.data_solution_safe(measurement, y_rep, config.get("backprojection_mode", "ridge_pinv"))
        noise = torch.randn_like(measurement.unflatten_img(x_data_rep_flat))
        out = p79.forward_with_noise(generator, measurement, y_rep, noise, config)
        samples_flat = out["x_hat_flat"]
        diff = samples_flat - x_rep_flat
        row_loss = p79.charbonnier_mean(p79.pr_exact(diff, a64, gram64))

        p0_samples = p79.p0_exact(samples_flat, a64, gram64).reshape(bsz, k_train, -1)
        p0_mean = p0_samples.mean(dim=1)
        p0_centered = p0_samples - p0_mean[:, None, :]
        p0_center_l1 = p0_centered.abs().mean()
        diversity_loss = F.relu(
            torch.tensor(float(args.target_p0_center_l1), device=device, dtype=p0_center_l1.dtype) - p0_center_l1
        )

        det_flat = deterministic_anchor_flat(det_generator, measurement, y, det_config)
        p0_det = p79.p0_exact(det_flat, a64, gram64)
        mean_anchor_loss = p79.charbonnier_mean(p0_mean - p0_det)

        fake_gauge_all = p79.gauge_from_flat(samples_flat, y_rep, measurement, a64, gram64, k64)
        adv_loss = -critic(fake_gauge_all).mean()
        g_loss = (
            row_loss
            + float(args.lambda_adv) * adv_loss
            + float(args.lambda_diversity) * diversity_loss
            + float(args.lambda_mean_anchor) * mean_anchor_loss
        )
        if not torch.isfinite(g_loss):
            raise RuntimeError(f"Non-finite generator loss at step {step}.")
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        rel = p79.relmeas_batch(samples_flat.detach(), y_rep, a64)
        p0_anchor_rmse = torch.sqrt(((p0_mean - p0_det) * (p0_mean - p0_det)).mean())
        row = {
            "step": step,
            "loss_g_total": float(g_loss.detach().cpu()),
            "loss_row_space": float(row_loss.detach().cpu()),
            "loss_adv": float(adv_loss.detach().cpu()),
            "loss_diversity_center_hinge": float(diversity_loss.detach().cpu()),
            "loss_mean_anchor": float(mean_anchor_loss.detach().cpu()),
            "p0_center_l1": float(p0_center_l1.detach().cpu()),
            "p0_mean_anchor_rmse": float(p0_anchor_rmse.detach().cpu()),
            "loss_d": float(d_loss.detach().cpu()),
            "d_accuracy": d_acc,
            "relmeas_train_max": float(rel.max().detach().cpu()),
        }
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            smoke = smoke_centered(
                generator,
                det_generator,
                measurement,
                val,
                config,
                det_config,
                device,
                a64,
                gram64,
                k=int(args.smoke_k),
                seed=int(args.seed) + 1000 + step,
            )
            row.update(smoke)
            append_log(
                f"eval step={step} std={smoke['smoke_mean_pixel_std']:.6g} "
                f"p0var={smoke['smoke_p0_variance']:.6g} "
                f"anchor_rmse={smoke['smoke_p0_mean_anchor_rmse']:.6g}"
            )
            ckpt_path = OUT / "checkpoints" / f"step_{step:05d}.pt"
            save_checkpoint(ckpt_path, generator, critic, opt_g, opt_d, config, row, args, start_checkpoint)
            checkpoint_rows.append(
                {
                    "kind": "step",
                    "step": step,
                    "path": str(ckpt_path),
                    "sha256": p69a.sha256_file(ckpt_path),
                }
            )
        rows.append(row)

    final_metrics = rows[-1]
    final_path = OUT / "checkpoints" / "final.pt"
    save_checkpoint(final_path, generator, critic, opt_g, opt_d, config, final_metrics, args, start_checkpoint)
    checkpoint_rows.append(
        {"kind": "final", "step": int(args.steps), "path": str(final_path), "sha256": p69a.sha256_file(final_path)}
    )
    write_csv(OUT / "training_log.csv", rows)
    write_csv(OUT / "checkpoint_hashes.csv", checkpoint_rows)
    save_json(
        OUT / "calibration_repair_summary.json",
        {
            "status": "completed",
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": p69a.sha256_file(final_path),
            "start_checkpoint_sha256": protocol["start_checkpoint_sha256"],
            "deterministic_anchor_checkpoint_sha256": protocol["deterministic_anchor_checkpoint_sha256"],
            "A_sha256_float32_bytes": protocol["A_sha256_float32_bytes"],
            "last_training_row": final_metrics,
            "protocol": protocol,
        },
    )
    append_log("complete")
    print(
        json.dumps(
            p69a.json_safe(
                {
                    "final_checkpoint": final_path,
                    "final_checkpoint_sha256": p69a.sha256_file(final_path),
                    "last_training_row": final_metrics,
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
