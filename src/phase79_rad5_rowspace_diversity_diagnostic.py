from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from . import phase73_overnight_gauge_gan_expansion as p73
from .models import build_generator
from .utils import set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase79_posterior_anti_collapse" / "rad5_rowspace_diversity_diagnostic"
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


def p0_exact(v: torch.Tensor, a64: torch.Tensor, gram64: torch.Tensor) -> torch.Tensor:
    v64 = v.to(torch.float64)
    av = v64 @ a64.T
    sol = torch.linalg.solve(gram64, av.T).T
    return v64 - sol @ a64


def pr_exact(v: torch.Tensor, a64: torch.Tensor, gram64: torch.Tensor) -> torch.Tensor:
    return v.to(torch.float64) - p0_exact(v, a64, gram64)


def charbonnier_mean(v: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(v * v + eps * eps).mean()


def forward_with_noise(generator, measurement, y: torch.Tensor, noise: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    y32 = y.float()
    x_data_flat = p69a.data_solution_safe(measurement, y32, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    residual = generator(x_data, noise, y=y32)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns = measurement.null_project(residual_flat) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = measurement.dc_project(v_stage0, y32) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, measurement.unflatten_img(x_stage1))
        v_pre = x_stage1 + measurement.flatten_img(refine.float())
    else:
        v_pre = x_stage1
    x_hat_flat = measurement.dc_project(v_pre, y32) if bool(config.get("use_final_dc_project", True)) else v_pre
    return {
        "x_data_flat": x_data_flat,
        "v_pre": v_pre,
        "x_hat_flat": x_hat_flat,
        "x_hat": measurement.unflatten_img(x_hat_flat),
        "correction_flat": x_hat_flat - v_pre,
    }


def gauge_from_flat(flat: torch.Tensor, y: torch.Tensor, measurement, a64: torch.Tensor, gram64: torch.Tensor, k64: torch.Tensor) -> torch.Tensor:
    b = p69a.blambda_y(y, a64, k64)
    p0 = p0_exact(flat, a64, gram64)
    return measurement.unflatten_img((p0 + b).to(torch.float32))


def relmeas_batch(x_hat_flat: torch.Tensor, y: torch.Tensor, a64: torch.Tensor) -> torch.Tensor:
    pred = x_hat_flat.to(torch.float64) @ a64.T
    y64 = y.to(torch.float64)
    return torch.linalg.norm(pred - y64, dim=1) / torch.linalg.norm(y64, dim=1).clamp_min(1e-12)


@torch.no_grad()
def smoke_sampling(generator, measurement, cache: p69b.SplitCache, config: dict[str, Any], device: torch.device, a64: torch.Tensor, gram64: torch.Tensor, *, k: int, seed: int) -> dict[str, float]:
    generator.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x = cache.x[:1].to(device)
    y = cache.y[:1].to(device)
    y_rep = y.repeat(int(k), 1)
    x_data_flat = p69a.data_solution_safe(measurement, y_rep, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    noise = torch.randn(x_data.shape, device=device, dtype=x_data.dtype, generator=gen)
    out = forward_with_noise(generator, measurement, y_rep, noise, config)
    samples = out["x_hat_flat"]
    centered = samples - samples.mean(dim=0, keepdim=True)
    p0 = p0_exact(centered, a64, gram64)
    pr = centered.to(torch.float64) - p0
    rel = relmeas_batch(samples, y_rep, a64)
    generator.train()
    return {
        "smoke_K": int(k),
        "smoke_mean_pixel_std": float(samples.std(dim=0, unbiased=False).mean().detach().cpu()),
        "smoke_p0_variance": float((p0 * p0).mean().detach().cpu()),
        "smoke_pr_variance": float((pr * pr).mean().detach().cpu()),
        "smoke_p0_pr_ratio": float((p0 * p0).mean().detach().cpu() / max(float((pr * pr).mean().detach().cpu()), 1e-30)),
        "smoke_relmeas_max": float(rel.max().detach().cpu()),
    }


def save_checkpoint(path: Path, generator, critic, opt_g, opt_d, config: dict[str, Any], metrics: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase79",
        "experiment": "rad5_rowspace_diversity_diagnostic",
        "step": int(metrics.get("step", -1)),
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(p73.REGIMES["rad5"]["checkpoint"]),
        "source_checkpoint_sha256": p69a.sha256_file(p73.REGIMES["rad5"]["checkpoint"]),
        "predeclared_hyperparameters": {
            "step_budget": int(args.steps),
            "lambda_adv": float(args.lambda_adv),
            "lambda_diversity": float(args.lambda_diversity),
            "target_p0_pair_l1": float(args.target_p0_pair_l1),
            "lr_g": float(args.lr_g),
            "lr_d": float(args.lr_d),
            "batch_size": BATCH_SIZE,
        },
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rad-5 row-space-only reconstruction + diversity diagnostic.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET)
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=79001)
    parser.add_argument("--lr_g", type=float, default=2e-5)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--lambda_adv", type=float, default=0.05)
    parser.add_argument("--lambda_diversity", type=float, default=1.0)
    parser.add_argument("--target_p0_pair_l1", type=float, default=0.02)
    parser.add_argument("--smoke_k", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(OUT)
    append_log("start")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(int(args.seed))

    config = p73.regime_config("rad5", device)
    config["output_dir"] = str(OUT)
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"
    config["phase79_diagnostic"] = {
        "hypothesis": "full image reconstruction loss suppresses null-space diversity",
        "reconstruction_loss": "exact row-space PR(x_hat - x) only",
        "zero_space_terms": "gauge adversarial + hinge diversity on P0 pairwise L1",
        "test_split_training": False,
    }
    measurement, a_tensor = p73.make_regime_measurement("rad5", config, device)
    generator, config = p73.load_regime_generator("rad5", config, measurement, device, train=True)
    config["output_dir"] = str(OUT)
    config["phase79_diagnostic"] = {
        "hypothesis": "full image reconstruction loss suppresses null-space diversity",
        "reconstruction_loss": "exact row-space PR(x_hat - x) only",
        "zero_space_terms": "gauge adversarial + hinge diversity on P0 pairwise L1",
        "test_split_training": False,
    }
    train, val, _test, split = p73.build_caches("rad5", config, measurement, device)
    save_json(OUT / "split_manifest.json", split)

    a64, gram64, k64 = p73.exact_projectors(measurement.A, float(config["lambda_solver"]))
    critic = p69a.PatchCritic(1).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(args.lr_g), betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(critic.parameters(), lr=float(args.lr_d), betas=(0.5, 0.9))
    loader = cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=int(args.seed) + 11))
    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    protocol = {
        "phase": "Phase79",
        "regime": "rad5",
        "source_checkpoint": str(p73.REGIMES["rad5"]["checkpoint"]),
        "source_checkpoint_sha256": p69a.sha256_file(p73.REGIMES["rad5"]["checkpoint"]),
        "source_config": str(p73.REGIMES["rad5"]["config"]),
        "source_A": str(p73.REGIMES["rad5"]["A"]),
        "A_sha256_float32_bytes": p69a.sha256_np(np.load(p73.REGIMES["rad5"]["A"]).astype(np.float32)),
        "steps": int(args.steps),
        "batch_size": BATCH_SIZE,
        "lambda_adv": float(args.lambda_adv),
        "lambda_diversity": float(args.lambda_diversity),
        "target_p0_pair_l1": float(args.target_p0_pair_l1),
        "lr_g": float(args.lr_g),
        "lr_d": float(args.lr_d),
        "train_source": split["train_source"],
        "val_source": split["val_source"],
        "test_source": "not used during training; final external criteria eval uses frozen main_rad5 cache",
        "predeclared_before_run": True,
    }
    save_json(OUT / "diagnostic_protocol.json", protocol)

    initial_smoke = smoke_sampling(
        generator,
        measurement,
        val,
        config,
        device,
        a64,
        gram64,
        k=int(args.smoke_k),
        seed=int(args.seed) + 101,
    )
    initial_smoke.update({"step": 0, "event": "initial_smoke"})
    rows.append(initial_smoke)
    append_log(f"initial_smoke std={initial_smoke['smoke_mean_pixel_std']:.6g} p0var={initial_smoke['smoke_p0_variance']:.6g}")

    for step in range(1, int(args.steps) + 1):
        x, y, _labels, _indices = next(loader)
        x = x.to(device)
        y = y.to(device)
        x_flat = measurement.flatten_img(x)

        generator.eval()
        critic.train()
        with torch.no_grad():
            x_data_flat = p69a.data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
            x_data = measurement.unflatten_img(x_data_flat)
            noise_d = torch.randn_like(x_data)
            out_d = forward_with_noise(generator, measurement, y, noise_d, config)
            real_gauge = gauge_from_flat(x_flat, y, measurement, a64, gram64, k64)
            fake_gauge = gauge_from_flat(out_d["x_hat_flat"], y, measurement, a64, gram64, k64)
        opt_d.zero_grad(set_to_none=True)
        real_score = critic(real_gauge)
        fake_score = critic(fake_gauge)
        d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
        d_loss.backward()
        opt_d.step()
        d_acc = float(0.5 * ((real_score.detach() > 0).float().mean() + (fake_score.detach() < 0).float().mean()).cpu())

        generator.train()
        critic.eval()
        with torch.no_grad():
            x_data_flat = p69a.data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
            x_data = measurement.unflatten_img(x_data_flat)
        noise1 = torch.randn_like(x_data)
        noise2 = torch.randn_like(x_data)
        out1 = forward_with_noise(generator, measurement, y, noise1, config)
        out2 = forward_with_noise(generator, measurement, y, noise2, config)
        diff1 = out1["x_hat_flat"] - x_flat
        diff2 = out2["x_hat_flat"] - x_flat
        row_loss = 0.5 * (
            charbonnier_mean(pr_exact(diff1, a64, gram64))
            + charbonnier_mean(pr_exact(diff2, a64, gram64))
        )
        p0_pair = p0_exact(out1["x_hat_flat"] - out2["x_hat_flat"], a64, gram64)
        p0_pair_l1 = p0_pair.abs().mean()
        diversity_loss = F.relu(torch.tensor(float(args.target_p0_pair_l1), device=device, dtype=p0_pair_l1.dtype) - p0_pair_l1)
        fake_gauge_1 = gauge_from_flat(out1["x_hat_flat"], y, measurement, a64, gram64, k64)
        fake_gauge_2 = gauge_from_flat(out2["x_hat_flat"], y, measurement, a64, gram64, k64)
        adv_loss = -critic(torch.cat([fake_gauge_1, fake_gauge_2], dim=0)).mean()
        g_loss = row_loss + float(args.lambda_adv) * adv_loss + float(args.lambda_diversity) * diversity_loss
        if not torch.isfinite(g_loss):
            raise RuntimeError(f"Non-finite generator loss at step {step}.")
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        rel1 = relmeas_batch(out1["x_hat_flat"].detach(), y, a64)
        row = {
            "step": step,
            "loss_g_total": float(g_loss.detach().cpu()),
            "loss_row_space": float(row_loss.detach().cpu()),
            "loss_adv": float(adv_loss.detach().cpu()),
            "loss_diversity_hinge": float(diversity_loss.detach().cpu()),
            "p0_pair_l1": float(p0_pair_l1.detach().cpu()),
            "loss_d": float(d_loss.detach().cpu()),
            "d_accuracy": d_acc,
            "relmeas_train_max": float(rel1.max().detach().cpu()),
        }
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            smoke = smoke_sampling(
                generator,
                measurement,
                val,
                config,
                device,
                a64,
                gram64,
                k=int(args.smoke_k),
                seed=int(args.seed) + 1000 + step,
            )
            row.update(smoke)
            append_log(
                f"eval step={step} std={smoke['smoke_mean_pixel_std']:.6g} "
                f"p0var={smoke['smoke_p0_variance']:.6g} prvar={smoke['smoke_pr_variance']:.6g}"
            )
            ckpt_path = OUT / "checkpoints" / f"step_{step:05d}.pt"
            save_checkpoint(ckpt_path, generator, critic, opt_g, opt_d, config, row, args)
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
    save_checkpoint(final_path, generator, critic, opt_g, opt_d, config, final_metrics, args)
    checkpoint_rows.append(
        {"kind": "final", "step": int(args.steps), "path": str(final_path), "sha256": p69a.sha256_file(final_path)}
    )
    write_csv(OUT / "training_log.csv", rows)
    write_csv(OUT / "checkpoint_hashes.csv", checkpoint_rows)
    save_json(
        OUT / "diagnostic_summary.json",
        {
            "status": "completed",
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": p69a.sha256_file(final_path),
            "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
            "A_sha256_float32_bytes": protocol["A_sha256_float32_bytes"],
            "last_training_row": final_metrics,
            "protocol": protocol,
        },
    )
    append_log("complete")
    print(json.dumps(p69a.json_safe({"final_checkpoint": final_path, "final_checkpoint_sha256": p69a.sha256_file(final_path), "last_training_row": final_metrics}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
