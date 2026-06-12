"""Mode C posterior-sampler trainer (g2r_ series).

Protocol: guarded TRAIN loader (train_split=unlabeled), held-out val
(val_split=train, verified disjoint), CheckpointManager with forced final
save, run IDs g2r_*, ALL metrics on unclipped tensors. The discriminator
receives exactly concat(candidate, x_data) and never any residual-derived
feature. No test-split evaluation at any point: every gate runs on val.

Noise/certificate semantics: the noise setting is identical to the paper-1
main results. With finite measurement noise the exact (lambda=0) x_star
audit fits the RECORDED noisy y by design — that is certificate semantics
(the certificate asserts consistency with the recorded measurements, not
with the unknown clean signal); lambda is recorded in the certificate
tuple. A Morozov-lambda variant is deferred to the noise phase.

Run:  python -m src.g2r_modec_train --config configs/g2r/<run>.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval.checker import check_results, pass_fail_table

from .datasets import get_val_dataloader
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config
from .g2r_modec import (
    CondPatchGAN,
    ModeCSampler,
    adv_ramp,
    exact_consistency_audit,
    grad_norm,
    hinge_d_loss,
    hinge_g_loss,
    load_p0_artifact,
    r1_penalty,
    rcgan_std_reward,
    unclipped_rel_meas_err,
)
from .models import build_generator
from .phase48_49_common import load_bundle_task
from .run_protocol import CheckpointManager, enforce_run_protocol
from .split_guard import (
    SplitViolationError,
    assert_train_loader_disjoint_from_test,
    assert_val_loader_held_out,
    get_train_dataloader_guarded,
)
from .utils import (
    apply_experiment_defaults,
    ensure_dir,
    load_config,
    reconstruct_from_measurements,
    save_config,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def build_frozen_baseline(bundle_config: dict, checkpoint_path: str, measurement, device):
    generator = build_generator(bundle_config, measurement=measurement).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("generator_ema") or payload.get("generator")
    if state is None:
        raise RuntimeError(f"No generator state in warm-start checkpoint {checkpoint_path}.")
    generator.load_state_dict(state)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return generator


@torch.no_grad()
def deterministic_baseline(generator, measurement, y, bundle_config):
    """Audited deterministic output (unclipped) + anchor x_data, both fp32."""
    xhat, x_data, extras = reconstruct_from_measurements(
        generator,
        measurement,
        y,
        use_null_project=bool(bundle_config.get("use_null_project", True)),
        use_dc_project=True,
        use_final_dc_project=True,
        backprojection_mode=bundle_config.get("backprojection_mode", "ridge_pinv"),
        enable_refiner=True,
        output_range_mode=bundle_config.get("output_range_mode", "clamp_eval_only"),
        return_extras=True,
    )
    return extras["x_hat_unclamped"].float(), x_data.float(), xhat.float()


@torch.no_grad()
def chunked_baseline(generator, measurement, y, bundle_config, chunk: int):
    xs, xd, bl = [], [], []
    for start in range(0, y.shape[0], chunk):
        a, b, c = deterministic_baseline(generator, measurement, y[start : start + chunk], bundle_config)
        xs.append(a)
        xd.append(b)
        bl.append(c)
    return torch.cat(xs), torch.cat(xd), torch.cat(bl)


@torch.no_grad()
def audited_anchor(measurement, x_data, y):
    flat = measurement.flatten_img(x_data.float())
    return measurement.unflatten_img(measurement.dc_project(flat, y.float()))


def compute_x_star(mode: str, x_star_det, x_data, measurement, y, exact_audit: bool = True):
    if mode == "deterministic":
        x_star = x_star_det
    elif mode == "anchor":
        x_star = audited_anchor(measurement, x_data, y)
    else:
        raise ValueError(f"x_star_mode must be 'deterministic' or 'anchor', got {mode!r}.")
    if exact_audit:
        # Exact (lambda=0) audit: without it x_star inherits the published
        # B_lambda floor (rel ~5e-3 on scr5) instead of the float32 floor.
        x_star = exact_consistency_audit(measurement, x_star, y)
    return x_star


def sample_k(sampler, x_data, x_star, k, z_dim, device, generator_seed=None):
    """Return unclipped samples [B, K, 1, H, W]; z folded into the batch dim."""
    b = x_data.shape[0]
    if generator_seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(generator_seed))
        z = torch.randn(b * k, z_dim, generator=gen).to(device)
    else:
        z = torch.randn(b * k, z_dim, device=device)
    x_data_rep = x_data.repeat_interleave(k, dim=0)
    x_star_rep = x_star.repeat_interleave(k, dim=0)
    x_hat = sampler(x_data_rep, x_star_rep, z)
    return x_hat.reshape(b, k, *x_hat.shape[1:])


@torch.no_grad()
def chunked_sample_k(sampler, x_data, x_star, k, z_dim, device, generator_seed, chunk: int):
    """Chunked eval sampling with z drawn ONCE so chunking never changes z."""
    n = x_data.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(generator_seed))
    z_all = torch.randn(n * k, z_dim, generator=gen)
    outs = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        z = z_all[start * k : stop * k].to(device)
        xd = x_data[start:stop].repeat_interleave(k, dim=0)
        xs = x_star[start:stop].repeat_interleave(k, dim=0)
        x_hat = sampler(xd, xs, z)
        outs.append(x_hat.reshape(stop - start, k, *x_hat.shape[1:]))
    return torch.cat(outs)


TRAJ_FIELDS = [
    "step", "n_images", "k_samples",
    "psnr_mean_db", "psnr_sample_db", "psnr_baseline_db",
    "lpips_sample", "lpips_baseline",
    "std_median", "nvr", "edge_spearman",
    "relmeas_median_f64", "relmeas_p95_f64", "relmeas_max_f64",
    "g_cal", "g_div", "g_nvr", "g_mean", "g_cert", "g_perc",
    "gates_passed",
]


class GateEvaluator:
    """Seed-pinned val-subset gate evaluation through eval/checker itself.

    The val tensors (images, noisy y, frozen baseline, x_star) are computed
    once with eval_seed so every arm, step, and training seed evaluates on
    identical inputs; trajectory evals use the prefix [:n_traj].
    """

    def __init__(self, config, measurement, baseline_gen, bundle_config, val_loader, device, p0_path):
        self.device = device
        self.eval_seed = int(config.get("eval_seed", config.get("seed", 1234)))
        self.k_eval = int(config.get("k_eval", 8))
        self.z_dim = int(config["z_dim"])
        self.chunk = int(config.get("eval_chunk_images", 16))
        self.bridge_A_path = str(config["bridge_A_path"])
        self.p0_path = str(p0_path)
        self.measurement = measurement
        self.exact_audit = bool(config.get("exact_x_star_audit", True))
        self.x_star_mode = str(config.get("x_star_mode", "deterministic"))

        x_val = next(iter(val_loader))[0].to(device)
        set_seed(self.eval_seed + 7)  # pin measurement noise across arms/steps
        self.y = measurement.measure(x_val)
        self.x = x_val
        x_star_det, self.x_data, self.baseline = chunked_baseline(
            baseline_gen, measurement, self.y, bundle_config, self.chunk
        )
        self.x_star = compute_x_star(
            self.x_star_mode, x_star_det, self.x_data, measurement, self.y, exact_audit=self.exact_audit
        )
        with np.load(self.bridge_A_path) as data:
            key = "A" if "A" in data.files else data.files[0]
            self.A64 = np.asarray(data[key], dtype=np.float64)
        live_A = measurement.get_current_A().detach().cpu().numpy()
        if not np.array_equal(self.A64.astype(np.float32), live_A.astype(np.float32)):
            raise RuntimeError("Bridge A artifact does not match the live operator; refusing to evaluate.")

    @torch.no_grad()
    def run(self, sampler, *, step: int, n: int, out_dir: Path, tag: str, distributional: bool):
        was_training = sampler.training
        sampler.eval()
        x, y = self.x[:n], self.y[:n]
        samples_unclipped = chunked_sample_k(
            sampler, self.x_data[:n], self.x_star[:n], self.k_eval, self.z_dim,
            self.device, self.eval_seed + 99, self.chunk,
        )
        if was_training:
            sampler.train()
        samples_clipped = samples_unclipped.clamp(0.0, 1.0)
        sample_mean = samples_unclipped.mean(dim=1).clamp(0.0, 1.0)

        dump_dir = ensure_dir(out_dir / "evals")
        dump_path = dump_dir / f"{tag}.npz"
        np.savez(
            dump_path,
            x=x.detach().cpu().numpy().reshape(n, -1),
            samples=samples_clipped.detach().cpu().numpy().reshape(n, self.k_eval, -1),
            samples_unclipped=samples_unclipped.detach().cpu().numpy().reshape(n, self.k_eval, -1),
            sample_mean=sample_mean.detach().cpu().numpy().reshape(n, -1),
            baseline=self.baseline[:n].detach().cpu().numpy().reshape(n, -1),
            y=y.detach().cpu().numpy(),
            ref_x=x.detach().cpu().numpy().reshape(n, -1),
            A_path=self.bridge_A_path,
            P0_path=self.p0_path,
        )
        report = check_results(
            dump_path,
            perceptual_backend="lpips",
            compute_distributional=distributional,
            device=str(self.device),
        )
        # RelMeasErr p95 in float64 (checker reports max/median only).
        s64 = samples_unclipped.detach().cpu().numpy().reshape(n * self.k_eval, -1).astype(np.float64)
        y64 = np.repeat(y.detach().cpu().numpy().astype(np.float64), self.k_eval, axis=0)
        resid = s64 @ self.A64.T - y64
        rel = np.linalg.norm(resid, axis=1) / np.maximum(np.linalg.norm(y64, axis=1), 1e-300)
        gates = report["gates"]
        row = {
            "step": step,
            "n_images": n,
            "k_samples": self.k_eval,
            "psnr_mean_db": gates["G-CAL"]["values"]["avg_sample_mean_psnr_db"],
            "psnr_sample_db": gates["G-CAL"]["values"]["avg_sample_psnr_db"],
            "psnr_baseline_db": gates["G-MEAN"]["values"]["avg_baseline_psnr_db"],
            "lpips_sample": gates["G-PERC"]["values"]["mean_sample_lpips"],
            "lpips_baseline": gates["G-PERC"]["values"]["mean_baseline_lpips"],
            "std_median": gates["G-DIV"]["values"]["median_pixel_std"],
            "nvr": gates["G-NVR"]["values"]["null_variance_ratio"],
            "edge_spearman": gates["G-DIV"]["values"]["spearman_std_vs_gt_sobel"],
            "relmeas_median_f64": float(np.median(rel)),
            "relmeas_p95_f64": float(np.percentile(rel, 95)),
            "relmeas_max_f64": float(np.max(rel)),
            "g_cal": gates["G-CAL"]["status"],
            "g_div": gates["G-DIV"]["status"],
            "g_nvr": gates["G-NVR"]["status"],
            "g_mean": gates["G-MEAN"]["status"],
            "g_cert": gates["G-CERT"]["status"],
            "g_perc": gates["G-PERC"]["status"],
            "gates_passed": sum(1 for g in gates.values() if g["passed"]),
        }
        csv_path = out_dir / "gate_trajectory.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJ_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return report, row, dump_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = str(config["run_id"])
    output_dir = Path(config["output_dir"])
    enforce_run_protocol(output_dir, {"run_id": run_id, "val_split": str(config.get("val_split", "train"))})
    out = ensure_dir(output_dir)
    save_config(config, out / "resolved_config.yaml")
    set_seed(int(config.get("seed", 1234)))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"

    # --- measurement + warm-start backbone from the scr5 noleak bundle -----
    info = load_bundle_task(config["bundle_root"], str(config.get("task", "scr5")))
    bundle_config = apply_experiment_defaults(dict(info["config"]))
    bundle_config["device"] = str(device)
    measurement = make_measurement(bundle_config, device)
    exact_info = apply_measurement_override_from_config(bundle_config, measurement, device)
    save_json(exact_info, out / "exact_A_info.json")
    baseline_gen = build_frozen_baseline(bundle_config, str(info["checkpoint_path"]), measurement, device)

    # --- P0 from the verified artifact (never rebuilt) ---------------------
    p0_manifest = json.loads(Path(config["p0_manifest"]).read_text(encoding="utf-8"))
    task_key = str(config.get("task", "scr5"))
    p0_entry = next(e for e in p0_manifest["tasks"] if e["task"] == task_key)
    P0 = load_p0_artifact(
        p0_entry["artifact"]["path"], device, expected_sha256=p0_entry["artifact"]["sha256"]
    )

    # --- guarded loaders ----------------------------------------------------
    eval_seed = int(config.get("eval_seed", config.get("seed", 1234)))
    raw_limit = config.get("limit_train_samples")
    train_loader = get_train_dataloader_guarded(
        dataset_root=config["dataset_root"],
        img_size=int(bundle_config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
        limit_train_samples=int(raw_limit) if raw_limit is not None else None,
        seed=int(config.get("seed", 1234)),
        train_split=str(config.get("train_split", "unlabeled")),
        pin_memory=device.type == "cuda",
        context="g2r_modec train loader",
    )
    n_val = int(config.get("limit_val_samples", 16))
    val_loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(bundle_config["img_size"]),
        batch_size=n_val,
        num_workers=0,
        limit_val_samples=n_val,
        seed=eval_seed,
        val_split=str(config.get("val_split", "train")),
    )
    heldout = assert_val_loader_held_out(val_loader, train_loader, context="g2r_modec val loader")
    print(f"[protocol] train guard + held-out val verified: {heldout}")

    # --- deliberate split-violation drill: the guard must fire -------------
    guard_fired = False
    try:
        forbidden = get_val_dataloader(
            dataset_root=config["dataset_root"],
            img_size=int(bundle_config["img_size"]),
            batch_size=8,
            num_workers=0,
            limit_val_samples=8,
            val_split="test",
        )
        assert_train_loader_disjoint_from_test(forbidden, context="deliberate violation drill")
    except SplitViolationError as exc:
        guard_fired = True
        print(f"[protocol] split guard fired as expected on test-split loader: {exc}")
    if not guard_fired:
        raise RuntimeError("Split guard did NOT fire on the deliberate violation; aborting.")

    # --- models / optimizers ------------------------------------------------
    k = int(config["k_samples"])
    z_dim = int(config["z_dim"])
    if not hasattr(baseline_gen, "refiner"):
        raise ValueError("Mode C requires a two-stage backbone with a refiner (hq_two_stage).")
    sampler = ModeCSampler(
        baseline_gen,
        P0,
        z_dim=z_dim,
        z_channels=int(config.get("z_channels", 64)),
        per_scale_injection=bool(config.get("per_scale_injection", False)),
        freeze_stage1=bool(config.get("freeze_stage1", True)),
        delta_mode=bool(config.get("sampler_delta_mode", True)),
    ).to(device)
    pack = k if bool(config.get("pacgan", False)) else 1
    discriminator = CondPatchGAN(base_channels=int(config.get("d_channels", 64)), pack=pack).to(device)

    opt_g = torch.optim.Adam(sampler.trainable_parameters(), lr=float(config.get("g_lr", 2e-5)), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(config.get("d_lr", 1e-4)), betas=(0.5, 0.999))
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = int(config["total_steps"])
    omega_adv = float(config.get("omega_adv", 3e-3))
    beta_sd = float(config.get("beta_sd", 1.0))
    r1_enabled = bool(config.get("r1_enabled", False))
    r1_gamma = float(config.get("r1_gamma", 1.0))
    r1_interval = int(config.get("r1_interval", 16))
    log_every = int(config.get("log_every", 25))
    x_star_mode = str(config.get("x_star_mode", "deterministic"))
    exact_x_star = bool(config.get("exact_x_star_audit", True))
    gate_eval_every = int(config.get("gate_eval_every", 0))
    gate_eval_n = int(config.get("gate_eval_n", min(128, n_val)))
    keep_step_checkpoints = int(config.get("keep_step_checkpoints", 3))

    evaluator = GateEvaluator(
        config, measurement, baseline_gen, bundle_config, val_loader, device, p0_entry["artifact"]["path"]
    )

    def save_fn(path: Path, context: dict) -> None:
        torch.save(
            {
                "sampler": sampler.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": config,
                "p0_sha256": p0_entry["artifact"]["sha256"],
                **context,
            },
            path,
        )
        # Prune old periodic checkpoints (keep the newest few + final).
        steps_files = sorted(path.parent.glob("step_*.pt"))
        for old in steps_files[:-keep_step_checkpoints]:
            old.unlink(missing_ok=True)

    # Collapse detector (pre-registered operationalization of "std median
    # falls for >4000 consecutive steps"): EMA (halflife ~200 steps) of the
    # train-batch median pixel std, sampled every log_every steps; the
    # consecutive-decline counter advances by log_every per declining sample
    # and resets on any non-decline. Trigger when counter > 4000 steps.
    ema_std = None
    ema_prev = None
    decline_steps = 0
    collapse_detected = False
    ema_alpha = 1.0 - 0.5 ** (log_every / 200.0)

    log_rows: list[dict] = []
    step = 0
    t0 = time.time()
    with CheckpointManager(
        out, run_id=run_id, save_fn=save_fn, save_every_steps=int(config.get("checkpoint_every", 100))
    ) as ckpt:
        data_iter = iter(train_loader)
        while step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x = batch[0].to(device, non_blocking=True)
            b = x.shape[0]
            y = measurement.measure(x)
            x_star_det, x_data, _ = deterministic_baseline(baseline_gen, measurement, y, bundle_config)
            x_star = compute_x_star(x_star_mode, x_star_det, x_data, measurement, y, exact_audit=exact_x_star)

            # ---------------- D step (TTUR: D faster) ----------------------
            with torch.autocast(device_type=device.type, enabled=use_amp):
                with torch.no_grad():
                    x_hat = sample_k(sampler, x_data, x_star, k, z_dim, device)  # [B,K,1,H,W]
                if pack > 1:
                    fake_cand = x_hat.reshape(b, k, *x_hat.shape[3:])  # pack K along channels
                    real_cand = x.expand(-1, 1, -1, -1).repeat(1, pack, 1, 1)
                    d_real = discriminator(real_cand, x_data)
                    d_fake = discriminator(fake_cand, x_data)
                else:
                    fake_flat = x_hat.reshape(b * k, *x_hat.shape[2:])
                    x_data_rep = x_data.repeat_interleave(k, dim=0)
                    d_real = discriminator(x, x_data)
                    d_fake = discriminator(fake_flat, x_data_rep)
                d_loss = hinge_d_loss(d_real, d_fake)
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            d_gnorm = grad_norm(discriminator.parameters())
            scaler_d.step(opt_d)
            scaler_d.update()

            r1_value = 0.0
            if r1_enabled and step % r1_interval == 0:
                r1 = r1_penalty(discriminator, x, x_data)
                opt_d.zero_grad(set_to_none=True)
                (0.5 * r1_gamma * r1 * r1_interval).backward()
                opt_d.step()
                r1_value = float(r1.detach())

            # ---------------- G step ---------------------------------------
            omega = omega_adv * adv_ramp(step + 1, total_steps, float(config.get("adv_ramp_frac", 0.15)))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                x_hat = sample_k(sampler, x_data, x_star, k, z_dim, device)
                mean_k = x_hat.mean(dim=1)
                # L_rec on the SAMPLE MEAN only (per-sample L1 collapses diversity).
                l_rec = torch.nn.functional.l1_loss(mean_k.float(), x.float())
                l_sd = rcgan_std_reward(x_hat.float(), k)
                if pack > 1:
                    g_logits = discriminator(x_hat.reshape(b, k, *x_hat.shape[3:]), x_data)
                else:
                    g_logits = discriminator(
                        x_hat.reshape(b * k, *x_hat.shape[2:]), x_data.repeat_interleave(k, dim=0)
                    )
                l_adv = hinge_g_loss(g_logits)
                g_loss = l_rec - beta_sd * l_sd + omega * l_adv
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            g_gnorm = grad_norm(sampler.trainable_parameters())
            scaler_g.step(opt_g)
            scaler_g.update()

            step += 1
            ckpt.step()

            if step % log_every == 0 or step == 1:
                with torch.no_grad():
                    flat_bk = x_hat.reshape(b * k, *x_hat.shape[2:])
                    rel = unclipped_rel_meas_err(measurement, flat_bk, y.repeat_interleave(k, dim=0))
                    pix_std = x_hat.float().std(dim=1, unbiased=False)
                row = {
                    "step": step,
                    "d_loss": float(d_loss.detach()),
                    "g_loss": float(g_loss.detach()),
                    "l_rec": float(l_rec.detach()),
                    "l_sd_reward": float(l_sd.detach()),
                    "l_adv": float(l_adv.detach()),
                    "omega_adv": omega,
                    "d_real_mean": float(d_real.detach().mean()),
                    "d_fake_mean": float(d_fake.detach().mean()),
                    "g_grad_norm": g_gnorm,
                    "d_grad_norm": d_gnorm,
                    "r1": r1_value,
                    "rel_meas_err_unclipped_median": float(rel.median()),
                    "rel_meas_err_unclipped_max": float(rel.max()),
                    "median_pixel_std": float(pix_std.median()),
                    "sec_per_step": (time.time() - t0) / step,
                }
                log_rows.append(row)
                print(
                    f"[{step:5d}/{total_steps}] d={row['d_loss']:.4f} g={row['g_loss']:.4f} "
                    f"l_rec={row['l_rec']:.4f} sd={row['l_sd_reward']:.5f} adv={row['l_adv']:.4f} "
                    f"omega={omega:.2e} relmeas_med={row['rel_meas_err_unclipped_median']:.3e} "
                    f"pix_std_med={row['median_pixel_std']:.4f} |gG|={g_gnorm:.3f} |gD|={d_gnorm:.3f}"
                )
                if not np.isfinite([row["d_loss"], row["g_loss"]]).all():
                    raise RuntimeError(f"Non-finite loss at step {step}: {row}")

                # collapse detection on the std EMA
                std_now = row["median_pixel_std"]
                ema_std = std_now if ema_std is None else (1 - ema_alpha) * ema_std + ema_alpha * std_now
                if ema_prev is not None:
                    if ema_std < ema_prev:
                        decline_steps += log_every
                    else:
                        decline_steps = 0
                ema_prev = ema_std
                if decline_steps > 4000 and not collapse_detected:
                    collapse_detected = True
                    ckpt.save("collapse_snapshot.pt", reason="collapse_snapshot")
                    print(
                        f"[collapse] std EMA declined for {decline_steps} consecutive steps at step {step}; "
                        "snapshot saved, stopping this arm."
                    )
                    break

            if gate_eval_every and step % gate_eval_every == 0:
                report, row_t, _ = evaluator.run(
                    sampler, step=step, n=gate_eval_n, out_dir=out,
                    tag=f"traj_step{step:06d}", distributional=False,
                )
                print(
                    f"[gate@{step}] psnr_mean={row_t['psnr_mean_db']:.2f} (base {row_t['psnr_baseline_db']:.2f}) "
                    f"psnr_sample={row_t['psnr_sample_db']:.2f} lpips={row_t['lpips_sample']:.4f} "
                    f"(base {row_t['lpips_baseline']:.4f}) std_med={row_t['std_median']:.4f} "
                    f"nvr={row_t['nvr']:.3f} edge_rho={row_t['edge_spearman']:.3f} "
                    f"relmeas_med={row_t['relmeas_median_f64']:.2e} passed={row_t['gates_passed']}/6"
                )

    (out / "train_log.json").write_text(json.dumps(log_rows, indent=2), encoding="utf-8")

    # ---------------- final gate eval (full val set, FID/KID included) ------
    sampler.eval()
    final_report, final_row, final_dump = evaluator.run(
        sampler, step=step, n=n_val, out_dir=out, tag="final", distributional=True
    )
    (out / "final_gate_report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pass_fail_table(final_report))

    # ---------------- checkpoint save/restore roundtrip ---------------------
    n_rt = min(16, n_val)
    roundtrip_path = out / "roundtrip_check.pt"
    save_fn(roundtrip_path, {"run_id": run_id, "step": step, "reason": "roundtrip_check"})
    sampler2 = ModeCSampler(
        baseline_gen,
        P0,
        z_dim=z_dim,
        z_channels=int(config.get("z_channels", 64)),
        per_scale_injection=bool(config.get("per_scale_injection", False)),
        freeze_stage1=bool(config.get("freeze_stage1", True)),
        delta_mode=bool(config.get("sampler_delta_mode", True)),
    ).to(device)
    payload = torch.load(roundtrip_path, map_location=device, weights_only=False)
    sampler2.load_state_dict(payload["sampler"])
    sampler2.eval()
    with torch.no_grad():
        ref = chunked_sample_k(
            sampler, evaluator.x_data[:n_rt], evaluator.x_star[:n_rt],
            evaluator.k_eval, z_dim, device, evaluator.eval_seed + 99, evaluator.chunk,
        )
        again = chunked_sample_k(
            sampler2, evaluator.x_data[:n_rt], evaluator.x_star[:n_rt],
            evaluator.k_eval, z_dim, device, evaluator.eval_seed + 99, evaluator.chunk,
        )
    roundtrip_max_diff = float((again - ref).abs().max())

    summary = {
        "run_id": run_id,
        "steps": step,
        "k_train": k,
        "k_eval": evaluator.k_eval,
        "omega_adv": omega_adv,
        "beta_sd": beta_sd,
        "seed": int(config.get("seed", 1234)),
        "eval_seed": evaluator.eval_seed,
        "guard_fired_on_deliberate_violation": guard_fired,
        "heldout_val": heldout,
        "collapse_detected": collapse_detected,
        "final_gates": {name: g["status"] for name, g in final_report["gates"].items()},
        "final_gates_passed": int(final_row["gates_passed"]),
        "final_metrics_row": final_row,
        "rel_meas_err_unclipped": {
            "median": final_row["relmeas_median_f64"],
            "p95": final_row["relmeas_p95_f64"],
            "max": final_row["relmeas_max_f64"],
        },
        "roundtrip_max_abs_diff": roundtrip_max_diff,
        "dump_path": str(final_dump),
        "final_log_row": log_rows[-1] if log_rows else None,
    }
    save_json(summary, out / "smoke_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
