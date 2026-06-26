from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from . import phase78_96px_rad5_one_seed_probe as p78
from .utils import set_seed


DATA_ROOT = Path(os.environ.get("NS_MC_GAN_GI_DATA_ROOT", "E:/ns_mc_gan_gi"))
OUT = DATA_ROOT / "outputs_phase81_96px_rad5_paper_completion"
PH78 = DATA_ROOT / "outputs_phase78_96px_rad5_one_seed_probe"

IMG_SIZE = p78.IMG_SIZE
STEP_BUDGET = p78.STEP_BUDGET
EVAL_EVERY = p78.EVAL_EVERY
SEEDS = list(range(1, 11))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return p69a.format_table(rows, columns) if rows else ""


def configure_phase78_globals(seed: int | None = None) -> None:
    p78.DATA_ROOT = DATA_ROOT
    p78.OUT = OUT
    p78.CERT_CACHE = DATA_ROOT / "results" / "cert_package_20260612" / "cache"
    p78.PROVENANCE_JSON = DATA_ROOT / "results" / "cert_package_20260612" / "PROVENANCE.json"
    p78.SPLIT_TRAIN = p78.CERT_CACHE / "split_train_indices_stl10_train_unlabeled.npy"
    p78.SPLIT_EVAL = p78.CERT_CACHE / "split_eval_indices_stl10_test.npy"
    p78.RAD5_CHECKPOINT = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "last.pt"
    p78.RAD5_CONFIG = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "resolved_config.yaml"
    p78.PHASE73_RAD5_DELTA = DATA_ROOT / "outputs_phase73_overnight_gauge_gan_expansion" / "rad5_seed_delta_metrics.csv"
    p78.PHASE71_SCR5_SEED01_DELTA = DATA_ROOT / "outputs_phase71_gauge_cgan_paired_seeds" / "seed01" / "paired_comparison_C_vs_B.csv"
    p78.OUT = OUT
    if os.environ.get("PHASE81_STL10_DOWNLOAD", "0").lower() in {"1", "true", "yes"}:
        def stl10_dataset_96_download(split: str):
            transform = p78.build_transform(IMG_SIZE, dataset_name="stl10", train=False, use_augmentation=False)
            return p78.datasets.STL10(root=str(DATA_ROOT / "data"), split=split, transform=transform, download=True)

        p78.stl10_dataset_96 = stl10_dataset_96_download
    if seed is not None:
        p78.SEED_ID = int(seed)


def set_all_seeds(seed: int) -> None:
    set_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def prepare_common(device: torch.device):
    configure_phase78_globals()
    config = p78.make_config(device)
    config["output_dir"] = str(OUT)
    measurement = p78.make_measurement(config, device)
    train, val, test, split = p78.build_caches(measurement, device)
    return config, measurement, train, val, test, split


def phase78_ckpt(seed: int, arm: str) -> Path:
    return PH78 / f"seed{seed:02d}" / arm / "checkpoints" / "best_by_val.pt"


def phase81_ckpt(seed: int, arm: str) -> Path:
    return OUT / f"seed{seed:02d}" / arm / "checkpoints" / "best_by_val.pt"


def best_ckpt(seed: int, arm: str) -> Path | None:
    if arm in {"B", "C"} and seed == 1 and phase78_ckpt(seed, arm).exists():
        return phase78_ckpt(seed, arm)
    path = phase81_ckpt(seed, arm)
    return path if path.exists() else None


def beta0_value(device: torch.device, config: dict[str, Any], measurement, train) -> tuple[float, dict[str, Any]]:
    rows = read_csv(PH78 / "beta_calibration_96px_rad5.csv")
    if rows:
        row = rows[0]
        return float(row["selected_beta0"]), {
            "source": str(PH78 / "beta_calibration_96px_rad5.csv"),
            "source_sha256": p69a.sha256_file(PH78 / "beta_calibration_96px_rad5.csv"),
            **row,
        }
    probe = p78.load_generator_96(config, measurement, device, train=False)
    beta, beta_rows = p78.beta_calibration(probe, measurement, train, config, device)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return beta, {"source": "phase81_recomputed", **beta_rows[0]}


def preflight() -> None:
    ensure_dir(OUT)
    configure_phase78_globals()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = p78.make_config(device)
    config["output_dir"] = str(OUT)
    measurement = p78.make_measurement(config, device)
    failures: list[str] = []
    required = [
        p78.RAD5_CHECKPOINT,
        p78.RAD5_CONFIG,
        p78.SPLIT_TRAIN,
        p78.SPLIT_EVAL,
    ]
    require_phase78_seed01 = os.environ.get("PHASE81_REQUIRE_PHASE78_SEED01", "1").lower() not in {"0", "false", "no"}
    if require_phase78_seed01:
        required.extend(
            [
                PH78 / "seed01" / "B" / "checkpoints" / "best_by_val.pt",
                PH78 / "seed01" / "C" / "checkpoints" / "best_by_val.pt",
                PH78 / "seed01" / "evaluation" / "per_sample_outputs_B.npz",
                PH78 / "seed01" / "evaluation" / "per_sample_outputs_C.npz",
            ]
        )
    for path in required:
        if not path.exists():
            failures.append(f"missing required path: {path}")
    if measurement.img_size != IMG_SIZE or measurement.n != IMG_SIZE * IMG_SIZE or measurement.m != 461:
        failures.append(f"unexpected measurement shape img={measurement.img_size} n={measurement.n} m={measurement.m}")
    payload = {
        "phase": "Phase81_96px_rad5_paper_completion",
        "output_dir": str(OUT),
        "device": str(device),
        "img_size": IMG_SIZE,
        "n": int(measurement.n),
        "m": int(measurement.m),
        "sampling_ratio_effective": float(measurement.m / measurement.n),
        "pattern_type": measurement.pattern_type,
        "matrix_normalization": measurement.matrix_normalization,
        "A_sha256_float32_bytes": p69a.sha256_np(measurement.A.detach().cpu().numpy().astype(np.float32)),
        "source_rad5_checkpoint": str(p78.RAD5_CHECKPOINT),
        "source_rad5_checkpoint_sha256": p69a.sha256_file(p78.RAD5_CHECKPOINT) if p78.RAD5_CHECKPOINT.exists() else "",
        "source_phase78_seed01_B_sha256": p69a.sha256_file(PH78 / "seed01" / "B" / "checkpoints" / "best_by_val.pt")
        if (PH78 / "seed01" / "B" / "checkpoints" / "best_by_val.pt").exists()
        else "",
        "source_phase78_seed01_C_sha256": p69a.sha256_file(PH78 / "seed01" / "C" / "checkpoints" / "best_by_val.pt")
        if (PH78 / "seed01" / "C" / "checkpoints" / "best_by_val.pt").exists()
        else "",
        "require_phase78_seed01": require_phase78_seed01,
        "failures": failures,
    }
    save_json(OUT / "preflight_checks.json", payload)
    write_text(
        OUT / "PHASE81_96PX_RAD5_PROTOCOL.md",
        "\n".join(
            [
                "# Phase81 96px Rad-5 Paper Completion Protocol",
                "",
                f"- output_dir: `{OUT}`",
                "- purpose: complete 96px Rad-5 paired B/C seeds and standard cGAN D control.",
                "- seed01 B/C are read-only Phase78 artifacts.",
                "- seed02/03 B/C start from the same verified Rad-5 checkpoint with paired seed/init and loader order.",
                "- D_standard uses a standard image-input PatchGAN: real `x`, fake deployed `x_hat`.",
                "- Gauge C uses real/fake gauge images `P0 x + B_lambda y` and `P0 v + B_lambda y`.",
                "- Train/val are STL10 train+unlabeled slices; official test split is evaluation-only.",
                f"- step_budget_per_arm: {STEP_BUDGET}",
                "",
            ]
        ),
    )
    if failures:
        write_text(OUT / "UNSAFE_TO_RUN.md", "# UNSAFE TO RUN\n\n" + "\n".join(f"- {f}" for f in failures) + "\n")
        raise RuntimeError("Phase81 preflight failed; see UNSAFE_TO_RUN.md")
    append_log("preflight_complete")


def save_phase81_checkpoint(
    path: Path,
    seed_id: int,
    arm: str,
    step: int,
    generator,
    opt_g,
    config: dict[str, Any],
    metrics: dict[str, Any],
    loader_seed: int,
    beta: float,
    adv_mode: str,
    critic=None,
    opt_d=None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase81_96px_rad5_paper_completion",
        "seed_id": int(seed_id),
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(p78.RAD5_CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(p78.RAD5_CHECKPOINT),
        "beta": float(beta),
        "adv_mode": adv_mode,
        "paired_loader_seed": int(loader_seed),
        "img_size": IMG_SIZE,
        "checkpoint_selection_rule": "best_by_val_rec_loss",
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if opt_d is not None:
        payload["optimizer_d"] = opt_d.state_dict()
    torch.save(payload, path)


def d_accuracy(real_score: torch.Tensor, fake_score: torch.Tensor) -> float:
    return float(0.5 * ((real_score.detach() > 0).float().mean() + (fake_score.detach() < 0).float().mean()).cpu())


def train_arm_mode(
    seed_id: int,
    arm: str,
    adv_mode: str,
    beta: float,
    config: dict[str, Any],
    measurement,
    train,
    val,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    if adv_mode not in {"none", "gauge", "standard"}:
        raise ValueError(f"Unsupported adv_mode: {adv_mode}")
    configure_phase78_globals(seed_id)
    seed_dir = ensure_dir(OUT / f"seed{seed_id:02d}")
    arm_dir = ensure_dir(seed_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    best_path = ckpt_dir / "best_by_val.pt"
    final_path = ckpt_dir / "final.pt"
    loader_seed = 785400 + int(seed_id)

    generator = p78.load_generator_96(config, measurement, device, train=True)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adv_mode in {"gauge", "standard"} else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = p78.cycle_loader(p78.make_loader(train, shuffle=True, seed=loader_seed))
    rows: list[dict[str, Any]] = []
    d_hist: list[float] = []
    best_val = float("inf")
    best_step = -1
    final_metrics: dict[str, Any] = {}
    append_log(f"train_start seed={seed_id} arm={arm} adv_mode={adv_mode} steps={STEP_BUDGET}")

    for step in range(1, STEP_BUDGET + 1):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)
        d_loss_v = float("nan")
        d_acc = float("nan")
        if critic is not None and opt_d is not None:
            generator.eval()
            critic.train()
            with torch.no_grad():
                od = p78.forward_candidate(generator, measurement, x, y, config)
            real_img = od["real_gauge"] if adv_mode == "gauge" else x
            fake_img = od["fake_gauge"] if adv_mode == "gauge" else od["x_hat"]
            opt_d.zero_grad(set_to_none=True)
            rs, fs = critic(real_img), critic(fake_img)
            d_loss = F.relu(1.0 - rs).mean() + F.relu(1.0 + fs).mean()
            d_loss.backward()
            opt_d.step()
            d_loss_v = float(d_loss.detach().cpu())
            d_acc = d_accuracy(rs, fs)
            d_hist.append(d_acc)
            generator.train()

        opt_g.zero_grad(set_to_none=True)
        out = p78.forward_candidate(generator, measurement, x, y, config)
        rec = p69b.charbonnier(out["x_hat"], x)
        adv = torch.zeros((), device=device)
        if critic is not None:
            critic.eval()
            fake_img = out["fake_gauge"] if adv_mode == "gauge" else out["x_hat"]
            adv = -critic(fake_img).mean()
        loss = rec + float(beta) * adv
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss seed={seed_id} arm={arm} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        row = {
            "seed": int(seed_id),
            "arm": arm,
            "step": step,
            "paired_loader_seed": loader_seed,
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec.detach().cpu()),
            "loss_adv": float(adv.detach().cpu()),
            "loss_d": d_loss_v,
            "d_accuracy": d_acc,
            "beta": float(beta),
            "adv_mode": adv_mode,
        }
        if step % EVAL_EVERY == 0 or step == STEP_BUDGET:
            vm = p78.eval_val_loss(generator, measurement, val, config, device)
            row.update(vm)
            final_metrics = vm
            append_log(f"train_eval seed={seed_id} arm={arm} step={step} val_rec={vm['val_rec_loss']:.6g}")
            if vm["val_rec_loss"] < best_val:
                best_val = float(vm["val_rec_loss"])
                best_step = step
                save_phase81_checkpoint(best_path, seed_id, arm, step, generator, opt_g, config, vm, loader_seed, beta, adv_mode, critic, opt_d)
        rows.append(row)

    save_phase81_checkpoint(final_path, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, adv_mode, critic, opt_d)
    if not best_path.exists():
        best_val = float(final_metrics.get("val_rec_loss", float("nan")))
        best_step = STEP_BUDGET
        save_phase81_checkpoint(best_path, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, adv_mode, critic, opt_d)
    summary = {
        "seed": int(seed_id),
        "arm": arm,
        "steps": STEP_BUDGET,
        "paired_loader_seed": int(loader_seed),
        "beta": float(beta),
        "adv_mode": adv_mode,
        "best_val_rec_loss": best_val,
        "best_step": best_step,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": p69a.sha256_file(best_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": p69a.sha256_file(final_path),
        "d_accuracy_last_mean": float(np.nanmean(d_hist[-50:])) if d_hist else float("nan"),
        "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_hist[-50:]) > 0.95) if d_hist else False,
    }
    write_csv(arm_dir / "training_log.csv", rows)
    save_json(arm_dir / "training_summary.json", summary)
    del generator, opt_g, critic, opt_d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_log(f"train_complete seed={seed_id} arm={arm} best_step={best_step} best_val={best_val:.6g}")
    return summary, rows, best_path


def load_checkpoint_for_eval(path: Path, config: dict[str, Any], measurement, device: torch.device):
    return p78.load_probe_checkpoint_for_eval(path, config, measurement, device)


def compute_lpips_any(seed_dir: Path, test, outputs: dict[str, np.ndarray], device: torch.device, per_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    try:
        import lpips
    except Exception as exc:
        write_csv(seed_dir / "lpips_or_dists_results.csv", [{"metric_package": "LPIPS", "available": False, "reason": repr(exc)}])
        return

    def prep(arr: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(arr.astype(np.float32))
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = x.repeat(1, 3, 1, 1)
        return x * 2.0 - 1.0

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true = test.x[:, 0].numpy().astype(np.float32)
    true_t = prep(true)
    vals_by_arm: dict[str, np.ndarray] = {}
    per_lp: list[dict[str, Any]] = []
    with torch.no_grad():
        for arm, arr in outputs.items():
            pred_t = prep(np.clip(arr.astype(np.float32), 0, 1))
            vals: list[float] = []
            for i in range(0, pred_t.shape[0], 16):
                vals.extend(loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device)).reshape(-1).detach().cpu().numpy().astype(float).tolist())
            vals_by_arm[arm] = np.asarray(vals, dtype=np.float64)
            for j, v in enumerate(vals):
                per_lp.append({"arm": arm, "sample_index": int(test.indices[j]), "sample_ordinal": j, "lpips": float(v)})
    write_csv(seed_dir / "lpips_per_sample.csv", per_lp)
    write_csv(
        seed_dir / "lpips_or_dists_results.csv",
        [
            {
                "metric_package": "LPIPS",
                "available": True,
                "arm": arm,
                "n": int(len(vals)),
                "lpips_mean": float(vals.mean()),
                "lpips_median": float(np.median(vals)),
                "lpips_std": float(vals.std()),
            }
            for arm, vals in vals_by_arm.items()
        ],
    )
    lookup = {(str(r["arm"]), int(r["sample_index"])): float(r["lpips"]) for r in per_lp}
    for row in per_rows:
        key = (str(row["arm"]), int(row["sample_index"]))
        if key in lookup:
            row["lpips"] = lookup[key]
    for row in eval_rows:
        arm = str(row["arm"])
        if arm in vals_by_arm:
            vals = vals_by_arm[arm]
            row["lpips_mean"] = float(vals.mean())
            row["lpips_median"] = float(np.median(vals))
            row["lpips_std"] = float(vals.std())


def paired_compare(rows: list[dict[str, Any]], left: str, right: str, seed: int) -> list[dict[str, Any]]:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), {})[int(row["sample_index"])] = row
    common = sorted(set(by_arm.get(left, {})) & set(by_arm.get(right, {})))
    metrics = [
        ("psnr", "higher"),
        ("ssim", "higher"),
        ("relmeaserr_unclipped_float64", "lower"),
        ("correction_norm_rel", "lower"),
        ("rapsd_distance", "lower"),
        ("gradient_mean_abs_error", "lower"),
        ("highfreq_ratio_abs_error", "lower"),
        ("p0_l2", "lower"),
        ("lpips", "lower"),
    ]
    out: list[dict[str, Any]] = []
    for metric, direction in metrics:
        if not common or metric not in by_arm[left][common[0]] or metric not in by_arm[right][common[0]]:
            continue
        lvals = np.asarray([float(by_arm[left][i][metric]) for i in common], dtype=np.float64)
        rvals = np.asarray([float(by_arm[right][i][metric]) for i in common], dtype=np.float64)
        delta = rvals - lvals
        improvement = delta if direction == "higher" else -delta
        mean, lo, hi = p69b.bootstrap_ci(improvement, seed=81000 + seed * 100 + len(out), n_boot=1000)
        out.append(
            {
                "seed": int(seed),
                "pair": f"{left}_vs_{right}",
                "left_arm": left,
                "right_arm": right,
                "metric": metric,
                "direction": direction,
                f"mean_{left}": float(np.nanmean(lvals)),
                f"mean_{right}": float(np.nanmean(rvals)),
                "mean_right_minus_left": float(np.nanmean(delta)),
                f"improvement_positive_means_{right}_better": mean,
                "ci_low": lo,
                "ci_high": hi,
                f"ci_excludes_zero_in_favor_of_{right}": bool(lo > 0),
            }
        )
    return out


def evaluate_seed(seed_id: int) -> None:
    configure_phase78_globals(seed_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, measurement, _train, _val, test, split = prepare_common(device)
    seed_dir = ensure_dir(OUT / f"seed{seed_id:02d}")
    eval_dir = ensure_dir(seed_dir / "evaluation")
    save_json(seed_dir / "split_manifest.json", split)
    arms: dict[str, Path | None] = {"A": None}
    for arm in ["B", "C", "D_standard"]:
        path = best_ckpt(seed_id, arm)
        if path is not None:
            arms[arm] = path
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, path in arms.items():
        if arm == "A":
            gen = p78.load_generator_96(config, measurement, device, train=False)
        else:
            gen = load_checkpoint_for_eval(path, config, measurement, device)
        agg, per, arr = p78.evaluate_arm(arm, gen, measurement, test, config, device, eval_dir)
        agg["seed"] = int(seed_id)
        agg["checkpoint"] = str(path) if path is not None else str(p78.RAD5_CHECKPOINT)
        agg["checkpoint_sha256"] = p69a.sha256_file(path) if path is not None else p69a.sha256_file(p78.RAD5_CHECKPOINT)
        eval_rows.append(agg)
        for row in per:
            row["seed"] = int(seed_id)
        per_rows.extend(per)
        outputs[arm] = arr
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    compute_lpips_any(seed_dir, test, outputs, device, per_rows, eval_rows)
    comp: list[dict[str, Any]] = []
    if "B" in outputs and "C" in outputs:
        comp.extend(paired_compare(per_rows, "B", "C", seed_id))
    if "B" in outputs and "D_standard" in outputs:
        comp.extend(paired_compare(per_rows, "B", "D_standard", seed_id))
    if "C" in outputs and "D_standard" in outputs:
        comp.extend(paired_compare(per_rows, "C", "D_standard", seed_id))
    write_csv(seed_dir / "evaluation_metrics.csv", eval_rows)
    write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
    write_csv(seed_dir / "pairwise_comparisons.csv", comp)
    p69b.save_visual_grid(seed_dir, p69b.SplitCache(test.name, test.x, test.y, test.labels, test.indices), outputs, n=6)
    p69b.save_rapsd_plot(seed_dir, p69b.SplitCache(test.name, test.x, test.y, test.labels, test.indices), outputs)
    save_seed_manifest(seed_id)
    append_log(f"evaluate_complete seed={seed_id} arms={','.join(outputs)}")


def train_pair(seed_id: int) -> None:
    if seed_id == 1:
        append_log("train_pair_skip seed=1 uses Phase78 read-only B/C")
        evaluate_seed(seed_id)
        return
    configure_phase78_globals(seed_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, measurement, train, val, _test, split = prepare_common(device)
    save_json(OUT / f"seed{seed_id:02d}" / "split_manifest.json", split)
    beta, beta_meta = beta0_value(device, config, measurement, train)
    save_json(OUT / f"seed{seed_id:02d}" / "beta_meta.json", beta_meta)
    base_seed = 785000 + 100 * int(seed_id)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for arm, adv_mode, arm_beta in [("B", "none", 0.0), ("C", "gauge", beta)]:
        ckpt = phase81_ckpt(seed_id, arm)
        if ckpt.exists():
            summary_path = OUT / f"seed{seed_id:02d}" / arm / "training_summary.json"
            summaries.append(read_json(summary_path) if summary_path.exists() else {"seed": seed_id, "arm": arm, "status": "existing"})
            append_log(f"train_pair_skip_existing seed={seed_id} arm={arm}")
            continue
        set_all_seeds(base_seed)
        summary, arm_rows, _best = train_arm_mode(seed_id, arm, adv_mode, arm_beta, config, measurement, train, val, device)
        summaries.append(summary)
        rows.extend(arm_rows)
    write_csv(OUT / f"seed{seed_id:02d}" / "training_log_pair.csv", rows)
    save_json(OUT / f"seed{seed_id:02d}" / "pair_training_summary.json", {"seed": int(seed_id), "summaries": summaries, "beta": beta})
    evaluate_seed(seed_id)


def train_standard(seed_id: int) -> None:
    configure_phase78_globals(seed_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, measurement, train, val, _test, split = prepare_common(device)
    save_json(OUT / f"seed{seed_id:02d}" / "split_manifest.json", split)
    beta, beta_meta = beta0_value(device, config, measurement, train)
    save_json(OUT / f"seed{seed_id:02d}" / "beta_meta.json", beta_meta)
    ckpt = phase81_ckpt(seed_id, "D_standard")
    if not ckpt.exists():
        base_seed = 785000 + 100 * int(seed_id)
        set_all_seeds(base_seed)
        summary, rows, _best = train_arm_mode(seed_id, "D_standard", "standard", beta, config, measurement, train, val, device)
        write_csv(OUT / f"seed{seed_id:02d}" / "training_log_standard.csv", rows)
        save_json(OUT / f"seed{seed_id:02d}" / "D_standard_summary.json", summary)
    else:
        append_log(f"train_standard_skip_existing seed={seed_id}")
    evaluate_seed(seed_id)


def save_seed_manifest(seed_id: int) -> None:
    seed_dir = OUT / f"seed{seed_id:02d}"
    rows: list[dict[str, Any]] = []
    for arm in ["B", "C", "D_standard"]:
        path = best_ckpt(seed_id, arm)
        if path is not None:
            rows.append({"kind": "checkpoint", "seed": seed_id, "arm": arm, "path": str(path), "sha256": p69a.sha256_file(path)})
    for npz in sorted((seed_dir / "evaluation").glob("per_sample_outputs_*.npz")):
        arm = npz.stem.replace("per_sample_outputs_", "")
        rows.append({"kind": "per_sample_outputs", "seed": seed_id, "arm": arm, "path": str(npz), "sha256": p69a.sha256_file(npz)})
    for path in [seed_dir / "per_sample_metrics.csv", seed_dir / "evaluation_metrics.csv", seed_dir / "pairwise_comparisons.csv"]:
        if path.exists():
            rows.append({"kind": "table", "seed": seed_id, "arm": "", "path": str(path), "sha256": p69a.sha256_file(path)})
    write_csv(seed_dir / "artifact_hashes.csv", rows)


def aggregate() -> None:
    all_metrics: list[dict[str, Any]] = []
    all_comp: list[dict[str, Any]] = []
    all_hashes: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in SEEDS:
        seed_dir = OUT / f"seed{seed:02d}"
        metrics_path = seed_dir / "evaluation_metrics.csv"
        comp_path = seed_dir / "pairwise_comparisons.csv"
        hashes_path = seed_dir / "artifact_hashes.csv"
        if not metrics_path.exists():
            missing.append(f"missing evaluation_metrics for seed{seed:02d}")
        else:
            all_metrics.extend(read_csv(metrics_path))
        if not comp_path.exists():
            missing.append(f"missing pairwise_comparisons for seed{seed:02d}")
        else:
            all_comp.extend(read_csv(comp_path))
        if hashes_path.exists():
            all_hashes.extend(read_csv(hashes_path))
    write_csv(OUT / "all_seed_evaluation_metrics.csv", all_metrics)
    write_csv(OUT / "all_seed_pairwise_comparisons.csv", all_comp)
    write_csv(OUT / "all_seed_artifact_hashes.csv", all_hashes)

    task2 = [
        r
        for r in all_comp
        if r.get("pair") == "B_vs_C" and r.get("metric") in {"lpips", "rapsd_distance"}
    ]
    write_csv(OUT / "task2_C_vs_B_lpips_rapsd_by_seed.csv", task2)
    by_seed_metric = {(int(r["seed"]), r["metric"]): float(r["mean_right_minus_left"]) for r in task2}
    task2_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        lp = by_seed_metric.get((seed, "lpips"), float("nan"))
        rp = by_seed_metric.get((seed, "rapsd_distance"), float("nan"))
        task2_rows.append(
            {
                "seed": seed,
                "C_minus_B_lpips": lp,
                "C_minus_B_rapsd_distance": rp,
                "lpips_C_better": bool(lp < 0) if math.isfinite(lp) else False,
                "rapsd_C_better": bool(rp < 0) if math.isfinite(rp) else False,
            }
        )
    task2_pass = all(r["lpips_C_better"] and r["rapsd_C_better"] for r in task2_rows)
    write_csv(OUT / "task2_seed_signs.csv", task2_rows)

    task3 = [
        r
        for r in all_comp
        if r.get("pair") == "C_vs_D_standard" and r.get("metric") in {"lpips", "rapsd_distance", "psnr", "ssim", "relmeaserr_unclipped_float64"}
    ]
    write_csv(OUT / "task3_C_vs_D_standard_by_seed.csv", task3)
    task3_focus: list[dict[str, Any]] = []
    for row in task3:
        metric = str(row.get("metric", ""))
        direction = str(row.get("direction", ""))
        if metric not in {"lpips", "rapsd_distance", "psnr", "ssim"}:
            continue
        delta = float(row["mean_right_minus_left"])
        d_better = delta > 0 if direction == "higher" else delta < 0
        task3_focus.append(
            {
                "seed": int(row["seed"]),
                "metric": metric,
                "direction": direction,
                "mean_C": float(row.get("mean_C", "nan")),
                "mean_D_standard": float(row.get("mean_D_standard", "nan")),
                "D_minus_C": delta,
                "D_better_by_mean": bool(d_better),
                "improvement_positive_D_better": float(row.get("improvement_positive_means_D_standard_better", "nan")),
                "improvement_ci_low": float(row.get("ci_low", "nan")),
                "improvement_ci_high": float(row.get("ci_high", "nan")),
            }
        )
    write_csv(OUT / "task3_standard_vs_gauge_focus.csv", task3_focus)

    standard_seed_count = len({int(r["seed"]) for r in all_metrics if r.get("arm") == "D_standard"})
    lines = [
        "# Phase81 96px Rad-5 Paper Completion Report",
        "",
        f"- output_dir: `{OUT}`",
        f"- missing_items: `{len(missing)}`",
        "- seed01 B/C source: Phase78 read-only.",
        "- seed02/03 B/C source: Phase81 trained if present.",
        "- D_standard: standard image-input cGAN, real x vs deployed fake x_hat.",
        "",
        "## Task2 C vs B LPIPS/RAPSD Signs",
        "",
        table(task2_rows, ["seed", "C_minus_B_lpips", "C_minus_B_rapsd_distance", "lpips_C_better", "rapsd_C_better"]),
        "",
        f"Task2 3/3 same-sign decision: `{task2_pass}`.",
        "",
        "## Task3 Standard D Coverage",
        "",
        f"- D_standard evaluated seeds: `{standard_seed_count}`.",
        "",
        "For this table, `D_minus_C` is the raw mean difference. Lower is better for LPIPS/RAPSD, higher is better for PSNR/SSIM.",
        "",
        table(task3_focus, ["seed", "metric", "mean_C", "mean_D_standard", "D_minus_C", "D_better_by_mean", "improvement_ci_low", "improvement_ci_high"]),
        "",
        "Full pairwise rows, including RelMeasErr, are in `task3_C_vs_D_standard_by_seed.csv`.",
        "",
        "## Missing",
        "",
        *[f"- {m}" for m in missing],
        "",
    ]
    write_text(OUT / "PHASE81_96PX_RAD5_COMPLETION_REPORT.md", "\n".join(lines))
    save_json(
        OUT / "PHASE81_MANIFEST.json",
        {
            "phase": "Phase81_96px_rad5_paper_completion",
            "output_dir": str(OUT),
            "seeds": SEEDS,
            "missing": missing,
            "task2_all_three_lpips_and_rapsd_C_better": task2_pass,
            "standard_seed_count": standard_seed_count,
            "artifact_hash_count": len(all_hashes),
        },
    )
    append_log(f"aggregate_complete missing={len(missing)} task2_pass={task2_pass}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase81 96px Rad-5 paper completion runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    pair_p = sub.add_parser("pair")
    pair_p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    std_p = sub.add_parser("standard")
    std_p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    sub.add_parser("aggregate")
    args = parser.parse_args()

    if args.cmd == "preflight":
        preflight()
    elif args.cmd == "pair":
        preflight()
        train_pair(int(args.seed))
    elif args.cmd == "standard":
        preflight()
        train_standard(int(args.seed))
    elif args.cmd == "evaluate":
        preflight()
        evaluate_seed(int(args.seed))
    elif args.cmd == "aggregate":
        aggregate()
    else:
        raise AssertionError(args.cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
