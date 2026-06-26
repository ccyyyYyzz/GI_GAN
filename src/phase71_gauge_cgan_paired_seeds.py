from __future__ import annotations

import csv
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from .eval import make_measurement
from .utils import set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"

SEEDS = [1, 2, 3]
TRAIN_COUNT = 1024
VAL_COUNT = 256
TEST_COUNT = 256
BATCH_SIZE = 8
STEP_BUDGET = 300
EVAL_EVERY = 100
RELMEASERR_ABS_DELTA_GUARDRAIL = 1e-4
PSNR_LOSS_GUARDRAIL_DB = 0.3


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
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |",
        "| " + " | ".join("-" * widths[c] for c in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) + " |")
    return "\n".join(lines)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "/")
    return json.dumps(text)


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k2, v2 in value.items():
                lines.append(f"  {k2}: {yaml_scalar(v2)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append("  -")
                    for k2, v2 in item.items():
                        lines.append(f"    {k2}: {yaml_scalar(v2)}")
                else:
                    lines.append(f"  - {yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    write_text(path, "\n".join(lines) + "\n")


def unsafe_stop(failures: list[str]) -> int:
    write_text(
        OUT / "UNSAFE_TO_RUN.md",
        "\n".join(
            [
                "# UNSAFE TO RUN",
                "",
                "Phase71 stopped before paired-seed training.",
                "",
                "## Critical Failures",
                "",
                *[f"- {failure}" for failure in failures],
                "",
                "No Phase71 seed training was run after this failure.",
                "",
            ]
        ),
    )
    append_log("unsafe_stop")
    return 2


def load_locked_protocol() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    for path in [PH69B, PH70, p69a.CHECKPOINT, p69a.A_SCR5, p69a.SPLIT_TRAIN, p69a.SPLIT_EVAL, p69a.EVAL_CACHE]:
        if not path.exists():
            failures.append(f"Missing required path: {path}")
    if failures:
        return {}, failures

    split = read_json(PH69B / "split_manifest.json")
    b_summary = read_json(PH69B / "pilot" / "armB" / "training_summary.json")
    c_summary = read_json(PH69B / "pilot" / "armC" / "training_summary.json")
    beta = read_csv(PH69B / "beta_calibration.csv")[0]
    preflight = read_json(PH69B / "preflight_checks.json")
    train_idx = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_idx = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    a_np = np.load(p69a.A_SCR5).astype(np.float32)

    if p69a.sha256_np(train_idx, sort_int64=True) != preflight["train_split_expected_sha256"]:
        failures.append("Train split hash cannot be reconstructed.")
    if p69a.sha256_np(eval_idx, sort_int64=True) != preflight["test_split_expected_sha256"]:
        failures.append("Eval/test split hash cannot be reconstructed.")
    if p69a.sha256_file(p69a.CHECKPOINT) != preflight["checkpoint_expected_sha256"]:
        failures.append("Published mean-mode checkpoint hash mismatch.")
    if str(b_summary["steps"]) != str(c_summary["steps"]):
        failures.append("Phase69B B/C step budgets differ; cannot lock protocol.")
    if int(split["train_count"]) != TRAIN_COUNT or int(split["val_count"]) != VAL_COUNT or int(split["test_count"]) != TEST_COUNT:
        failures.append("Phase69B split counts differ from Phase71 locked constants.")

    phase69b_source = Path(p69b.__file__).read_text(encoding="utf-8")
    phase69b_data_order_compatible = "seed=69040 + (1 if adversarial else 0)" not in phase69b_source
    protocol = {
        "phase": "Phase71",
        "output_dir": str(OUT),
        "source_phase69B": str(PH69B),
        "source_phase70": str(PH70),
        "phase69B_reference_seed_counted": False,
        "phase69B_reference_seed_not_counted_reason": "Phase69B train_arm used different B/C loader seeds (69040 vs 69041), so it is reference-only under Phase71 paired-data-order rules.",
        "phase69B_source_data_order_compatible": bool(phase69b_data_order_compatible),
        "source_checkpoint": str(p69a.CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(p69a.CHECKPOINT),
        "scr5_A_path": str(p69a.A_SCR5),
        "scr5_A_sha256_float32": p69a.sha256_np(a_np),
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "train_full_sorted_sha256": split["train_full_sorted_sha256"],
        "eval_full_sorted_sha256": split["eval_full_sorted_sha256"],
        "train_indices_sha256": split["train_indices_sha256"],
        "val_indices_sha256": split["val_indices_sha256"],
        "test_indices_sha256": split["test_indices_sha256"],
        "beta0": float(beta["selected_beta0"]),
        "batch_size": BATCH_SIZE,
        "step_budget": STEP_BUDGET,
        "eval_every": EVAL_EVERY,
        "generator_optimizer": "Adam(lr=2e-5, betas=(0.9, 0.999))",
        "critic_optimizer": "Adam(lr=2e-4, betas=(0.5, 0.9))",
        "checkpoint_selection_rule": "best_by_val_rec_loss over validation split at every 100 steps",
        "D_architecture": "phase69A.PatchCritic(in_channels=1), gauge-only input",
        "gauge_real": "tilde_x_real = P0 x + B_lambda y",
        "gauge_fake": "tilde_x_fake = P0 v_theta + B_lambda y",
        "deployment": "x_hat = Pi_y^lambda(v_theta)",
        "forbidden_D_inputs": "Au-y, RelMeasErr, correction vector, Pi_y(v)-v, B_lambda(Av-y)",
        "relmeaserr_definition": "unclipped float64 against recorded y",
        "psnr_ssim_definition": "clipped/display image",
        "seed_ids": SEEDS,
        "psnr_loss_guardrail_db": PSNR_LOSS_GUARDRAIL_DB,
        "relmeaserr_abs_delta_guardrail": RELMEASERR_ABS_DELTA_GUARDRAIL,
    }
    return protocol, failures


def write_protocol_lock(protocol: dict[str, Any]) -> None:
    write_simple_yaml(OUT / "phase71_protocol_config.yaml", protocol)
    lines = [
        "# Phase71 Protocol Lock",
        "",
        f"Output directory: `{OUT}`",
        "",
        "## Locked Sources",
        "",
        f"- Phase69B: `{PH69B}`",
        f"- Phase70: `{PH70}`",
        f"- Scr-5 checkpoint: `{protocol['source_checkpoint']}`",
        f"- checkpoint SHA256: `{protocol['source_checkpoint_sha256']}`",
        f"- Scr-5 A SHA256 float32: `{protocol['scr5_A_sha256_float32']}`",
        "",
        "## Phase69B Reference Seed",
        "",
        "- Counted as Phase71 seed: `False`",
        f"- Reason: {protocol['phase69B_reference_seed_not_counted_reason']}",
        "",
        "## Split Hashes",
        "",
        table(
            [
                {"name": "train_full_sorted", "sha256": protocol["train_full_sorted_sha256"]},
                {"name": "eval_full_sorted", "sha256": protocol["eval_full_sorted_sha256"]},
                {"name": "train_subset", "sha256": protocol["train_indices_sha256"]},
                {"name": "val_subset", "sha256": protocol["val_indices_sha256"]},
                {"name": "test_subset", "sha256": protocol["test_indices_sha256"]},
            ],
            ["name", "sha256"],
        ),
        "",
        "## Training Lock",
        "",
        f"- seeds: `{SEEDS}`",
        f"- beta0: `{protocol['beta0']}`",
        f"- step budget: `{STEP_BUDGET}`",
        f"- batch size: `{BATCH_SIZE}`",
        f"- generator optimizer: `{protocol['generator_optimizer']}`",
        f"- critic optimizer: `{protocol['critic_optimizer']}`",
        f"- checkpoint selection: `{protocol['checkpoint_selection_rule']}`",
        "- B and C use identical init checkpoint, train split, data-order seed, generator optimizer, step budget, validation split, and selection rule.",
        "- C differs only by the gauge-equalized adversarial branch.",
        "",
        "## Gauge / Deployment",
        "",
        f"- real gauge: `{protocol['gauge_real']}`",
        f"- fake gauge: `{protocol['gauge_fake']}`",
        f"- deployment: `{protocol['deployment']}`",
        "- D input is residual-shortcut-free; forbidden residual/correction/certificate features are not supplied.",
        "",
        "No first-paper checkpoint, table, title, abstract, or main result is modified.",
        "",
    ]
    write_text(OUT / "PHASE71_PROTOCOL_LOCK.md", "\n".join(lines))


def make_phase71_config(device: torch.device) -> tuple[dict[str, Any], Any, torch.Tensor, float]:
    config = p69b.make_config(str(device), BATCH_SIZE)
    config["output_dir"] = str(OUT)
    config["batch_size"] = BATCH_SIZE
    config["num_workers"] = 0
    measurement = make_measurement(config, device)
    a_np = np.load(p69a.A_SCR5).astype(np.float32)
    A = torch.from_numpy(a_np).to(device)
    measurement.set_A_override(A, metadata={"phase": "phase71", "tensor_sha256": p69a.sha256_np(a_np)}, rebuild_cache=True)
    lambda_dc = float(config["lambda_solver"])
    return config, measurement, A, lambda_dc


def save_phase71_checkpoint(
    path: Path,
    seed_id: int,
    arm: str,
    step: int,
    generator,
    optimizer_g,
    config: dict[str, Any],
    metrics: dict[str, Any],
    loader_seed: int,
    critic=None,
    optimizer_d=None,
    beta: float = 0.0,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase71",
        "seed_id": int(seed_id),
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(p69a.CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(p69a.CHECKPOINT),
        "beta": float(beta),
        "paired_loader_seed": int(loader_seed),
        "checkpoint_selection_rule": "best_by_val_rec_loss",
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    torch.save(payload, path)


def train_arm_paired(
    seed_id: int,
    arm: str,
    generator,
    train: p69b.SplitCache,
    val: p69b.SplitCache,
    A: torch.Tensor,
    lambda_dc: float,
    config: dict[str, Any],
    device: torch.device,
    seed_dir: Path,
    loader_seed: int,
    beta: float,
    adversarial: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    arm_dir = ensure_dir(seed_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    p69b.append_log(seed_dir, f"train_start seed={seed_id} arm={arm} loader_seed={loader_seed} adversarial={adversarial}")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adversarial else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = p69b.cycle_loader(p69b.make_loader(train, int(config["batch_size"]), shuffle=True, seed=loader_seed))
    log_rows: list[dict[str, Any]] = []
    d_acc_history: list[float] = []
    best_val = float("inf")
    best_step = -1
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics: dict[str, Any] = {}

    for step in range(1, STEP_BUDGET + 1):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)
        d_loss_value = float("nan")
        d_acc_value = float("nan")
        if adversarial and critic is not None and opt_d is not None:
            generator.eval()
            critic.train()
            with torch.no_grad():
                out_d = p69b.forward_candidate(generator, x, y, A, lambda_dc, config)
            opt_d.zero_grad(set_to_none=True)
            real_score = critic(out_d["real_gauge"])
            fake_score = critic(out_d["fake_gauge"])
            d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            d_loss.backward()
            opt_d.step()
            d_loss_value = float(d_loss.detach().cpu())
            d_acc_value = p69b.d_accuracy(real_score, fake_score)
            d_acc_history.append(d_acc_value)
            generator.train()

        opt_g.zero_grad(set_to_none=True)
        out = p69b.forward_candidate(generator, x, y, A, lambda_dc, config)
        rec_loss = p69b.charbonnier(out["x_hat"], x)
        adv_loss = torch.zeros((), device=device)
        if adversarial and critic is not None:
            critic.eval()
            adv_loss = -critic(out["fake_gauge"]).mean()
        loss = rec_loss + float(beta) * adv_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss seed={seed_id} arm={arm} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        row = {
            "seed": seed_id,
            "arm": arm,
            "step": step,
            "paired_loader_seed": loader_seed,
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec_loss.detach().cpu()),
            "loss_adv": float(adv_loss.detach().cpu()),
            "loss_d": d_loss_value,
            "d_accuracy": d_acc_value,
            "beta": float(beta),
        }
        if step % EVAL_EVERY == 0 or step == STEP_BUDGET:
            val_metrics = p69b.evaluate_val_loss(generator, val, A, lambda_dc, config, device, max_batches=None)
            row.update(val_metrics)
            final_metrics = val_metrics
            if val_metrics["val_rec_loss"] < best_val:
                best_val = float(val_metrics["val_rec_loss"])
                best_step = step
                save_phase71_checkpoint(best_path, seed_id, arm, step, generator, opt_g, config, val_metrics, loader_seed, critic=critic, optimizer_d=opt_d, beta=beta)
            p69b.append_log(seed_dir, f"train_eval seed={seed_id} arm={arm} step={step} val_rec={val_metrics['val_rec_loss']:.6g}")
        log_rows.append(row)

    final_path = ckpt_dir / "final.pt"
    save_phase71_checkpoint(final_path, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, critic=critic, optimizer_d=opt_d, beta=beta)
    if not best_path.exists():
        best_step = STEP_BUDGET
        best_val = float(final_metrics.get("val_rec_loss", float("nan")))
        save_phase71_checkpoint(best_path, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, critic=critic, optimizer_d=opt_d, beta=beta)

    summary = {
        "seed": int(seed_id),
        "arm": arm,
        "steps": STEP_BUDGET,
        "paired_loader_seed": int(loader_seed),
        "finite_losses": True,
        "best_val_rec_loss": best_val,
        "best_step": int(best_step),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": p69a.sha256_file(best_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": p69a.sha256_file(final_path),
        "d_accuracy_last_mean": float(np.nanmean(d_acc_history[-50:])) if d_acc_history else float("nan"),
        "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_acc_history[-50:]) > 0.95) if d_acc_history else False,
    }
    write_csv(arm_dir / "training_log.csv", log_rows)
    save_json(arm_dir / "training_summary.json", summary)
    p69b.append_log(seed_dir, f"train_complete seed={seed_id} arm={arm} best_step={best_step} best_val={best_val:.6g}")
    return summary, log_rows, best_path


def prep_lpips(arr: np.ndarray) -> torch.Tensor:
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    tensor = torch.from_numpy(arr[:, None, :, :])
    return tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0


def compute_lpips(seed_dir: Path, test: p69b.SplitCache, outputs: dict[str, np.ndarray], device: torch.device, per_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    try:
        import lpips
    except Exception as exc:
        write_csv(
            seed_dir / "lpips_or_dists_results.csv",
            [{"metric_package": "LPIPS", "module": "lpips", "available": False, "note": str(exc)}],
        )
        return

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true = test.x[:, 0].numpy().astype(np.float32)
    true_t = prep_lpips(true)
    lpips_rows: list[dict[str, Any]] = []
    per_lpips_rows: list[dict[str, Any]] = []
    lpips_by_arm: dict[str, np.ndarray] = {}
    sample_indices = test.indices.numpy().astype(int).tolist()
    with torch.no_grad():
        for arm in ["A", "B", "C"]:
            pred = outputs[arm].astype(np.float32)
            pred_t = prep_lpips(pred)
            vals: list[float] = []
            for i in range(0, pred_t.shape[0], 16):
                dist = loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device))
                vals.extend(dist.reshape(-1).detach().cpu().numpy().astype(float).tolist())
            vals_np = np.asarray(vals, dtype=np.float64)
            lpips_by_arm[arm] = vals_np
            lpips_rows.append(
                {
                    "metric_package": "LPIPS",
                    "module": "lpips",
                    "available": True,
                    "arm": arm,
                    "n": int(vals_np.shape[0]),
                    "lpips_mean": float(vals_np.mean()),
                    "lpips_median": float(np.median(vals_np)),
                    "lpips_std": float(vals_np.std()),
                    "note": "",
                }
            )
            for ordinal, value in enumerate(vals):
                per_lpips_rows.append(
                    {
                        "arm": arm,
                        "sample_ordinal": ordinal,
                        "sample_index": sample_indices[ordinal],
                        "lpips": float(value),
                    }
                )

    lpips_rows.extend(
        [
            {"metric_package": "DISTS", "module": "DISTS_pytorch", "available": False, "arm": "", "n": "", "lpips_mean": "", "lpips_median": "", "lpips_std": "", "note": "module unavailable"},
            {"metric_package": "KID", "module": "torchmetrics/cleanfid", "available": "not_run", "arm": "", "n": int(test.x.shape[0]), "lpips_mean": "", "lpips_median": "", "lpips_std": "", "note": "small-sample KID not used for decision"},
        ]
    )
    write_csv(seed_dir / "lpips_or_dists_results.csv", lpips_rows)
    write_csv(seed_dir / "lpips_per_sample.csv", per_lpips_rows)

    lpips_lookup = {(row["arm"], int(row["sample_index"])): float(row["lpips"]) for row in per_lpips_rows}
    for row in per_rows:
        key = (row["arm"], int(row["sample_index"]))
        if key in lpips_lookup:
            row["lpips"] = lpips_lookup[key]
    for row in eval_rows:
        arm = row["arm"]
        vals = lpips_by_arm.get(arm)
        if vals is not None:
            row["lpips_mean"] = float(vals.mean())
            row["lpips_median"] = float(np.median(vals))
            row["lpips_std"] = float(vals.std())

    b = lpips_by_arm["B"]
    c = lpips_by_arm["C"]
    improvement = b - c
    mean, lo, hi = p69b.bootstrap_ci(improvement, seed=71070 + int(seed_dir.name.replace("seed", "")), n_boot=1000)
    comparison_rows.append(
        {
            "metric": "lpips",
            "direction": "lower",
            "mean_B": float(b.mean()),
            "mean_C": float(c.mean()),
            "mean_C_minus_B": float((c - b).mean()),
            "improvement_positive_means_C_better": mean,
            "ci_low": lo,
            "ci_high": hi,
            "ci_excludes_zero_in_favor_of_C": bool(lo > 0),
        }
    )


def evaluate_seed(seed_id: int, seed_dir: Path, b_best: Path, c_best: Path, config: dict[str, Any], measurement, test: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    eval_dir = ensure_dir(seed_dir / "evaluation")
    gen_a, _ = p69b.load_generator_from_checkpoint(config, measurement, device)
    gen_a.eval()
    gen_b = p69b.load_generator_checkpoint_for_eval(b_best, config, measurement, device)
    gen_c = p69b.load_generator_checkpoint_for_eval(c_best, config, measurement, device)
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, gen in [("A", gen_a), ("B", gen_b), ("C", gen_c)]:
        agg, per, out_arr = p69b.evaluate_arm(arm, gen, test, A, lambda_dc, config, device, eval_dir)
        agg["seed"] = seed_id
        eval_rows.append(agg)
        for row in per:
            row["seed"] = seed_id
        per_rows.extend(per)
        outputs[arm] = out_arr
    comparison_rows = p69b.paired_comparison(per_rows)
    compute_lpips(seed_dir, test, outputs, device, per_rows, comparison_rows, eval_rows)
    write_csv(seed_dir / "evaluation_metrics.csv", eval_rows)
    write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
    write_csv(seed_dir / "paired_comparison_C_vs_B.csv", comparison_rows)
    p69b.save_visual_grid(seed_dir, test, outputs, n=6)
    p69b.save_rapsd_plot(seed_dir, test, outputs)
    return eval_rows, per_rows, comparison_rows, outputs


def run_one_seed(seed_id: int, protocol: dict[str, Any], config: dict[str, Any], measurement, train: p69b.SplitCache, val: p69b.SplitCache, test: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, device: torch.device) -> dict[str, Any]:
    seed_dir = ensure_dir(OUT / f"seed{seed_id:02d}")
    if (seed_dir / "SEED_DONE.json").exists():
        append_log(f"seed{seed_id:02d}_already_complete")
        return read_json(seed_dir / "SEED_DONE.json")
    if any(seed_dir.iterdir()):
        raise RuntimeError(f"Seed directory is non-empty and incomplete; refusing to overwrite: {seed_dir}")

    write_text(seed_dir / "RUNLOG.md", f"# Phase71 seed{seed_id:02d} Runlog\n")
    loader_seed = 710400 + seed_id
    save_json(
        seed_dir / "seed_config.json",
        {
            "seed": seed_id,
            "paired_loader_seed": loader_seed,
            "source_checkpoint": protocol["source_checkpoint"],
            "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
            "beta0": protocol["beta0"],
            "step_budget": STEP_BUDGET,
            "eval_every": EVAL_EVERY,
            "B_C_same_data_order": True,
        },
    )
    p69b.append_log(seed_dir, f"seed_start seed={seed_id} paired_loader_seed={loader_seed}")

    seed_base = 710000 + seed_id * 100
    set_seed(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    gen_b, _ = p69b.load_generator_from_checkpoint(config, measurement, device)
    b_summary, b_rows, b_best = train_arm_paired(seed_id, "B", gen_b, train, val, A, lambda_dc, config, device, seed_dir, loader_seed, beta=0.0, adversarial=False)
    del gen_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    gen_c, _ = p69b.load_generator_from_checkpoint(config, measurement, device)
    c_summary, c_rows, c_best = train_arm_paired(seed_id, "C", gen_c, train, val, A, lambda_dc, config, device, seed_dir, loader_seed, beta=float(protocol["beta0"]), adversarial=True)
    del gen_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_csv(seed_dir / "training_log.csv", b_rows + c_rows)
    eval_rows, per_rows, comparison_rows, outputs = evaluate_seed(seed_id, seed_dir, b_best, c_best, config, measurement, test, A, lambda_dc, device)
    done = {
        "seed": seed_id,
        "paired_loader_seed": loader_seed,
        "armB_summary": b_summary,
        "armC_summary": c_summary,
        "evaluation_metrics_csv": str(seed_dir / "evaluation_metrics.csv"),
        "paired_comparison_csv": str(seed_dir / "paired_comparison_C_vs_B.csv"),
        "per_sample_metrics_csv": str(seed_dir / "per_sample_metrics.csv"),
        "per_sample_outputs_dir": str(seed_dir / "evaluation"),
        "d_accuracy_last_mean": c_summary["d_accuracy_last_mean"],
        "d_saturated_last_mean_gt_0p95": c_summary["d_saturated_last_mean_gt_0p95"],
    }
    save_json(seed_dir / "SEED_DONE.json", done)
    p69b.append_log(seed_dir, f"seed_complete seed={seed_id}")
    append_log(f"seed{seed_id:02d}_complete")
    return done


def build_combined_outputs(seed_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed_id = int(result["seed"])
        seed_dir = OUT / f"seed{seed_id:02d}"
        for row in read_csv(seed_dir / "evaluation_metrics.csv"):
            row["seed"] = seed_id
            metrics_rows.append(row)
        for row in read_csv(seed_dir / "paired_comparison_C_vs_B.csv"):
            row["seed"] = seed_id
            delta_rows.append(row)
        for arm_key, arm_name in [("armB_summary", "B"), ("armC_summary", "C")]:
            summary = result[arm_key]
            checkpoint_rows.extend(
                [
                    {
                        "seed": seed_id,
                        "arm": arm_name,
                        "kind": "best_by_val",
                        "path": summary["best_checkpoint"],
                        "sha256": summary["best_checkpoint_sha256"],
                    },
                    {
                        "seed": seed_id,
                        "arm": arm_name,
                        "kind": "final",
                        "path": summary["final_checkpoint"],
                        "sha256": summary["final_checkpoint_sha256"],
                    },
                ]
            )
    write_csv(OUT / "scr5_seed_metrics.csv", metrics_rows)
    write_csv(OUT / "scr5_seed_delta_metrics.csv", delta_rows)
    write_csv(OUT / "checkpoint_hashes.csv", checkpoint_rows)
    return metrics_rows, delta_rows, checkpoint_rows


def metric_direction(metric: str) -> str:
    higher = {"psnr", "ssim"}
    return "higher" if metric in higher else "lower"


def seed_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    idx = np.arange(values.shape[0])
    boot = []
    for _ in range(5000):
        sample = rng.choice(idx, size=idx.shape[0], replace=True)
        boot.append(float(np.nanmean(values[sample])))
    return float(np.nanmean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stability_analysis(delta_rows: list[dict[str, Any]], seed_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        by_metric.setdefault(row["metric"], []).append(row)
    stability_rows: list[dict[str, Any]] = []
    for metric, rows in sorted(by_metric.items()):
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        improvements = np.asarray([float(r["improvement_positive_means_C_better"]) for r in rows], dtype=np.float64)
        mean, lo, hi = seed_ci(improvements, seed=71100 + len(stability_rows))
        stability_rows.append(
            {
                "metric": metric,
                "n_seeds": len(rows),
                "direction": rows[0]["direction"],
                "seeds_C_better": int(np.sum(improvements > 0)),
                "all_seeds_C_better": bool(np.all(improvements > 0)),
                "mean_improvement_positive_C_better": mean,
                "std_improvement": float(np.nanstd(improvements, ddof=1)) if len(improvements) > 1 else 0.0,
                "seed_ci_low": lo,
                "seed_ci_high": hi,
                "per_seed_improvements": ";".join(f"seed{int(r['seed']):02d}:{float(r['improvement_positive_means_C_better']):.8g}" for r in rows),
            }
        )
    write_csv(OUT / "scr5_seed_stability.csv", stability_rows)

    metric_map = {row["metric"]: row for row in stability_rows}
    delta_by_seed_metric = {(int(row["seed"]), row["metric"]): row for row in delta_rows}
    lpips_ok = bool(metric_map.get("lpips", {}).get("seeds_C_better") == len(SEEDS))
    rapsd_ok = bool(metric_map.get("rapsd_distance", {}).get("seeds_C_better") == len(SEEDS))
    psnr_ok = True
    rel_ok = True
    for seed_id in SEEDS:
        psnr_delta = float(delta_by_seed_metric[(seed_id, "psnr")]["mean_C_minus_B"])
        rel_delta = float(delta_by_seed_metric[(seed_id, "relmeaserr_unclipped_float64")]["mean_C_minus_B"])
        if psnr_delta < -PSNR_LOSS_GUARDRAIL_DB:
            psnr_ok = False
        if abs(rel_delta) > RELMEASERR_ABS_DELTA_GUARDRAIL:
            rel_ok = False
    d_not_saturated = all(not bool(result["armC_summary"]["d_saturated_last_mean_gt_0p95"]) for result in seed_results)
    success = bool(lpips_ok and rapsd_ok and psnr_ok and rel_ok and d_not_saturated)
    decision = "PROCEED_TO_PHASE72_SCR10_OR_WORKSHOP" if success else "STOP_GAN_PAPER_KEEP_PROJECT_SUPPLEMENT"
    summary = {
        "n_counted_seeds": len(SEEDS),
        "lpips_3_of_3": lpips_ok,
        "rapsd_3_of_3": rapsd_ok,
        "psnr_guardrail_all": psnr_ok,
        "relmeaserr_guardrail_all": rel_ok,
        "d_not_saturated_all": d_not_saturated,
        "success": success,
        "decision": decision,
    }
    save_json(OUT / "seed_stability_summary.json", summary)
    return stability_rows, summary


def plot_training_curves(seed_results: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for result in seed_results:
        seed_id = int(result["seed"])
        rows = read_csv(OUT / f"seed{seed_id:02d}" / "training_log.csv")
        for arm in ["B", "C"]:
            arm_rows = [r for r in rows if r["arm"] == arm]
            axes[0].plot([int(r["step"]) for r in arm_rows], [float(r["loss_rec"]) for r in arm_rows], label=f"s{seed_id}{arm}")
            val_rows = [r for r in arm_rows if r.get("val_rec_loss")]
            axes[1].plot([int(r["step"]) for r in val_rows], [float(r["val_rec_loss"]) for r in val_rows], marker="o", label=f"s{seed_id}{arm}")
        c_rows = [r for r in rows if r["arm"] == "C" and r.get("d_accuracy") and str(r["d_accuracy"]).lower() != "nan"]
        axes[2].plot([int(r["step"]) for r in c_rows], [float(r["d_accuracy"]) for r in c_rows], label=f"s{seed_id}C")
    axes[0].set_title("train rec loss")
    axes[1].set_title("val rec loss")
    axes[2].set_title("C critic accuracy")
    axes[2].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "scr5_D_training_curves.png", dpi=180)
    plt.close(fig)


def plot_visual_grid() -> None:
    fig, axes = plt.subplots(len(SEEDS), 4, figsize=(8, 2.2 * len(SEEDS)))
    for row_idx, seed_id in enumerate(SEEDS):
        seed_dir = OUT / f"seed{seed_id:02d}"
        z = np.load(p69a.EVAL_CACHE, allow_pickle=False)
        gt = z["x"][0].reshape(64, 64)
        b = np.load(seed_dir / "evaluation" / "per_sample_outputs_B.npz")["x_hat_unclipped"][0]
        c = np.load(seed_dir / "evaluation" / "per_sample_outputs_C.npz")["x_hat_unclipped"][0]
        a = np.load(seed_dir / "evaluation" / "per_sample_outputs_A.npz")["x_hat_unclipped"][0]
        for col, (title, img) in enumerate([("GT", gt), ("A", a), ("B", b), ("C", c)]):
            ax = axes[row_idx, col]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"seed{seed_id:02d}")
    fig.tight_layout()
    fig.savefig(OUT / "scr5_seed_visual_grid.png", dpi=180)
    plt.close(fig)


def plot_stability(delta_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> None:
    focus = ["lpips", "rapsd_distance", "psnr", "ssim", "relmeaserr_unclipped_float64"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.22
    for i, metric in enumerate(focus):
        rows = sorted([r for r in delta_rows if r["metric"] == metric], key=lambda r: int(r["seed"]))
        vals = [float(r["improvement_positive_means_C_better"]) for r in rows]
        xs = np.arange(len(SEEDS)) + i * width
        ax.bar(xs, vals, width=width, label=metric)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(SEEDS)) + width * (len(focus) - 1) / 2)
    ax.set_xticklabels([f"seed{s:02d}" for s in SEEDS])
    ax.set_ylabel("improvement positive = C better")
    ax.set_title("C vs B improvements by seed")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "scr5_seed_stability_plot.png", dpi=180)
    plt.close(fig)

    ci_rows = [r for r in stability_rows if r["metric"] in focus]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = np.asarray([float(r["mean_improvement_positive_C_better"]) for r in ci_rows])
    lows = np.asarray([float(r["seed_ci_low"]) for r in ci_rows])
    highs = np.asarray([float(r["seed_ci_high"]) for r in ci_rows])
    ax.bar(range(len(ci_rows)), vals, yerr=np.vstack([vals - lows, highs - vals]), capsize=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(ci_rows)))
    ax.set_xticklabels([r["metric"] for r in ci_rows], rotation=25, ha="right")
    ax.set_ylabel("mean improvement across seeds")
    ax.set_title("Across-seed 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(OUT / "scr5_C_vs_B_ci_bars.png", dpi=180)
    plt.close(fig)


def write_reports(delta_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]], stability: dict[str, Any], checkpoint_rows: list[dict[str, Any]]) -> None:
    metric_focus = ["lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error", "psnr", "ssim", "relmeaserr_unclipped_float64"]
    per_seed_table = [r for r in delta_rows if r["metric"] in metric_focus]
    per_seed_table = sorted(per_seed_table, key=lambda r: (int(r["seed"]), metric_focus.index(r["metric"])))
    stability_focus = [r for r in stability_rows if r["metric"] in metric_focus]
    d_rows = [
        {
            "seed": seed_id,
            "d_accuracy_last_mean": read_json(OUT / f"seed{seed_id:02d}" / "SEED_DONE.json")["d_accuracy_last_mean"],
            "d_saturated": read_json(OUT / f"seed{seed_id:02d}" / "SEED_DONE.json")["d_saturated_last_mean_gt_0p95"],
        }
        for seed_id in SEEDS
    ]
    conclusion = "stable_success" if stability["success"] else "unstable_or_guardrail_fail"
    workshop = "defensible" if stability["success"] else "only project supplement / negative result"
    strong_gap = [
        "Run Scr-10 paired regime next.",
        "Run Rad-5 robustness or explicitly scope to Scrambled-Hadamard.",
        "Complete beta sweep: 0, 0.3 beta0, beta0, 3 beta0.",
        "Package reproducible scripts/configs and command logs.",
    ]
    if not stability["success"]:
        strong_gap.insert(0, "Resolve Scr-5 seed instability or guardrail failure before any strong-conference claim.")

    answers = [
        {"question": "Does cGAN beat supervised-only across seeds?", "answer": "Yes on the primary LPIPS/RAPSD criteria." if stability["lpips_3_of_3"] and stability["rapsd_3_of_3"] else "No; primary metrics are not 3/3 stable."},
        {"question": "Is effect stable or seed-fragile?", "answer": "Stable across the counted Scr-5 paired seeds." if stability["success"] else "Seed-fragile or blocked by guardrail."},
        {"question": "Does certificate remain unchanged?", "answer": "Yes. RelMeasErr is evaluated unclipped float64 against recorded y; certificate remains Pi_y^lambda, not D."},
        {"question": "Does PSNR remain within budget?", "answer": "Yes, all seed PSNR losses are within 0.3 dB." if stability["psnr_guardrail_all"] else "No, at least one seed exceeds the 0.3 dB PSNR-loss guardrail."},
        {"question": "Is workshop paper now defensible?", "answer": "Yes, with the paired-seed evidence and cautious framing." if stability["success"] else "No as a positive claim; only a negative/project-supplement account is defensible."},
        {"question": "What remains for strong conference?", "answer": " ".join(strong_gap)},
        {"question": "Should Scr-10 be run next?", "answer": "Yes." if stability["success"] else "No, unless the goal is a negative stress test."},
        {"question": "Should Rad-5 be run next?", "answer": "After Scr-10; Rad-5 is the robustness gate." if stability["success"] else "No, not before resolving Scr-5."},
        {"question": "Should GAN paper continue or stop?", "answer": "Continue to Phase72 Scr-10 or write workshop." if stability["success"] else "Stop positive GAN-paper claims; keep as project supplement."},
    ]

    write_text(
        OUT / "SCR5_SEED_STABILITY_REPORT.md",
        "\n".join(
            [
                "# Scr-5 Seed Stability Report",
                "",
                f"Counted paired seeds: `{len(SEEDS)}`",
                f"Conclusion: `{conclusion}`",
                "",
                "## Per-Seed C vs B",
                "",
                table(per_seed_table, ["seed", "metric", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
                "",
                "## Across-Seed Stability",
                "",
                table(stability_focus, ["metric", "seeds_C_better", "all_seeds_C_better", "mean_improvement_positive_C_better", "std_improvement", "seed_ci_low", "seed_ci_high"]),
                "",
                "## D Saturation",
                "",
                table(d_rows, ["seed", "d_accuracy_last_mean", "d_saturated"]),
                "",
            ]
        ),
    )

    write_text(
        OUT / "PHASE71_PAIRED_SEED_REPORT.md",
        "\n".join(
            [
                "# Phase71 Paired-Seed Validation Report",
                "",
                f"Output directory: `{OUT}`",
                f"Decision: `{stability['decision']}`",
                "",
                "## Required Answers",
                "",
                table(answers, ["question", "answer"]),
                "",
                "## Per-Seed Primary / Guardrail Metrics",
                "",
                table(per_seed_table, ["seed", "metric", "mean_B", "mean_C", "mean_C_minus_B", "improvement_positive_means_C_better"]),
                "",
                "## Checkpoint Hashes",
                "",
                table(checkpoint_rows, ["seed", "arm", "kind", "sha256"]),
                "",
                "No first-paper checkpoint, table, title, abstract, or main result was modified.",
                "",
            ]
        ),
    )
    write_text(
        OUT / "WORKSHOP_READINESS_AFTER_SEEDS.md",
        "\n".join(
            [
                "# Workshop Readiness After Seeds",
                "",
                f"Decision: `{workshop}`",
                "",
                table(answers[:5], ["question", "answer"]),
                "",
                "The adversarial branch should remain framed as a gauge-equalized adversarial prior, not a measurement certificate.",
                "",
            ]
        ),
    )
    write_text(
        OUT / "STRONG_CONFERENCE_GAP_AFTER_SEEDS.md",
        "\n".join(["# Strong Conference Gap After Seeds", "", *[f"- {item}" for item in strong_gap], ""]),
    )
    allowed_claims = [
        "B/C are now paired by init checkpoint, data order, optimizer, step budget, validation split, and checkpoint-selection rule.",
        "D receives only gauge-equalized images and no residual/correction/certificate shortcut features.",
        "Measurement accountability remains with Pi_y^lambda.",
    ]
    if stability["success"]:
        allowed_claims.append("Across three Scr-5 paired seeds, C improves the primary LPIPS and RAPSD metrics under PSNR/RelMeasErr/D-saturation guardrails.")
    else:
        allowed_claims.append("The paired-seed validation did not support a stable positive cGAN claim.")
    write_text(OUT / "CLAIMS_AFTER_SEEDS.md", "\n".join(["# Claims After Seeds", "", *[f"- {item}" for item in allowed_claims], ""]))


def write_manifest() -> None:
    files = sorted(p.name for p in OUT.iterdir())
    write_text(
        OUT / "MANIFEST.md",
        "\n".join(["# Phase71 Manifest", "", f"Output directory: `{OUT}`", "", "## Top-Level Files", "", *[f"- `{name}`" for name in files], ""]),
    )


def main() -> int:
    ensure_dir(OUT)
    if not (OUT / "RUNLOG.md").exists():
        write_text(OUT / "RUNLOG.md", f"# Phase71 Runlog\n- {now()} runner_start\n")
    else:
        append_log("runner_start_existing_output_dir")

    protocol, failures = load_locked_protocol()
    if failures:
        return unsafe_stop(failures)
    write_protocol_lock(protocol)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_log(f"device={device}")
    config, measurement, A, lambda_dc = make_phase71_config(device)
    probe_gen, config = p69b.load_generator_from_checkpoint(config, measurement, device)
    config["output_dir"] = str(OUT)
    config["batch_size"] = BATCH_SIZE
    config["num_workers"] = 0
    del probe_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_log("build_caches_start")
    train, val, test, split_info = p69b.build_caches(config, measurement, device, TRAIN_COUNT, VAL_COUNT, TEST_COUNT)
    save_json(OUT / "split_manifest.json", split_info)
    append_log("build_caches_complete")

    seed_results = []
    for seed_id in SEEDS:
        seed_results.append(run_one_seed(seed_id, protocol, config, measurement, train, val, test, A, lambda_dc, device))
    metrics_rows, delta_rows, checkpoint_rows = build_combined_outputs(seed_results)
    stability_rows, stability = stability_analysis(delta_rows, seed_results)
    plot_training_curves(seed_results)
    plot_visual_grid()
    plot_stability(delta_rows, stability_rows)
    write_reports(delta_rows, stability_rows, stability, checkpoint_rows)
    write_manifest()
    append_log(f"runner_complete decision={stability['decision']} main_results_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
