from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.compatibility_data import (
    RangeNullPairDataset,
    compute_train_normalization,
    load_rad5_96_components,
    make_derangement,
    make_semihard_donors,
    normalize_images,
    save_json,
    save_split_cache,
    split_manifest,
    verify_feasible_pairs,
    write_csv,
)
from src.compatibility_eval import e1_gate, evaluate_critic_split
from src.compatibility_model import CompatibilityCritic, margin_ranking_from_scores, symmetric_infonce_loss
from src.projections import exact_null_project, get_exact_projector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase-1 feasible-counterfactual compatibility critic.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output_dir", default=None, help="Override config output_dir.")
    parser.add_argument("--device", default=None, help="Override config device.")
    parser.add_argument("--resume", default=None, help="Resume checkpoint path.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = str(path)
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "UNKNOWN"


def atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_run_files(out: Path, cfg: dict[str, Any], argv: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text("$ " + " ".join(argv) + "\n", encoding="utf-8")
    with (out / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def projector_runtime_checks(split, measurement, *, device: torch.device, sample_count: int = 16) -> dict[str, Any]:
    count = min(int(sample_count), split.size)
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    flat = split.x[:count].to(device).reshape(count, -1).double()
    y = projector.A_forward(flat)
    n = projector.null_project_flat(flat)
    r = projector.row_project_flat(flat)
    n2 = projector.null_project_flat(n)
    denom_y = torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    ap0_rel = torch.linalg.norm(projector.A_forward(n) - torch.zeros_like(y), dim=1) / denom_y
    recon = torch.linalg.norm((r + n) - flat, dim=1) / torch.linalg.norm(flat, dim=1).clamp_min(1e-12)
    dot = torch.sum(r * n, dim=1).abs() / (
        torch.linalg.norm(r, dim=1).clamp_min(1e-12) * torch.linalg.norm(n, dim=1).clamp_min(1e-12)
    )
    idem = torch.linalg.norm(n2 - n, dim=1) / torch.linalg.norm(flat, dim=1).clamp_min(1e-12)
    anchor = projector.data_anchor_flat(y)
    anchor_rel = torch.linalg.norm(projector.A_forward(anchor) - y, dim=1) / denom_y
    info = projector.info_dict()
    return {
        "sample_count": count,
        "float64_A_P0_rel_max": float(ap0_rel.max().item()),
        "reconstruction_rel_max": float(recon.max().item()),
        "orthogonality_cos_max": float(dot.max().item()),
        "idempotence_rel_max": float(idem.max().item()),
        "exact_data_anchor_rel_max": float(anchor_rel.max().item()),
        "pass": bool(
            ap0_rel.max().item() < 1e-9
            and recon.max().item() < 1e-10
            and dot.max().item() < 1e-8
            and idem.max().item() < 1e-9
            and anchor_rel.max().item() < 1e-9
        ),
        "projector_info": info,
    }


def make_loader(dataset, cfg: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)) and torch.cuda.is_available(),
        drop_last=bool(cfg.get("drop_last", True)) if shuffle else False,
    )


def run_tiny_overfit(train_ds, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    steps = int(cfg.get("tiny_overfit_steps", 0))
    if steps <= 0:
        return {"enabled": False}
    subset_count = min(int(cfg.get("tiny_overfit_samples", 64)), len(train_ds))
    subset = Subset(train_ds, list(range(subset_count)))
    loader = DataLoader(subset, batch_size=min(16, subset_count), shuffle=True, num_workers=0, drop_last=True)
    model = CompatibilityCritic(
        embed_dim=int(cfg.get("embed_dim", 128)),
        base_channels=int(cfg.get("base_channels", 24)),
        temperature=float(cfg.get("temperature", 0.07)),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    losses = []
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        r = batch["r"].to(device)
        n = batch["n"].to(device)
        loss = symmetric_infonce_loss(model.score_matrix(r, n))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "enabled": True,
        "samples": subset_count,
        "steps": steps,
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "loss_ratio_last_over_first": (losses[-1] / losses[0]) if losses and losses[0] else float("nan"),
        "pass_loss_decreased": bool(losses and losses[-1] < losses[0]),
    }


def train(cfg: dict[str, Any], out: Path, device: torch.device) -> dict[str, Any]:
    seed = int(cfg.get("seed", 1))
    set_all_seeds(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    measurement, rad_config, splits, split_info = load_rad5_96_components(cfg, output_dir=out, device=device)
    normalization = compute_train_normalization(splits["train"])
    save_json(out / "split_manifest.json", split_manifest(splits, measurement, split_info))
    save_json(out / "normalization_train_only.json", normalization)
    if bool(cfg.get("save_component_cache", True)):
        cache_paths = save_split_cache(out, splits)
    else:
        cache_paths = {}

    random_donors = make_derangement(splits["train"].size, seed + 11)
    semihard_donors = make_semihard_donors(splits["train"], seed=seed + 12, pool_size=int(cfg.get("semihard_pool_size", 96)))
    feasible_random = verify_feasible_pairs(splits["train"], measurement, random_donors, device=device, max_pairs=int(cfg.get("feasible_check_pairs", 128)))
    feasible_semihard = verify_feasible_pairs(splits["train"], measurement, semihard_donors, device=device, max_pairs=int(cfg.get("feasible_check_pairs", 128)))
    projector_checks = projector_runtime_checks(splits["train"], measurement, device=device, sample_count=int(cfg.get("projector_check_samples", 16)))

    train_ds = RangeNullPairDataset(splits["train"], normalization=normalization)
    val_ds = RangeNullPairDataset(splits["val"], normalization=normalization)
    tiny_overfit = run_tiny_overfit(train_ds, cfg, device)

    model = CompatibilityCritic(
        embed_dim=int(cfg.get("embed_dim", 128)),
        base_channels=int(cfg.get("base_channels", 24)),
        temperature=float(cfg.get("temperature", 0.07)),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0
    global_step = 0
    resume = cfg.get("resume")
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        opt.load_state_dict(payload["optimizer"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))

    best_score = -float("inf")
    best_path = out / "checkpoints" / "best_by_val.pt"
    history: list[dict[str, Any]] = []
    train_loader = make_loader(train_ds, cfg, shuffle=True, seed=seed + 21)
    semi_for_train = make_semihard_donors(splits["train"], seed=seed + 22, pool_size=int(cfg.get("semihard_pool_size", 96)))
    for epoch in range(start_epoch, int(cfg.get("epochs", 5))):
        model.train()
        losses = []
        for batch in train_loader:
            r = batch["r"].to(device, non_blocking=True)
            n = batch["n"].to(device, non_blocking=True)
            local_idx = batch["local_idx"].long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                matrix = model.score_matrix(r, n)
                loss = symmetric_infonce_loss(matrix)
                if epoch >= int(cfg.get("stage2_start_epoch", 999999)):
                    donor_idx = semi_for_train[local_idx].long()
                    neg_flat = splits["train"].n[donor_idx]
                    n_neg = normalize_images(neg_flat, img_size=splits["train"].img_size, key="n", normalization=normalization).to(device)
                    pos_score = model.score_pairs(r, n)
                    neg_score = model.score_pairs(r, n_neg)
                    loss = loss + float(cfg.get("margin_weight", 0.25)) * margin_ranking_from_scores(
                        pos_score, neg_score, margin=float(cfg.get("margin", 0.1))
                    )
            scaler.scale(loss).backward()
            if float(cfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
            global_step += 1
        val_metrics, val_rows = evaluate_critic_split(
            model,
            splits["val"],
            normalization,
            device=device,
            seed=seed + 100 + epoch,
            donors_per_anchor=int(cfg.get("donors_per_anchor", 32)),
            batch_size=int(cfg.get("eval_batch_size", 128)),
        )
        score = float(val_metrics["semi_hard_auc"]) + float(val_metrics["recall_at_1"]) - max(0.0, float(val_metrics["score_p0_error_spearman"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), **val_metrics}
        history.append(row)
        save_json(out / "history_latest.json", {"history": history})
        if score > best_score:
            best_score = score
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if amp else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": cfg,
                    "normalization": normalization,
                    "val_metrics": val_metrics,
                    "rad_config": rad_config,
                },
                best_path,
            )
            write_csv(out / "val_per_image_latest.csv", val_rows)

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    val_metrics, val_rows = evaluate_critic_split(
        model,
        splits["val"],
        normalization,
        device=device,
        seed=seed + 500,
        donors_per_anchor=int(cfg.get("donors_per_anchor", 32)),
        batch_size=int(cfg.get("eval_batch_size", 128)),
    )
    test_metrics, test_rows = evaluate_critic_split(
        model,
        splits["test"],
        normalization,
        device=device,
        seed=seed + 600,
        donors_per_anchor=int(cfg.get("donors_per_anchor", 32)),
        batch_size=int(cfg.get("eval_batch_size", 128)),
    )
    write_csv(out / "val_per_image.csv", val_rows)
    write_csv(out / "test_per_image.csv", test_rows)
    write_csv(out / "training_history.csv", history)
    val_gate = e1_gate(val_metrics)
    allowed_e2 = bool(
        val_gate["pass"]
        and projector_checks["pass"]
        and feasible_random["pass_float32_proxy"]
        and feasible_semihard["pass_float32_proxy"]
    )
    runtime = time.time() - start_time
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "phase": "E1_compatibility_critic_rad5_96_pilot",
        "status": "complete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": str(Path.cwd()),
        "git_commit": git_commit(Path.cwd()),
        "seed": seed,
        "device": str(device),
        "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": peak_gpu,
        "config_path": cfg.get("config_path"),
        "output_dir": str(out),
        "component_cache_paths": cache_paths,
        "measurement": {
            "img_size": int(measurement.img_size),
            "m": int(measurement.m),
            "n": int(measurement.n),
            "pattern_type": str(measurement.pattern_type),
            "matrix_normalization": str(measurement.matrix_normalization),
        },
        "projector_tests": projector_checks,
        "feasible_counterfactual_tests": {
            "random_negative": feasible_random,
            "semi_hard_negative": feasible_semihard,
            "pass": bool(feasible_random["pass_float32_proxy"] and feasible_semihard["pass_float32_proxy"]),
        },
        "baseline_reproduction_differences": {
            "status": "not_run_in_E1_critic_pilot",
            "reason": "This phase freezes the existing generator and trains only the compatibility critic; no GAN baseline was re-evaluated.",
        },
        "tiny_overfit": tiny_overfit,
        "best_checkpoint": str(best_path),
        "best_checkpoint_selection": "max(val semi_hard_auc + recall_at_1 - positive_spearman_penalty)",
        "validation_metrics": val_metrics,
        "semi_hard_auc": val_metrics.get("semi_hard_auc"),
        "recall_at_1_relative_to_random": float(val_metrics.get("recall_at_1", 0.0))
        / max(float(val_metrics.get("random_recall_at_1", 1.0 / 32.0)), 1e-12),
        "score_p0_error_spearman": val_metrics.get("score_p0_error_spearman"),
        "oracle_best_of_16_headroom": "not_run_e2_gate_failed" if not allowed_e2 else "pending_e2",
        "critic_fraction_of_oracle_gain": "not_run_e2_gate_failed" if not allowed_e2 else "pending_e2",
        "test_metrics_reported_once_no_tuning": test_metrics,
        "e1_gate_on_validation": val_gate,
        "recommend_enter_generator_finetuning": False,
        "allowed_to_run_e2_candidate_selection": allowed_e2,
        "notes": [
            "Generator and GAN loss were not modified.",
            "Exact counterfactuals use clean Ax and exact AA^T solves, not ridge lambda_dc.",
            "Test metrics are reported only after checkpoint selection from validation metrics.",
        ],
    }
    save_json(out / "gate_report_e1.json", report)
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device
    if args.resume:
        cfg["resume"] = args.resume
    out = Path(cfg.get("output_dir", "outputs/compatibility/rad5_96_pilot")).resolve()
    save_run_files(out, cfg, sys.argv)
    device = resolve_device(str(cfg.get("device", "cuda")))
    report = train(cfg, out, device)
    print(json.dumps({"gate_report_e1": str(out / "gate_report_e1.json"), "e1_pass": report["e1_gate_on_validation"]["pass"]}, indent=2))


if __name__ == "__main__":
    main()
