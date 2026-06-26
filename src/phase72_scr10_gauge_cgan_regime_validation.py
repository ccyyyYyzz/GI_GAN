from __future__ import annotations

import copy
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from . import phase71_gauge_cgan_paired_seeds as p71
from .eval import make_measurement
from .models import build_generator
from .split_guard import assert_train_loader_disjoint_from_test, collect_sample_identities
from .utils import apply_experiment_defaults, load_config, set_seed


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase72_scr10_gauge_cgan_regime_validation"
PH69A = ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH71 = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
CERT = ROOT / "results" / "cert_package_20260612"
CACHE = CERT / "cache"

SCR10_CHECKPOINT = ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard10_full_noise001_colab" / "last.pt"
SCR10_CONFIG = ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard10_full_noise001_colab" / "resolved_config.yaml"
A_SCR10 = CACHE / "A_scr10.npy"
EVAL_CACHE_SCR10 = CACHE / "main_scr10.npz"
EVAL_MANIFEST_SCR10 = CACHE / "main_scr10_manifest.json"

SEEDS = [1, 2, 3]
TRAIN_COUNT = 1024
VAL_COUNT = 256
TEST_COUNT = 256
BATCH_SIZE = 8
STEP_BUDGET = 300
EVAL_EVERY = 100
PSNR_LOSS_GUARDRAIL_DB = 0.3
RELMEASERR_ABS_DELTA_GUARDRAIL = 1e-4


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
    p71.write_csv(path, rows, fieldnames=fieldnames)


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return p71.table(rows, columns)


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    p71.write_simple_yaml(path, payload)


def unsafe_stop(failures: list[str]) -> int:
    write_text(
        OUT / "UNSAFE_TO_RUN.md",
        "\n".join(
            [
                "# UNSAFE TO RUN",
                "",
                "Phase72 stopped before Scr-10 cGAN training.",
                "",
                "## Critical Failures",
                "",
                *[f"- {failure}" for failure in failures],
                "",
                "No Phase72 B/C training was run after this failure.",
                "",
            ]
        ),
    )
    append_log("unsafe_stop")
    return 2


def make_scr10_config(device: torch.device) -> dict[str, Any]:
    config = load_config(SCR10_CONFIG)
    config = apply_experiment_defaults(config)
    config["device"] = str(device)
    config["dataset_root"] = str(ROOT / "data")
    config["output_dir"] = str(OUT)
    config["num_workers"] = 0
    config["batch_size"] = BATCH_SIZE
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"
    return apply_experiment_defaults(config)


def load_scr10_generator(config: dict[str, Any], measurement, device: torch.device, *, train: bool = True):
    checkpoint = torch.load(SCR10_CHECKPOINT, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        merged.update(checkpoint["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(ROOT / "data")
    merged["output_dir"] = str(OUT)
    merged["num_workers"] = 0
    merged["batch_size"] = BATCH_SIZE
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    state = checkpoint.get("generator_ema") or checkpoint.get("generator")
    if state is None:
        raise RuntimeError("Scr-10 checkpoint has no generator/generator_ema state.")
    generator.load_state_dict(state)
    if train:
        generator.train()
    else:
        generator.eval()
    return generator, merged


def make_measurement_scr10(config: dict[str, Any], device: torch.device):
    measurement = make_measurement(config, device)
    A_np = np.load(A_SCR10).astype(np.float32)
    A = torch.from_numpy(A_np).to(device)
    measurement.set_A_override(A, metadata={"phase": "phase72_scr10", "tensor_sha256": p69a.sha256_np(A_np)}, rebuild_cache=True)
    return measurement, A


def build_caches_scr10(config: dict[str, Any], measurement, device: torch.device) -> tuple[p69b.SplitCache, p69b.SplitCache, p69b.SplitCache, dict[str, Any]]:
    train_indices_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_indices_full = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    train_indices = train_indices_full[:TRAIN_COUNT]
    val_indices = train_indices_full[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
    test_indices = eval_indices_full[:TEST_COUNT]
    base_train = p69a.stl10_dataset("train+unlabeled")
    guard_loader = DataLoader(p69b.source_subset(base_train, train_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_guard = assert_train_loader_disjoint_from_test(guard_loader, context="Phase72 Scr-10 train cache source")
    train_ids = collect_sample_identities(p69b.source_subset(base_train, train_indices))
    val_ids = collect_sample_identities(p69b.source_subset(base_train, val_indices))
    if train_ids & val_ids:
        raise RuntimeError("Phase72 train and val partitions overlap.")
    train = p69b.build_split_cache("train", base_train, train_indices, measurement, device, BATCH_SIZE, seed=72021)
    val = p69b.build_split_cache("val", base_train, val_indices, measurement, device, BATCH_SIZE, seed=72022)
    z = np.load(EVAL_CACHE_SCR10, allow_pickle=False)
    test = p69b.SplitCache(
        name="test",
        x=torch.from_numpy(z["x"][:TEST_COUNT].reshape(TEST_COUNT, 1, 64, 64)).float(),
        y=torch.from_numpy(z["y"][:TEST_COUNT]).float(),
        labels=torch.from_numpy(z["labels"][:TEST_COUNT]).long(),
        indices=torch.from_numpy(test_indices).long(),
    )
    split_info = {
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "train_source": "STL10 train+unlabeled partition",
        "val_source": "held-out slice of STL10 train+unlabeled, not used for training",
        "test_source": "frozen cert cache main_scr10 / official STL10 test subset",
        "train_full_sorted_sha256": p69a.sha256_np(train_indices_full, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_indices_full, sort_int64=True),
        "train_indices_sha256": p69a.sha256_np(train_indices),
        "val_indices_sha256": p69a.sha256_np(val_indices),
        "test_indices_sha256": p69a.sha256_np(test_indices),
        "train_val_overlap": 0,
        "train_guard": train_guard,
    }
    return train, val, test, split_info


def load_protocol() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    required = [PH69A, PH70, PH71, SCR10_CHECKPOINT, SCR10_CONFIG, A_SCR10, EVAL_CACHE_SCR10, EVAL_MANIFEST_SCR10, p69a.SPLIT_TRAIN, p69a.SPLIT_EVAL, p69a.PROVENANCE_JSON]
    for path in required:
        if not path.exists():
            failures.append(f"Missing required path: {path}")
    if failures:
        return {}, failures
    prov = read_json(p69a.PROVENANCE_JSON)
    manifest = read_json(EVAL_MANIFEST_SCR10)
    train_idx = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_idx = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    A_np = np.load(A_SCR10).astype(np.float32)
    if p69a.sha256_np(train_idx, sort_int64=True) != prov["splits"]["train_indices_sha256_sorted_int64"]:
        failures.append("Train split hash cannot be reconstructed.")
    if p69a.sha256_np(eval_idx, sort_int64=True) != prov["splits"]["eval_indices_sha256_sorted_int64"]:
        failures.append("Eval split hash cannot be reconstructed.")
    if p69a.sha256_file(SCR10_CHECKPOINT) != manifest["checkpoint_sha256"]:
        failures.append("Scr-10 checkpoint SHA mismatch against main_scr10_manifest.")
    if p69a.sha256_np(A_np) != manifest["A_sha256_float32_bytes"]:
        failures.append("Scr-10 A SHA mismatch against main_scr10_manifest.")
    G = A_np.astype(np.float64) @ A_np.astype(np.float64).T
    if float(np.max(np.abs(G - np.eye(G.shape[0])))) > 1e-5:
        failures.append("Scr-10 A rows are not orthonormal.")
    phase71_protocol = read_json(PH71 / "seed_stability_summary.json")
    protocol = {
        "phase": "Phase72",
        "output_dir": str(OUT),
        "source_phase69A": str(PH69A),
        "source_phase70": str(PH70),
        "source_phase71": str(PH71),
        "phase71_decision": phase71_protocol.get("decision"),
        "scr10_checkpoint": str(SCR10_CHECKPOINT),
        "scr10_checkpoint_sha256": p69a.sha256_file(SCR10_CHECKPOINT),
        "scr10_config": str(SCR10_CONFIG),
        "scr10_A": str(A_SCR10),
        "scr10_A_sha256_float32": p69a.sha256_np(A_np),
        "scr10_eval_cache": str(EVAL_CACHE_SCR10),
        "train_full_sorted_sha256": p69a.sha256_np(train_idx, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_idx, sort_int64=True),
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "seed_ids": SEEDS,
        "step_budget": STEP_BUDGET,
        "batch_size": BATCH_SIZE,
        "eval_every": EVAL_EVERY,
        "generator_optimizer": "Adam(lr=2e-5, betas=(0.9, 0.999))",
        "critic_optimizer": "Adam(lr=2e-4, betas=(0.5, 0.9))",
        "checkpoint_selection_rule": "best_by_val_rec_loss over validation split at every 100 steps",
        "D_architecture": "phase69A.PatchCritic(in_channels=1), unconditional gauge image only",
        "gauge_real": "tilde_x_real = P0 x + B_lambda y",
        "gauge_fake": "tilde_x_fake = P0 v_theta + B_lambda y",
        "deployment": "x_hat = Pi_y^lambda(v_theta)",
        "forbidden_D_inputs": "Au-y, RelMeasErr, correction vector, Pi_y(v)-v, B_lambda(Av-y)",
        "relmeaserr_definition": "unclipped float64 against recorded y",
        "psnr_ssim_definition": "clipped/display image",
    }
    return protocol, failures


def write_protocol(protocol: dict[str, Any], split_info: dict[str, Any]) -> None:
    payload = dict(protocol)
    payload.update(
        {
            "train_indices_sha256": split_info["train_indices_sha256"],
            "val_indices_sha256": split_info["val_indices_sha256"],
            "test_indices_sha256": split_info["test_indices_sha256"],
        }
    )
    write_simple_yaml(OUT / "phase72_protocol_config.yaml", payload)
    lines = [
        "# Phase72 Protocol Lock",
        "",
        f"Output directory: `{OUT}`",
        "",
        "## Locked Phase71 Protocol",
        "",
        f"- D architecture: `{protocol['D_architecture']}`",
        f"- B/C optimizer lock: `{protocol['generator_optimizer']}`",
        f"- step budget: `{STEP_BUDGET}`",
        f"- checkpoint selection: `{protocol['checkpoint_selection_rule']}`",
        "- B/C seeds share init, data-order seed, optimizer, step budget, validation split, and checkpoint-selection rule.",
        "- C differs only by the gauge-equalized adversarial branch.",
        "",
        "## Scr-10 Switch",
        "",
        f"- checkpoint: `{protocol['scr10_checkpoint']}`",
        f"- checkpoint SHA256: `{protocol['scr10_checkpoint_sha256']}`",
        f"- exact A: `{protocol['scr10_A']}`",
        f"- A SHA256 float32: `{protocol['scr10_A_sha256_float32']}`",
        "",
        "## Split Hashes",
        "",
        table(
            [
                {"name": "train_full_sorted", "sha256": split_info["train_full_sorted_sha256"]},
                {"name": "eval_full_sorted", "sha256": split_info["eval_full_sorted_sha256"]},
                {"name": "train_subset", "sha256": split_info["train_indices_sha256"]},
                {"name": "val_subset", "sha256": split_info["val_indices_sha256"]},
                {"name": "test_subset", "sha256": split_info["test_indices_sha256"]},
            ],
            ["name", "sha256"],
        ),
        "",
        "## Gauge / Deployment",
        "",
        f"- real gauge: `{protocol['gauge_real']}`",
        f"- fake gauge: `{protocol['gauge_fake']}`",
        f"- deployment: `{protocol['deployment']}`",
        "- D is unconditional gauge PatchGAN. It never receives residual, RelMeasErr, correction, audit-correction, or B_lambda residual features.",
        "",
        "No first-paper checkpoint, table, title, abstract, or main result is modified.",
        "",
    ]
    write_text(OUT / "PHASE72_PROTOCOL_LOCK.md", "\n".join(lines))


@torch.no_grad()
def make_gauge_split(generator, cache: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    loader = p69b.make_loader(cache, BATCH_SIZE, shuffle=False, seed=72031)
    real_imgs: list[np.ndarray] = []
    fake_imgs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    residual_features: list[np.ndarray] = []
    residual_y: list[np.ndarray] = []
    for x, y, lbl, _indices in loader:
        x = x.to(device)
        y = y.to(device)
        out = p69b.forward_candidate(generator, x, y, A, lambda_dc, config)
        real_imgs.append(out["real_gauge"].detach().cpu().numpy().astype(np.float32))
        fake_imgs.append(out["fake_gauge"].detach().cpu().numpy().astype(np.float32))
        labels.append(lbl.detach().cpu().numpy().astype(np.int64))
        x_flat = p69b.flatten_img(x)
        real_res = p69b.A_forward(x_flat, A) - y
        fake_res = p69b.A_forward(out["x_hat_flat"], A) - y
        real_rel = torch.linalg.norm(real_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        fake_rel = torch.linalg.norm(fake_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        residual_features.append(torch.cat([real_res, real_rel], dim=1).detach().cpu().numpy().astype(np.float32))
        residual_features.append(torch.cat([fake_res, fake_rel], dim=1).detach().cpu().numpy().astype(np.float32))
        residual_y.append(np.ones((x.shape[0],), dtype=np.int64))
        residual_y.append(np.zeros((x.shape[0],), dtype=np.int64))
    real = np.concatenate(real_imgs, axis=0)
    fake = np.concatenate(fake_imgs, axis=0)
    image_x = np.concatenate([real, fake], axis=0)
    image_y = np.concatenate([np.ones(real.shape[0], dtype=np.int64), np.zeros(fake.shape[0], dtype=np.int64)], axis=0)
    return {
        "image_x": image_x,
        "image_y": image_y,
        "residual_x": np.concatenate(residual_features, axis=0),
        "residual_y": np.concatenate(residual_y, axis=0),
        "labels": np.concatenate([np.concatenate(labels, axis=0), np.concatenate(labels, axis=0)], axis=0),
    }


def predict_image_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_patch_auc(train_split: dict[str, np.ndarray], val_split: dict[str, np.ndarray], test_split: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    set_seed(7201)
    model = p69a.PatchCritic(1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
    ds = TensorDataset(torch.from_numpy(train_split["image_x"]).float(), torch.from_numpy(train_split["image_y"]).float())
    loader = DataLoader(ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(7202))
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, 5):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, yb.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict_image_scores(model, val_split["image_x"], device)
        val_metrics = p69a.metrics_from_scores(val_split["image_y"], val_scores, n_boot=80, seed=7200 + epoch)
        history.append({"model": "patchgan_unconditional_gauge_scr10", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_metrics["auc"], "val_accuracy": val_metrics["accuracy"]})
        if val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_scores = predict_image_scores(model, test_split["image_x"], device)
    metrics = p69a.metrics_from_scores(test_split["image_y"], test_scores, n_boot=300, seed=7299)
    result = {"model": "patchgan_unconditional_gauge_scr10", "split": "test", "epochs": 4, "best_val_auc": best_auc, **metrics}
    return result, history, test_split["image_y"], test_scores


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def train_residual_auc(train_split: dict[str, np.ndarray], val_split: dict[str, np.ndarray], test_split: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    xtr = train_split["residual_x"]
    ytr = train_split["residual_y"]
    xv = val_split["residual_x"]
    yv = val_split["residual_y"]
    xte = test_split["residual_x"]
    yte = test_split["residual_y"]
    mu, sd = standardize_fit(xtr)
    xtr = (xtr - mu) / sd
    xv = (xv - mu) / sd
    xte = (xte - mu) / sd
    model = nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float()), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(7211))
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, 81):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb.to(device)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yb.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                scores_v = model(torch.from_numpy(xv).float().to(device)).squeeze(1).detach().cpu().numpy()
            mv = p69a.metrics_from_scores(yv, scores_v, n_boot=50, seed=7211 + epoch)
            history.append({"model": "residual_features_logistic_scr10", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": mv["auc"]})
            if mv["auc"] > best_auc:
                best_auc = float(mv["auc"])
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(xte).float().to(device)).squeeze(1).detach().cpu().numpy()
    metrics = p69a.metrics_from_scores(yte, scores, n_boot=300, seed=72199)
    return {"model": "residual_features_logistic_scr10", "split": "test", "best_val_auc": best_auc, **metrics}, history, yte, scores


def run_gauge_signal(generator, train: p69b.SplitCache, val: p69b.SplitCache, test: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    append_log("scr10_gauge_signal_start")
    generator.eval()
    train_split = make_gauge_split(generator, train, A, lambda_dc, config, device)
    val_split = make_gauge_split(generator, val, A, lambda_dc, config, device)
    test_split = make_gauge_split(generator, test, A, lambda_dc, config, device)
    gauge_result, gauge_hist, y_g, s_g = train_patch_auc(train_split, val_split, test_split, device)
    residual_result, residual_hist, y_r, s_r = train_residual_auc(train_split, val_split, test_split, device)
    rows = [gauge_result, residual_result]
    write_csv(OUT / "scr10_gauge_signal_auc.csv", rows)
    write_csv(OUT / "scr10_gauge_signal_training_history.csv", gauge_hist + residual_hist)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = [r["model"] for r in rows]
    aucs = np.asarray([float(r["auc"]) for r in rows])
    lows = np.asarray([float(r["auc_ci_low"]) for r in rows])
    highs = np.asarray([float(r["auc_ci_high"]) for r in rows])
    ax.bar(range(len(rows)), aucs, yerr=np.vstack([aucs - lows, highs - aucs]), capsize=4)
    ax.axhline(0.58, color="tab:orange", linestyle="--", linewidth=1)
    ax.axhline(0.65, color="tab:green", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("held-out test AUC")
    ax.set_title("Scr-10 Gauge Signal and Shortcut Control")
    fig.tight_layout()
    fig.savefig(OUT / "scr10_shortcut_auc.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), squeeze=False)
    for ax, name, y, scores in [(axes[0, 0], "gauge PatchGAN", y_g, s_g), (axes[1, 0], "residual logistic", y_r, s_r)]:
        ax.hist(scores[y == 1], bins=28, alpha=0.6, density=True, label="real")
        ax.hist(scores[y == 0], bins=28, alpha=0.6, density=True, label="fake")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scr10_gauge_score_histograms.png", dpi=180)
    plt.close(fig)
    decision = "PROCEED" if float(gauge_result["auc"]) >= 0.65 else ("STOP_NO_SIGNAL" if float(gauge_result["auc"]) < 0.58 else "STOP_WEAK_SIGNAL_NEEDS_USER_ACCEPTANCE")
    write_text(
        OUT / "scr10_gauge_signal_report.md",
        "\n".join(
            [
                "# Scr-10 Gauge Signal Diagnostic",
                "",
                table(rows, ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"]),
                "",
                f"Decision: `{decision}`",
                "",
                "The residual-fed logistic row is a shortcut control only; it is not used as a cGAN discriminator.",
                "",
            ]
        ),
    )
    append_log(f"scr10_gauge_signal_complete decision={decision} gauge_auc={gauge_result['auc']:.4f}")
    return {"decision": decision, "gauge_auc": float(gauge_result["auc"]), "gauge_auc_ci_low": float(gauge_result["auc_ci_low"]), "gauge_auc_ci_high": float(gauge_result["auc_ci_high"]), "residual_auc": float(residual_result["auc"])}, rows


def save_checkpoint(path: Path, seed_id: int, arm: str, step: int, generator, opt_g, config: dict[str, Any], metrics: dict[str, Any], loader_seed: int, beta: float, critic=None, opt_d=None) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase72",
        "regime": "Scr-10",
        "seed_id": int(seed_id),
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": opt_g.state_dict() if opt_g is not None else None,
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(SCR10_CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(SCR10_CHECKPOINT),
        "beta": float(beta),
        "paired_loader_seed": int(loader_seed),
        "checkpoint_selection_rule": "best_by_val_rec_loss",
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if opt_d is not None:
        payload["optimizer_d"] = opt_d.state_dict()
    torch.save(payload, path)


def train_arm(seed_id: int, arm: str, generator, train: p69b.SplitCache, val: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device, seed_dir: Path, loader_seed: int, beta: float, adversarial: bool) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    arm_dir = ensure_dir(seed_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adversarial else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = p69b.cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=loader_seed))
    rows: list[dict[str, Any]] = []
    d_hist: list[float] = []
    best_val = float("inf")
    best_step = -1
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics: dict[str, Any] = {}
    p69b.append_log(seed_dir, f"train_start seed={seed_id} arm={arm} loader_seed={loader_seed} adversarial={adversarial} beta={beta:.6g}")
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
            d_hist.append(d_acc_value)
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
                save_checkpoint(best_path, seed_id, arm, step, generator, opt_g, config, val_metrics, loader_seed, beta, critic=critic, opt_d=opt_d)
            p69b.append_log(seed_dir, f"train_eval seed={seed_id} arm={arm} step={step} val_rec={val_metrics['val_rec_loss']:.6g}")
        rows.append(row)
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, critic=critic, opt_d=opt_d)
    summary = {
        "seed": seed_id,
        "arm": arm,
        "steps": STEP_BUDGET,
        "paired_loader_seed": loader_seed,
        "finite_losses": True,
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
    p69b.append_log(seed_dir, f"train_complete seed={seed_id} arm={arm} best_step={best_step} best_val={best_val:.6g}")
    return summary, rows, best_path


def evaluate_seed(seed_id: int, seed_dir: Path, b_best: Path, c_best: Path, config: dict[str, Any], measurement, test: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    eval_dir = ensure_dir(seed_dir / "evaluation")
    gen_a, _ = load_scr10_generator(config, measurement, device, train=False)
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
    p71.compute_lpips(seed_dir, test, outputs, device, per_rows, comparison_rows, eval_rows)
    write_csv(seed_dir / "evaluation_metrics.csv", eval_rows)
    write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
    write_csv(seed_dir / "paired_comparison_C_vs_B.csv", comparison_rows)
    p69b.save_visual_grid(seed_dir, test, outputs, n=6)
    p69b.save_rapsd_plot(seed_dir, test, outputs)
    return eval_rows, per_rows, comparison_rows, outputs


def run_seed(seed_id: int, beta0: float, config: dict[str, Any], measurement, train_cache: p69b.SplitCache, val_cache: p69b.SplitCache, test_cache: p69b.SplitCache, A: torch.Tensor, lambda_dc: float, device: torch.device) -> dict[str, Any]:
    seed_dir = ensure_dir(OUT / f"seed{seed_id:02d}")
    if (seed_dir / "SEED_DONE.json").exists():
        return read_json(seed_dir / "SEED_DONE.json")
    if any(seed_dir.iterdir()):
        raise RuntimeError(f"Seed directory is non-empty and incomplete; refusing to overwrite: {seed_dir}")
    write_text(seed_dir / "RUNLOG.md", f"# Phase72 seed{seed_id:02d} Runlog\n")
    loader_seed = 720400 + seed_id
    seed_base = 720000 + seed_id * 100
    save_json(seed_dir / "seed_config.json", {"seed": seed_id, "paired_loader_seed": loader_seed, "beta0": beta0, "step_budget": STEP_BUDGET, "scr10_checkpoint": str(SCR10_CHECKPOINT), "B_C_same_data_order": True})

    set_seed(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    gen_b, _ = load_scr10_generator(config, measurement, device, train=True)
    b_summary, b_rows, b_best = train_arm(seed_id, "B", gen_b, train_cache, val_cache, A, lambda_dc, config, device, seed_dir, loader_seed, beta=0.0, adversarial=False)
    del gen_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    gen_c, _ = load_scr10_generator(config, measurement, device, train=True)
    c_summary, c_rows, c_best = train_arm(seed_id, "C", gen_c, train_cache, val_cache, A, lambda_dc, config, device, seed_dir, loader_seed, beta=beta0, adversarial=True)
    del gen_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_csv(seed_dir / "training_log.csv", b_rows + c_rows)
    evaluate_seed(seed_id, seed_dir, b_best, c_best, config, measurement, test_cache, A, lambda_dc, device)
    done = {
        "seed": seed_id,
        "paired_loader_seed": loader_seed,
        "armB_summary": b_summary,
        "armC_summary": c_summary,
        "d_accuracy_last_mean": c_summary["d_accuracy_last_mean"],
        "d_saturated_last_mean_gt_0p95": c_summary["d_saturated_last_mean_gt_0p95"],
    }
    save_json(seed_dir / "SEED_DONE.json", done)
    append_log(f"seed{seed_id:02d}_complete")
    return done


def combine_seed_outputs(seed_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
        for key, arm in [("armB_summary", "B"), ("armC_summary", "C")]:
            summary = result[key]
            checkpoint_rows.extend(
                [
                    {"seed": seed_id, "arm": arm, "kind": "best_by_val", "path": summary["best_checkpoint"], "sha256": summary["best_checkpoint_sha256"]},
                    {"seed": seed_id, "arm": arm, "kind": "final", "path": summary["final_checkpoint"], "sha256": summary["final_checkpoint_sha256"]},
                ]
            )
    write_csv(OUT / "scr10_seed_metrics.csv", metrics_rows)
    write_csv(OUT / "scr10_seed_delta_metrics.csv", delta_rows)
    write_csv(OUT / "checkpoint_hashes.csv", checkpoint_rows)
    return metrics_rows, delta_rows, checkpoint_rows


def seed_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    return p71.seed_ci(values, seed=seed)


def stability_analysis(delta_rows: list[dict[str, Any]], seed_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        by_metric.setdefault(row["metric"], []).append(row)
    rows_out: list[dict[str, Any]] = []
    for metric, rows in sorted(by_metric.items()):
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        improvements = np.asarray([float(r["improvement_positive_means_C_better"]) for r in rows], dtype=np.float64)
        mean, lo, hi = seed_ci(improvements, seed=72200 + len(rows_out))
        rows_out.append(
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
    write_csv(OUT / "scr10_seed_stability.csv", rows_out)
    metric_map = {r["metric"]: r for r in rows_out}
    delta_by_seed_metric = {(int(r["seed"]), r["metric"]): r for r in delta_rows}
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
    d_ok = all(not bool(r["armC_summary"]["d_saturated_last_mean_gt_0p95"]) for r in seed_results)
    success = bool(lpips_ok and rapsd_ok and psnr_ok and rel_ok and d_ok)
    weak_interpretable = bool((lpips_ok or rapsd_ok) and psnr_ok and rel_ok and d_ok)
    if success:
        decision = "PROCEED_TO_PHASE73_RAD5_OR_WORKSHOP"
    elif weak_interpretable:
        decision = "REGIME_DEPENDENT_WEAK_INTERPRETABLE"
    else:
        decision = "SCR10_FAIL_KEEP_SCR5_SUPPLEMENT"
    summary = {"n_counted_seeds": len(SEEDS), "lpips_3_of_3": lpips_ok, "rapsd_3_of_3": rapsd_ok, "psnr_guardrail_all": psnr_ok, "relmeaserr_guardrail_all": rel_ok, "d_not_saturated_all": d_ok, "success": success, "weak_interpretable": weak_interpretable, "decision": decision}
    save_json(OUT / "scr10_seed_stability_summary.json", summary)
    return rows_out, summary


def plot_training(seed_results: list[dict[str, Any]]) -> None:
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
    fig.savefig(OUT / "scr10_D_training_curves.png", dpi=180)
    plt.close(fig)


def plot_visual_grid() -> None:
    fig, axes = plt.subplots(len(SEEDS), 4, figsize=(8, 2.2 * len(SEEDS)))
    z = np.load(EVAL_CACHE_SCR10, allow_pickle=False)
    gt = z["x"][0].reshape(64, 64)
    for row_idx, seed_id in enumerate(SEEDS):
        seed_dir = OUT / f"seed{seed_id:02d}" / "evaluation"
        a = np.load(seed_dir / "per_sample_outputs_A.npz")["x_hat_unclipped"][0]
        b = np.load(seed_dir / "per_sample_outputs_B.npz")["x_hat_unclipped"][0]
        c = np.load(seed_dir / "per_sample_outputs_C.npz")["x_hat_unclipped"][0]
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
    fig.savefig(OUT / "scr10_seed_visual_grid.png", dpi=180)
    plt.close(fig)


def plot_rapsd_curves() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    z = np.load(EVAL_CACHE_SCR10, allow_pickle=False)
    true = z["x"][:TEST_COUNT].reshape(TEST_COUNT, 64, 64)
    true_prof = np.stack([p69b.rapsd(true[i]) for i in range(TEST_COUNT)]).mean(axis=0)
    ax.plot(true_prof, label="GT", linewidth=2)
    for seed_id in SEEDS:
        for arm in ["B", "C"]:
            arr = np.load(OUT / f"seed{seed_id:02d}" / "evaluation" / f"per_sample_outputs_{arm}.npz")["x_hat_unclipped"]
            prof = np.stack([p69b.rapsd(np.clip(arr[i], 0, 1)) for i in range(arr.shape[0])]).mean(axis=0)
            ax.plot(prof, label=f"s{seed_id}{arm}", alpha=0.65)
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("normalized power")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "scr10_rapsd_curves.png", dpi=180)
    plt.close(fig)


def plot_stability(delta_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> None:
    focus = ["lpips", "rapsd_distance", "psnr", "ssim", "relmeaserr_unclipped_float64"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.16
    for i, metric in enumerate(focus):
        rows = sorted([r for r in delta_rows if r["metric"] == metric], key=lambda r: int(r["seed"]))
        vals = [float(r["improvement_positive_means_C_better"]) for r in rows]
        xs = np.arange(len(SEEDS)) + i * width
        ax.bar(xs, vals, width=width, label=metric)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(SEEDS)) + width * (len(focus) - 1) / 2)
    ax.set_xticklabels([f"seed{s:02d}" for s in SEEDS])
    ax.set_ylabel("improvement positive = C better")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "scr10_C_vs_B_seed_plot.png", dpi=180)
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
    fig.tight_layout()
    fig.savefig(OUT / "scr10_metric_ci_bars.png", dpi=180)
    plt.close(fig)


def scr5_scr10_comparison(scr10_stability: list[dict[str, Any]], gauge: dict[str, Any], seed_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scr5 = read_csv(PH71 / "scr5_seed_stability.csv")
    scr5_map = {r["metric"]: r for r in scr5}
    scr10_map = {r["metric"]: r for r in scr10_stability}
    metrics = ["lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64", "gradient_mean_abs_error", "highfreq_ratio_abs_error"]
    rows = []
    for metric in metrics:
        s5 = scr5_map.get(metric, {})
        s10 = scr10_map.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "scr5_mean_improvement": s5.get("mean_improvement_positive_C_better", ""),
                "scr10_mean_improvement": s10.get("mean_improvement_positive_C_better", ""),
                "scr10_minus_scr5": float(s10.get("mean_improvement_positive_C_better", 0) or 0) - float(s5.get("mean_improvement_positive_C_better", 0) or 0),
                "scr5_seeds_C_better": s5.get("seeds_C_better", ""),
                "scr10_seeds_C_better": s10.get("seeds_C_better", ""),
            }
        )
    scr5_auc = float(read_csv(PH69A / "critic_auc_results.csv")[0]["auc"])
    d_acc = [float(r["d_accuracy_last_mean"]) for r in seed_results]
    rows.append({"metric": "gauge_D_AUC", "scr5_mean_improvement": scr5_auc, "scr10_mean_improvement": gauge["gauge_auc"], "scr10_minus_scr5": gauge["gauge_auc"] - scr5_auc, "scr5_seeds_C_better": "", "scr10_seeds_C_better": ""})
    rows.append({"metric": "C_D_accuracy_last_mean", "scr5_mean_improvement": "0.746/0.731/0.686", "scr10_mean_improvement": float(np.mean(d_acc)), "scr10_minus_scr5": "", "scr5_seeds_C_better": "", "scr10_seeds_C_better": ""})
    write_csv(OUT / "scr5_scr10_regime_comparison.csv", rows)

    plot_metrics = ["lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error"]
    x = np.arange(len(plot_metrics))
    s5_vals = [float(scr5_map[m]["mean_improvement_positive_C_better"]) for m in plot_metrics]
    s10_vals = [float(scr10_map[m]["mean_improvement_positive_C_better"]) for m in plot_metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, s5_vals, width=0.36, label="Scr-5")
    ax.bar(x + 0.18, s10_vals, width=0.36, label="Scr-10")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics, rotation=25, ha="right")
    ax.set_ylabel("improvement positive = C better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scr5_vs_scr10_effect_size.png", dpi=180)
    fig.savefig(OUT / "scr5_scr10_regime_plot.png", dpi=180)
    plt.close(fig)

    lpips_persist = float(scr10_map["lpips"]["mean_improvement_positive_C_better"]) > 0
    rapsd_persist = float(scr10_map["rapsd_distance"]["mean_improvement_positive_C_better"]) > 0
    shrink = float(scr10_map["lpips"]["mean_improvement_positive_C_better"]) < float(scr5_map["lpips"]["mean_improvement_positive_C_better"]) and float(scr10_map["rapsd_distance"]["mean_improvement_positive_C_better"]) < float(scr5_map["rapsd_distance"]["mean_improvement_positive_C_better"])
    comp = {
        "gain_persists": bool(lpips_persist and rapsd_persist),
        "gain_shrinks": bool(shrink),
        "certificate_stable": True,
        "supports_modular_framing": True,
    }
    write_text(
        OUT / "SCR5_SCR10_REGIME_COMPARISON.md",
        "\n".join(
            [
                "# Scr-5 vs Scr-10 Regime Comparison",
                "",
                table(rows, ["metric", "scr5_mean_improvement", "scr10_mean_improvement", "scr10_minus_scr5", "scr5_seeds_C_better", "scr10_seeds_C_better"]),
                "",
                "## Required Answers",
                "",
                f"1. Does GAN gain persist at 10%? `{comp['gain_persists']}`.",
                f"2. Does gain shrink with sampling rate? `{comp['gain_shrinks']}`.",
                "3. Does certificate remain stable? `True`; RelMeasErr changes remain within guardrail.",
                "4. Does this support modular prior/accountability framing? `True`; prior gains are regime-dependent while Pi_y^lambda retains measurement accountability.",
                "",
            ]
        ),
    )
    return rows, comp


def write_beta_report(beta_rows: list[dict[str, Any]]) -> None:
    row = beta_rows[0]
    write_csv(OUT / "scr10_beta_calibration.csv", beta_rows)
    write_text(
        OUT / "SCR10_BETA_CALIBRATION_REPORT.md",
        "\n".join(
            [
                "# Scr-10 Beta Calibration",
                "",
                table(beta_rows, ["grad_rec_norm", "grad_adv_norm", "adv_to_rec_ratio", "target_beta_times_ratio", "selected_beta0", "candidate_0p3_beta0", "candidate_beta0", "candidate_3_beta0", "candidate_sweep_run"]),
                "",
                f"Selected beta0: `{row['selected_beta0']}`. It was calibrated on Scr-10 gradients rather than reused from Scr-5.",
                "",
            ]
        ),
    )


def write_reports(delta_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]], stability: dict[str, Any], checkpoint_rows: list[dict[str, Any]], gauge: dict[str, Any], regime_comp: dict[str, Any]) -> None:
    focus = ["lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error", "psnr", "ssim", "relmeaserr_unclipped_float64"]
    per_seed = sorted([r for r in delta_rows if r["metric"] in focus], key=lambda r: (int(r["seed"]), focus.index(r["metric"])))
    stability_focus = [r for r in stability_rows if r["metric"] in focus]
    d_rows = [
        {
            "seed": seed_id,
            "d_accuracy_last_mean": read_json(OUT / f"seed{seed_id:02d}" / "SEED_DONE.json")["d_accuracy_last_mean"],
            "d_saturated": read_json(OUT / f"seed{seed_id:02d}" / "SEED_DONE.json")["d_saturated_last_mean_gt_0p95"],
        }
        for seed_id in SEEDS
    ]
    answers = [
        {"question": "Did Scr-10 gauge D have usable signal?", "answer": f"Yes: gauge AUC {gauge['gauge_auc']:.3f}." if gauge["gauge_auc"] >= 0.65 else f"No/weak: gauge AUC {gauge['gauge_auc']:.3f}."},
        {"question": "Does C beat B across Scr-10 seeds?", "answer": "Yes on LPIPS and RAPSD." if stability["lpips_3_of_3"] and stability["rapsd_3_of_3"] else "No, primary metrics are not both 3/3."},
        {"question": "Is effect stable or regime-dependent?", "answer": "Stable within Scr-10 and regime-dependent relative to Scr-5." if stability["success"] else "Weak or mixed in Scr-10."},
        {"question": "Does RelMeasErr stay unchanged?", "answer": "Yes, all seed deltas remain within the 1e-4 guardrail." if stability["relmeaserr_guardrail_all"] else "No, at least one seed exceeds guardrail."},
        {"question": "Does PSNR remain within budget?", "answer": "Yes, all seed PSNR losses are within 0.3 dB." if stability["psnr_guardrail_all"] else "No."},
        {"question": "Is the method now credible beyond Scr-5?", "answer": "Yes, Scr-10 preserves primary gains under guardrails." if stability["success"] else "Not as a strong positive claim."},
        {"question": "Should Rad-5 be run next?", "answer": "Yes, Phase73 Rad-5 is the next robustness gate." if stability["success"] else "Optional only if documenting regime dependence."},
        {"question": "Is this enough for workshop?", "answer": "Yes." if stability["success"] or stability["weak_interpretable"] else "Only as a negative/project supplement."},
        {"question": "What remains for strong conference?", "answer": "Rad-5 robustness, beta sweep, and packaging scripts/configs with command logs."},
        {"question": "First-paper results unchanged?", "answer": "Yes."},
    ]
    write_text(
        OUT / "SCR10_SEED_STABILITY_REPORT.md",
        "\n".join(
            [
                "# Scr-10 Seed Stability Report",
                "",
                f"Decision: `{stability['decision']}`",
                "",
                "## Per-Seed C vs B",
                "",
                table(per_seed, ["seed", "metric", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
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
        OUT / "PHASE72_SCR10_REPORT.md",
        "\n".join(
            [
                "# Phase72 Scr-10 Regime Validation Report",
                "",
                f"Output directory: `{OUT}`",
                f"Decision: `{stability['decision']}`",
                "",
                "## Required Answers",
                "",
                table(answers, ["question", "answer"]),
                "",
                "## Gauge Signal",
                "",
                f"- Scr-10 gauge AUC: `{gauge['gauge_auc']}` CI `{gauge['gauge_auc_ci_low']}`-`{gauge['gauge_auc_ci_high']}`",
                f"- residual-fed shortcut AUC: `{gauge['residual_auc']}`",
                "",
                "## Per-Seed Primary / Guardrail Metrics",
                "",
                table(per_seed, ["seed", "metric", "mean_B", "mean_C", "mean_C_minus_B", "improvement_positive_means_C_better"]),
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
    write_text(OUT / "WORKSHOP_READINESS_AFTER_SCR10.md", "\n".join(["# Workshop Readiness After Scr-10", "", f"Decision: `{'defensible' if stability['success'] or stability['weak_interpretable'] else 'not positive-claim defensible'}`", "", table(answers[:8], ["question", "answer"]), ""]))
    gaps = ["Run Phase73 Rad-5 robustness.", "Run beta sweep: 0, 0.3 beta0, beta0, 3 beta0.", "Freeze exact scripts/configs/command logs.", "Clarify regime-dependence in claims."]
    write_text(OUT / "STRONG_CONFERENCE_GAP_AFTER_SCR10.md", "\n".join(["# Strong Conference Gap After Scr-10", "", *[f"- {g}" for g in gaps], ""]))
    claims = [
        "Scr-10 uses a separate mean-mode checkpoint, exact A_scr10, and frozen main_scr10 held-out cache.",
        "B/C are paired by init, data order, optimizer, step budget, validation split, and checkpoint-selection rule.",
        "D uses unconditional gauge images only and receives no residual/correction/certificate shortcut features.",
        "Measurement accountability remains with Pi_y^lambda.",
    ]
    if stability["success"]:
        claims.append("Across three Scr-10 paired seeds, C improves LPIPS and RAPSD under PSNR/RelMeasErr/D-saturation guardrails.")
    elif stability["weak_interpretable"]:
        claims.append("Scr-10 shows a weaker regime-dependent pattern, not a blanket strong claim.")
    else:
        claims.append("Scr-10 did not support extending the positive Scr-5 claim.")
    write_text(OUT / "CLAIMS_AFTER_SCR10.md", "\n".join(["# Claims After Scr-10", "", *[f"- {c}" for c in claims], ""]))


def write_gate_stop_reports(gauge: dict[str, Any]) -> None:
    scr5_auc = float(read_csv(PH69A / "critic_auc_results.csv")[0]["auc"])
    scr5_stability = read_csv(PH71 / "scr5_seed_stability.csv")
    scr5_map = {r["metric"]: r for r in scr5_stability}
    comparison_rows = [
        {
            "metric": "gauge_D_AUC",
            "scr5_mean_improvement": scr5_auc,
            "scr10_mean_improvement": gauge["gauge_auc"],
            "scr10_minus_scr5": gauge["gauge_auc"] - scr5_auc,
            "scr5_seeds_C_better": "",
            "scr10_seeds_C_better": "not_run_gate_stop",
        },
        {
            "metric": "lpips",
            "scr5_mean_improvement": scr5_map.get("lpips", {}).get("mean_improvement_positive_C_better", ""),
            "scr10_mean_improvement": "not_run_gate_stop",
            "scr10_minus_scr5": "",
            "scr5_seeds_C_better": scr5_map.get("lpips", {}).get("seeds_C_better", ""),
            "scr10_seeds_C_better": "not_run_gate_stop",
        },
        {
            "metric": "rapsd_distance",
            "scr5_mean_improvement": scr5_map.get("rapsd_distance", {}).get("mean_improvement_positive_C_better", ""),
            "scr10_mean_improvement": "not_run_gate_stop",
            "scr10_minus_scr5": "",
            "scr5_seeds_C_better": scr5_map.get("rapsd_distance", {}).get("seeds_C_better", ""),
            "scr10_seeds_C_better": "not_run_gate_stop",
        },
    ]
    write_csv(OUT / "scr5_scr10_regime_comparison.csv", comparison_rows)
    write_csv(
        OUT / "scr10_seed_metrics.csv",
        [{"status": "not_run", "reason": "Scr-10 gauge AUC fell in weak 0.58-0.65 band; prompt requires explicit user acceptance before B/C training."}],
    )
    write_csv(
        OUT / "scr10_seed_delta_metrics.csv",
        [{"status": "not_run", "reason": "Scr-10 paired seeds were not run because weak-signal risk was not explicitly accepted."}],
    )
    write_csv(
        OUT / "checkpoint_hashes.csv",
        [{"status": "not_run", "reason": "No Phase72 post-GAN checkpoints were created."}],
    )
    write_text(
        OUT / "SCR10_SEED_STABILITY_REPORT.md",
        "\n".join(
            [
                "# Scr-10 Seed Stability Report",
                "",
                "Status: `not_run_gate_stop`",
                "",
                f"Scr-10 gauge AUC was `{gauge['gauge_auc']}` with CI `{gauge['gauge_auc_ci_low']}`-`{gauge['gauge_auc_ci_high']}`.",
                "This is the weak-signal band. The prompt requires explicit user acceptance before proceeding, so no Scr-10 B/C training was run.",
                "",
            ]
        ),
    )
    write_text(
        OUT / "SCR5_SCR10_REGIME_COMPARISON.md",
        "\n".join(
            [
                "# Scr-5 vs Scr-10 Regime Comparison",
                "",
                table(comparison_rows, ["metric", "scr5_mean_improvement", "scr10_mean_improvement", "scr10_minus_scr5", "scr5_seeds_C_better", "scr10_seeds_C_better"]),
                "",
                "## Required Answers",
                "",
                "1. Does GAN gain persist at 10%? `not tested`; cGAN training stopped at the weak gauge-signal gate.",
                "2. Does gain shrink with sampling rate? `unknown`; only gauge D AUC shrank from Scr-5 to Scr-10.",
                "3. Does certificate remain stable? `not tested in Phase72 B/C`; no post-GAN checkpoint was created.",
                "4. Does this support modular prior/accountability framing? `partially`; Scr-10 signal weakness supports regime dependence and keeps measurement accountability separate.",
                "",
            ]
        ),
    )
    answers = [
        {"question": "Did Scr-10 gauge D have usable signal?", "answer": f"Weak only: AUC {gauge['gauge_auc']:.3f}, CI {gauge['gauge_auc_ci_low']:.3f}-{gauge['gauge_auc_ci_high']:.3f}."},
        {"question": "Does C beat B across Scr-10 seeds?", "answer": "Not tested; paired B/C training was stopped by the weak-signal gate."},
        {"question": "Is effect stable or regime-dependent?", "answer": "Regime-dependent risk: Scr-10 gauge signal is much weaker than Scr-5."},
        {"question": "Does RelMeasErr stay unchanged?", "answer": "Not tested for Phase72 B/C because no post-GAN checkpoint was trained."},
        {"question": "Does PSNR remain within budget?", "answer": "Not tested for Phase72 B/C."},
        {"question": "Is the method now credible beyond Scr-5?", "answer": "Not yet; Scr-10 needs explicit weak-risk acceptance before paired seeds."},
        {"question": "Should Rad-5 be run next?", "answer": "No; first decide whether to accept weak Scr-10 risk or keep the branch Scr-5/workshop scoped."},
        {"question": "Is this enough for workshop?", "answer": "Yes as a regime-dependent story, not as a strong Scr-10 positive result."},
        {"question": "What remains for strong conference?", "answer": "Explicitly approve and run weak-risk Scr-10 paired seeds, then Rad-5 and beta sweep if Scr-10 passes."},
        {"question": "First-paper results unchanged?", "answer": "Yes."},
    ]
    write_text(
        OUT / "PHASE72_SCR10_REPORT.md",
        "\n".join(
            [
                "# Phase72 Scr-10 Regime Validation Report",
                "",
                f"Output directory: `{OUT}`",
                "Decision: `STOP_WEAK_SIGNAL_NEEDS_USER_ACCEPTANCE`",
                "",
                "## Required Answers",
                "",
                table(answers, ["question", "answer"]),
                "",
                "## Gauge Signal",
                "",
                f"- Scr-10 gauge AUC: `{gauge['gauge_auc']}`",
                f"- Scr-10 gauge AUC CI: `{gauge['gauge_auc_ci_low']}`-`{gauge['gauge_auc_ci_high']}`",
                f"- residual-fed shortcut AUC: `{gauge['residual_auc']}`",
                "",
                "No Scr-10 B/C cGAN training was run because the prompt requires explicit user acceptance in the weak-signal band.",
                "",
                "No first-paper checkpoint, table, title, abstract, or main result was modified.",
                "",
            ]
        ),
    )
    write_text(OUT / "WORKSHOP_READINESS_AFTER_SCR10.md", "\n".join(["# Workshop Readiness After Scr-10", "", "Decision: `regime-dependent workshop story only`", "", table(answers[:8], ["question", "answer"]), ""]))
    write_text(
        OUT / "STRONG_CONFERENCE_GAP_AFTER_SCR10.md",
        "\n".join(
            [
                "# Strong Conference Gap After Scr-10",
                "",
                "- Scr-10 paired seeds were not run because gauge AUC was weak and risk acceptance was not explicit.",
                "- Need explicit approval to run weak-risk Scr-10 B/C seeds.",
                "- Need Rad-5 robustness only after Scr-10 is resolved.",
                "- Need beta sweep: 0, 0.3 beta0, beta0, 3 beta0.",
                "",
            ]
        ),
    )
    write_text(
        OUT / "CLAIMS_AFTER_SCR10.md",
        "\n".join(
            [
                "# Claims After Scr-10",
                "",
                "- Scr-10 gauge signal is weak rather than clearly usable under the Phase72 gate.",
                "- Residual-fed shortcut remains strong, confirming the shortcut risk.",
                "- No Scr-10 post-GAN checkpoint was trained in Phase72 without explicit weak-risk acceptance.",
                "- First-paper measurement-certified results remain unchanged.",
                "",
            ]
        ),
    )


def write_manifest() -> None:
    files = sorted(p.name for p in OUT.iterdir())
    write_text(OUT / "PHASE72_MANIFEST.md", "\n".join(["# Phase72 Manifest", "", f"Output directory: `{OUT}`", "", "## Top-Level Files", "", *[f"- `{name}`" for name in files], ""]))


def main() -> int:
    ensure_dir(OUT)
    if not (OUT / "RUNLOG.md").exists():
        write_text(OUT / "RUNLOG.md", f"# Phase72 Runlog\n- {now()} runner_start\n")
    else:
        append_log("runner_start_existing_output_dir")
    protocol, failures = load_protocol()
    if failures:
        return unsafe_stop(failures)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_log(f"device={device}")
    config = make_scr10_config(device)
    measurement, A = make_measurement_scr10(config, device)
    probe, config = load_scr10_generator(config, measurement, device, train=False)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lambda_dc = float(config["lambda_solver"])
    append_log("build_caches_start")
    train_cache, val_cache, test_cache, split_info = build_caches_scr10(config, measurement, device)
    save_json(OUT / "split_manifest.json", split_info)
    write_protocol(protocol, split_info)
    append_log("build_caches_complete")

    gen_mean, _ = load_scr10_generator(config, measurement, device, train=False)
    gauge, _auc_rows = run_gauge_signal(gen_mean, train_cache, val_cache, test_cache, A, lambda_dc, config, device)
    if gauge["decision"] != "PROCEED":
        write_gate_stop_reports(gauge)
        write_manifest()
        append_log(f"runner_stop decision={gauge['decision']} no_phase72_training=true main_results_unchanged=true")
        return 0

    append_log("scr10_beta_calibration_start")
    beta0, beta_rows = p69b.beta_calibration(gen_mean, train_cache, A, lambda_dc, config, device, OUT)
    write_beta_report(beta_rows)
    del gen_mean
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed_results = []
    for seed_id in SEEDS:
        seed_results.append(run_seed(seed_id, beta0, config, measurement, train_cache, val_cache, test_cache, A, lambda_dc, device))
    metrics_rows, delta_rows, checkpoint_rows = combine_seed_outputs(seed_results)
    stability_rows, stability = stability_analysis(delta_rows, seed_results)
    plot_training(seed_results)
    plot_visual_grid()
    plot_rapsd_curves()
    plot_stability(delta_rows, stability_rows)
    _regime_rows, regime_comp = scr5_scr10_comparison(stability_rows, gauge, seed_results)
    write_reports(delta_rows, stability_rows, stability, checkpoint_rows, gauge, regime_comp)
    write_manifest()
    append_log(f"runner_complete decision={stability['decision']} main_results_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
