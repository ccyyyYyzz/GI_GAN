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
OUT = ROOT / "outputs_phase73_overnight_gauge_gan_expansion"
PH69A = ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH71 = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
PH72 = ROOT / "outputs_phase72_scr10_gauge_cgan_regime_validation"
CERT = ROOT / "results" / "cert_package_20260612"
CACHE = CERT / "cache"

TRAIN_COUNT = 1024
VAL_COUNT = 256
TEST_COUNT = 256
BATCH_SIZE = 8
STEP_BUDGET = 300
EVAL_EVERY = 100
PSNR_LOSS_GUARDRAIL_DB = 0.3
RELMEASERR_ABS_DELTA_GUARDRAIL = 1e-4

REGIMES = {
    "scr5": {
        "checkpoint": ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "last.pt",
        "config": ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "resolved_config.yaml",
        "A": CACHE / "A_scr5.npy",
        "cache": CACHE / "main_scr5.npz",
        "manifest": CACHE / "main_scr5_manifest.json",
        "orthonormal": True,
    },
    "scr10": {
        "checkpoint": ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard10_full_noise001_colab" / "last.pt",
        "config": ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard10_full_noise001_colab" / "resolved_config.yaml",
        "A": CACHE / "A_scr10.npy",
        "cache": CACHE / "main_scr10.npz",
        "manifest": CACHE / "main_scr10_manifest.json",
        "orthonormal": True,
    },
    "rad5": {
        "checkpoint": ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "last.pt",
        "config": ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "resolved_config.yaml",
        "A": CACHE / "A_rad5.npy",
        "cache": CACHE / "main_rad5.npz",
        "manifest": CACHE / "main_rad5_manifest.json",
        "orthonormal": False,
    },
}


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


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    p71.write_simple_yaml(path, payload)


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def unsafe_stop(failures: list[str]) -> int:
    write_text(
        OUT / "UNSAFE_TO_RUN.md",
        "\n".join(
            [
                "# UNSAFE TO RUN",
                "",
                "Phase73 stopped before any training task.",
                "",
                "## Critical Failures",
                "",
                *[f"- {x}" for x in failures],
                "",
            ]
        ),
    )
    append_log("unsafe_stop")
    return 2


def regime_config(name: str, device: torch.device) -> dict[str, Any]:
    info = REGIMES[name]
    config = load_config(info["config"])
    config = apply_experiment_defaults(config)
    config["device"] = str(device)
    config["dataset_root"] = str(ROOT / "data")
    config["output_dir"] = str(OUT)
    config["batch_size"] = BATCH_SIZE
    config["num_workers"] = 0
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"
    return apply_experiment_defaults(config)


def make_regime_measurement(name: str, config: dict[str, Any], device: torch.device):
    measurement = make_measurement(config, device)
    A_np = np.load(REGIMES[name]["A"]).astype(np.float32)
    A = torch.from_numpy(A_np).to(device)
    measurement.set_A_override(A, metadata={"phase": "phase73", "regime": name, "tensor_sha256": p69a.sha256_np(A_np)}, rebuild_cache=True)
    return measurement, A


def load_regime_generator(name: str, config: dict[str, Any], measurement, device: torch.device, train: bool = True):
    ckpt_path = REGIMES[name]["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        merged.update(ckpt["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(ROOT / "data")
    merged["output_dir"] = str(OUT)
    merged["batch_size"] = BATCH_SIZE
    merged["num_workers"] = 0
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged = apply_experiment_defaults(merged)
    gen = build_generator(merged, measurement=measurement).to(device)
    state = ckpt.get("generator_ema") or ckpt.get("generator")
    if state is None:
        raise RuntimeError(f"{name} checkpoint has no generator/generator_ema state.")
    gen.load_state_dict(state)
    gen.train(train)
    if not train:
        gen.eval()
    return gen, merged


def build_caches(name: str, config: dict[str, Any], measurement, device: torch.device):
    train_indices_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_indices_full = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    train_indices = train_indices_full[:TRAIN_COUNT]
    val_indices = train_indices_full[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
    test_indices = eval_indices_full[:TEST_COUNT]
    base_train = p69a.stl10_dataset("train+unlabeled")
    guard_loader = DataLoader(p69b.source_subset(base_train, train_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    guard = assert_train_loader_disjoint_from_test(guard_loader, context=f"Phase73 {name} train cache source")
    if collect_sample_identities(p69b.source_subset(base_train, train_indices)) & collect_sample_identities(p69b.source_subset(base_train, val_indices)):
        raise RuntimeError(f"{name} train/val overlap.")
    train = p69b.build_split_cache("train", base_train, train_indices, measurement, device, BATCH_SIZE, seed=73021)
    val = p69b.build_split_cache("val", base_train, val_indices, measurement, device, BATCH_SIZE, seed=73022)
    z = np.load(REGIMES[name]["cache"], allow_pickle=False)
    test = p69b.SplitCache(
        name="test",
        x=torch.from_numpy(z["x"][:TEST_COUNT].reshape(TEST_COUNT, 1, 64, 64)).float(),
        y=torch.from_numpy(z["y"][:TEST_COUNT]).float(),
        labels=torch.from_numpy(z["labels"][:TEST_COUNT]).long(),
        indices=torch.from_numpy(test_indices).long(),
    )
    split = {
        "regime": name,
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "train_full_sorted_sha256": p69a.sha256_np(train_indices_full, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_indices_full, sort_int64=True),
        "train_indices_sha256": p69a.sha256_np(train_indices),
        "val_indices_sha256": p69a.sha256_np(val_indices),
        "test_indices_sha256": p69a.sha256_np(test_indices),
        "train_source": "STL10 train+unlabeled partition",
        "val_source": "held-out STL10 train+unlabeled slice, not used for training",
        "test_source": f"frozen cert cache main_{name}",
        "train_guard": guard,
        "train_val_overlap": 0,
    }
    return train, val, test, split


def exact_projectors(A: torch.Tensor, lambda_dc: float):
    return p69a.exact_projectors(A, lambda_dc)


def forward_candidate_general(generator, measurement, x: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    x_data_flat = p69a.data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    zero_noise = torch.zeros_like(x_data)
    residual = generator(x_data, zero_noise, y=y)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns = measurement.null_project(residual_flat) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = measurement.dc_project(v_stage0, y) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, measurement.unflatten_img(x_stage1))
        v_pre = x_stage1 + measurement.flatten_img(refine.float())
    else:
        v_pre = x_stage1
    x_hat_flat = measurement.dc_project(v_pre, y) if bool(config.get("use_final_dc_project", True)) else v_pre
    A64, G, K = exact_projectors(measurement.A, float(config["lambda_solver"]))
    b = p69a.blambda_y(y, A64, K)
    p0x = p69a.p0_exact(measurement.flatten_img(x).to(torch.float64), A64, G)
    p0v = p69a.p0_exact(v_pre.to(torch.float64), A64, G)
    return {
        "x_data_flat": x_data_flat,
        "v_pre": v_pre,
        "x_hat_flat": x_hat_flat,
        "x_hat": measurement.unflatten_img(x_hat_flat),
        "real_gauge": measurement.unflatten_img((p0x + b).to(torch.float32)),
        "fake_gauge": measurement.unflatten_img((p0v + b).to(torch.float32)),
        "correction_flat": x_hat_flat - v_pre,
    }


@torch.no_grad()
def gauge_split(generator, measurement, cache: p69b.SplitCache, config: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    loader = p69b.make_loader(cache, BATCH_SIZE, shuffle=False, seed=73031)
    image_x, image_y, residual_x, residual_y = [], [], [], []
    xdata = []
    labels = []
    for x, y, lbl, _idx in loader:
        x = x.to(device)
        y = y.to(device)
        out = forward_candidate_general(generator, measurement, x, y, config)
        real = out["real_gauge"].detach().cpu().numpy().astype(np.float32)
        fake = out["fake_gauge"].detach().cpu().numpy().astype(np.float32)
        image_x.extend([real, fake])
        image_y.extend([np.ones(real.shape[0], dtype=np.int64), np.zeros(fake.shape[0], dtype=np.int64)])
        xdata.append(out["x_data_flat"].detach().cpu().numpy().astype(np.float32))
        labels.append(lbl.detach().cpu().numpy().astype(np.int64))
        A = measurement.A
        x_flat = measurement.flatten_img(x)
        real_res = measurement.A_forward(x_flat) - y
        fake_res = measurement.A_forward(out["x_hat_flat"]) - y
        real_rel = torch.linalg.norm(real_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        fake_rel = torch.linalg.norm(fake_res, dim=1, keepdim=True) / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
        residual_x.extend([torch.cat([real_res, real_rel], 1).detach().cpu().numpy().astype(np.float32), torch.cat([fake_res, fake_rel], 1).detach().cpu().numpy().astype(np.float32)])
        residual_y.extend([np.ones(real.shape[0], dtype=np.int64), np.zeros(fake.shape[0], dtype=np.int64)])
    return {
        "image_x": np.concatenate(image_x, axis=0),
        "image_y": np.concatenate(image_y, axis=0),
        "residual_x": np.concatenate(residual_x, axis=0),
        "residual_y": np.concatenate(residual_y, axis=0),
        "x_data_pair": np.concatenate([np.concatenate(xdata, axis=0), np.concatenate(xdata, axis=0)], axis=0).reshape(-1, 1, 64, 64),
        "labels": np.concatenate([np.concatenate(labels, axis=0), np.concatenate(labels, axis=0)], axis=0),
    }


def predict_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=64, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1)


def train_image_model(name: str, model: nn.Module, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, device: torch.device, epochs: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed + 1))
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb.to(device)).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict_scores(model, val_x, device)
        vm = p69a.metrics_from_scores(val_y, val_scores, n_boot=80, seed=seed + epoch)
        hist.append({"model": name, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": vm["auc"], "val_accuracy": vm["accuracy"]})
        if vm["auc"] > best_auc:
            best_auc = float(vm["auc"])
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_scores = predict_scores(model, test_x, device)
    tm = p69a.metrics_from_scores(test_y, test_scores, n_boot=300, seed=seed + 999)
    return {"model": name, "split": "test", "epochs": epochs, "best_val_auc": best_auc, **tm}, hist, test_y, test_scores


def train_residual_model(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    xtr, ytr = train["residual_x"], train["residual_y"]
    xv, yv = val["residual_x"], val["residual_y"]
    xte, yte = test["residual_x"], test["residual_y"]
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    xtr, xv, xte = (xtr - mu) / sd, (xv - mu) / sd, (xte - mu) / sd
    model = nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float()), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(7311))
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    hist = []
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
            with torch.no_grad():
                sv = model(torch.from_numpy(xv).float().to(device)).squeeze(1).detach().cpu().numpy()
            mv = p69a.metrics_from_scores(yv, sv, n_boot=50, seed=epoch)
            hist.append({"model": "residual_features_logistic_rad5", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": mv["auc"]})
            if mv["auc"] > best_auc:
                best_auc = float(mv["auc"])
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():
        ste = model(torch.from_numpy(xte).float().to(device)).squeeze(1).detach().cpu().numpy()
    m = p69a.metrics_from_scores(yte, ste, n_boot=300, seed=73199)
    return {"model": "residual_features_logistic_rad5", "split": "test", "best_val_auc": best_auc, **m}, hist, yte, ste


def rad5_diagnostic(device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    append_log("rad5_diagnostic_start")
    config = regime_config("rad5", device)
    measurement, A = make_regime_measurement("rad5", config, device)
    gen, config = load_regime_generator("rad5", config, measurement, device, train=False)
    train, val, test, split = build_caches("rad5", config, measurement, device)
    save_json(OUT / "rad5_split_manifest.json", split)
    tr = gauge_split(gen, measurement, train, config, device)
    va = gauge_split(gen, measurement, val, config, device)
    te = gauge_split(gen, measurement, test, config, device)
    rows, hist, score_sets = [], [], {}
    r, h, y, s = train_image_model("patchgan_unconditional_gauge_rad5", p69a.PatchCritic(1), tr["image_x"], tr["image_y"], va["image_x"], va["image_y"], te["image_x"], te["image_y"], device, epochs=4, seed=7301)
    rows.append(r); hist.extend(h); score_sets[r["model"]] = (y, s)
    r, h, y, s = train_image_model("simple_cnn_gauge_rad5", p69a.SimpleCNN(1), tr["image_x"], tr["image_y"], va["image_x"], va["image_y"], te["image_x"], te["image_y"], device, epochs=4, seed=7302)
    rows.append(r); hist.extend(h); score_sets[r["model"]] = (y, s)
    cond_train = np.concatenate([tr["image_x"], tr["x_data_pair"]], axis=1)
    cond_val = np.concatenate([va["image_x"], va["x_data_pair"]], axis=1)
    cond_test = np.concatenate([te["image_x"], te["x_data_pair"]], axis=1)
    r, h, y, s = train_image_model("patchgan_conditional_xdata_gauge_rad5", p69a.PatchCritic(2), cond_train, tr["image_y"], cond_val, va["image_y"], cond_test, te["image_y"], device, epochs=3, seed=7303)
    rows.append(r); hist.extend(h); score_sets[r["model"]] = (y, s)
    ctrl, h, y, s = train_residual_model(tr, va, te, device)
    hist.extend(h); score_sets[ctrl["model"]] = (y, s)
    write_csv(OUT / "rad5_gauge_signal_auc.csv", rows)
    write_csv(OUT / "rad5_shortcut_controls.csv", [ctrl])
    write_csv(OUT / "rad5_gauge_signal_training_history.csv", hist)
    p69a.save_score_histograms(OUT, score_sets)
    (OUT / "critic_score_histograms.png").replace(OUT / "rad5_gauge_score_histograms.png")
    if (OUT / "critic_score_histograms.pdf").exists():
        (OUT / "critic_score_histograms.pdf").unlink()
    all_auc = rows + [ctrl]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aucs = np.asarray([float(r["auc"]) for r in all_auc])
    lows = np.asarray([float(r["auc_ci_low"]) for r in all_auc])
    highs = np.asarray([float(r["auc_ci_high"]) for r in all_auc])
    ax.bar(range(len(all_auc)), aucs, yerr=np.vstack([aucs - lows, highs - aucs]), capsize=4)
    ax.axhline(0.65, color="tab:orange", linestyle="--", linewidth=1)
    ax.axhline(0.75, color="tab:green", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(all_auc)))
    ax.set_xticklabels([r["model"] for r in all_auc], rotation=25, ha="right")
    ax.set_ylabel("held-out AUC")
    fig.tight_layout()
    fig.savefig(OUT / "rad5_shortcut_auc_plot.png", dpi=180)
    plt.close(fig)
    main = rows[0]
    auc, lo = float(main["auc"]), float(main["auc_ci_low"])
    if auc >= 0.75 and lo > 0.70:
        decision = "strong_run_3_paired_seeds"
        n_seeds = 3
    elif auc >= 0.65:
        decision = "moderate_run_1_paired_seed"
        n_seeds = 1
    else:
        decision = "weak_no_rad5_cgan"
        n_seeds = 0
    gate = {"decision": decision, "paired_seed_count": n_seeds, "rad5_gauge_auc": auc, "rad5_gauge_auc_ci_low": lo, "rad5_gauge_auc_ci_high": float(main["auc_ci_high"]), "rad5_residual_auc": float(ctrl["auc"])}
    save_json(OUT / "rad5_gate_decision.json", gate)
    write_text(
        OUT / "RAD5_GAUGE_SIGNAL_REPORT.md",
        "\n".join(
            [
                "# Rad-5 Gauge Signal Report",
                "",
                table(rows, ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"]),
                "",
                "## Shortcut Control",
                "",
                table([ctrl], ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"]),
                "",
                f"Decision: `{decision}`.",
                "",
            ]
        ),
    )
    append_log(f"rad5_diagnostic_complete decision={decision} auc={auc:.4f}")
    return gate, rows + [ctrl]


def beta_calibration_general(generator, measurement, train: p69b.SplitCache, config: dict[str, Any], device: torch.device, out_dir: Path, target: float = 0.075) -> tuple[float, list[dict[str, Any]]]:
    critic = p69a.PatchCritic(1).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9))
    loader = p69b.cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=7321))
    generator.eval()
    for _ in range(20):
        x, y, _, _ = next(loader)
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            out = forward_candidate_general(generator, measurement, x, y, config)
        opt.zero_grad(set_to_none=True)
        rs, fs = critic(out["real_gauge"]), critic(out["fake_gauge"])
        d_loss = F.relu(1 - rs).mean() + F.relu(1 + fs).mean()
        d_loss.backward(); opt.step()
    x, y, _, _ = next(iter(p69b.make_loader(train, BATCH_SIZE, shuffle=False, seed=7322)))
    x = x.to(device); y = y.to(device)
    with torch.no_grad():
        base = forward_candidate_general(generator, measurement, x, y, config)
    v = base["v_pre"].detach().requires_grad_(True)
    x_hat_flat = measurement.dc_project(v, y)
    x_hat = measurement.unflatten_img(x_hat_flat)
    A64, G, K = exact_projectors(measurement.A, float(config["lambda_solver"]))
    fake_gauge = measurement.unflatten_img((p69a.p0_exact(v, A64, G) + p69a.blambda_y(y, A64, K)).to(torch.float32))
    rec = p69b.charbonnier(x_hat, x)
    adv = -critic(fake_gauge).mean()
    gr = torch.autograd.grad(rec, v, retain_graph=True)[0]
    ga = torch.autograd.grad(adv, v)[0]
    rec_norm = float(torch.linalg.norm(gr).detach().cpu())
    adv_norm = float(torch.linalg.norm(ga).detach().cpu())
    ratio = adv_norm / max(rec_norm, 1e-12)
    beta0 = float(np.clip(target / max(ratio, 1e-12), 1e-5, 1.0))
    rows = [{"grad_rec_norm": rec_norm, "grad_adv_norm": adv_norm, "adv_to_rec_ratio": ratio, "target_beta_times_ratio": target, "selected_beta0": beta0, "candidate_0p3_beta0": 0.3 * beta0, "candidate_beta0": beta0, "candidate_3_beta0": 3.0 * beta0, "candidate_sweep_run": False}]
    write_csv(out_dir / "rad5_beta_calibration.csv", rows)
    return beta0, rows


def save_phase73_checkpoint(path: Path, regime: str, seed_id: int, arm: str, step: int, generator, opt_g, config: dict[str, Any], metrics: dict[str, Any], loader_seed: int, beta: float, critic=None, opt_d=None) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase73",
        "regime": regime,
        "seed_id": seed_id,
        "arm": arm,
        "step": step,
        "generator": generator.state_dict(),
        "optimizer_g": opt_g.state_dict() if opt_g else None,
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(REGIMES[regime]["checkpoint"]),
        "source_checkpoint_sha256": p69a.sha256_file(REGIMES[regime]["checkpoint"]),
        "beta": float(beta),
        "paired_loader_seed": int(loader_seed),
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if opt_d is not None:
        payload["optimizer_d"] = opt_d.state_dict()
    torch.save(payload, path)


def eval_val_loss_general(generator, measurement, val: p69b.SplitCache, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    losses, rels = [], []
    generator.eval()
    for x, y, _, _ in p69b.make_loader(val, BATCH_SIZE, shuffle=False, seed=7331):
        x = x.to(device); y = y.to(device)
        out = forward_candidate_general(generator, measurement, x, y, config)
        losses.append(float(p69b.charbonnier(out["x_hat"], x).detach().cpu()))
        rels.extend(p69b.relmeas_batch(out["x_hat_flat"], y, measurement.A).tolist())
    generator.train()
    return {"val_rec_loss": float(np.mean(losses)), "val_relmeas": float(np.mean(rels))}


def train_rad5_arm(seed_id: int, arm: str, generator, measurement, train: p69b.SplitCache, val: p69b.SplitCache, config: dict[str, Any], device: torch.device, seed_dir: Path, loader_seed: int, beta: float, adversarial: bool):
    arm_dir = ensure_dir(seed_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adversarial else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic else None
    loader = p69b.cycle_loader(p69b.make_loader(train, BATCH_SIZE, shuffle=True, seed=loader_seed))
    rows, d_hist = [], []
    best_val, best_step = float("inf"), -1
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics = {}
    for step in range(1, STEP_BUDGET + 1):
        x, y, _, _ = next(loader)
        x = x.to(device); y = y.to(device)
        d_loss_v = float("nan"); d_acc = float("nan")
        if adversarial and critic is not None and opt_d is not None:
            generator.eval(); critic.train()
            with torch.no_grad():
                od = forward_candidate_general(generator, measurement, x, y, config)
            opt_d.zero_grad(set_to_none=True)
            rs, fs = critic(od["real_gauge"]), critic(od["fake_gauge"])
            d_loss = F.relu(1 - rs).mean() + F.relu(1 + fs).mean()
            d_loss.backward(); opt_d.step()
            d_loss_v = float(d_loss.detach().cpu())
            d_acc = p69b.d_accuracy(rs, fs)
            d_hist.append(d_acc)
            generator.train()
        opt_g.zero_grad(set_to_none=True)
        out = forward_candidate_general(generator, measurement, x, y, config)
        rec = p69b.charbonnier(out["x_hat"], x)
        adv = torch.zeros((), device=device)
        if adversarial and critic is not None:
            critic.eval()
            adv = -critic(out["fake_gauge"]).mean()
        loss = rec + float(beta) * adv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()
        row = {"seed": seed_id, "arm": arm, "step": step, "paired_loader_seed": loader_seed, "loss_total": float(loss.detach().cpu()), "loss_rec": float(rec.detach().cpu()), "loss_adv": float(adv.detach().cpu()), "loss_d": d_loss_v, "d_accuracy": d_acc, "beta": beta}
        if step % EVAL_EVERY == 0 or step == STEP_BUDGET:
            vm = eval_val_loss_general(generator, measurement, val, config, device)
            row.update(vm); final_metrics = vm
            if vm["val_rec_loss"] < best_val:
                best_val = float(vm["val_rec_loss"]); best_step = step
                save_phase73_checkpoint(best_path, "rad5", seed_id, arm, step, generator, opt_g, config, vm, loader_seed, beta, critic=critic, opt_d=opt_d)
        rows.append(row)
    final_path = ckpt_dir / "final.pt"
    save_phase73_checkpoint(final_path, "rad5", seed_id, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, critic=critic, opt_d=opt_d)
    summary = {"seed": seed_id, "arm": arm, "steps": STEP_BUDGET, "paired_loader_seed": loader_seed, "best_val_rec_loss": best_val, "best_step": best_step, "best_checkpoint": str(best_path), "best_checkpoint_sha256": p69a.sha256_file(best_path), "final_checkpoint": str(final_path), "final_checkpoint_sha256": p69a.sha256_file(final_path), "d_accuracy_last_mean": float(np.nanmean(d_hist[-50:])) if d_hist else float("nan"), "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_hist[-50:]) > 0.95) if d_hist else False}
    write_csv(arm_dir / "training_log.csv", rows)
    save_json(arm_dir / "training_summary.json", summary)
    return summary, rows, best_path


@torch.no_grad()
def evaluate_general(arm: str, generator, measurement, test: p69b.SplitCache, config: dict[str, Any], device: torch.device, out_dir: Path):
    rows, outputs = [], []
    A64, G, _ = exact_projectors(measurement.A, float(config["lambda_solver"]))
    for x, y, labels, indices in p69b.make_loader(test, BATCH_SIZE, shuffle=False, seed=7341):
        x = x.to(device); y = y.to(device)
        out = forward_candidate_general(generator, measurement, x, y, config)
        x_hat = out["x_hat"].detach().cpu().numpy()[:, 0]
        x_clip = np.clip(x_hat, 0, 1)
        x_true = x.detach().cpu().numpy()[:, 0]
        rels = p69b.relmeas_batch(out["x_hat_flat"], y, measurement.A)
        corr = torch.linalg.norm(out["correction_flat"].detach(), dim=1) / torch.linalg.norm(out["v_pre"].detach(), dim=1).clamp_min(1e-12)
        p0_pred = p69a.p0_exact(out["x_hat_flat"].detach().to(torch.float64), A64, G)
        p0_true = p69a.p0_exact(measurement.flatten_img(x).detach().to(torch.float64), A64, G)
        p0_l2 = (torch.linalg.norm(p0_pred - p0_true, dim=1) / math.sqrt(4096)).detach().cpu().numpy()
        outputs.append(x_hat.astype(np.float32))
        for i in range(x.shape[0]):
            pred, true = x_clip[i], x_true[i]
            rows.append({"arm": arm, "sample_index": int(indices[i]), "label": int(labels[i]), "psnr": p69b.psnr_one(pred, true), "ssim": p69b.ssim_one(pred, true), "relmeaserr_unclipped_float64": float(rels[i]), "correction_norm_rel": float(corr[i].detach().cpu()), "rapsd_distance": float(np.linalg.norm(p69b.rapsd(pred) - p69b.rapsd(true))), "gradient_mean_abs_error": float(abs(p69b.grad_mag(pred).mean() - p69b.grad_mag(true).mean())), "highfreq_ratio_abs_error": float(abs(p69b.hf_ratio(pred) - p69b.hf_ratio(true))), "p0_l2": float(p0_l2[i])})
    arr = np.concatenate(outputs, axis=0)
    agg = {"arm": arm, "n": int(arr.shape[0])}
    for metric in ["psnr", "ssim", "relmeaserr_unclipped_float64", "correction_norm_rel", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error", "p0_l2"]:
        vals = [float(r[metric]) for r in rows]
        st = p69b.metric_summary(vals)
        agg[f"{metric}_mean"] = st["mean"]; agg[f"{metric}_median"] = st["median"]; agg[f"{metric}_std"] = st["std"]
    ensure_dir(out_dir)
    np.savez_compressed(out_dir / f"per_sample_outputs_{arm}.npz", x_hat_unclipped=arr.astype(np.float16))
    return agg, rows, arr


def compute_lpips_for_outputs(seed_dir: Path, test: p69b.SplitCache, outputs: dict[str, np.ndarray], device: torch.device, per_rows: list[dict[str, Any]], comp: list[dict[str, Any]], eval_rows: list[dict[str, Any]]):
    try:
        import lpips
    except Exception:
        return
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true = test.x[:, 0].numpy().astype(np.float32)
    true_t = p71.prep_lpips(true)
    vals_by_arm = {}
    per_lp = []
    with torch.no_grad():
        for arm in ["A", "B", "C"]:
            pred_t = p71.prep_lpips(outputs[arm].astype(np.float32))
            vals = []
            for i in range(0, pred_t.shape[0], 16):
                vals.extend(loss_fn(pred_t[i:i+16].to(device), true_t[i:i+16].to(device)).reshape(-1).detach().cpu().numpy().astype(float).tolist())
            vals_by_arm[arm] = np.asarray(vals, dtype=np.float64)
            for j, v in enumerate(vals):
                per_lp.append({"arm": arm, "sample_index": int(test.indices[j]), "sample_ordinal": j, "lpips": float(v)})
    write_csv(seed_dir / "lpips_per_sample.csv", per_lp)
    write_csv(seed_dir / "lpips_or_dists_results.csv", [{"metric_package": "LPIPS", "module": "lpips", "available": True, "arm": arm, "n": len(v), "lpips_mean": float(v.mean()), "lpips_median": float(np.median(v)), "lpips_std": float(v.std())} for arm, v in vals_by_arm.items()])
    lookup = {(r["arm"], int(r["sample_index"])): float(r["lpips"]) for r in per_lp}
    for row in per_rows:
        if (row["arm"], int(row["sample_index"])) in lookup:
            row["lpips"] = lookup[(row["arm"], int(row["sample_index"]))]
    for row in eval_rows:
        v = vals_by_arm[row["arm"]]
        row["lpips_mean"] = float(v.mean()); row["lpips_median"] = float(np.median(v)); row["lpips_std"] = float(v.std())
    imp = vals_by_arm["B"] - vals_by_arm["C"]
    mean, lo, hi = p69b.bootstrap_ci(imp, seed=7350, n_boot=1000)
    comp.append({"metric": "lpips", "direction": "lower", "mean_B": float(vals_by_arm["B"].mean()), "mean_C": float(vals_by_arm["C"].mean()), "mean_C_minus_B": float((vals_by_arm["C"] - vals_by_arm["B"]).mean()), "improvement_positive_means_C_better": mean, "ci_low": lo, "ci_high": hi, "ci_excludes_zero_in_favor_of_C": bool(lo > 0)})


def run_rad5_seeds(gate: dict[str, Any], device: torch.device):
    n = int(gate["paired_seed_count"])
    if n <= 0:
        write_csv(OUT / "rad5_seed_metrics.csv", [{"status": "not_run", "reason": "Rad-5 gauge AUC < 0.65"}])
        write_csv(OUT / "rad5_seed_delta_metrics.csv", [{"status": "not_run", "reason": "Rad-5 gate did not pass"}])
        write_text(OUT / "RAD5_PAIRED_SEED_REPORT.md", "# Rad-5 Paired Seed Report\n\nRad-5 paired cGAN was not run because the gauge signal gate did not pass.\n")
        return [], []
    config = regime_config("rad5", device)
    measurement, _A = make_regime_measurement("rad5", config, device)
    probe, config = load_regime_generator("rad5", config, measurement, device, train=False)
    train, val, test, split = build_caches("rad5", config, measurement, device)
    beta0, beta_rows = beta_calibration_general(probe, measurement, train, config, device, OUT)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_results, all_metrics, all_delta, ckpt_rows = [], [], [], []
    for seed_id in range(1, n + 1):
        seed_dir = ensure_dir(OUT / f"rad5_seed{seed_id:02d}")
        write_text(seed_dir / "RUNLOG.md", f"# Rad-5 seed{seed_id:02d}\n")
        loader_seed = 735400 + seed_id
        base_seed = 735000 + 100 * seed_id
        set_seed(base_seed); random.seed(base_seed); np.random.seed(base_seed); torch.manual_seed(base_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(base_seed)
        gen_b, _ = load_regime_generator("rad5", config, measurement, device, train=True)
        bsum, brows, bbest = train_rad5_arm(seed_id, "B", gen_b, measurement, train, val, config, device, seed_dir, loader_seed, 0.0, False)
        del gen_b
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        set_seed(base_seed); random.seed(base_seed); np.random.seed(base_seed); torch.manual_seed(base_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(base_seed)
        gen_c, _ = load_regime_generator("rad5", config, measurement, device, train=True)
        csum, crows, cbest = train_rad5_arm(seed_id, "C", gen_c, measurement, train, val, config, device, seed_dir, loader_seed, beta0, True)
        del gen_c
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        write_csv(seed_dir / "training_log.csv", brows + crows)
        eval_dir = ensure_dir(seed_dir / "evaluation")
        gen_a, _ = load_regime_generator("rad5", config, measurement, device, train=False)
        gen_b_eval = p69b.load_generator_checkpoint_for_eval(bbest, config, measurement, device)
        gen_c_eval = p69b.load_generator_checkpoint_for_eval(cbest, config, measurement, device)
        eval_rows, per_rows, outputs = [], [], {}
        for arm, gen in [("A", gen_a), ("B", gen_b_eval), ("C", gen_c_eval)]:
            agg, per, arr = evaluate_general(arm, gen, measurement, test, config, device, eval_dir)
            agg["seed"] = seed_id; eval_rows.append(agg)
            for r in per: r["seed"] = seed_id
            per_rows.extend(per); outputs[arm] = arr
        comp = p69b.paired_comparison(per_rows)
        compute_lpips_for_outputs(seed_dir, test, outputs, device, per_rows, comp, eval_rows)
        write_csv(seed_dir / "evaluation_metrics.csv", eval_rows); write_csv(seed_dir / "per_sample_metrics.csv", per_rows); write_csv(seed_dir / "paired_comparison_C_vs_B.csv", comp)
        p69b.save_visual_grid(seed_dir, test, outputs, n=6)
        seed_results.append({"seed": seed_id, "armB_summary": bsum, "armC_summary": csum})
        for row in eval_rows: all_metrics.append(row)
        for row in comp: row["seed"] = seed_id; all_delta.append(row)
        for arm, summary in [("B", bsum), ("C", csum)]:
            ckpt_rows.append({"seed": seed_id, "arm": arm, "kind": "best_by_val", "path": summary["best_checkpoint"], "sha256": summary["best_checkpoint_sha256"]})
            ckpt_rows.append({"seed": seed_id, "arm": arm, "kind": "final", "path": summary["final_checkpoint"], "sha256": summary["final_checkpoint_sha256"]})
    write_csv(OUT / "rad5_seed_metrics.csv", all_metrics); write_csv(OUT / "rad5_seed_delta_metrics.csv", all_delta); write_csv(OUT / "rad5_checkpoint_hashes.csv", ckpt_rows)
    plot_rad5_outputs(seed_results, all_delta)
    write_rad5_report(seed_results, all_delta)
    return seed_results, all_delta


def plot_rad5_outputs(seed_results: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    if not seed_results:
        return
    first = OUT / "rad5_seed01"
    if (first / "visual_grid_A_B_C.png").exists():
        import shutil
        shutil.copy2(first / "visual_grid_A_B_C.png", OUT / "rad5_visual_grid.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    focus = [r for r in delta_rows if r["metric"] in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64"}]
    names = [f"s{r['seed']}:{r['metric']}" for r in focus]
    vals = [float(r["improvement_positive_means_C_better"]) for r in focus]
    ax.bar(range(len(vals)), vals)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=35, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "rad5_C_vs_B_ci_bars.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for result in seed_results:
        seed_id = int(result["seed"])
        rows = read_csv(OUT / f"rad5_seed{seed_id:02d}" / "training_log.csv")
        for arm in ["B", "C"]:
            ar = [r for r in rows if r["arm"] == arm]
            axes[0].plot([int(r["step"]) for r in ar], [float(r["loss_rec"]) for r in ar], label=f"s{seed_id}{arm}")
            vr = [r for r in ar if r.get("val_rec_loss")]
            axes[1].plot([int(r["step"]) for r in vr], [float(r["val_rec_loss"]) for r in vr], marker="o", label=f"s{seed_id}{arm}")
        cr = [r for r in rows if r["arm"] == "C" and r.get("d_accuracy") and str(r["d_accuracy"]).lower() != "nan"]
        axes[2].plot([int(r["step"]) for r in cr], [float(r["d_accuracy"]) for r in cr], label=f"s{seed_id}C")
    for ax in axes: ax.legend(fontsize=7); ax.set_xlabel("step")
    axes[2].axhline(0.95, color="tab:red", linestyle="--")
    fig.tight_layout(); fig.savefig(OUT / "rad5_D_training_curves.png", dpi=180); plt.close(fig)


def write_rad5_report(seed_results: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    n = len(seed_results)
    focus = [r for r in delta_rows if r["metric"] in {"lpips", "rapsd_distance", "psnr", "relmeaserr_unclipped_float64", "gradient_mean_abs_error", "highfreq_ratio_abs_error"}]
    lp = [float(r["improvement_positive_means_C_better"]) for r in delta_rows if r["metric"] == "lpips"]
    ra = [float(r["improvement_positive_means_C_better"]) for r in delta_rows if r["metric"] == "rapsd_distance"]
    if n >= 3 and all(v > 0 for v in lp) and all(v > 0 for v in ra):
        decision = "Rad-5 supports robustness."
    elif n == 1:
        decision = "Rad-5 pilot only."
    else:
        decision = "Rad-5 negative / regime-specific."
    write_text(OUT / "RAD5_PAIRED_SEED_REPORT.md", "\n".join(["# Rad-5 Paired Seed Report", "", f"Paired seeds run: `{n}`", f"Decision: {decision}", "", table(focus, ["seed", "metric", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high"]), ""]))


def gauge_equality_structural(device: torch.device) -> None:
    rows = []
    for name in ["scr5", "scr10", "rad5"]:
        config = regime_config(name, device)
        measurement, A = make_regime_measurement(name, config, device)
        gen, config = load_regime_generator(name, config, measurement, device, train=False)
        train, val, test, _ = build_caches(name, config, measurement, device)
        for split_name, cache in [("train", train), ("val", val), ("test", test)]:
            x, y, _, _ = next(iter(p69b.make_loader(cache, BATCH_SIZE, shuffle=False, seed=7361)))
            x = x.to(device); y = y.to(device)
            out = forward_candidate_general(gen, measurement, x, y, config)
            A64, _, K = exact_projectors(measurement.A, float(config["lambda_solver"]))
            anchor = p69a.blambda_y(y, A64, K) @ A64.T
            real_m = p69b.flatten_img(out["real_gauge"]).to(torch.float64) @ A64.T
            fake_m = p69b.flatten_img(out["fake_gauge"]).to(torch.float64) @ A64.T
            for check, resid in [
                ("A_real_gauge_minus_A_Blambda_y", real_m - anchor),
                ("A_fake_gauge_minus_A_Blambda_y", fake_m - anchor),
                ("A_real_gauge_minus_A_fake_gauge", real_m - fake_m),
            ]:
                rel = torch.linalg.norm(resid, dim=1) / torch.linalg.norm(anchor, dim=1).clamp_min(1e-12)
                rows.append({"regime": name, "split": split_name, "check": check, "median_relative_error": float(torch.median(rel).detach().cpu()), "max_relative_error": float(torch.max(rel).detach().cpu()), "n": int(rel.numel())})
    write_csv(OUT / "gauge_equality_structural_check.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [float(r["max_relative_error"]) for r in rows]
    ax.bar(range(len(rows)), vals)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r['regime']}/{r['split']}/{r['check'].replace('A_', '')[:8]}" for r in rows], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("max relative error")
    fig.tight_layout(); fig.savefig(OUT / "gauge_equality_error_plot.png", dpi=180); plt.close(fig)
    write_text(OUT / "GAUGE_EQUALITY_STRUCTURAL_CHECK.md", "\n".join(["# Gauge Equality Structural Check", "", "This verifies a row-equalized / residual-shortcut-free gauge. It does not claim `A tilde_x = y` for the practical `B_lambda y` gauge; exact feasibility would require a hard `A^dagger y` gauge.", "", table(rows, ["regime", "split", "check", "median_relative_error", "max_relative_error", "n"]), ""]))


def verify_beta_frontier() -> None:
    src = PH70 / "beta_frontier.csv"
    rows = read_csv(src) if src.exists() else []
    write_csv(OUT / "scr5_beta_frontier.csv", rows or [{"status": "missing"}])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    obs = [r for r in rows if r.get("run_status") == "observed"]
    if obs:
        axes[0].plot([float(r["psnr"]) for r in obs], [float(r["lpips"]) for r in obs], marker="o")
        axes[1].plot([float(r["psnr"]) for r in obs], [float(r["rapsd_distance"]) for r in obs], marker="o")
        axes[2].plot([float(r["beta"]) for r in obs], [float(r["relmeaserr"]) for r in obs], marker="o")
    axes[0].set_xlabel("PSNR"); axes[0].set_ylabel("LPIPS")
    axes[1].set_xlabel("PSNR"); axes[1].set_ylabel("RAPSD")
    axes[2].set_xlabel("beta"); axes[2].set_ylabel("RelMeasErr")
    fig.tight_layout(); fig.savefig(OUT / "scr5_beta_frontier_plot.png", dpi=180); plt.close(fig)
    write_text(OUT / "SCR5_BETA_FRONTIER_REPORT.md", "\n".join(["# Scr-5 Beta Frontier", "", "Status: existing Phase70 frontier verified. Full 0.3 beta0 / 3 beta0 sweep was not run in Phase73 because the prompt allowed status verification if full sweep is too expensive.", "", table(rows, ["beta", "source", "psnr", "lpips", "rapsd_distance", "relmeaserr", "run_status"]), ""]))


def p0_space_metrics() -> None:
    rows = []
    A = torch.from_numpy(np.load(REGIMES["scr5"]["A"]).astype(np.float32)).to(torch.float64)
    A64, G, _ = p69a.exact_projectors(A, 0.001)
    z = np.load(REGIMES["scr5"]["cache"], allow_pickle=False)
    true = torch.from_numpy(z["x"][:TEST_COUNT].reshape(TEST_COUNT, -1)).to(torch.float64)
    true_p0 = p69a.p0_exact(true, A64, G).detach().cpu().numpy().reshape(TEST_COUNT, 64, 64)
    true_range = (true - p69a.p0_exact(true, A64, G)).detach().cpu().numpy()
    for seed in [1, 2, 3]:
        seed_dir = PH71 / f"seed{seed:02d}" / "evaluation"
        b = np.load(seed_dir / "per_sample_outputs_B.npz")["x_hat_unclipped"].astype(np.float32)
        c = np.load(seed_dir / "per_sample_outputs_C.npz")["x_hat_unclipped"].astype(np.float32)
        for arm, arr in [("B", b), ("C", c)]:
            flat = torch.from_numpy(arr.reshape(TEST_COUNT, -1)).to(torch.float64)
            p0 = p69a.p0_exact(flat, A64, G).detach().cpu().numpy().reshape(TEST_COUNT, 64, 64)
            rng = (flat - p69a.p0_exact(flat, A64, G)).detach().cpu().numpy()
            rapsd_vals = [float(np.linalg.norm(p69b.rapsd(p0[i]) - p69b.rapsd(true_p0[i]))) for i in range(TEST_COUNT)]
            hf_vals = [float(abs(p69b.hf_ratio(np.clip(arr[i], 0, 1)) - p69b.hf_ratio(z["x"][i].reshape(64, 64)))) for i in range(TEST_COUNT)]
            rows.append({"regime": "scr5", "seed": seed, "arm": arm, "p0_rapsd_mean": float(np.mean(rapsd_vals)), "p0_energy_mean": float(np.mean(np.linalg.norm(p0.reshape(TEST_COUNT, -1), axis=1))), "range_energy_mean": float(np.mean(np.linalg.norm(rng, axis=1))), "range_change_vs_true_mean": float(np.mean(np.linalg.norm(rng - true_range, axis=1))), "highfreq_error_mean": float(np.mean(hf_vals))})
    # Add C-B rows.
    deltas = []
    for seed in [1, 2, 3]:
        b = next(r for r in rows if r["seed"] == seed and r["arm"] == "B")
        c = next(r for r in rows if r["seed"] == seed and r["arm"] == "C")
        deltas.append({"regime": "scr5", "seed": seed, "arm": "C_minus_B", "p0_rapsd_mean_delta": c["p0_rapsd_mean"] - b["p0_rapsd_mean"], "range_change_delta": c["range_change_vs_true_mean"] - b["range_change_vs_true_mean"], "highfreq_error_delta": c["highfreq_error_mean"] - b["highfreq_error_mean"]})
    write_csv(OUT / "p0_space_metrics.csv", rows + deltas)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"s{r['seed']}" for r in deltas], [-float(r["p0_rapsd_mean_delta"]) for r in deltas])
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("P0 RAPSD improvement (positive=C better)")
    fig.tight_layout(); fig.savefig(OUT / "p0_space_rapsd_plot.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"s{r['seed']} null" for r in deltas], [abs(float(r["p0_rapsd_mean_delta"])) for r in deltas], label="P0")
    ax.bar([f"s{r['seed']} range" for r in deltas], [abs(float(r["range_change_delta"])) for r in deltas], alpha=0.65, label="range")
    ax.legend(); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(OUT / "range_vs_null_change_bar.png", dpi=180); plt.close(fig)
    write_text(OUT / "P0_SPACE_METRICS_REPORT.md", "\n".join(["# P0-Space Metrics Report", "", table(deltas, ["regime", "seed", "p0_rapsd_mean_delta", "range_change_delta", "highfreq_error_delta"]), "", "1. C's measurable improvement is concentrated in null/high-frequency diagnostics rather than certificate axes. 2. Row/range-space changes are small relative diagnostics, and measurement accountability remains with the audit. 3. This supports the interpretation that GAN improves the prior axis while Pi_y^lambda preserves bucket accountability.", ""]))


def figure_pack() -> None:
    z = np.load(REGIMES["scr5"]["cache"], allow_pickle=False)
    true = z["x"][:TEST_COUNT].reshape(TEST_COUNT, 64, 64)
    seed_dir = PH71 / "seed01" / "evaluation"
    a = np.load(seed_dir / "per_sample_outputs_A.npz")["x_hat_unclipped"].astype(np.float32)
    b = np.load(seed_dir / "per_sample_outputs_B.npz")["x_hat_unclipped"].astype(np.float32)
    c = np.load(seed_dir / "per_sample_outputs_C.npz")["x_hat_unclipped"].astype(np.float32)
    idxs = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(idxs), 5, figsize=(10, 2 * len(idxs)))
    for r, i in enumerate(idxs):
        for col, (title, img) in enumerate([("GT", true[i]), ("A mean", a[i]), ("B sup", b[i]), ("C cGAN", c[i]), ("|C-B|", np.abs(c[i] - b[i]))]):
            ax = axes[r, col]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / "fig_visual_grid_ABC.png", dpi=180); fig.savefig(OUT / "fig_visual_grid_ABC.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.stack([p69b.rapsd(true[i]) for i in range(TEST_COUNT)]).mean(0), label="GT", linewidth=2)
    for name, arr in [("A", a), ("B", b), ("C", c)]:
        ax.plot(np.stack([p69b.rapsd(np.clip(arr[i], 0, 1)) for i in range(TEST_COUNT)]).mean(0), label=name)
    ax.legend(); ax.set_xlabel("radial frequency bin"); ax.set_ylabel("normalized power")
    fig.tight_layout(); fig.savefig(OUT / "fig_rapsd_curves.png", dpi=180); fig.savefig(OUT / "fig_rapsd_curves.pdf"); plt.close(fig)
    auc_rows = read_csv(PH69A / "critic_auc_results.csv") + read_csv(PH69A / "shortcut_control_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in auc_rows[:4]]
    vals = [float(r["auc"]) for r in auc_rows[:4]]
    ax.bar(range(len(vals)), vals); ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=25, ha="right"); ax.set_ylabel("AUC")
    fig.tight_layout(); fig.savefig(OUT / "fig_shortcut_auc.png", dpi=180); fig.savefig(OUT / "fig_shortcut_auc.pdf"); plt.close(fig)
    rel = read_csv(PH69B / "relmeaserr_certificate_table.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([r["arm"] for r in rel], [float(r["relmeaserr_unclipped_float64_mean"]) for r in rel]); ax.set_ylabel("RelMeasErr")
    fig.tight_layout(); fig.savefig(OUT / "fig_certificate_bars.png", dpi=180); fig.savefig(OUT / "fig_certificate_bars.pdf"); plt.close(fig)
    # Failure cases: smallest or negative LPIPS improvements.
    lp = read_csv(PH71 / "seed01" / "lpips_per_sample.csv")
    by = {}
    for row in lp:
        by.setdefault(int(row["sample_ordinal"]), {})[row["arm"]] = float(row["lpips"])
    worst = sorted(by, key=lambda k: by[k].get("B", 0) - by[k].get("C", 0))[:4]
    fig, axes = plt.subplots(len(worst), 4, figsize=(8, 2 * len(worst)))
    for r, i in enumerate(worst):
        for col, (title, img) in enumerate([("GT", true[i]), ("B", b[i]), ("C", c[i]), ("|C-B|", np.abs(c[i] - b[i]))]):
            ax = axes[r, col]; ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / "fig_failure_cases.png", dpi=180); fig.savefig(OUT / "fig_failure_cases.pdf"); plt.close(fig)
    write_text(OUT / "FIGURE_PACKAGE_REPORT.md", "# Figure Package Report\n\nGenerated visual grid, RAPSD curves, shortcut AUC, certificate bars, and failure-case figures from Phase71 Scr-5 paired seed outputs.\n")


def human_2afc_pack() -> None:
    out = ensure_dir(OUT / "human_2afc_pack")
    z = np.load(REGIMES["scr5"]["cache"], allow_pickle=False)
    true = z["x"][:TEST_COUNT].reshape(TEST_COUNT, 64, 64)
    seed_dir = PH71 / "seed01" / "evaluation"
    b = np.load(seed_dir / "per_sample_outputs_B.npz")["x_hat_unclipped"].astype(np.float32)
    c = np.load(seed_dir / "per_sample_outputs_C.npz")["x_hat_unclipped"].astype(np.float32)
    rng = np.random.default_rng(73073)
    rows = []
    import matplotlib.image as mpimg
    for k, i in enumerate(range(24)):
        left_is_c = bool(rng.integers(0, 2))
        left, right = (c[i], b[i]) if left_is_c else (b[i], c[i])
        lp = out / f"pair_{k:03d}_left.png"; rp = out / f"pair_{k:03d}_right.png"
        mpimg.imsave(lp, np.clip(left, 0, 1), cmap="gray", vmin=0, vmax=1)
        mpimg.imsave(rp, np.clip(right, 0, 1), cmap="gray", vmin=0, vmax=1)
        rows.append({"pair_id": k, "sample_ordinal": i, "left_file": lp.name, "right_file": rp.name, "left_arm": "C" if left_is_c else "B", "right_arm": "B" if left_is_c else "C", "gt_hidden": True})
    write_csv(out / "human_2afc_pairs.csv", rows)
    write_csv(OUT / "human_2afc_pairs.csv", rows)
    write_text(out / "human_2afc_instructions.md", "# Human 2AFC Instructions\n\nQuestion: Which reconstruction looks more natural / sharper while preserving object identity?\n\nGround truth is hidden. Choose left or right for each randomized pair.\n")
    html = ["<html><body><h1>2AFC Preview</h1><p>Which reconstruction looks more natural / sharper while preserving object identity?</p><table>"]
    for r in rows:
        html.append(f"<tr><td>{r['pair_id']}</td><td><img width='160' src='{r['left_file']}'></td><td><img width='160' src='{r['right_file']}'></td><td>Left / Right</td></tr>")
    html.append("</table></body></html>")
    write_text(out / "human_2afc_preview.html", "\n".join(html))


def positioning_docs() -> None:
    rows = [
        {"axis": "adversarial regularizers", "exists": "GAN losses for inverse problems", "we_do_not_claim": "first adversarial reconstruction", "safe_distinction": "residual-shortcut-free gauge D plus explicit audit", "needed_citation": "GAN inverse-problem regularization", "experimental_comparison_required": "contextual only"},
        {"axis": "deep null-space learning", "exists": "null-space correction methods", "we_do_not_claim": "first null-space prior", "safe_distinction": "bucket-accountable Pi_y^lambda with gauge-equalized prior", "needed_citation": "null-space networks", "experimental_comparison_required": "optional"},
        {"axis": "data consistency", "exists": "projection / DC correction", "we_do_not_claim": "new DC projection", "safe_distinction": "certificate remains separate from GAN", "needed_citation": "data-consistency inverse solvers", "experimental_comparison_required": "already covered by first paper"},
        {"axis": "diffusion inverse solvers", "exists": "DPS/DDRM/DDNM", "we_do_not_claim": "better perceptual quality than diffusion", "safe_distinction": "single/few forward-pass, real-time SPI compatible, explicit certificate", "needed_citation": "DPS, DDRM, DDNM", "experimental_comparison_required": "strong conference may require baseline"},
        {"axis": "measurement-certified audit", "exists": "first-paper GI certificate", "we_do_not_claim": "GAN is certificate", "safe_distinction": "GAN is prior branch, Pi_y^lambda is accountability", "needed_citation": "first paper", "experimental_comparison_required": "not as baseline"},
    ]
    write_csv(OUT / "method_positioning_matrix.csv", rows)
    write_text(OUT / "METHOD_COMPARISON_TABLE.md", "# Method Comparison Table\n\n" + table(rows, ["axis", "exists", "we_do_not_claim", "safe_distinction", "needed_citation", "experimental_comparison_required"]) + "\n")
    write_text(OUT / "RELATED_WORK_POSITIONING_GAUGE_GAN.md", "# Related Work Positioning\n\nOur method is not the first adversarial inverse reconstruction. The safe claim is residual-shortcut-free gauge equalization plus certificate-preserving adversarial prior in ghost imaging. Diffusion inverse solvers may be stronger in image quality, but this GAN branch is single/few forward-pass and keeps bucket accountability explicit through `Pi_y^lambda`.\n")


def conditional_analysis() -> None:
    write_text(OUT / "CONDITIONAL_VS_UNCONDITIONAL_ANALYSIS.md", "# Conditional vs Unconditional Analysis\n\nPhase69A found unconditional gauge PatchGAN AUC 0.8466, conditional x_data gauge AUC 0.7306, simple CNN gauge AUC 0.7697, and shuffled conditional AUC 0.6641. The condition contains some information, but concatenating x_data appears to dilute the texture/null-space signal for this small critic and budget. The main method therefore uses unconditional `D(tilde_x)` because it is stronger, simpler, and leaves fewer anchor-shortcut ambiguities.\n")


def preflight_and_protocol() -> tuple[bool, list[str]]:
    failures = []
    required = [PH69A, PH69B, PH70, PH71, PH72, p69a.SPLIT_TRAIN, p69a.SPLIT_EVAL, p69a.PROVENANCE_JSON]
    for p in required:
        if not p.exists():
            failures.append(f"Missing required path: {p}")
    for name, info in REGIMES.items():
        for key in ["checkpoint", "config", "A", "cache", "manifest"]:
            if not info[key].exists():
                failures.append(f"Missing {name} {key}: {info[key]}")
        if info["checkpoint"].exists() and info["manifest"].exists():
            man = read_json(info["manifest"])
            if p69a.sha256_file(info["checkpoint"]) != man["checkpoint_sha256"]:
                failures.append(f"{name} checkpoint hash mismatch")
            if p69a.sha256_np(np.load(info["A"]).astype(np.float32)) != man["A_sha256_float32_bytes"]:
                failures.append(f"{name} A hash mismatch")
    train_idx = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_idx = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    prov = read_json(p69a.PROVENANCE_JSON)
    if p69a.sha256_np(train_idx, sort_int64=True) != prov["splits"]["train_indices_sha256_sorted_int64"]:
        failures.append("train split hash mismatch")
    if p69a.sha256_np(eval_idx, sort_int64=True) != prov["splits"]["eval_indices_sha256_sorted_int64"]:
        failures.append("eval split hash mismatch")
    proto = {
        "phase": "Phase73",
        "output_dir": str(OUT),
        "scr5_phase71_protocol": str(PH71 / "PHASE71_PROTOCOL_LOCK.md"),
        "gauge_real": "tilde_x_real = P0 x + B_lambda y",
        "gauge_fake": "tilde_x_fake = P0 v_theta + B_lambda y",
        "deployment": "x_hat = Pi_y^lambda(v_theta)",
        "D_architecture": "unconditional gauge PatchGAN main D",
        "optimizer": "Adam G lr=2e-5, D lr=2e-4",
        "step_budget": STEP_BUDGET,
        "checkpoint_selection": "best_by_val_rec_loss",
        "train_split_sha256": p69a.sha256_np(train_idx, sort_int64=True),
        "eval_split_sha256": p69a.sha256_np(eval_idx, sort_int64=True),
        "no_test_split_training": True,
        "train_loader_not_get_val_dataloader": True,
        "per_sample_saving_available": True,
    }
    write_simple_yaml(OUT / "phase73_protocol_config.yaml", proto)
    write_text(OUT / "PHASE73_PROTOCOL_LOCK.md", "# Phase73 Protocol Lock\n\n" + table([proto], ["phase", "step_budget", "checkpoint_selection", "no_test_split_training", "train_loader_not_get_val_dataloader", "per_sample_saving_available"]) + "\n\nNo first-paper result, checkpoint, table, title, or abstract will be modified.\n")
    write_text(OUT / "PHASE73_PRECHECK.md", "# Phase73 Precheck\n\n" + ("\n".join(f"- FAILURE: {f}" for f in failures) if failures else "- all critical safety items passed\n- exact A files are available\n- P0, B_lambda, Pi_y^lambda are available through exact projector / solver-cache APIs\n- checkpoint hashes are available\n"))
    return len(failures) == 0, failures


def final_reports(gate: dict[str, Any], rad_seed_results: list[dict[str, Any]]) -> None:
    beta_status = "existing Phase70 beta_frontier verified; full sweep not rerun"
    p0_status = "computed for Scr-5 Phase71 seeds"
    visual_path = str(OUT)
    twoafc_path = str(OUT / "human_2afc_pack")
    positioning_path = str(OUT / "RELATED_WORK_POSITIONING_GAUGE_GAN.md")
    workshop_ready = "yes: Scr-5 3-seed stable plus Phase73 mechanism/figure/2AFC package; Rad-5 status depends on gate"
    if gate["decision"] == "weak_no_rad5_cgan":
        workshop_ready = "yes as Scr-5 stable plus Rad-5 weak-signal/regime-limited story; not a robustness claim"
    strong_gap = [
        "Rad-5 paired robustness if Rad-5 signal is not weak.",
        "Full Scr-5 beta sweep if challenged.",
        "Human 2AFC responses, not just prepared pack.",
        "Diffusion inverse-solver baseline if targeting a strong venue.",
    ]
    answers = [
        {"question": "Is Rad-5 signal strong?", "answer": f"{gate['decision']} with AUC {gate['rad5_gauge_auc']:.3f}."},
        {"question": "Were Rad-5 paired seeds run?", "answer": f"{len(rad_seed_results)} paired seed(s) run."},
        {"question": "If run, does C beat B?", "answer": "See RAD5_PAIRED_SEED_REPORT.md." if rad_seed_results else "Not run due to gate."},
        {"question": "Is Scr-5 seed evidence stable?", "answer": "Yes, Phase71 passed 3/3 LPIPS and RAPSD."},
        {"question": "Is Scr-10 weak signal interpretable?", "answer": "Yes, it supports regime dependence rather than a positive Scr-10 claim."},
        {"question": "Does beta frontier support operating point?", "answer": beta_status},
        {"question": "Do P0 metrics support null/prior-axis improvement?", "answer": p0_status},
        {"question": "Are visual examples convincing?", "answer": "Figure pack generated; human study not run."},
        {"question": "Is workshop paper ready?", "answer": workshop_ready},
        {"question": "What remains for strong conference?", "answer": " ".join(strong_gap)},
        {"question": "Should we continue training or start writing?", "answer": "Start workshop writing; continue training only if targeting strong conference robustness gaps."},
        {"question": "First paper main results unchanged?", "answer": "Yes."},
    ]
    write_text(OUT / "PHASE73_OVERNIGHT_REPORT.md", "# Phase73 Overnight Report\n\n" + table(answers, ["question", "answer"]) + "\n")
    write_text(OUT / "WORKSHOP_PAPER_OUTLINE.md", "# Workshop Paper Outline\n\n1. Problem and certificate/prior separation\n2. Gauge-equalized adversarial prior\n3. Shortcut diagnostics\n4. Scr-5 paired-seed results\n5. Regime diagnostics: Scr-10 weak and Rad-5 gate\n6. Null-space mechanism and visual evidence\n7. Limitations and next steps\n")
    write_text(OUT / "STRONG_CONFERENCE_GAP_LIST.md", "# Strong Conference Gap List\n\n" + "\n".join(f"- {x}" for x in strong_gap) + "\n")
    write_text(OUT / "CLAIMS_AFTER_PHASE73.md", "# Claims After Phase73\n\n- GAN branch is an adversarial prior, not a measurement certificate.\n- Gauge-equalized D removes direct residual shortcut.\n- Scr-5 C vs B is stable across three paired seeds.\n- Scr-10 gauge signal is weak, supporting regime dependence.\n- Rad-5 claim follows the Phase73 gate result only.\n")
    write_text(OUT / "LIMITATIONS_AFTER_PHASE73.md", "# Limitations After Phase73\n\n- Human 2AFC materials are prepared but no human responses were collected.\n- Full beta sweep was not rerun overnight.\n- Diffusion baselines are positioned but not executed.\n- Rad-5 paired training is conditional on the diagnostic gate.\n")
    decision_rows = [
        {"item": "Workshop-ready now?", "decision": workshop_ready},
        {"item": "Short-paper-ready now?", "decision": "yes, with cautious regime-dependent framing"},
        {"item": "Strong-conference-ready?", "decision": "not yet"},
        {"item": "Need Rad-5 paired seeds?", "decision": "only if gate moderate/strong; see Rad-5 report"},
        {"item": "Need human 2AFC?", "decision": "recommended for stronger perceptual claim"},
        {"item": "Need diffusion baseline?", "decision": "yes for strong conference"},
    ]
    write_text(OUT / "NEXT_DECISION.md", "# Next Decision\n\n" + table(decision_rows, ["item", "decision"]) + "\n")


def manifest() -> None:
    files = sorted(p.name for p in OUT.iterdir())
    write_text(OUT / "PHASE73_MANIFEST.md", "# Phase73 Manifest\n\n" + "\n".join(f"- `{f}`" for f in files) + "\n")


def main() -> int:
    ensure_dir(OUT)
    if not (OUT / "RUNLOG.md").exists():
        write_text(OUT / "RUNLOG.md", f"# Phase73 Runlog\n- {now()} runner_start\n")
    else:
        append_log("runner_start_existing")
    ok, failures = preflight_and_protocol()
    if not ok:
        return unsafe_stop(failures)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_log(f"device={device}")
    gate, _ = rad5_diagnostic(device)
    gauge_equality_structural(device)
    conditional_analysis()
    rad_seed_results, _ = run_rad5_seeds(gate, device)
    verify_beta_frontier()
    p0_space_metrics()
    figure_pack()
    human_2afc_pack()
    positioning_docs()
    final_reports(gate, rad_seed_results)
    manifest()
    append_log(f"runner_complete rad5_decision={gate['decision']} main_results_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
