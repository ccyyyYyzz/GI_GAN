from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src import phase73_overnight_gauge_gan_expansion as p73
from src.models import build_generator
from src.phase79_rad5_rowspace_diversity_diagnostic import forward_with_noise
from src.projections import exact_data_anchor, exact_null_project, get_exact_projector, relative_measurement_error
from src.utils import apply_experiment_defaults
from src.compatibility_model import CompatibilityCritic
from src.phase1_1_controls import pair_features, sum_image_features, random_derangement


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DATA_ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
PHASE79_CKPT = DATA_ROOT / "outputs_phase79_posterior_anti_collapse" / "rad5_rowspace_diversity_diagnostic" / "checkpoints" / "final.pt"
PHASE1_1 = ROOT / "outputs" / "compatibility" / "phase1_1_corrected_rad5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.2 Rad-5/64 candidate coverage and transfer pipeline.")
    parser.add_argument("--output_dir", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dev_images", type=int, default=96)
    parser.add_argument("--kmax", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--checkpoint", default=str(PHASE79_CKPT))
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "UNKNOWN"
    try:
        diff_stat = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--stat"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        diff_stat = "UNKNOWN"
    return {"git_commit": commit, "git_diff_stat": diff_stat}


def make_phase79_measurement(device: torch.device):
    config = p73.regime_config("rad5", device)
    measurement, A = p73.make_regime_measurement("rad5", config, device)
    return measurement, A, config


def load_phase79_generator(checkpoint: Path, base_config: dict[str, Any], measurement, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    merged = dict(base_config)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        merged.update(ckpt["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(DATA_ROOT / "data")
    merged["output_dir"] = str(OUT)
    merged["batch_size"] = 8
    merged["num_workers"] = 0
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged["output_range_mode"] = "clamp_eval_only"
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    state_key = "generator_ema" if isinstance(ckpt, dict) and ckpt.get("generator_ema") is not None else "generator"
    state = ckpt.get(state_key) if isinstance(ckpt, dict) else None
    if state is None:
        raise RuntimeError(f"No generator/generator_ema state in {checkpoint}")
    load = generator.load_state_dict(state, strict=False)
    strict_missing = list(load.missing_keys)
    strict_unexpected = list(load.unexpected_keys)
    if strict_missing or strict_unexpected:
        raise RuntimeError(f"Phase79 checkpoint did not load strictly: missing={strict_missing}, unexpected={strict_unexpected}")
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return generator, merged, ckpt, state_key, strict_missing, strict_unexpected


def build_dev_cache(measurement, device: torch.device, count: int):
    _train, _val, test, split = p73.build_caches("rad5", p73.regime_config("rad5", device), measurement, device)
    n = min(int(count), int(test.x.shape[0]))
    x = test.x[:n].float()
    labels = test.labels[:n].long()
    indices = test.indices[:n].long()
    # Use clean measurements for the coverage protocol, not cached noisy y.
    ys = []
    for start in range(0, n, 32):
        xb = x[start : start + 32].to(device)
        ys.append(measurement.A_forward(measurement.flatten_img(xb)).detach().cpu())
    y = torch.cat(ys, 0)
    return {"x": x, "y": y, "labels": labels, "indices": indices, "split": split, "name": "legacy_seen_dev_64"}


def pairwise_rmse(mat: torch.Tensor) -> torch.Tensor:
    if mat.shape[0] < 2:
        return torch.zeros(1)
    diffs = mat[:, None, :] - mat[None, :, :]
    d = torch.sqrt(torch.mean(diffs * diffs, dim=-1))
    iu = torch.triu_indices(mat.shape[0], mat.shape[0], offset=1)
    return d[iu[0], iu[1]]


def effective_rank(centered: torch.Tensor) -> float:
    if centered.shape[0] < 2:
        return 1.0
    s = torch.linalg.svdvals(centered.float())
    p = (s * s) / (s * s).sum().clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(ent).item())


def tv_and_freq(x_flat: torch.Tensor, img_size: int) -> dict[str, float]:
    img = x_flat.reshape(x_flat.shape[0], img_size, img_size)
    dx = img[:, :, 1:] - img[:, :, :-1]
    dy = img[:, 1:, :] - img[:, :-1, :]
    tv = (dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))).mean()
    grad = torch.sqrt((dx * dx).mean(dim=(1, 2)) + (dy * dy).mean(dim=(1, 2)) + 1e-12).mean()
    fft = torch.fft.fft2(img)
    power = (fft.real * fft.real + fft.imag * fft.imag)
    h, w = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    rr = torch.sqrt((yy.float() - h / 2) ** 2 + (xx.float() - w / 2) ** 2)
    rr = torch.fft.fftshift(rr).to(power.device)
    mask_hi = (rr / rr.max()) >= 0.35
    hi = (power[:, mask_hi].sum(dim=1) / power.reshape(power.shape[0], -1).sum(dim=1).clamp_min(1e-12)).mean()
    return {"tv_mean": float(tv.item()), "grad_rms_mean": float(grad.item()), "high_freq_fraction_mean": float(hi.item())}


def clean_y_for_x(measurement, x: torch.Tensor, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    ys = []
    for start in range(0, int(x.shape[0]), int(batch_size)):
        xb = x[start : start + batch_size].to(device)
        ys.append(measurement.A_forward(measurement.flatten_img(xb)).detach().cpu())
    return torch.cat(ys, 0)


def split_subset_from_cache(cache, count: int, name: str, measurement, device: torch.device) -> dict[str, Any]:
    n = min(int(count), int(cache.x.shape[0]))
    x = cache.x[:n].float()
    return {
        "name": name,
        "x": x,
        "y": clean_y_for_x(measurement, x, device),
        "labels": cache.labels[:n].long(),
        "indices": cache.indices[:n].long(),
    }


@torch.no_grad()
def generate_pool(generator, measurement, config: dict[str, Any], y: torch.Tensor, *, kmax: int, base_seed: int, device: torch.device):
    y_rep = y.repeat(int(kmax), 1).to(device)
    x_data_flat = measurement.data_solution(y_rep.float(), config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    noise_maps = []
    seeds = []
    for k in range(int(kmax)):
        seed = int(base_seed + k)
        gen = torch.Generator(device=device).manual_seed(seed)
        noise_maps.append(torch.randn(x_data[:1].shape, device=device, generator=gen, dtype=x_data.dtype))
        seeds.append(seed)
    noise = torch.cat(noise_maps, dim=0)
    out = forward_with_noise(generator, measurement, y_rep, noise, config)
    return out, seeds


@torch.no_grad()
def coverage_eval(generator, measurement, config: dict[str, Any], cache: dict[str, Any], *, out: Path, kmax: int, seed: int, device: torch.device):
    img_size = int(measurement.img_size)
    n_pix = int(measurement.n)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"kmax": int(kmax), "images": []}
    all_pixel_std = []
    all_p0_var = []
    all_dup = []
    all_near = []
    all_eff = []
    all_rel = []
    curve: dict[int, list[float]] = {k: [] for k in [1, 4, 8, 16, 32] if k <= kmax}
    det_errors = []
    rand_errors = {k: [] for k in curve}
    best_errors = {k: [] for k in curve}
    posterior_errors = {k: [] for k in curve}
    winner_indices = {k: [] for k in curve}
    rel_best_minus_random = {k: [] for k in curve}
    candidate_dist_rows: list[dict[str, Any]] = []
    lpips_available = False
    try:
        import lpips  # noqa: F401

        lpips_available = True
    except Exception:
        lpips_available = False

    for i in range(int(cache["x"].shape[0])):
        x = cache["x"][i : i + 1].to(device)
        y = cache["y"][i : i + 1].to(device)
        x_flat = measurement.flatten_img(x)
        true_n = exact_null_project(x_flat, measurement, dtype=torch.float64, device=device)
        r_y = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False)
        zero_noise = torch.zeros(1, 1, img_size, img_size, device=device)
        det_out = forward_with_noise(generator, measurement, y, zero_noise, config)
        det_native = det_out["x_hat_flat"]
        det_n = exact_null_project(det_native, measurement, dtype=torch.float64, device=device)
        det_canon = r_y + det_n
        det_p0 = float(torch.sqrt(torch.mean((det_n - true_n) ** 2)).item())
        det_errors.append(det_p0)
        pool, seeds = generate_pool(generator, measurement, config, y, kmax=kmax, base_seed=seed + i * 1000, device=device)
        native = pool["x_hat_flat"]
        raw_vpre = pool["v_pre"]
        cand_n = exact_null_project(native, measurement, dtype=torch.float64, device=device)
        canon = r_y.repeat(kmax, 1) + cand_n
        p0_err = torch.sqrt(torch.mean((cand_n - true_n.repeat(kmax, 1)) ** 2, dim=1)).detach().cpu()
        full_rmse = torch.sqrt(torch.mean((canon - x_flat.repeat(kmax, 1).to(canon.dtype)) ** 2, dim=1)).detach().cpu()
        psnr = (-20.0 * torch.log10(full_rmse.clamp_min(1e-12))).detach().cpu()
        rel = relative_measurement_error(canon.float(), y.repeat(kmax, 1).float(), measurement).detach().cpu()
        all_rel.extend(rel.tolist())
        centered = canon - canon.mean(dim=0, keepdim=True)
        pixel_std = float(canon.std(dim=0, unbiased=False).mean().detach().cpu().item())
        p0_var = float(cand_n.var(dim=0, unbiased=False).mean().detach().cpu().item())
        pair_p0 = pairwise_rmse(cand_n.detach().cpu()).numpy()
        rounded = torch.round(canon.detach().cpu() * 1e6).to(torch.int64).numpy()
        uniq = {hashlib.sha256(row.tobytes()).hexdigest() for row in rounded}
        dup_ratio = 1.0 - len(uniq) / float(kmax)
        near_ratio = float(np.mean(pair_p0 < 1e-5)) if pair_p0.size else 0.0
        eff = effective_rank(centered.detach().cpu())
        all_pixel_std.append(pixel_std)
        all_p0_var.append(p0_var)
        all_dup.append(dup_ratio)
        all_near.append(near_ratio)
        all_eff.append(eff)
        dist = tv_and_freq(canon.float().detach().cpu(), img_size)
        below = float((canon < 0).float().mean().detach().cpu().item())
        above = float((canon > 1).float().mean().detach().cpu().item())
        candidate_dist_rows.append(
            {
                "sample_ordinal": i,
                "source_index": int(cache["indices"][i].item()),
                "pixel_std": pixel_std,
                "p0_variance": p0_var,
                "pairwise_p0_rmse_mean": float(pair_p0.mean()) if pair_p0.size else 0.0,
                "exact_duplicate_ratio": dup_ratio,
                "near_duplicate_ratio": near_ratio,
                "effective_rank": eff,
                "fraction_below_0": below,
                "fraction_above_1": above,
                **dist,
                "relmeaserr_max": float(rel.max().item()),
                "lpips_available": bool(lpips_available),
            }
        )
        for k in curve:
            prefix = slice(0, int(k))
            pe = p0_err[prefix]
            best_idx = int(torch.argmin(pe).item())
            best = float(pe[best_idx].item())
            random_expect = float(pe.mean().item())
            mean_flat = canon[prefix].mean(dim=0, keepdim=True)
            mean_n = exact_null_project(mean_flat, measurement, dtype=torch.float64, device=device)
            post = float(torch.sqrt(torch.mean((mean_n - true_n) ** 2)).item())
            curve[k].append((det_p0 - best) / max(det_p0, 1e-12))
            rand_errors[k].append(random_expect)
            best_errors[k].append(best)
            posterior_errors[k].append(post)
            winner_indices[k].append(best_idx)
            rel_best_minus_random[k].append((random_expect - best) / max(random_expect, 1e-12))
        rows.append(
            {
                "sample_ordinal": i,
                "source_index": int(cache["indices"][i].item()),
                "deterministic_p0_rmse": det_p0,
                "random_k32_expected_p0_rmse": float(p0_err.mean().item()),
                "posterior_mean_k32_p0_rmse": posterior_errors[max(curve.keys())][-1],
                "best_k32_p0_rmse": best_errors[max(curve.keys())][-1],
                "best_k32_full_rmse": float(full_rmse[torch.argmin(p0_err)].item()),
                "best_k32_psnr": float(psnr[torch.argmin(p0_err)].item()),
                "oracle_improved": bool(best_errors[max(curve.keys())][-1] < det_p0),
                "oracle_winner_k32": int(winner_indices[max(curve.keys())][-1]),
                "relmeaserr_max_canonical": float(rel.max().item()),
                "lpips_oracle": "[DATA MISSING]" if not lpips_available else "[NOT_COMPUTED]",
            }
        )
        manifest["images"].append(
            {
                "sample_ordinal": i,
                "source_index": int(cache["indices"][i].item()),
                "candidate_seeds": seeds,
                "k_prefixes": {str(k): list(range(k)) for k in curve},
                "raw_output_name": "v_pre",
                "native_audited_output_name": "x_hat_flat",
                "canonicalized_output": "exact_data_anchor(y)+exact_null_project(x_hat_flat)",
            }
        )
    write_csv(out / "reports" / "per_image_coverage_64.csv", rows)
    write_csv(out / "reports" / "candidate_distribution_64_per_image.csv", candidate_dist_rows)
    save_json(out / "manifests" / "candidate_pool_dev_64.json", manifest)
    summary_curve = {}
    for k in curve:
        det = np.asarray(det_errors)
        rnd = np.asarray(rand_errors[k])
        best = np.asarray(best_errors[k])
        post = np.asarray(posterior_errors[k])
        improvement = (det - best) / np.maximum(det, 1e-12)
        headroom = rnd - best
        summary_curve[str(k)] = {
            "deterministic_p0_rmse_mean": float(det.mean()),
            "random_expected_p0_rmse_mean": float(rnd.mean()),
            "posterior_mean_p0_rmse_mean": float(post.mean()),
            "oracle_best_p0_rmse_mean": float(best.mean()),
            "oracle_relative_improvement_vs_det_mean": float(improvement.mean()),
            "oracle_improved_fraction": float(np.mean(best < det)),
            "oracle_headroom_vs_random_mean": float(headroom.mean()),
            "oracle_winner_index_histogram": {str(int(v)): int(np.sum(np.asarray(winner_indices[k]) == v)) for v in sorted(set(winner_indices[k]))},
        }
    diversity = {
        "fixed_y_pixel_std_mean": float(np.mean(all_pixel_std)),
        "exact_p0_variance_mean": float(np.mean(all_p0_var)),
        "exact_duplicate_ratio_mean": float(np.mean(all_dup)),
        "near_duplicate_ratio_mean": float(np.mean(all_near)),
        "covariance_effective_rank_mean": float(np.mean(all_eff)),
        "relmeaserr_max": float(np.max(all_rel)),
        "lpips_available": bool(lpips_available),
    }
    coverage_k16 = summary_curve.get("16") or summary_curve[str(max(curve.keys()))]
    duplicate_ok = diversity["exact_duplicate_ratio_mean"] < 0.10
    p0_var_ok = diversity["exact_p0_variance_mean"] > 1e-8
    rel_improve = float(coverage_k16["oracle_relative_improvement_vs_det_mean"])
    frac_improved = float(coverage_k16["oracle_improved_fraction"])
    headroom_ok = rel_improve >= 0.05 or (rel_improve >= 0.02 and frac_improved >= 0.25)
    gate_checks = {
        "exact_duplicate_ratio_lt_0_10": duplicate_ok,
        "p0_variation_above_numeric_noise": p0_var_ok,
        "k16_oracle_headroom": headroom_ok,
        "at_least_25_percent_images_improve": frac_improved >= 0.25,
        "same_row_audit_protocol": True,
        "relmeaserr_not_driving_oracle": diversity["relmeaserr_max"] < 1e-2,
    }
    if not duplicate_ok or not p0_var_ok:
        classification = "FAIL_CANDIDATE_COLLAPSE"
    elif not headroom_ok or frac_improved < 0.25:
        classification = "FAIL_NO_ORACLE_HEADROOM"
    elif diversity["relmeaserr_max"] >= 1e-2:
        classification = "INCONCLUSIVE_ALIGNMENT_FAILURE"
    else:
        classification = "PASS_USEFUL_CANDIDATE_COVERAGE"
    report = {
        "phase": "E2a_64_candidate_coverage",
        "classification": classification,
        "gate_checks": gate_checks,
        "dev_image_count": int(cache["x"].shape[0]),
        "kmax": int(kmax),
        "candidate_diversity": diversity,
        "coverage_curve": summary_curve,
        "primary_oracle": "argmin exact P0 RMSE of canonicalized candidates",
        "outputs": {
            "per_image_csv": str(out / "reports" / "per_image_coverage_64.csv"),
            "candidate_manifest": str(out / "manifests" / "candidate_pool_dev_64.json"),
        },
    }
    save_json(out / "reports" / "gate_report_e2a_64_coverage.json", report)
    save_json(out / "reports" / "candidate_distribution_64.json", {"summary": diversity, "per_image_csv": str(out / "reports" / "candidate_distribution_64_per_image.csv")})
    return report


@torch.no_grad()
def build_candidate_cache(
    generator,
    measurement,
    config: dict[str, Any],
    split: dict[str, Any],
    *,
    out: Path,
    k: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    img_size = int(measurement.img_size)
    n_pix = int(measurement.n)
    r_list, true_n_list, cand_n_list, err_list, det_err_list, post_err_list, seed_rows = [], [], [], [], [], [], []
    raw_min, raw_max, native_min, native_max = [], [], [], []
    for i in range(int(split["x"].shape[0])):
        x = split["x"][i : i + 1].to(device)
        y = split["y"][i : i + 1].to(device)
        x_flat = measurement.flatten_img(x)
        r_y = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False).float()
        true_n = exact_null_project(x_flat, measurement, dtype=torch.float64, device=device).float()
        zero = torch.zeros(1, 1, img_size, img_size, device=device)
        det = forward_with_noise(generator, measurement, y, zero, config)["x_hat_flat"]
        det_n = exact_null_project(det, measurement, dtype=torch.float64, device=device).float()
        det_err = torch.sqrt(torch.mean((det_n - true_n) ** 2, dim=1))
        pool, seeds = generate_pool(generator, measurement, config, y, kmax=k, base_seed=seed + i * 1000, device=device)
        native = pool["x_hat_flat"].float()
        raw = pool["v_pre"].float()
        cand_n = exact_null_project(native, measurement, dtype=torch.float64, device=device).float()
        err = torch.sqrt(torch.mean((cand_n - true_n.repeat(k, 1)) ** 2, dim=1))
        post_n = cand_n.mean(dim=0, keepdim=True)
        post_err = torch.sqrt(torch.mean((post_n - true_n) ** 2, dim=1))
        r_list.append(r_y.detach().cpu())
        true_n_list.append(true_n.detach().cpu())
        cand_n_list.append(cand_n.detach().cpu())
        err_list.append(err.detach().cpu())
        det_err_list.append(det_err.detach().cpu())
        post_err_list.append(post_err.detach().cpu())
        raw_min.append(float(raw.min().item()))
        raw_max.append(float(raw.max().item()))
        native_min.append(float(native.min().item()))
        native_max.append(float(native.max().item()))
        seed_rows.append({"sample_ordinal": i, "source_index": int(split["indices"][i].item()), "candidate_seeds": seeds})
    cache = {
        "name": split["name"],
        "x": split["x"].reshape(split["x"].shape[0], -1).float(),
        "y": split["y"].float(),
        "r": torch.cat(r_list, 0).float(),
        "true_n": torch.cat(true_n_list, 0).float(),
        "cand_n": torch.stack(cand_n_list, 0).float(),
        "p0_error": torch.stack(err_list, 0).float(),
        "deterministic_p0_error": torch.cat(det_err_list, 0).float(),
        "posterior_mean_p0_error": torch.cat(post_err_list, 0).float(),
        "indices": split["indices"].long(),
        "labels": split["labels"].long(),
        "k": int(k),
        "img_size": img_size,
        "n": n_pix,
        "raw_range": {"min": float(np.min(raw_min)), "max": float(np.max(raw_max))},
        "native_range": {"min": float(np.min(native_min)), "max": float(np.max(native_max))},
        "candidate_seed_rows": seed_rows,
    }
    path = out / "candidate_cache" / f"{split['name']}_k{k}.pt"
    ensure(path.parent)
    torch.save(cache, path)
    write_csv(out / "manifests" / f"{split['name']}_candidate_seeds.csv", seed_rows)
    return cache


def feature_matrix_for_cache(cache: dict[str, Any], mode: str) -> tuple[np.ndarray, list[str]]:
    n_img, k, n_pix = cache["cand_n"].shape
    r = cache["r"][:, None, :].repeat(1, k, 1).reshape(n_img * k, n_pix)
    cn = cache["cand_n"].reshape(n_img * k, n_pix)
    if mode == "pair":
        return pair_features(r, cn, int(cache["img_size"]))
    if mode == "sum":
        return sum_image_features(r, cn, int(cache["img_size"]))
    raise ValueError(mode)


def evaluate_scores(cache: dict[str, Any], scores: np.ndarray, method: str) -> dict[str, Any]:
    err = cache["p0_error"].numpy()
    n_img, k = err.shape
    score = scores.reshape(n_img, k)
    selected = np.argmax(score, axis=1)
    selected_err = err[np.arange(n_img), selected]
    random_err = err.mean(axis=1)
    oracle_idx = np.argmin(err, axis=1)
    oracle_err = err[np.arange(n_img), oracle_idx]
    det = cache["deterministic_p0_error"].numpy()
    post = cache["posterior_mean_p0_error"].numpy()
    denom = random_err - oracle_err
    gain = np.where(np.abs(denom) > 1e-12, (random_err - selected_err) / denom, np.nan)
    return {
        "method": method,
        "selected_p0_rmse_mean": float(selected_err.mean()),
        "random_expected_p0_rmse_mean": float(random_err.mean()),
        "oracle_p0_rmse_mean": float(oracle_err.mean()),
        "deterministic_p0_rmse_mean": float(det.mean()),
        "posterior_mean_p0_rmse_mean": float(post.mean()),
        "selection_regret_mean": float((selected_err - oracle_err).mean()),
        "oracle_gain_fraction_mean": float(np.nanmean(gain)),
        "top_oracle_hit_rate": float(np.mean(selected == oracle_idx)),
        "selected_rank_mean": float(np.mean([1 + int(np.where(np.argsort(err[i]) == selected[i])[0][0]) for i in range(n_img)])),
        "selected_beats_random_fraction": float(np.mean(selected_err < random_err)),
    }


def evaluate_baselines(cache: dict[str, Any]) -> dict[str, Any]:
    err = cache["p0_error"].numpy()
    n_img, k = err.shape
    random_err = err.mean(axis=1)
    oracle_err = err.min(axis=1)
    det = cache["deterministic_p0_error"].numpy()
    post = cache["posterior_mean_p0_error"].numpy()
    denom = random_err - oracle_err
    return {
        "deterministic": {"p0_rmse_mean": float(det.mean())},
        "random_expectation": {"p0_rmse_mean": float(random_err.mean())},
        "posterior_mean": {"p0_rmse_mean": float(post.mean())},
        "oracle_best_of_k": {"p0_rmse_mean": float(oracle_err.mean())},
        "oracle_gain_available_mean": float(np.mean(denom)),
    }


def train_scalar_selector(train_cache: dict[str, Any], val_cache: dict[str, Any], mode: str) -> tuple[dict[str, Any], np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    xtr, names = feature_matrix_for_cache(train_cache, mode)
    ytr = -train_cache["p0_error"].reshape(-1).numpy()
    xv, _ = feature_matrix_for_cache(val_cache, mode)
    candidates = []
    for name, model in [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=44)),
    ]:
        model.fit(xtr, ytr)
        scores = model.predict(xv)
        metrics = evaluate_scores(val_cache, scores, f"{mode}_{name}")
        candidates.append((metrics["selected_p0_rmse_mean"], name, model, scores, metrics))
    _loss, name, model, scores, metrics = min(candidates, key=lambda t: t[0])
    metrics["selected_model"] = name
    metrics["feature_count"] = int(len(names))
    return metrics, scores


class CandidateRankDataset(Dataset):
    def __init__(self, cache: dict[str, Any], mode: str = "global") -> None:
        self.cache = cache
        self.mode = mode

    def __len__(self) -> int:
        return int(self.cache["r"].shape[0])

    def _prep(self, flat: torch.Tensor) -> torch.Tensor:
        img = flat.reshape(*flat.shape[:-1], 1, self.cache["img_size"], self.cache["img_size"]).float()
        if self.mode == "zscore":
            mean = img.mean(dim=(-1, -2, -3), keepdim=True)
            std = img.std(dim=(-1, -2, -3), unbiased=False, keepdim=True).clamp_min(1e-6)
            return (img - mean) / std
        if self.mode == "rms":
            rms = torch.sqrt(torch.mean(img * img, dim=(-1, -2, -3), keepdim=True) + 1e-8)
            return img / rms
        return img

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "r": self._prep(self.cache["r"][idx]),
            "n": self._prep(self.cache["cand_n"][idx]),
            "err": self.cache["p0_error"][idx].float(),
            "true_n": self._prep(self.cache["true_n"][idx]),
        }


def pretrain_counterfactual(model, cache: dict[str, Any], kind: str, device: torch.device, *, seed: int, epochs: int = 1, mode: str = "global") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(cache["r"].shape[0])
    donors = random_derangement(n, seed)
    target = cache["p0_error"].reshape(-1).numpy()
    target = rng.choice(target, size=n, replace=True)
    true_n = cache["true_n"]
    negs = []
    alphas = []
    for i in range(n):
        j = int(donors[i])
        if kind == "raw":
            negs.append(true_n[j])
            alphas.append(1.0)
        else:
            diff = true_n[j] - true_n[i]
            denom = torch.linalg.norm(diff).item()
            alpha = float(target[i] * math.sqrt(cache["n"]) / max(denom, 1e-12))
            alpha = float(np.clip(alpha, 0.05, 1.5))
            negs.append(true_n[i] + alpha * diff)
            alphas.append(alpha)
    negs_t = torch.stack(negs, 0).float()
    ds = CandidateRankDataset(cache, mode=mode)
    loader = DataLoader(torch.arange(n), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for _ in range(int(epochs)):
        model.train()
        for idx in loader:
            r = ds._prep(cache["r"][idx]).to(device).reshape(-1, 1, cache["img_size"], cache["img_size"])
            pos = ds._prep(true_n[idx]).to(device).reshape(-1, 1, cache["img_size"], cache["img_size"])
            neg = ds._prep(negs_t[idx]).to(device).reshape(-1, 1, cache["img_size"], cache["img_size"])
            sp = model.score_pairs(r, pos)
            sn = model.score_pairs(r, neg)
            loss = F.softplus(-(sp - sn)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {
        "kind": kind,
        "alpha_mean": float(np.mean(alphas)),
        "alpha_min": float(np.min(alphas)),
        "alpha_max": float(np.max(alphas)),
        "donor_unique_fraction": float(len(np.unique(donors)) / n),
    }


def train_ranker(
    train_cache: dict[str, Any],
    val_cache: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    pretrain: str | None = None,
    structural: bool = False,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mode = "zscore" if structural else "global"
    model = CompatibilityCritic(embed_dim=128, base_channels=24, temperature=0.07).to(device)
    pre_report = {"kind": "none"}
    if pretrain:
        pre_report = pretrain_counterfactual(model, train_cache, pretrain, device, seed=seed + 99, epochs=1, mode=mode)
    ds = CandidateRankDataset(train_cache, mode=mode)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + 1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for _epoch in range(2):
        model.train()
        for batch in loader:
            r = batch["r"].to(device)
            n = batch["n"].to(device)
            err = batch["err"].to(device)
            b, k = err.shape
            r_rep = r[:, None].repeat(1, k, 1, 1, 1).reshape(b * k, 1, train_cache["img_size"], train_cache["img_size"])
            n_flat = n.reshape(b * k, 1, train_cache["img_size"], train_cache["img_size"])
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                scores = model.score_pairs(r_rep, n_flat).reshape(b, k)
                q = torch.softmax(-err / err.std(dim=1, keepdim=True).clamp_min(1e-4), dim=1)
                loss = -(q * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    val_ds = CandidateRankDataset(val_cache, mode=mode)
    scores_all = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0):
            r = batch["r"].to(device)
            n = batch["n"].to(device)
            b, k = batch["err"].shape
            r_rep = r[:, None].repeat(1, k, 1, 1, 1).reshape(b * k, 1, val_cache["img_size"], val_cache["img_size"])
            n_flat = n.reshape(b * k, 1, val_cache["img_size"], val_cache["img_size"]).to(device)
            scores_all.append(model.score_pairs(r_rep, n_flat).reshape(b, k).detach().cpu())
    scores = torch.cat(scores_all, 0).numpy()
    method = ("structural_" if structural else "") + (pretrain or "scratch") + "_dual_ranker"
    metrics = evaluate_scores(val_cache, scores, method)
    return metrics, scores, pre_report


def run_selector_stage(
    generator,
    measurement,
    config: dict[str, Any],
    train_split,
    val_split,
    *,
    out: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    k = 16
    train_cache = build_candidate_cache(generator, measurement, config, train_split, out=out, k=k, seed=seed + 20000, device=device)
    val_cache = build_candidate_cache(generator, measurement, config, val_split, out=out, k=k, seed=seed + 30000, device=device)
    baselines = evaluate_baselines(val_cache)
    pair_metrics, pair_scores = train_scalar_selector(train_cache, val_cache, "pair")
    sum_metrics, sum_scores = train_scalar_selector(train_cache, val_cache, "sum")
    ranker_reports = {}
    pre_reports = {}
    directions = []
    for run_seed in [1, 2, 3]:
        for name, pre, structural in [
            ("scratch", None, False),
            ("raw_fcc", "raw", False),
            ("dm_fcc", "dm", False),
            ("structural_dm_fcc", "dm", True),
        ]:
            metrics, _scores, pre_report = train_ranker(train_cache, val_cache, device=device, seed=seed + run_seed * 100 + len(name), pretrain=pre, structural=structural)
            ranker_reports[f"{name}_seed{run_seed}"] = metrics
            pre_reports[f"{name}_seed{run_seed}"] = pre_report
        directions.append(
            ranker_reports[f"dm_fcc_seed{run_seed}"]["selected_p0_rmse_mean"]
            < ranker_reports[f"scratch_seed{run_seed}"]["selected_p0_rmse_mean"]
        )
    dm_vals = [ranker_reports[f"dm_fcc_seed{s}"]["selected_p0_rmse_mean"] for s in [1, 2, 3]]
    scratch_vals = [ranker_reports[f"scratch_seed{s}"]["selected_p0_rmse_mean"] for s in [1, 2, 3]]
    best_dual_key = min(ranker_reports, key=lambda k2: ranker_reports[k2]["selected_p0_rmse_mean"])
    best_dual = ranker_reports[best_dual_key]
    best_natural = min(pair_metrics["selected_p0_rmse_mean"], sum_metrics["selected_p0_rmse_mean"])
    selector_gate = {
        "best_dual_key": best_dual_key,
        "best_dual_beats_random": best_dual["selected_p0_rmse_mean"] < baselines["random_expectation"]["p0_rmse_mean"],
        "best_dual_oracle_gain_fraction_ge_0_2": best_dual["oracle_gain_fraction_mean"] >= 0.2,
        "dm_fcc_beats_scratch_2_of_3": int(sum(directions)) >= 2,
        "dual_beats_best_naturalness": best_dual["selected_p0_rmse_mean"] < best_natural,
        "k": k,
    }
    if not selector_gate["best_dual_beats_random"]:
        classification = "COVERAGE_EXISTS_BUT_NO_SELECTOR_SIGNAL"
    elif not selector_gate["dual_beats_best_naturalness"]:
        classification = "SELECTOR_WORKS_ONLY_VIA_SIMPLE_NATURALNESS"
    elif not selector_gate["dm_fcc_beats_scratch_2_of_3"]:
        classification = "DIRECT_SUPERVISED_RANKING_WORKS_NO_FCC_GAIN"
    else:
        classification = "DM_FCC_ADDS_VALUE"
    raw_dm_report = {
        "raw_fcc_pretraining": {k2: v for k2, v in pre_reports.items() if "raw_fcc" in k2},
        "dm_fcc_pretraining": {k2: v for k2, v in pre_reports.items() if "dm_fcc" in k2},
        "dm_vs_scratch_selected_p0_rmse": {"dm": dm_vals, "scratch": scratch_vals, "dm_better_flags": directions},
        "candidate_error_distribution_train": {
            "mean": float(train_cache["p0_error"].mean().item()),
            "std": float(train_cache["p0_error"].std(unbiased=False).item()),
            "min": float(train_cache["p0_error"].min().item()),
            "max": float(train_cache["p0_error"].max().item()),
        },
    }
    report = {
        "classification": classification,
        "baselines": baselines,
        "scalar_pair_selector": pair_metrics,
        "sum_image_selector": sum_metrics,
        "rankers": ranker_reports,
        "selector_gate": selector_gate,
        "pretraining_reports": pre_reports,
        "raw_dm_report": raw_dm_report,
        "final_locked_test_run": False,
    }
    save_json(out / "reports" / "dm_fcc_negative_report.json", raw_dm_report)
    save_json(out / "reports" / "selector_validation_ablation.json", report)
    save_json(out / "reports" / "gate_report_e2b_64_selector.json", report)
    write_csv(
        out / "reports" / "selector_validation_summary.csv",
        [
            {"method": "scalar_pair", **pair_metrics},
            {"method": "sum_image", **sum_metrics},
            *[{"method": k2, **v} for k2, v in ranker_reports.items()],
        ],
    )
    return report


def write_final_locked_64_manifest(out: Path) -> dict[str, Any]:
    parent = PHASE1_1 / "reports" / "final_locked_test_manifest.json"
    parent_indices = PHASE1_1 / "reports" / "final_locked_test_indices.npy"
    if not parent.exists() or not parent_indices.exists():
        report = {"status": "blocked", "reason": "Phase1.1 final locked manifest not found"}
        save_json(out / "manifests" / "final_locked_test_64_manifest.json", report)
        return report
    parent_manifest = json.loads(parent.read_text(encoding="utf-8"))
    indices = np.load(parent_indices).astype(np.int64)
    ds = p69a.stl10_dataset("test")
    hashes = []
    labels = []
    for idx in indices:
        x, label = ds[int(idx)]
        hashes.append(hashlib.sha256(x.numpy().tobytes()).hexdigest())
        labels.append(int(label))
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    report = {
        "status": "derived_locked_not_evaluated",
        "source_indices_unchanged": True,
        "source_indices_count": int(indices.size),
        "source_indices_sha256": sha256_np(indices),
        "parent_manifest": str(parent),
        "parent_manifest_sha256": sha256_file(parent),
        "parent_indices_sha256": parent_manifest.get("indices_sha256"),
        "resize": "phase69A.stl10_dataset split=test, build_transform(64), grayscale tensor",
        "normalization": "repository STL10 64px transform, no final-test metric evaluation",
        "image_hashes_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
        "label_histogram": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "final_test_evaluated": False,
    }
    save_json(out / "manifests" / "final_locked_test_64_manifest.json", report)
    return report


def alignment_reports(out: Path, checkpoint: Path, measurement, A: torch.Tensor, base_config: dict[str, Any], generator, gen_config: dict[str, Any], ckpt: dict[str, Any], state_key: str, missing: list[str], unexpected: list[str], device: torch.device):
    A_np = A.detach().cpu().numpy().astype(np.float32)
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    op_report = {
        "A_shape": list(A_np.shape),
        "A_sha256": sha256_np(A_np),
        "A_source": str(p73.REGIMES["rad5"]["A"]),
        "A_source_sha256_file": sha256_file(p73.REGIMES["rad5"]["A"]),
        "m": int(measurement.m),
        "n": int(measurement.n),
        "img_size": int(measurement.img_size),
        "sampling_ratio": float(measurement.sampling_ratio),
        "operator_seed": int(getattr(measurement, "seed", -1)),
        "pattern_type": str(measurement.pattern_type),
        "normalization": str(measurement.matrix_normalization),
        "hadamard_include_dc": bool(getattr(measurement, "hadamard_include_dc", False)),
        "noise_std": float(measurement.noise_std),
        "soft_audit_lambda": float(measurement.lambda_dc),
        "exact_projector": projector.info_dict(),
        "shape_gate_pass": list(A_np.shape) == [205, 4096],
    }
    save_json(out / "reports" / "operator_alignment_64.json", op_report)
    x_data = torch.zeros(2, 1, int(measurement.img_size), int(measurement.img_size), device=device)
    noise_shape = list(x_data.shape)
    chk_report = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "phase": ckpt.get("phase") if isinstance(ckpt, dict) else None,
        "experiment": ckpt.get("experiment") if isinstance(ckpt, dict) else None,
        "step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "generator_architecture": gen_config.get("model_type"),
        "loaded_key": state_key,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "noise_input_shape": noise_shape,
        "forward_with_noise_api": "forward_with_noise(generator, measurement, y, noise, config)",
        "checkpoint_img_size": ckpt.get("config", {}).get("img_size") if isinstance(ckpt, dict) else None,
        "checkpoint_sampling_ratio": ckpt.get("config", {}).get("sampling_ratio") if isinstance(ckpt, dict) else None,
        "output_protocol": {
            "raw_generator_stage": "v_pre",
            "native_audited_output": "x_hat_flat from forward_with_noise",
            "contains_internal_soft_audit": "dc_project after residual and optional final dc_project",
        },
    }
    save_json(out / "reports" / "checkpoint_alignment_64.json", chk_report)
    return op_report, chk_report


@torch.no_grad()
def alignment_smoke(out: Path, generator, measurement, config, cache, device: torch.device) -> dict[str, Any]:
    x = cache["x"][:1].to(device)
    y = cache["y"][:1].to(device)
    zero = torch.zeros(1, 1, measurement.img_size, measurement.img_size, device=device)
    a = forward_with_noise(generator, measurement, y, zero, config)["x_hat_flat"]
    b = forward_with_noise(generator, measurement, y, zero, config)["x_hat_flat"]
    gen1 = torch.Generator(device=device).manual_seed(1)
    gen2 = torch.Generator(device=device).manual_seed(2)
    n1 = torch.randn_like(zero, generator=gen1) if False else torch.randn(zero.shape, device=device, generator=gen1)
    n2 = torch.randn(zero.shape, device=device, generator=gen2)
    o1 = forward_with_noise(generator, measurement, y, n1, config)["x_hat_flat"]
    o2 = forward_with_noise(generator, measurement, y, n2, config)["x_hat_flat"]
    p01 = exact_null_project(o1, measurement, dtype=torch.float64, device=device)
    p02 = exact_null_project(o2, measurement, dtype=torch.float64, device=device)
    p0_delta = float(torch.sqrt(torch.mean((p01 - p02) ** 2)).item())
    report = {
        "zero_noise_repeat_max_abs": float((a - b).abs().max().item()),
        "zero_noise_stable": bool((a - b).abs().max().item() < 1e-8),
        "different_noise_p0_rmse": p0_delta,
        "different_noise_changes_p0": bool(p0_delta > 1e-6),
        "raw_output_range_vpre": {
            "min": float(forward_with_noise(generator, measurement, y, n1, config)["v_pre"].min().item()),
            "max": float(forward_with_noise(generator, measurement, y, n1, config)["v_pre"].max().item()),
        },
    }
    save_json(out / "reports" / "alignment_smoke_64.json", report)
    return report


def write_static_reports(out: Path) -> None:
    rationale = """# Phase 1.2 Scientific Rationale

Phase 1.2 switches from synthetic raw donor splices at 96px to the resolution
and operator actually used by the stochastic Phase79 checkpoint: Rad-5/STL10 at
64x64 with m=205.  The first question is no longer whether raw splices are easy
to identify, but whether a frozen stochastic generator supplies a candidate
pool with oracle headroom under exact range-null canonicalization.

The selector stages are allowed only if candidate coverage passes.  This avoids
training a selector when the candidate pool is collapsed or has no useful
best-of-K improvement.
"""
    (out / "reports" / "phase1_2_scientific_rationale.md").write_text(rationale, encoding="utf-8")


def main() -> int:
    args = parse_args()
    start = time.time()
    out = ensure(Path(args.output_dir))
    ensure(out / "reports")
    ensure(out / "manifests")
    (out / "command.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    write_static_reports(out)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = Path(args.checkpoint)
    try:
        measurement, A, base_config = make_phase79_measurement(device)
        generator, gen_config, ckpt, state_key, missing, unexpected = load_phase79_generator(checkpoint, base_config, measurement, device)
        op_report, ckpt_report = alignment_reports(out, checkpoint, measurement, A, base_config, generator, gen_config, ckpt, state_key, missing, unexpected, device)
        if not op_report["shape_gate_pass"]:
            raise RuntimeError("PHASE79_OPERATOR_NOT_REPRODUCIBLE: A shape does not match expected [205,4096].")
        cache = build_dev_cache(measurement, device, args.dev_images)
        smoke = alignment_smoke(out, generator, measurement, gen_config, cache, device)
        if not smoke["zero_noise_stable"]:
            raise RuntimeError("ALIGNMENT_FAILURE: zero-noise forward is not repeatable.")
        if not smoke["different_noise_changes_p0"]:
            coverage = {
                "phase": "E2a_64_candidate_coverage",
                "classification": "FAIL_CANDIDATE_COLLAPSE",
                "reason": "Different noise seeds did not change exact P0 above threshold in alignment smoke.",
            }
            save_json(out / "reports" / "gate_report_e2a_64_coverage.json", coverage)
        else:
            coverage = coverage_eval(generator, measurement, gen_config, cache, out=out, kmax=args.kmax, seed=args.seed, device=device)
        final64 = write_final_locked_64_manifest(out)
        coverage_pass = coverage.get("classification") == "PASS_USEFUL_CANDIDATE_COVERAGE"
        if coverage_pass:
            train_full, val_full, _test_full, _split2 = p73.build_caches("rad5", gen_config, measurement, device)
            train_split = split_subset_from_cache(train_full, 512, "train_64_selector", measurement, device)
            val_split = split_subset_from_cache(val_full, 128, "val_64_selector", measurement, device)
            selector_report = run_selector_stage(
                generator,
                measurement,
                gen_config,
                train_split,
                val_split,
                out=out,
                seed=args.seed,
                device=device,
            )
            conclusion_class = selector_report.get("classification", "INCONCLUSIVE_IMPLEMENTATION_OR_ALIGNMENT_FAILURE")
        else:
            dm_report = {
                "status": "skipped",
                "reason": "Coverage gate did not pass; per protocol selector/cache/DM-FCC stages stop here.",
            }
            selector_report = {
                "status": "skipped",
                "reason": "E2b selector stages require PASS_USEFUL_CANDIDATE_COVERAGE; current classification is " + str(coverage.get("classification")),
            }
            save_json(out / "reports" / "dm_fcc_negative_report.json", dm_report)
            save_json(out / "reports" / "selector_validation_ablation.json", selector_report)
            save_json(out / "reports" / "gate_report_e2b_64_selector.json", selector_report)
            conclusion_class = (
                "NO_USEFUL_CANDIDATE_COVERAGE"
                if coverage.get("classification") in {"FAIL_CANDIDATE_COLLAPSE", "FAIL_NO_ORACLE_HEADROOM"}
                else "INCONCLUSIVE_IMPLEMENTATION_OR_ALIGNMENT_FAILURE"
            )
        conclusion = {
            "classification": conclusion_class,
            "coverage_classification": coverage.get("classification"),
            "selector_training_run": bool(coverage_pass),
            "final_locked_test_run": False,
            "true_bottleneck": "candidate_coverage" if conclusion_class == "NO_USEFUL_CANDIDATE_COVERAGE" else "alignment_or_unimplemented_downstream",
        }
        if coverage_pass:
            if conclusion_class in {"SELECTOR_WORKS_ONLY_VIA_SIMPLE_NATURALNESS", "DIRECT_SUPERVISED_RANKING_WORKS_NO_FCC_GAIN"}:
                conclusion["true_bottleneck"] = "fcc_specific_gain"
            elif conclusion_class == "DM_FCC_ADDS_VALUE":
                conclusion["true_bottleneck"] = "ready_for_locked_test_after_freeze"
        save_json(out / "reports" / "scientific_conclusion_phase1_2.json", conclusion)
        if not coverage_pass:
            (out / "reports" / "BLOCKERS.md").write_text(
                "# BLOCKERS\n\n"
                f"Coverage gate did not pass: `{coverage.get('classification')}`. "
                "Per Phase 1.2 protocol, selector training and final locked test evaluation were not run.\n",
                encoding="utf-8",
            )
        else:
            (out / "reports" / "BLOCKERS.md").write_text("# BLOCKERS\n\nFinal locked test was not run; selector configuration has not been frozen for final evaluation in this run.\n", encoding="utf-8")
        status = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(out),
            "coverage_classification": coverage.get("classification"),
            "selector_run": bool(coverage_pass),
            "final_test_run": False,
            "runtime_seconds": time.time() - start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            **git_info(),
        }
        save_json(out / "implementation_status_phase1_2.json", status)
        (out / "decision_log_phase1_2.md").write_text(
            f"# Phase 1.2 Decision Log\n\nCoverage classification: `{coverage.get('classification')}`.\n\n"
            "Selector stages and final locked test were skipped unless coverage passed.\n",
            encoding="utf-8",
        )
        (out / "RUNBOOK_PHASE1_2.md").write_text(
            "# RUNBOOK Phase 1.2\n\n"
            "```powershell\nD:\\Anacondar\\anaconda3\\python.exe phase1_2_rad5_64_pipeline.py\n```\n",
            encoding="utf-8",
        )
        print(json.dumps({"coverage": coverage.get("classification"), "output_dir": str(out)}, indent=2))
        return 0
    except Exception as exc:
        report = {
            "phase": "Phase1.2",
            "status": "blocked",
            "STOP_REASON": "PHASE79_OPERATOR_NOT_REPRODUCIBLE" if "PHASE79_OPERATOR_NOT_REPRODUCIBLE" in str(exc) else "INCONCLUSIVE_ALIGNMENT_FAILURE",
            "error": repr(exc),
        }
        save_json(out / "reports" / "gate_report_e2a_64_coverage.json", report)
        save_json(out / "implementation_status_phase1_2.json", {**report, "runtime_seconds": time.time() - start})
        (out / "reports" / "BLOCKERS.md").write_text(f"# BLOCKERS\n\n{repr(exc)}\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
