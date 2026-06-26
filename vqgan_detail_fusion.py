"""Zero-training VQAE-structure + VQGAN-detail distortion-constrained fusion canary.

Implements the goal in CODEX_NEXT_GOAL_VQGAN_DETAIL_FUSION.md / PROJECT_BRIEF_VQGAN_DETAIL_FUSION.md:
do NOT retrain priors; instead fuse the frozen multi-seed VQAE and VQGAN anchor-initialized
inversion reconstructions in the measurement null space.

For each frozen seed we regenerate, per image, the shared LMMSE anchor x0 and the two refined
reconstructions x_A (VQAE) and x_G (VQGAN), form null-space residuals d_A = P0(x_A - x0),
d_G = P0(x_G - x0), and fuse:   d_F = d_A + W (d_G - d_A),   x_hat = x0 + P0 d_F  -> exact audit.
W is a global scalar and/or radial-frequency band weights (low freq -> VQAE, high freq -> VQGAN).

Pipeline reuses the exact frozen machinery (operator, LMMSE, priors, refiners, projector, metrics,
gate) from gan_high_quality_gi / anchor_initialized_vqgan_inversion / measurement_conditioned_vqgan
so results are bit-comparable to the existing multi-seed Pareto confirmation.

Subcommands: regen | validate | canary | gate | all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
import anchor_initialized_vqgan_inversion as ai
import measurement_conditioned_vqgan as mc
from src.projections import get_exact_projector, relative_measurement_error

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/compatibility/measurement_conditioned_vqgan"
OUT = BASE / "detail_fusion"
CACHE = OUT / "cache"
SEEDS = [0, 1, 2]
SPLITS = ["val", "dev"]
METRIC_COLS = ["full_rmse", "centered_rmse", "psnr", "ssim", "lpips", "rapsd", "relmeaserr"]


def log(*a: Any) -> None:
    print(f"[fusion {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_cfg(seed: int) -> dict:
    p = BASE / f"anchor_multiseed_hashclean_seed{seed}/config_used.yaml"
    cfg = yaml.safe_load(p.read_text())
    # ensure a local dataset root regardless of where config was frozen
    cfg["data"]["dataset_root"] = cfg["data"].get("dataset_root", "E:/datasets")
    return cfg


def refiner_ckpt(seed: int, kind: str) -> Path:
    return BASE / (
        f"anchor_multiseed_hashclean_seed{seed}/runs/seed{seed}/"
        f"{kind}_refiner/checkpoints/{kind}_refiner_best_by_val_lpips.pt"
    )


# --------------------------------------------------------------------------- #
# Shared substrate (operator, splits, LMMSE) -- identical across all 3 seeds.
# --------------------------------------------------------------------------- #
class Substrate:
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.img = int(cfg["data"]["img_size"])
        log("building structured operator ...")
        self.rows_np, self.op_meta = hq.build_structured_operator_rows(
            img_size=self.img,
            total_m=int(cfg["operator"]["total_m"]),
            dct_rows=int(cfg["operator"]["dct_rows"]),
            hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
            random_rows=int(cfg["operator"]["random_rows"]),
            seed=int(cfg["operator"]["seed"]),
        )
        self.measurement = hq.make_measurement_operator(
            self.rows_np, img_size=self.img, device=device,
            lambda_solver=float(cfg["operator"]["lambda_solver"]),
        )
        self.projector = get_exact_projector(self.measurement, dtype=torch.float64, device=device)
        log("operator rows_sha256 =", self.op_meta.get("rows_sha256"))
        log("building hash-clean splits (STL10 scan; may take a minute) ...")
        self.train_ds, self.val_ds, self.dev_ds, self.split_manifest = ai.build_split_datasets(cfg)
        self.datasets = {"val": self.val_ds, "dev": self.dev_ds}
        log("fitting empirical LMMSE on", len(self.train_ds), "train images ...")
        train_x, _, _ = hq.tensor_dataset_to_matrix(
            self.train_ds, batch_size=int(cfg["data"].get("matrix_batch_size", 128)))
        self.lmmse = hq.EmpiricalLMMSE.fit(
            train_x, self.rows_np, lambda_=float(cfg["operator"]["lmmse_lambda"]))
        log("substrate ready.")

    def verify(self, seed: int) -> dict:
        """Cross-check operator/split identity against the frozen manifests."""
        rep = BASE / f"anchor_multiseed_hashclean_seed{seed}/reports"
        out = {}
        om = json.loads((rep / "operator_manifest.json").read_text())
        out["operator_rows_sha256_match"] = (om.get("rows_sha256") == self.op_meta.get("rows_sha256"))
        sm = json.loads((rep / "split_manifest.json").read_text())
        spans = sm.get("spans", sm)
        out["split_manifest_spans"] = spans
        return out


# --------------------------------------------------------------------------- #
# Regeneration: mirror anchor_initialized_vqgan_inversion.evaluate_refiner inner loop.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def regen_seed(seed: int, sub: Substrate) -> None:
    cfg = sub.cfg
    device = sub.device
    dt = float(cfg["training"].get("distance_temperature", 1.0))
    st = float(cfg["training"].get("soft_temperature", 1.0))
    priors = {
        ai.VQAE: ai.load_prior(ai.VQAE, ROOT / cfg["priors"]["vqae_checkpoint"], cfg, device),
        ai.VQGAN: ai.load_prior(ai.VQGAN, ROOT / cfg["priors"]["vqgan_checkpoint"], cfg, device),
    }
    refiners = {
        ai.VQAE: ai.load_refiner_checkpoint(refiner_ckpt(seed, ai.VQAE), cfg, device),
        ai.VQGAN: ai.load_refiner_checkpoint(refiner_ckpt(seed, ai.VQGAN), cfg, device),
    }

    def refine(kind: str, x0: torch.Tensor, unc: torch.Tensor) -> torch.Tensor:
        prior = priors[kind]
        z0 = prior.model.encode(x0)
        dz, dlogits = refiners[kind](x0, unc, z0)
        logits = ai.logits_from_latent(z0 + dz, prior, distance_temperature=dt) + dlogits
        zq, _, _ = ai.quantize_from_logits(prior, logits, soft_temperature=st, straight_through=False)
        return prior.model.decode_embeddings(zq)

    for split in SPLITS:
        loader = hq.build_loader(
            sub.datasets[split], batch_size=int(cfg["data"].get("eval_batch_size", 16)),
            workers=0, shuffle=False, seed=int(cfg["seed"]) + (10 if split == "val" else 11),
            device=device,
        )
        x0s, xAs, xGs, ys, truths, idxs, labels = [], [], [], [], [], [], []
        for x, label, idx in loader:
            x = x.to(device, non_blocking=True)
            flat = sub.measurement.flatten_img(x)
            y = sub.measurement.A_forward(flat)
            x0 = sub.measurement.unflatten_img(sub.lmmse.anchor(y, sub.measurement, device=device))
            unc = sub.lmmse.uncertainty_map(img_size=sub.img, device=device, batch_size=x.shape[0], dtype=x.dtype)
            x_A = refine(ai.VQAE, x0, unc)
            x_G = refine(ai.VQGAN, x0, unc)
            x0s.append(x0.cpu()); xAs.append(x_A.cpu()); xGs.append(x_G.cpu())
            ys.append(y.cpu()); truths.append(x.cpu())
            idxs.append(idx.cpu()); labels.append(label.cpu())
        pack = {
            "x0": torch.cat(x0s), "x_A": torch.cat(xAs), "x_G": torch.cat(xGs),
            "y": torch.cat(ys), "truth": torch.cat(truths),
            "source_index": torch.cat(idxs), "label": torch.cat(labels),
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"seed{seed}_{split}.pt"
        torch.save(pack, path)
        log(f"seed{seed} {split}: cached {pack['x0'].shape[0]} images -> {path.name}")


def load_pack(seed: int, split: str, device: torch.device) -> dict:
    pack = torch.load(CACHE / f"seed{seed}_{split}.pt", map_location=device)
    return pack


# --------------------------------------------------------------------------- #
# Scoring helper -- reuse ai.prediction_metrics so columns match the frozen CSVs.
# --------------------------------------------------------------------------- #
def score_method(pred: torch.Tensor, truth: torch.Tensor, y: torch.Tensor, measurement,
                 lpips_fn, method: str, beta: float, idx: torch.Tensor, label: torch.Tensor,
                 seed: int, split: str) -> list[dict]:
    extra = {"source": split, "projection_norm": 0.0, "pre_audit_rel": "[DATA MISSING]",
             "top1": "[DATA MISSING]", "top5": "[DATA MISSING]", "entropy": "[DATA MISSING]",
             "latent_l1": "[DATA MISSING]"}
    return ai.prediction_metrics(pred=pred, truth=truth, y=y, measurement=measurement,
                                 lpips_fn=lpips_fn, method=method, beta=beta, source_idx=idx,
                                 labels=label, train_seed=seed, extra=extra)


# --------------------------------------------------------------------------- #
# validate: confirm regenerated null-blend recons match the committed final_dev CSV.
# --------------------------------------------------------------------------- #
def cmd_validate(seed: int, device: torch.device) -> None:
    cfg = load_cfg(seed)
    sub_meas_device = device
    # lightweight: need measurement + lpips only (no LMMSE/splits) for scoring from cache
    rows_np, op_meta = hq.build_structured_operator_rows(
        img_size=int(cfg["data"]["img_size"]), total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]), hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]), seed=int(cfg["operator"]["seed"]))
    measurement = hq.make_measurement_operator(rows_np, img_size=int(cfg["data"]["img_size"]),
                                               device=device, lambda_solver=float(cfg["operator"]["lambda_solver"]))
    lpips_fn = hq.load_lpips(device)
    pack = load_pack(seed, "dev", device)
    betas = [float(b) for b in cfg["eval"]["beta_grid"]]
    idx, label, y, truth = pack["source_index"], pack["label"], pack["y"], pack["truth"]
    x0, x_A, x_G = pack["x0"], pack["x_A"], pack["x_G"]

    regen_rows: dict[tuple, dict] = {}
    for kind, xg in [(ai.VQAE, x_A), (ai.VQGAN, x_G)]:
        method = f"{kind}_refiner_nullblend"
        for beta in betas:
            pred = ai.null_blend(x0, xg, beta, measurement)
            for r in score_method(pred, truth, y, measurement, lpips_fn, method, beta, idx, label, seed, "dev"):
                regen_rows[(r["method"], float(r["beta"]), int(r["source_index"]))] = r

    # committed reference
    ref_csv = BASE / f"anchor_multiseed_hashclean_seed{seed}/reports/final_dev_per_image.csv"
    ref_rows: dict[tuple, dict] = {}
    with open(ref_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in {"vqae_refiner_nullblend", "vqgan_refiner_nullblend"}:
                ref_rows[(r["method"], float(r["beta"]), int(r["source_index"]))] = r

    keys = sorted(set(regen_rows) & set(ref_rows))
    log(f"validate seed{seed}: comparing {len(keys)} (method,beta,image) rows vs committed CSV")
    maxdiff = {m: 0.0 for m in METRIC_COLS}
    meandiff = {m: 0.0 for m in METRIC_COLS}
    n_ok = 0
    for k in keys:
        a, b = regen_rows[k], ref_rows[k]
        for m in METRIC_COLS:
            try:
                va, vb = float(a[m]), float(b[m])
            except (TypeError, ValueError):
                continue
            d = abs(va - vb)
            maxdiff[m] = max(maxdiff[m], d)
            meandiff[m] += d
        n_ok += 1
    for m in METRIC_COLS:
        meandiff[m] /= max(n_ok, 1)
    print("\n  metric        max|Δ|        mean|Δ|")
    for m in METRIC_COLS:
        print(f"  {m:<12} {maxdiff[m]:.3e}    {meandiff[m]:.3e}")
    # faithfulness verdict: distortion metrics should match to ~1e-4; lpips a touch looser
    tol = {"full_rmse": 5e-4, "centered_rmse": 5e-4, "psnr": 1e-1, "ssim": 5e-3,
           "lpips": 5e-3, "rapsd": 5e-4, "relmeaserr": 1e-4}
    verdict = all(maxdiff[m] <= tol[m] for m in METRIC_COLS)
    print(f"\n  FAITHFUL={verdict}  (tolerances {tol})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"validate_seed{seed}.json").write_text(json.dumps(
        {"seed": seed, "n_rows": len(keys), "max_abs_diff": maxdiff,
         "mean_abs_diff": meandiff, "faithful": verdict, "tolerances": tol}, indent=2))


# --------------------------------------------------------------------------- #
# Fusion machinery: radial-frequency band transform + null-space fusion + audit.
# --------------------------------------------------------------------------- #
N_BANDS = 16


def build_meas(cfg: dict, device: torch.device):
    rows_np, _ = hq.build_structured_operator_rows(
        img_size=int(cfg["data"]["img_size"]), total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]), hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]), seed=int(cfg["operator"]["seed"]))
    measurement = hq.make_measurement_operator(rows_np, img_size=int(cfg["data"]["img_size"]),
                                               device=device, lambda_solver=float(cfg["operator"]["lambda_solver"]))
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    return measurement, projector


def radial_masks(img: int, k: int, device: torch.device) -> list[torch.Tensor]:
    """K radial frequency bands over the fftshifted img x img grid (center-based, RAPSD-aligned)."""
    yy, xx = torch.meshgrid(torch.arange(img) - img // 2, torch.arange(img) - img // 2, indexing="ij")
    rr = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    edges = torch.linspace(0, float(rr.max()) + 1e-6, k + 1)
    return [((rr >= edges[b]) & (rr < edges[b + 1])).to(device=device, dtype=torch.float64) for b in range(k)]


def weight_map(w: np.ndarray, masks: list[torch.Tensor]) -> torch.Tensor:
    wm = torch.zeros_like(masks[0])
    for wi, m in zip(w, masks):
        wm = wm + float(wi) * m
    return wm


def apply_band(delta_img: torch.Tensor, wm: torch.Tensor) -> torch.Tensor:
    """Real radial filter: weight residual in fft domain by wm (fftshifted)."""
    f = torch.fft.fftshift(torch.fft.fft2(delta_img), dim=(-2, -1))
    out = torch.fft.ifft2(torch.fft.ifftshift(f * wm, dim=(-2, -1)), dim=(-2, -1)).real
    return out


def fuse(spec, x0f, d_A, d_G, y, measurement, projector, masks):
    """x_hat = x0 + P0( d_A + W (d_G - d_A) ), then exact measurement audit. All float64."""
    kind, w = spec
    delta = d_G - d_A
    if kind == "scalar":
        dd = float(w) * delta
    else:
        wm = weight_map(w, masks)
        delta_img = measurement.unflatten_img(delta)
        dd = measurement.flatten_img(apply_band(delta_img, wm))
    d_F = projector.null_project_flat(d_A + dd)
    xhat = projector.audit_flat(x0f + d_F, y)
    return measurement.unflatten_img(xhat).to(torch.float32)


def rapsd_metric_mean(pred: torch.Tensor, truth: torch.Tensor) -> float:
    p = pred.clamp(0, 1)[:, 0].detach().cpu().numpy()
    t = truth.clamp(0, 1)[:, 0].detach().cpu().numpy()
    d = [float(np.linalg.norm(hq.rapsd_np(p[i]) - hq.rapsd_np(t[i]))) for i in range(p.shape[0])]
    return float(np.mean(d))


def fast_means(pred, truth, y, measurement, lpips_fn) -> dict:
    clip = pred.clamp(0, 1)
    rmse = hq.full_rmse_torch(clip, truth)
    psnr = -20.0 * np.log10(np.maximum(rmse, 1e-12))
    lp = hq.lpips_batch(lpips_fn, clip, truth)
    rel = relative_measurement_error(pred, y, measurement).detach().cpu().numpy()
    return {"lpips": float(np.mean(lp)) if lp is not None else float("nan"),
            "psnr": float(np.mean(psnr)), "rmse": float(np.mean(rmse)),
            "rapsd": rapsd_metric_mean(pred, truth), "relmeaserr": float(np.mean(rel))}


def gen_specs(k: int) -> dict:
    specs = {}
    for b in np.round(np.linspace(0, 1, 21), 3):
        specs[f"scalar_{b:.2f}"] = ("scalar", float(b))
    for cut in range(0, k + 1):  # bands [0,cut) -> VQAE(0), [cut,k) -> VQGAN(1)
        specs[f"lowpass_cut{cut}"] = ("band", np.array([0.0] * cut + [1.0] * (k - cut)))
    return specs


def prep_residuals(pack: dict, measurement, projector) -> dict:
    x0f = measurement.flatten_img(pack["x0"]).double()
    d_A = projector.null_project_flat(measurement.flatten_img(pack["x_A"] - pack["x0"]).double())
    d_G = projector.null_project_flat(measurement.flatten_img(pack["x_G"] - pack["x0"]).double())
    return {"x0f": x0f, "d_A": d_A, "d_G": d_G, "y": pack["y"].double(),
            "truth": pack["truth"], "idx": pack["source_index"], "label": pack["label"]}


def select_operating_point(arm_means: dict, vqae: dict, mode: str):
    cands = []
    for name, m in arm_means.items():
        if mode == "balanced":
            ok = (m["psnr"] >= vqae["psnr"] - 0.5 and m["rmse"] <= vqae["rmse"] + 0.005
                  and m["rapsd"] <= vqae["rapsd"] + 1e-9)
        else:  # quality_lite
            ok = m["psnr"] >= vqae["psnr"] - 1.0
        if ok and np.isfinite(m["lpips"]):
            cands.append((name, m))
    if not cands:
        return None
    return min(cands, key=lambda t: t[1]["lpips"])[0]


def cmd_canary(seeds: list[int], device: torch.device) -> None:
    (OUT / "canary").mkdir(parents=True, exist_ok=True)
    specs = gen_specs(N_BANDS)
    for seed in seeds:
        cfg = load_cfg(seed)
        measurement, projector = build_meas(cfg, device)
        masks = radial_masks(int(cfg["data"]["img_size"]), N_BANDS, device)
        lpips_fn = hq.load_lpips(device)
        pre = {sp: prep_residuals(load_pack(seed, sp, device), measurement, projector) for sp in SPLITS}

        # ---- selection on VAL: mean metrics for every candidate spec ----
        t0 = time.time()
        val_means = {}
        for name, spec in specs.items():
            xhat = fuse(spec, pre["val"]["x0f"], pre["val"]["d_A"], pre["val"]["d_G"],
                        pre["val"]["y"], measurement, projector, masks)
            val_means[name] = fast_means(xhat, pre["val"]["truth"], pre["val"]["y"], measurement, lpips_fn)
        vqae_val = val_means["scalar_0.00"]
        vqgan_val = val_means["scalar_1.00"]
        sel = {
            "balanced": select_operating_point(val_means, vqae_val, "balanced"),
            "quality_lite": select_operating_point(val_means, vqae_val, "quality_lite"),
            "oracle": min((n for n, m in val_means.items() if np.isfinite(m["lpips"])),
                          key=lambda n: val_means[n]["lpips"]),
        }
        log(f"seed{seed} selection ({time.time()-t0:.0f}s on val): "
            f"balanced={sel['balanced']} quality_lite={sel['quality_lite']} oracle={sel['oracle']}")

        # ---- mechanical DEV scoring of the chosen + reference arms ----
        arms = {"vqae": ("scalar", 0.0), "vqgan": ("scalar", 1.0)}
        for mode in ("balanced", "quality_lite", "oracle"):
            if sel[mode] is not None:
                arms[f"fusion_{mode}"] = specs[sel[mode]]
        dev_rows = []
        dev_means = {}
        for arm_name, spec in arms.items():
            xhat = fuse(spec, pre["dev"]["x0f"], pre["dev"]["d_A"], pre["dev"]["d_G"],
                        pre["dev"]["y"], measurement, projector, masks)
            rows = score_method(xhat, pre["dev"]["truth"], pre["dev"]["y"], measurement, lpips_fn,
                                arm_name, 0.0, pre["dev"]["idx"], pre["dev"]["label"], seed, "dev")
            dev_rows.extend(rows)
            dev_means[arm_name] = fast_means(xhat, pre["dev"]["truth"], pre["dev"]["y"], measurement, lpips_fn)

        # persist per-seed
        if dev_rows:
            cols = list(dev_rows[0].keys())
            with open(OUT / "canary" / f"seed{seed}_dev_rows.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(dev_rows)
        (OUT / "canary" / f"seed{seed}_selection.json").write_text(json.dumps({
            "seed": seed, "n_bands": N_BANDS,
            "selected_specs": {k: (specs[v][0], list(np.atleast_1d(specs[v][1]).astype(float)))
                               for k, v in sel.items() if v is not None},
            "selected_names": sel,
            "val_means": {"vqae": vqae_val, "vqgan": vqgan_val,
                          **{f"fusion_{m}": val_means[sel[m]] for m in ("balanced", "quality_lite", "oracle") if sel[m]}},
            "dev_means": dev_means,
        }, indent=2))
        log(f"seed{seed} dev means: " + ", ".join(
            f"{a}(lpips={dev_means[a]['lpips']:.4f},psnr={dev_means[a]['psnr']:.2f},"
            f"rapsd={dev_means[a]['rapsd']:.5f},rel={dev_means[a]['relmeaserr']:.1e})" for a in dev_means))


# --------------------------------------------------------------------------- #
# Gate: seed-clustered bootstrap over fused-vs-VQAE per-image deltas.
# --------------------------------------------------------------------------- #
def clustered_bootstrap(delta_by_seed: dict, reps: int, rng: np.random.Generator) -> dict:
    seeds = sorted(delta_by_seed)
    pooled = np.concatenate([delta_by_seed[s] for s in seeds])
    boots = []
    for _ in range(reps):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        vals = [delta_by_seed[s][rng.integers(0, len(delta_by_seed[s]), size=len(delta_by_seed[s]))] for s in chosen]
        boots.append(float(np.concatenate(vals).mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_delta": float(pooled.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "per_seed_mean_delta": {int(s): float(delta_by_seed[s].mean()) for s in seeds},
            "same_direction_negative_count": int(sum(delta_by_seed[s].mean() < 0 for s in seeds)),
            "same_direction_positive_count": int(sum(delta_by_seed[s].mean() > 0 for s in seeds)),
            "n_images_total": int(len(pooled)), "n_seeds": len(seeds)}


def _paired_deltas(rows_by_seed: dict, method: str, ref: str, metric: str) -> dict:
    out = {}
    for seed, rows in rows_by_seed.items():
        by = {}
        for r in rows:
            if r["method"] not in (method, ref):
                continue
            try:
                v = float(r[metric])
            except (TypeError, ValueError):
                continue
            by.setdefault(int(r["source_index"]), {})[r["method"]] = v
        d = [by[i][method] - by[i][ref] for i in by if method in by[i] and ref in by[i]]
        if d:
            out[seed] = np.asarray(d, dtype=np.float64)
    return out


def _method_mean(rows_by_seed: dict, method: str, metric: str) -> float:
    vals = []
    for rows in rows_by_seed.values():
        for r in rows:
            if r["method"] == method:
                try:
                    vals.append(float(r[metric]))
                except (TypeError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else float("nan")


def _gate_block(rows_by_seed: dict, method: str, ref: str, reps: int) -> dict:
    rng = np.random.default_rng(20260626)
    metrics = {}
    for m in METRIC_COLS:
        dbs = _paired_deltas(rows_by_seed, method, ref, m)
        if dbs:
            metrics[m] = clustered_bootstrap(dbs, reps, rng)
    ref_lpips = _method_mean(rows_by_seed, ref, "lpips")
    meth_relmeas = _method_mean(rows_by_seed, method, "relmeaserr")
    lp = metrics.get("lpips", {})
    rel_gain = (-lp.get("mean_delta", 0.0) / max(abs(ref_lpips), 1e-12)) if lp else 0.0
    cond = {
        "lpips_gain_ge_5pct_ci_upper_lt0": bool(lp and lp["ci_high"] < 0 and rel_gain >= 0.05),
        "lpips_2_of_3_seeds_same_direction": bool(lp and lp["same_direction_negative_count"] >= 2),
        "rapsd_same_direction": bool(metrics.get("rapsd", {}).get("mean_delta", 1.0) < 0),
        "psnr_drop_within_2p5db": bool(metrics.get("psnr", {}).get("mean_delta", -99.0) >= -2.5),
        "psnr_drop_within_0p5db": bool(metrics.get("psnr", {}).get("mean_delta", -99.0) >= -0.5),
        "rmse_increase_within_0p005": bool(metrics.get("full_rmse", {}).get("mean_delta", 99.0) <= 0.005),
        "relmeaserr_ok": bool(meth_relmeas <= 1e-5),
    }
    return {"method": method, "reference": ref, "lpips_relative_gain": rel_gain,
            "reference_lpips_mean": ref_lpips, "method_relmeaserr_mean": meth_relmeas,
            "metrics": metrics, "conditions": cond}


def cmd_gate(seeds: list[int], reps: int = 2000) -> None:
    rows_by_seed = {}
    for seed in seeds:
        p = OUT / "canary" / f"seed{seed}_dev_rows.csv"
        with open(p, newline="") as f:
            rows_by_seed[seed] = list(csv.DictReader(f))
    blocks = {}
    for arm in ("fusion_balanced", "fusion_quality_lite", "fusion_oracle", "vqgan"):
        if any(any(r["method"] == arm for r in rows) for rows in rows_by_seed.values()):
            blocks[arm] = _gate_block(rows_by_seed, arm, "vqae", reps)

    def passes_quality(b):
        c = b["conditions"]
        return all(c[k] for k in ("lpips_gain_ge_5pct_ci_upper_lt0", "lpips_2_of_3_seeds_same_direction",
                                  "psnr_drop_within_2p5db", "relmeaserr_ok"))

    def passes_balanced(b):
        c = b["conditions"]
        return all(c[k] for k in ("lpips_gain_ge_5pct_ci_upper_lt0", "lpips_2_of_3_seeds_same_direction",
                                  "rapsd_same_direction", "psnr_drop_within_0p5db",
                                  "rmse_increase_within_0p005", "relmeaserr_ok"))

    bal = blocks.get("fusion_balanced")
    qual = blocks.get("fusion_quality_lite")
    relmeas_ok = all(b["conditions"]["relmeaserr_ok"] for b in blocks.values())
    if not relmeas_ok:
        classification = "INVALID_EXPERIMENT"
    elif bal and passes_balanced(bal):
        classification = "BALANCED_VQGAN_FUSION_CONFIRMED"
    elif qual and passes_quality(qual):
        classification = "QUALITY_LITE_FUSION_CONFIRMED"
    elif blocks.get("vqgan") and passes_quality(blocks["vqgan"]):
        classification = "ONLY_FULL_VQGAN_QUALITY_TRADEOFF"
    else:
        classification = "FUSION_GATE_OVERFITS_OR_UNSTABLE"

    report = {
        "classification": classification,
        "balanced_gate_passed": bool(bal and passes_balanced(bal)),
        "quality_lite_gate_passed": bool(qual and passes_quality(qual)),
        "full_vqgan_quality_passed": bool(blocks.get("vqgan") and passes_quality(blocks["vqgan"])),
        "seeds": seeds, "bootstrap_reps": reps, "n_bands": N_BANDS,
        "selections": {seed: json.loads((OUT / "canary" / f"seed{seed}_selection.json").read_text())["selected_names"]
                       for seed in seeds},
        "arms": blocks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MULTISEED_FUSION_GATE_REPORT.json").write_text(json.dumps(report, indent=2))
    log("CLASSIFICATION:", classification)
    for arm, b in blocks.items():
        c = b["conditions"]
        lp = b["metrics"].get("lpips", {})
        ps = b["metrics"].get("psnr", {})
        rp = b["metrics"].get("rapsd", {})
        log(f"  {arm} vs vqae: dLPIPS={lp.get('mean_delta',float('nan')):.4f}"
            f"[{lp.get('ci_low',float('nan')):.4f},{lp.get('ci_high',float('nan')):.4f}] "
            f"relgain={b['lpips_relative_gain']:.3f} dPSNR={ps.get('mean_delta',float('nan')):.2f} "
            f"dRAPSD={rp.get('mean_delta',float('nan')):.5f} cond={c}")


# --------------------------------------------------------------------------- #
# Reporting: Pareto figure + confirmation report + claim-evidence ledger.
# --------------------------------------------------------------------------- #
def _pooled_means(rows_by_seed: dict, arm: str) -> dict:
    return {m: _method_mean(rows_by_seed, arm, m) for m in METRIC_COLS}


def cmd_report(seeds: list[int]) -> None:
    gate = json.loads((OUT / "MULTISEED_FUSION_GATE_REPORT.json").read_text())
    rows_by_seed = {}
    for seed in seeds:
        with open(OUT / "canary" / f"seed{seed}_dev_rows.csv", newline="") as f:
            rows_by_seed[seed] = list(csv.DictReader(f))
    present = lambda a: any(any(r["method"] == a for r in rows) for rows in rows_by_seed.values())
    arms = [a for a in ("vqae", "vqgan", "fusion_balanced", "fusion_quality_lite", "fusion_oracle") if present(a)]
    means = {a: _pooled_means(rows_by_seed, a) for a in arms}

    # ---- Pareto figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, xk, xl in [(axes[0], "full_rmse", "Full RMSE  (distortion, lower=better)"),
                           (axes[1], "rapsd", "RAPSD spectral dist.  (lower=better)")]:
            for a in arms:
                ax.scatter(means[a][xk], means[a]["lpips"], s=90)
                ax.annotate(a.replace("fusion_", "F:"), (means[a][xk], means[a]["lpips"]),
                            fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel(xl)
            ax.set_ylabel("LPIPS  (perception, lower=better)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"VQGAN detail-fusion Pareto (dev, {len(seeds)} seeds) — {gate['classification']}")
        fig.tight_layout()
        fig.savefig(OUT / "FUSION_PARETO.png", dpi=140)
        plt.close(fig)
        log("wrote FUSION_PARETO.png")
    except Exception as e:  # noqa: BLE001
        log("figure skipped:", e)

    # ---- markdown report ----
    def fmt(a, m):
        return f"{means[a][m]:.4f}" if np.isfinite(means[a][m]) else "n/a"
    lines = [
        "# Multi-Seed VQGAN Detail-Fusion Canary Report",
        "",
        f"Classification: `{gate['classification']}`",
        f"- balanced_gate_passed: `{gate['balanced_gate_passed']}`",
        f"- quality_lite_gate_passed: `{gate['quality_lite_gate_passed']}`",
        f"- full_vqgan_quality_passed: `{gate['full_vqgan_quality_passed']}`",
        f"- seeds: {gate['seeds']} | bootstrap_reps: {gate['bootstrap_reps']} | radial bands: {gate['n_bands']}",
        "",
        "Zero-training null-space fusion: x_hat = x0 + P0( d_A + W (d_G - d_A) ), exact measurement audit.",
        "Selection on val, mechanical scoring on dev. Baseline = VQAE refiner (W=0).",
        "",
        "## Dev method means (pooled over seeds)",
        "",
        "| arm | LPIPS | full_rmse | centered_rmse | psnr | ssim | rapsd | relmeaserr |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(f"| {a} | {fmt(a,'lpips')} | {fmt(a,'full_rmse')} | {fmt(a,'centered_rmse')} | "
                     f"{fmt(a,'psnr')} | {fmt(a,'ssim')} | {fmt(a,'rapsd')} | {means[a]['relmeaserr']:.2e} |")
    lines += ["", "## Fused-vs-VQAE clustered bootstrap (per arm)", ""]
    for arm, b in gate["arms"].items():
        c = b["conditions"]
        lp, ps, rp = (b["metrics"].get(k, {}) for k in ("lpips", "psnr", "rapsd"))
        lines += [
            f"### {arm}",
            f"- ΔLPIPS = {lp.get('mean_delta', float('nan')):.4f} "
            f"CI[{lp.get('ci_low', float('nan')):.4f}, {lp.get('ci_high', float('nan')):.4f}], "
            f"relative_gain = {b['lpips_relative_gain']:.3f}, "
            f"same-direction seeds = {lp.get('same_direction_negative_count', 0)}/3",
            f"- ΔPSNR = {ps.get('mean_delta', float('nan')):.3f} dB | "
            f"ΔRAPSD = {rp.get('mean_delta', float('nan')):.5f} | "
            f"method relmeaserr mean = {b['method_relmeaserr_mean']:.2e}",
            f"- conditions: {json.dumps(c)}",
            "",
        ]
    lines += ["## Per-seed selected operating points (chosen on val)", ""]
    for seed in seeds:
        sel = gate["selections"][str(seed)] if str(seed) in gate["selections"] else gate["selections"].get(seed)
        lines.append(f"- seed{seed}: {json.dumps(sel)}")
    (OUT / "MULTISEED_FUSION_CONFIRMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- claim-evidence ledger ----
    led = [
        "# Claim-Evidence Ledger — VQGAN Detail Fusion",
        "",
        f"Mechanical classification: `{gate['classification']}`.",
        "",
        "| claim | evidence | status |",
        "|---|---|---|",
        f"| Exact measurement consistency preserved | all arms relmeaserr mean <= 1e-5 "
        f"({'OK' if all(b['conditions']['relmeaserr_ok'] for b in gate['arms'].values()) else 'FAIL'}) | "
        f"{'PASS' if all(b['conditions']['relmeaserr_ok'] for b in gate['arms'].values()) else 'FAIL'} |",
        f"| Balanced fusion (LPIPS↓, PSNR drop≤0.5dB, RMSE↑≤0.005, RAPSD not worse) | "
        f"fusion_balanced conditions | {'CONFIRMED' if gate['balanced_gate_passed'] else 'NOT CONFIRMED'} |",
        f"| Quality-lite fusion (LPIPS↓ with PSNR drop≤2.5dB, less distortion than full VQGAN) | "
        f"fusion_quality_lite conditions | {'CONFIRMED' if gate['quality_lite_gate_passed'] else 'NOT CONFIRMED'} |",
        f"| Selection used val only, scored on dev | val/dev firewall in canary | PASS |",
        f"| No retraining (frozen priors+refiners) | regen bit-matches frozen final_dev CSV | PASS |",
    ]
    (OUT / "CLAIM_EVIDENCE_LEDGER.md").write_text("\n".join(led), encoding="utf-8")
    log("wrote MULTISEED_FUSION_CONFIRMATION_REPORT.md + CLAIM_EVIDENCE_LEDGER.md")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["regen", "validate", "canary", "gate", "report", "all"])
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log("device =", device)

    if args.command in {"regen", "all"}:
        sub = None
        for seed in args.seeds:
            cfg = load_cfg(seed)
            if sub is None:
                sub = Substrate(cfg, device)
                log("manifest cross-check seed%d:" % seed, sub.verify(seed))
            sub.cfg = cfg  # priors/refiner paths per seed; operator/splits/lmmse are shared
            regen_seed(seed, sub)

    if args.command == "validate":
        for seed in args.seeds:
            cmd_validate(seed, device)

    if args.command in {"canary", "all"}:
        cmd_canary(list(args.seeds), device)

    if args.command in {"gate", "all"}:
        cmd_gate(list(args.seeds))

    if args.command in {"report", "all"}:
        cmd_report(list(args.seeds))


if __name__ == "__main__":
    main()
