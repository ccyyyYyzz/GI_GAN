"""A-tier paper-thickening studies (zero retraining, frozen method).

Does NOT touch the confirmed balanced locked claim or re-select B. Adds orthogonal axes:
  noise   : measurement-noise robustness sweep (frozen pipeline, locked images)
  bcurve  : fine-grained B perception-distortion frontier (dev cached recons)
  kid     : dataset-level KID on the locked recons (extra metric)
  report  : assemble figures + EXTRA_RESULTS.md (incl. scalar-vs-band ablation from dev)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gan_high_quality_gi as hq
import anchor_initialized_vqgan_inversion as ai
import vqgan_detail_fusion as vdf
import vqgan_detail_fusion_locked as vlk

OUT = vlk.LOCK.parent / "detail_fusion_paper"
SEEDS = [0, 1, 2]
NOISE = [0.0, 0.005, 0.01, 0.02, 0.05]


def log(*a):
    vdf.log(*a)


@torch.no_grad()
def _load_models(seed, cfg, device):
    priors = {ai.VQAE: ai.load_prior(ai.VQAE, vdf.ROOT / cfg["priors"]["vqae_checkpoint"], cfg, device),
              ai.VQGAN: ai.load_prior(ai.VQGAN, vdf.ROOT / cfg["priors"]["vqgan_checkpoint"], cfg, device)}
    refs = {ai.VQAE: ai.load_refiner_checkpoint(vdf.refiner_ckpt(seed, ai.VQAE), cfg, device),
            ai.VQGAN: ai.load_refiner_checkpoint(vdf.refiner_ckpt(seed, ai.VQGAN), cfg, device)}
    return priors, refs


@torch.no_grad()
def _refine(kind, x0, unc, priors, refs, cfg):
    p = priors[kind]
    dt = float(cfg["training"].get("distance_temperature", 1.0))
    st = float(cfg["training"].get("soft_temperature", 1.0))
    z0 = p.model.encode(x0)
    dz, dl = refs[kind](x0, unc, z0)
    logits = ai.logits_from_latent(z0 + dz, p, distance_temperature=dt) + dl
    zq, _, _ = ai.quantize_from_logits(p, logits, soft_temperature=st, straight_through=False)
    return p.model.decode_embeddings(zq)


def cmd_noise(device):
    cfg0 = vdf.load_cfg(0)
    sub = vdf.Substrate(cfg0, device)
    locked_ds = vlk.load_locked_ds(sub)
    bal, _ = vlk.frozen_B()
    measurement = sub.measurement
    projector = sub.projector
    lpips_fn = hq.load_lpips(device)
    g = torch.Generator(device=device).manual_seed(20260626)
    rows = []
    for seed in SEEDS:
        cfg = vdf.load_cfg(seed)
        priors, refs = _load_models(seed, cfg, device)
        loader = hq.build_loader(locked_ds, batch_size=int(cfg["data"]["eval_batch_size"]),
                                 workers=0, shuffle=False, seed=int(cfg["seed"]) + 99, device=device)
        for ns in NOISE:
            acc = defaultdict(list)
            for x, label, idx in loader:
                x = x.to(device)
                flat = measurement.flatten_img(x)
                y = measurement.A_forward(flat)
                if ns > 0:
                    y = y + ns * torch.randn(y.shape, generator=g, device=device, dtype=y.dtype)
                x0 = measurement.unflatten_img(sub.lmmse.anchor(y, measurement, device=device))
                unc = sub.lmmse.uncertainty_map(img_size=sub.img, device=device, batch_size=x.shape[0], dtype=x.dtype)
                x0f = measurement.flatten_img(x0).double()
                d_A = projector.null_project_flat(measurement.flatten_img(_refine(ai.VQAE, x0, unc, priors, refs, cfg) - x0).double())
                d_G = projector.null_project_flat(measurement.flatten_img(_refine(ai.VQGAN, x0, unc, priors, refs, cfg) - x0).double())
                yb = y.double()
                for arm, B in [("vqae", 0.0), ("balanced", bal[seed]), ("vqgan", 1.0)]:
                    xhat = vdf.fuse(("scalar", B), x0f, d_A, d_G, yb, measurement, projector, [])
                    m = vlk.per_image_metrics(xhat, x, y, measurement, lpips_fn)
                    for k in ("lpips", "psnr", "full_rmse"):
                        acc[(arm, k)].extend(m[k].tolist())
            for arm in ("vqae", "balanced", "vqgan"):
                rows.append({"seed": seed, "noise_std": ns, "method": arm,
                             "lpips": float(np.mean(acc[(arm, "lpips")])),
                             "psnr": float(np.mean(acc[(arm, "psnr")])),
                             "full_rmse": float(np.mean(acc[(arm, "full_rmse")]))})
            log(f"noise seed{seed} std={ns}: "
                + " ".join(f"{a}(lpips={np.mean(acc[(a,'lpips')]):.3f},psnr={np.mean(acc[(a,'psnr')]):.2f})" for a in ("vqae", "balanced", "vqgan")))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "noise_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "noise_std", "method", "lpips", "psnr", "full_rmse"])
        w.writeheader(); w.writerows(rows)
    # figure: LPIPS & PSNR vs noise (pooled over seeds)
    pooled = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pooled[(r["method"], r["noise_std"])]["lpips"].append(r["lpips"])
        pooled[(r["method"], r["noise_std"])]["psnr"].append(r["psnr"])
    col = {"vqae": "#5F5E5A", "balanced": "#0F6E56", "vqgan": "#993C1D"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, mk, yl in [(axes[0], "lpips", "LPIPS (lower better)"), (axes[1], "psnr", "PSNR dB (higher better)")]:
        for arm in ("vqae", "balanced", "vqgan"):
            ys = [float(np.mean(pooled[(arm, ns)][mk])) for ns in NOISE]
            ax.plot(NOISE, ys, "-o", color=col[arm], label=arm)
        ax.set_xlabel("measurement noise std"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Measurement-noise robustness on the locked split (frozen pipeline, 3 seeds)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"NOISE_ROBUSTNESS.{e}", dpi=150)
    plt.close(fig)
    log("wrote noise_sweep.csv + NOISE_ROBUSTNESS.png/pdf")


def cmd_bcurve(device):
    cfg0 = vdf.load_cfg(0)
    measurement, projector = vdf.build_meas(cfg0, device)
    lpips_fn = hq.load_lpips(device)
    grid = [round(b, 3) for b in np.linspace(0, 1, 21)]
    agg = defaultdict(lambda: defaultdict(list))
    for seed in SEEDS:
        pk = vdf.load_pack(seed, "dev", device)
        pre = vdf.prep_residuals(pk, measurement, projector)
        for B in grid:
            xhat = vdf.fuse(("scalar", B), pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], measurement, projector, [])
            m = vlk.per_image_metrics(xhat, pre["truth"], pre["y"], measurement, lpips_fn)
            agg[B]["lpips"].append(float(np.mean(m["lpips"])))
            agg[B]["psnr"].append(float(np.mean(m["psnr"])))
            agg[B]["full_rmse"].append(float(np.mean(m["full_rmse"])))
    rows = [{"B": B, "lpips": float(np.mean(agg[B]["lpips"])), "psnr": float(np.mean(agg[B]["psnr"])),
             "full_rmse": float(np.mean(agg[B]["full_rmse"]))} for B in grid]
    with open(OUT / "b_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["B", "lpips", "psnr", "full_rmse"]); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lp = [r["lpips"] for r in rows]; ps = [r["psnr"] for r in rows]
    sc = ax.scatter(ps, lp, c=[r["B"] for r in rows], cmap="viridis", s=45, zorder=3)
    ax.plot(ps, lp, color="gray", alpha=0.4, zorder=1)
    for r in rows:
        if r["B"] in (0.0, 0.5, 0.55, 1.0):
            ax.annotate(f"B={r['B']}", (r["psnr"], r["lpips"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    fig.colorbar(sc, label="fusion weight B")
    ax.set_xlabel("PSNR dB (higher better)"); ax.set_ylabel("LPIPS (lower better)")
    ax.set_title("Fine-grained perception-distortion frontier (dev, B in [0,1], 3 seeds)")
    ax.grid(alpha=0.3); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"B_CURVE.{e}", dpi=150)
    plt.close(fig)
    log("wrote b_curve.csv + B_CURVE.png/pdf")


def cmd_kid(device):
    cfg0 = vdf.load_cfg(0)
    measurement, projector = vdf.build_meas(cfg0, device)
    bal, ql = vlk.frozen_B()
    arms = {"vqae": lambda s: 0.0, "fusion_balanced": lambda s: bal[s],
            "fusion_quality_lite": lambda s: ql[s], "vqgan": lambda s: 1.0}
    preds = {a: [] for a in arms}
    truths = []
    for seed in SEEDS:
        pk = torch.load(vlk.CACHE / f"locked_seed{seed}.pt", map_location=device)
        pre = vdf.prep_residuals(pk, measurement, projector)
        truths.append(pk["truth"].cpu())
        for a, bf in arms.items():
            xhat = vdf.fuse(("scalar", bf(seed)), pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], measurement, projector, [])
            preds[a].append(xhat.clamp(0, 1).cpu())
    truth = torch.cat(truths)
    tf = hq.inception_features(truth, device=device, max_images=512)
    kid = {}
    for a in arms:
        ff = hq.inception_features(torch.cat(preds[a]), device=device, max_images=512)
        kid[a] = float(hq.kid_from_features(tf, ff)) if (tf is not None and ff is not None) else None
    (OUT / "locked_kid.json").write_text(json.dumps(kid, indent=2))
    log("locked KID:", json.dumps(kid))


def cmd_report():
    extra = ["# Extra Results (paper-thickening, zero retraining)", "",
             "These orthogonal studies use the frozen method; they do not re-touch the balanced locked claim or re-select B.", ""]
    # noise
    if (OUT / "noise_sweep.csv").exists():
        extra += ["## Measurement-noise robustness (locked split, 3 seeds)", "",
                  "See `NOISE_ROBUSTNESS.png`. Mean over seeds:", "",
                  "| noise std | VQAE LPIPS | balanced LPIPS | VQGAN LPIPS |", "|---|---|---|---|"]
        agg = defaultdict(dict)
        with open(OUT / "noise_sweep.csv") as f:
            for r in csv.DictReader(f):
                agg.setdefault(float(r["noise_std"]), defaultdict(list))[r["method"]].append(float(r["lpips"]))
        for ns in sorted(agg):
            row = agg[ns]
            extra.append(f"| {ns} | {np.mean(row['vqae']):.3f} | {np.mean(row['balanced']):.3f} | {np.mean(row['vqgan']):.3f} |")
        extra.append("")
    # kid
    if (OUT / "locked_kid.json").exists():
        kid = json.loads((OUT / "locked_kid.json").read_text())
        extra += ["## Locked KID (dataset-level, lower better)", "",
                  "| arm | KID |", "|---|---|"]
        for a, v in kid.items():
            extra.append(f"| {a} | {v:.5f} |" if v is not None else f"| {a} | n/a |")
        extra.append("")
    # ablation from dev gate
    dg = json.loads((vlk.DEV / "MULTISEED_FUSION_GATE_REPORT.json").read_text())
    sel = dg.get("selections", {})
    extra += ["## Fusion-mechanism ablation (development selection)", "",
              "On validation the simple global scalar B beat the 16-band frequency / low-pass-cutoff variants "
              "(the band/cutoff arms collapsed to the full-VQGAN endpoint). Per-seed val-selected operating points:", "",
              "| seed | balanced | quality-lite | oracle (unconstrained) |", "|---|---|---|---|"]
    for s in ("0", "1", "2"):
        v = sel.get(s, {})
        extra.append(f"| {s} | {v.get('balanced')} | {v.get('quality_lite')} | {v.get('oracle')} |")
    extra += ["", "Takeaway: the confirmed result is the *simplest* fusion (a single scalar), not a learned gate or "
              "frequency-band weighting — these did not improve over the scalar in development.", ""]
    (OUT / "EXTRA_RESULTS.md").write_text("\n".join(extra), encoding="utf-8")
    log("wrote EXTRA_RESULTS.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["noise", "bcurve", "kid", "report", "all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log("device =", device)
    if args.command in ("bcurve", "all"):
        cmd_bcurve(device)
    if args.command in ("kid", "all"):
        cmd_kid(device)
    if args.command in ("noise", "all"):
        cmd_noise(device)
    if args.command in ("report", "all"):
        cmd_report()


if __name__ == "__main__":
    main()
