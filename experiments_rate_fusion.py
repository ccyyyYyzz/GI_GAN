"""Cross-sampling-rate generalization (B-tier, supplementary, development-level).

For each new rate (2%, 10%) the rate-agnostic priors are reused and only the anchor refiner is
retrained (see make_rate_configs.py / anchor_rate{R}_seed{S}_local). Here we run the SAME global
scalar null-space fusion: regenerate x0/x_A/x_G, select balanced B on val, score on dev, and report
whether the balanced fusion still beats VQAE at the new rate. Does NOT touch the 5% locked claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gan_high_quality_gi as hq
import anchor_initialized_vqgan_inversion as ai
import vqgan_detail_fusion as vdf
import vqgan_detail_fusion_locked as vlk

BASE = vdf.BASE
OUT = BASE / "detail_fusion_paper"
SEEDS = [0, 1, 2]
RATE_PCT = {"02": 2.0, "10": 10.0}


def log(*a):
    vdf.log(*a)


def rate_dir(rate, seed):
    # Colab artifacts extract to anchor_rate{R}_seed{S}; local runs used the _local suffix.
    for suffix in ("", "_local"):
        d = BASE / f"anchor_rate{rate}_seed{seed}{suffix}"
        if (d / "config_used.yaml").exists():
            return d
    return BASE / f"anchor_rate{rate}_seed{seed}"


def rate_cfg(rate, seed):
    return rate_dir(rate, seed) / "config_used.yaml"


def rate_refiner(rate, seed, kind):
    return rate_dir(rate, seed) / "runs" / f"seed{seed}" / f"{kind}_refiner" / "checkpoints" / f"{kind}_refiner_best_by_val_lpips.pt"


@torch.no_grad()
def recon_split(seed, cfg, sub, ds, priors, refs, device):
    dt = float(cfg["training"].get("distance_temperature", 1.0))
    st = float(cfg["training"].get("soft_temperature", 1.0))

    def refine(kind, x0, unc):
        p = priors[kind]
        z0 = p.model.encode(x0)
        dz, dl = refs[kind](x0, unc, z0)
        logits = ai.logits_from_latent(z0 + dz, p, distance_temperature=dt) + dl
        zq, _, _ = ai.quantize_from_logits(p, logits, soft_temperature=st, straight_through=False)
        return p.model.decode_embeddings(zq)

    loader = hq.build_loader(ds, batch_size=int(cfg["data"].get("eval_batch_size", 16)),
                             workers=0, shuffle=False, seed=int(cfg["seed"]) + 7, device=device)
    acc = defaultdict(list)
    for x, label, idx in loader:
        x = x.to(device)
        y = sub.measurement.A_forward(sub.measurement.flatten_img(x))
        x0 = sub.measurement.unflatten_img(sub.lmmse.anchor(y, sub.measurement, device=device))
        unc = sub.lmmse.uncertainty_map(img_size=sub.img, device=device, batch_size=x.shape[0], dtype=x.dtype)
        acc["x0"].append(x0); acc["x_A"].append(refine(ai.VQAE, x0, unc))
        acc["x_G"].append(refine(ai.VQGAN, x0, unc))
        acc["y"].append(y); acc["truth"].append(x)
        acc["source_index"].append(idx.to(device)); acc["label"].append(label.to(device))
    return {k: torch.cat(v) for k, v in acc.items()}


def _save(per_rate, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(OUT / "rate_generalization.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / "rate_generalization.json").write_text(json.dumps(per_rate, indent=2))


def cmd_run(rates, device):
    lpips_fn = hq.load_lpips(device)
    grid = [round(b, 3) for b in np.linspace(0, 1, 21)]
    # resume: load any already-computed rates so a mid-run death doesn't lose progress
    per_rate, rows = {}, []
    pj = OUT / "rate_generalization.json"
    if pj.exists():
        per_rate = json.loads(pj.read_text())
    pc = OUT / "rate_generalization.csv"
    if pc.exists():
        with open(pc, newline="") as f:
            rows = [{k: (float(v) if k != "rate_pct" or True else v) for k, v in r.items()} for r in csv.DictReader(f)]
    for rate in rates:
        if rate in per_rate:
            log(f"rate{rate}: already computed (resume) — skip")
            continue
        if not rate_cfg(rate, 0).exists():
            log(f"rate{rate}: config_used.yaml not found yet (training incomplete) — skip")
            continue
        seed_stats = []
        for seed in SEEDS:
            cfg = yaml.safe_load(rate_cfg(rate, seed).read_text())
            cfg["data"]["dataset_root"] = "E:/datasets"  # configs were frozen with the Colab path
            sub = vdf.Substrate(cfg, device)
            priors = {ai.VQAE: ai.load_prior(ai.VQAE, vdf.ROOT / cfg["priors"]["vqae_checkpoint"], cfg, device),
                      ai.VQGAN: ai.load_prior(ai.VQGAN, vdf.ROOT / cfg["priors"]["vqgan_checkpoint"], cfg, device)}
            refs = {ai.VQAE: ai.load_refiner_checkpoint(rate_refiner(rate, seed, ai.VQAE), cfg, device),
                    ai.VQGAN: ai.load_refiner_checkpoint(rate_refiner(rate, seed, ai.VQGAN), cfg, device)}
            val = vdf.prep_residuals(recon_split(seed, cfg, sub, sub.val_ds, priors, refs, device), sub.measurement, sub.projector)
            dev = vdf.prep_residuals(recon_split(seed, cfg, sub, sub.dev_ds, priors, refs, device), sub.measurement, sub.projector)
            # val-select balanced B (global scalar, frozen rule)
            vqae_val = vdf.fast_means(vdf.fuse(("scalar", 0.0), val["x0f"], val["d_A"], val["d_G"], val["y"], sub.measurement, sub.projector, []), val["truth"], val["y"], sub.measurement, lpips_fn)
            best = None
            for B in grid:
                m = vdf.fast_means(vdf.fuse(("scalar", B), val["x0f"], val["d_A"], val["d_G"], val["y"], sub.measurement, sub.projector, []), val["truth"], val["y"], sub.measurement, lpips_fn)
                if (m["psnr"] >= vqae_val["psnr"] - 0.5 and m["rmse"] <= vqae_val["rmse"] + 0.005
                        and m["rapsd"] <= vqae_val["rapsd"] + 1e-9 and np.isfinite(m["lpips"])):
                    if best is None or m["lpips"] < best[1]:
                        best = (B, m["lpips"])
            Bsel = best[0] if best else 0.0
            # dev scoring
            d_vqae = vlk.per_image_metrics(vdf.fuse(("scalar", 0.0), dev["x0f"], dev["d_A"], dev["d_G"], dev["y"], sub.measurement, sub.projector, []), dev["truth"], dev["y"], sub.measurement, lpips_fn)
            d_bal = vlk.per_image_metrics(vdf.fuse(("scalar", Bsel), dev["x0f"], dev["d_A"], dev["d_G"], dev["y"], sub.measurement, sub.projector, []), dev["truth"], dev["y"], sub.measurement, lpips_fn)
            dl = float(np.mean(d_bal["lpips"] - d_vqae["lpips"]))
            dp = float(np.mean(d_bal["psnr"] - d_vqae["psnr"]))
            dr = float(np.mean(d_bal["full_rmse"] - d_vqae["full_rmse"]))
            dra = float(np.mean(d_bal["rapsd"] - d_vqae["rapsd"]))
            seed_stats.append({"seed": seed, "B": Bsel, "dlpips": dl, "dpsnr": dp, "drmse": dr, "drapsd": dra,
                               "vqae_lpips": float(np.mean(d_vqae["lpips"]))})
            log(f"rate{rate} seed{seed}: B={Bsel} dLPIPS={dl:.4f} dPSNR={dp:.2f} dRMSE={dr:+.4f} dRAPSD={dra:+.5f}")
        if seed_stats:
            dlp = np.array([s["dlpips"] for s in seed_stats])
            rel = -float(np.mean(dlp)) / max(float(np.mean([s["vqae_lpips"] for s in seed_stats])), 1e-12)
            summary = {"rate_pct": RATE_PCT[rate], "n_seeds": len(seed_stats),
                       "mean_dlpips": float(np.mean(dlp)), "lpips_relative_gain": rel,
                       "mean_dpsnr": float(np.mean([s["dpsnr"] for s in seed_stats])),
                       "mean_drmse": float(np.mean([s["drmse"] for s in seed_stats])),
                       "mean_drapsd": float(np.mean([s["drapsd"] for s in seed_stats])),
                       "seeds_improved": int(np.sum(dlp < 0)), "per_seed_B": {s["seed"]: s["B"] for s in seed_stats}}
            per_rate[rate] = summary
            for s in seed_stats:
                rows.append({"rate_pct": RATE_PCT[rate], **s})
            log(f"rate{rate} POOLED: dLPIPS={summary['mean_dlpips']:.4f} (rel {rel:.3f}) "
                f"dPSNR={summary['mean_dpsnr']:.2f} improved={summary['seeds_improved']}/{len(seed_stats)}")
            _save(per_rate, rows)  # incremental: survive a mid-run death
            log(f"rate{rate}: saved (resumable)")
    _save(per_rate, rows)
    # cross-rate table incl. the 5% locked anchor point
    locked = json.loads((vlk.LOCK / "LOCKED_GATE_REPORT.json").read_text())["arms"]["fusion_balanced"]
    table = {"02": per_rate.get("02"), "05_locked": {
        "rate_pct": 5.0, "mean_dlpips": locked["metrics"]["lpips"]["mean"],
        "lpips_relative_gain": locked["lpips_relative_gain"], "mean_dpsnr": locked["metrics"]["psnr"]["mean"],
        "seeds_improved": locked["lpips_same_direction_neg"], "n_seeds": 3}, "10": per_rate.get("10")}
    (OUT / "rate_generalization_table.json").write_text(json.dumps(table, indent=2))
    log("wrote rate_generalization.{csv,json} + rate_generalization_table.json")


def cmd_fig():
    """Render the cross-rate generalization figure from the saved table (no recompute)."""
    table = json.loads((OUT / "rate_generalization_table.json").read_text())
    order = [("02", "2%"), ("05_locked", "5%\n(locked)"), ("10", "10%")]
    pts = [(lab, table[k]) for k, lab in order if table.get(k)]
    xs = np.arange(len(pts))
    gains = [100.0 * p[1]["lpips_relative_gain"] for p in pts]
    dpsnr = [p[1]["mean_dpsnr"] for p in pts]
    labels = [p[0] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    bars = ax1.bar(xs, gains, color=["#4c72b0", "#c44e52", "#4c72b0"], width=0.6)
    ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_ylabel("LPIPS relative gain (%)\nbalanced fusion vs VQAE")
    ax1.set_title("(a) Perceptual gain generalizes across rate")
    ax1.set_ylim(0, max(gains) * 1.25)
    for b, g in zip(bars, gains):
        ax1.text(b.get_x() + b.get_width() / 2, g + 0.8, f"{g:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.axhline(0, color="k", lw=0.8)

    ax2.bar(xs, dpsnr, color="#8c8c8c", width=0.6)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels)
    ax2.set_ylabel("ΔPSNR (dB)")
    ax2.set_title("(b) PSNR cost stays within tolerance")
    ax2.axhline(-0.5, color="#c44e52", ls="--", lw=1.0, label="−0.5 dB val tolerance")
    ax2.set_ylim(min(dpsnr) * 1.4, 0.1)
    ax2.legend(fontsize=8, loc="lower right")
    for sp in (ax1, ax2):
        sp.spines["top"].set_visible(False); sp.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rate_generalization_figure.{ext}", dpi=200, bbox_inches="tight")
    log(f"wrote rate_generalization_figure.png/.pdf  (rates={labels} gains={[round(g,1) for g in gains]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "fig"])
    ap.add_argument("--rates", nargs="*", default=["02", "10"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.command == "fig":
        cmd_fig()
        return
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log("device =", device)
    cmd_run(args.rates, device)


if __name__ == "__main__":
    main()
