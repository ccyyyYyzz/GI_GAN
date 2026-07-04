"""C2 — Lock the 'INR strictly beats VQAE as a training-free structure prior' claim.

The claim currently rests on ONE cell: K=64, seed-0, 5% sampling. This upgrades it to a
(3 seeds) x (up to 3 rates) grid. The key structural fact the lock exploits: at a fixed rate
the operator / y / x0 are SHARED across the 3 anchor seeds (only the VQAE/VQGAN refiners differ),
so d_INR is seed-invariant -- one fixed training-free INR fit is pitted against 3 independently
trained VQAE refiners. A per-image win-rate is reported alongside the means.

Stages (results checkpointed after each rate so a partial run is still useful):
  1. rate 05  -- cached packs (seed{0,1,2}_dev.pt), reuse existing dINR_seed0_dev_K64.pt.  [near-free]
  2. rate 02  -- recon_split per seed off the rate refiners; one fresh dINR fit.
  3. rate 10  -- same as 02.

x_hat = x0 + P0(d), exact audit, for every arm (A x_hat = y holds).  No priors are retrained.
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn, yaml
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
import anchor_initialized_vqgan_inversion as ai
import experiments_rate_fusion as erf
from src.projections import get_exact_projector

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = vdf.BASE / "detail_fusion_paper"
K = 64
DATA_ROOT = "E:/GAN_FCC_WORK/datasets"   # the frozen configs point at E:/datasets which no longer exists


def log(*a): vdf.log(*a)


# ----- SIREN (identical to inr_vqgan_fusion.py so 5% numbers match the prior probe) -----
class Siren(nn.Module):
    def __init__(self, w=256, depth=4, w0=30.0):
        super().__init__(); self.w0 = w0
        self.hid = nn.ModuleList([nn.Linear(2, w)] + [nn.Linear(w, w) for _ in range(depth - 1)]); self.out = nn.Linear(w, 1)
        with torch.no_grad():
            self.hid[0].weight.uniform_(-1 / 2, 1 / 2)
            for l in self.hid[1:]: l.weight.uniform_(-np.sqrt(6 / w) / w0, np.sqrt(6 / w) / w0)
    def forward(self, c):
        h = torch.sin(self.w0 * self.hid[0](c))
        for l in self.hid[1:]: h = torch.sin(self.w0 * l(h))
        return torch.sigmoid(self.out(h))

def tv(img): return (img[..., 1:, :] - img[..., :-1, :]).abs().mean() + (img[..., :, 1:] - img[..., :, :-1]).abs().mean()

def fit_inr(coords, yk, meas, steps=1200):
    torch.manual_seed(0); f = Siren().to(DEV); opt = torch.optim.Adam(f.parameters(), lr=1e-4)
    for _ in range(steps):
        img = f(coords).reshape(1, 1, 64, 64)
        loss = ((meas.A_forward(meas.flatten_img(img)) - yk) ** 2).mean() + 0.02 * tv(img)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return f(coords).reshape(1, 1, 64, 64)

def build_dINR(pre, meas, proj, cache):
    if cache.exists():
        log(f"loaded cached d_INR {cache.name}"); return torch.load(cache).to(DEV)
    x0f, y = pre["x0f"], pre["y"]
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, 64, device=DEV), torch.linspace(-1, 1, 64, device=DEV), indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    log(f"fitting INR for K={K} images -> {cache.name} ...")
    dl = [proj.null_project_flat(meas.flatten_img(fit_inr(coords, y[k:k + 1].float(), meas)).double() - x0f[k:k + 1]) for k in range(K)]
    dINR = torch.cat(dl, 0); torch.save(dINR.cpu(), cache); log("cached d_INR"); return dINR


# ----- per-image metric helpers -----
def per_img(d, x0K, yK, tK, meas, proj, lp):
    """x_hat = x0 + P0(d), audit; return per-image (psnr[K], lpips[K])."""
    pred = meas.unflatten_img(proj.audit_flat(x0K + proj.null_project_flat(d), yK)).float().clamp(0, 1)
    r = np.atleast_1d(np.asarray(hq.full_rmse_torch(pred, tK)))
    ps = -20 * np.log10(np.maximum(r, 1e-12))
    lps = np.atleast_1d(np.asarray(hq.lpips_batch(lp, pred, tK)))
    return ps, lps

def balanced(dS, x0K, dGK, yK, tK, meas, proj, lp, ref_psnr):
    """min-LPIPS scalar B on the dS->dG chord under PSNR >= ref-0.5 (mirrors inr_vqgan_fusion.py)."""
    best = None
    for B in [round(b, 2) for b in np.linspace(0, 1, 21)]:
        ps, lps = per_img(dS + B * (dGK - dS), x0K, yK, tK, meas, proj, lp)
        mps, mlp = float(ps.mean()), float(lps.mean())
        if mps >= ref_psnr - 0.5 and (best is None or mlp < best[2]): best = (B, mps, mlp)
    return best if best else (None, 0.0, 9.0)


def eval_rate(tag, cfg, meas, proj, packs_by_seed, dINR, lp):
    """packs_by_seed[seed] = prep_residuals output (x0f,d_A,d_G,y,truth,...). dINR shared across seeds."""
    x0K = packs_by_seed[0]["x0f"][:K]; yK = packs_by_seed[0]["y"][:K]; tK = packs_by_seed[0]["truth"][:K]
    dINR = dINR[:K]
    inr_ps, inr_lp = per_img(dINR, x0K, yK, tK, meas, proj, lp)
    inr_only = (float(inr_ps.mean()), float(inr_lp.mean()))
    seed_rows = {}
    for seed, pre in packs_by_seed.items():
        dAK, dGK = pre["d_A"][:K], pre["d_G"][:K]
        a_ps, a_lp = per_img(dAK, x0K, yK, tK, meas, proj, lp)
        g_ps, g_lp = per_img(dGK, x0K, yK, tK, meas, proj, lp)
        vqae_only = (float(a_ps.mean()), float(a_lp.mean()))
        vqgan_only = (float(g_ps.mean()), float(g_lp.mean()))
        # per-image dominance: INR beats THIS seed's VQAE on BOTH axes
        both = int(np.sum((inr_ps > a_ps) & (inr_lp < a_lp)))
        psnr_win = int(np.sum(inr_ps > a_ps)); lpips_win = int(np.sum(inr_lp < a_lp))
        b_vqae = balanced(dAK, x0K, dGK, yK, tK, meas, proj, lp, vqae_only[0])
        b_inr = balanced(dINR, x0K, dGK, yK, tK, meas, proj, lp, inr_only[0])
        dominates = bool(inr_only[0] > vqae_only[0] and inr_only[1] < vqae_only[1])
        seed_rows[seed] = {
            "vqae_only": vqae_only, "vqgan_only": vqgan_only,
            "inr_dominates_vqae_mean": dominates,
            "per_image_both_axes_win": both, "psnr_win": psnr_win, "lpips_win": lpips_win, "K": K,
            "vqae_vqgan_balanced": b_vqae, "inr_vqgan_balanced": b_inr,
            "inr_minus_vqae_balanced_lpips": (b_inr[2] - b_vqae[2]) if (b_inr[0] is not None and b_vqae[0] is not None) else None,
        }
        log(f"  [{tag} seed{seed}] INR {inr_only[0]:.2f}/{inr_only[1]:.3f}  VQAE {vqae_only[0]:.2f}/{vqae_only[1]:.3f}  "
            f"dominate={dominates}  both-axes win {both}/{K}  bal INR {b_inr[2]:.3f} vs VQAE {b_vqae[2]:.3f}")
    return {"tag": tag, "inr_only": inr_only, "seeds": seed_rows}


def rate05(lp):
    cfg = vdf.load_cfg(0)
    meas, proj = vdf.build_meas(cfg, DEV)
    packs = {s: vdf.prep_residuals(vdf.load_pack(s, "dev", DEV), meas, proj) for s in vdf.SEEDS}
    dINR = build_dINR(packs[0], meas, proj, OUT / f"dINR_seed0_dev_K{K}.pt")
    return eval_rate("05", cfg, meas, proj, packs, dINR, lp)


def rate_lowhi(rate, lp):
    """rate in {'02','10'} -- recon_split each seed off its rate refiner."""
    cfg0 = yaml.safe_load(erf.rate_cfg(rate, 0).read_text()); cfg0["data"]["dataset_root"] = DATA_ROOT
    sub = vdf.Substrate(cfg0, DEV)                      # operator/splits/lmmse shared across seeds at this rate
    meas, proj = sub.measurement, sub.projector
    priors = {ai.VQAE: ai.load_prior(ai.VQAE, vdf.ROOT / cfg0["priors"]["vqae_checkpoint"], cfg0, DEV),
              ai.VQGAN: ai.load_prior(ai.VQGAN, vdf.ROOT / cfg0["priors"]["vqgan_checkpoint"], cfg0, DEV)}
    packs = {}
    for seed in vdf.SEEDS:
        cfg = yaml.safe_load(erf.rate_cfg(rate, seed).read_text()); cfg["data"]["dataset_root"] = DATA_ROOT
        refs = {ai.VQAE: ai.load_refiner_checkpoint(erf.rate_refiner(rate, seed, ai.VQAE), cfg, DEV),
                ai.VQGAN: ai.load_refiner_checkpoint(erf.rate_refiner(rate, seed, ai.VQGAN), cfg, DEV)}
        pack = erf.recon_split(seed, cfg, sub, sub.dev_ds, priors, refs, DEV)
        packs[seed] = vdf.prep_residuals(pack, meas, proj)
        log(f"  [{rate}] recon seed{seed}: {pack['x0'].shape[0]} dev images")
    dINR = build_dINR(packs[0], meas, proj, OUT / f"dINR_rate{rate}_dev_K{K}.pt")
    return eval_rate(rate, cfg0, meas, proj, packs, dINR, lp)


def main():
    log("device =", DEV)
    lp = hq.load_lpips(DEV)
    outp = OUT / "inr_lock_multiseed_rate.json"
    results = json.loads(outp.read_text()) if outp.exists() else {}
    stages = [("05", rate05), ("02", lambda l: rate_lowhi("02", l)), ("10", lambda l: rate_lowhi("10", l))]
    for tag, fn in stages:
        if tag in results:
            log(f"rate{tag}: already in results -- skip"); continue
        try:
            results[tag] = fn(lp)
            outp.write_text(json.dumps(results, indent=2)); log(f"rate{tag}: checkpointed -> {outp.name}")
        except Exception as e:  # noqa: BLE001
            log(f"rate{tag}: FAILED ({type(e).__name__}: {e}) -- keeping earlier stages");
            import traceback; traceback.print_exc()
    # summary
    log("=" * 60)
    ndom = 0; ncells = 0
    for tag, r in results.items():
        for seed, sr in r["seeds"].items():
            ncells += 1; ndom += int(sr["inr_dominates_vqae_mean"])
    log(f"INR-dominates-VQAE (mean, both axes) in {ndom}/{ncells} (rate x seed) cells")
    outp.write_text(json.dumps(results, indent=2)); log(f"wrote {outp.name}")


if __name__ == "__main__":
    main()
