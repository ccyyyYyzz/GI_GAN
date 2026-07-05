# -*- coding: utf-8 -*-
"""Optional reviewer item: the no-adaptation lemma's empirical face as one figure.
(a) oracle per-image B* vs the GT-free selector's prediction B(y) (with the constant baseline)
(b) LPIPS regret of the three rules relative to the per-image oracle.
Recomputes the per-image B-curves (val for training the selector, dev for evaluation),
mirroring experiments_gtfree_selector.py."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import gan_high_quality_gi as hq
import vqgan_detail_fusion as vdf

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(r"E:\ns_mc_gan_gi_code_fcc_phase1")
GRID = np.array([round(b, 2) for b in np.linspace(0, 1, 21)])


def curves(pre, meas, proj, lp):
    x0f, dA, dG, y, truth = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"]
    N = truth.shape[0]
    L = np.zeros((N, len(GRID)))
    for j, B in enumerate(GRID):
        xh = vdf.fuse(("scalar", float(B)), x0f, dA, dG, y, meas, proj, []).float().clamp(0, 1)
        L[:, j] = np.atleast_1d(np.asarray(hq.lpips_batch(lp, xh, truth)))
    return L


def feats(pre):
    dA, dG = pre["d_A"], pre["d_G"]
    na = dA.norm(dim=1).cpu().numpy(); ng = dG.norm(dim=1).cpu().numpy()
    chord = (dG - dA).norm(dim=1).cpu().numpy()
    cos = (((dA * dG).sum(1)) / (dA.norm(dim=1) * dG.norm(dim=1) + 1e-12)).cpu().numpy()
    return np.stack([na, ng, chord, cos], 1)


def knn(Ftr, Btr, Fte, k=16):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A = (Ftr - mu) / sd; Bq = (Fte - mu) / sd
    d2 = ((Bq[:, None, :] - A[None, :, :]) ** 2).sum(2)
    return Btr[np.argsort(d2, 1)[:, :k]].mean(1)


def main():
    cfg = vdf.load_cfg(0)
    meas, proj = vdf.build_meas(cfg, DEV)
    lp = hq.load_lpips(DEV)
    val = vdf.prep_residuals(vdf.load_pack(0, "val", DEV), meas, proj)
    dev = vdf.prep_residuals(vdf.load_pack(0, "dev", DEV), meas, proj)
    Lv, Ld = curves(val, meas, proj, lp), curves(dev, meas, proj, lp)
    Bv, Bd = GRID[Lv.argmin(1)], GRID[Ld.argmin(1)]
    Fv, Fd = feats(val), feats(dev)
    Bpred = np.clip(np.round(knn(Fv, Bv, Fd) * 20) / 20, 0, 1)
    Bconst = np.full_like(Bd, float(GRID[Lv.mean(0).argmin()]))    # val-optimal constant

    def achieved(L, Bsel):
        idx = np.array([int(round(b * 20)) for b in Bsel])
        return L[np.arange(L.shape[0]), idx]
    l_orc = achieved(Ld, Bd); l_prd = achieved(Ld, Bpred); l_cst = achieved(Ld, Bconst)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    jit = (np.random.default_rng(0).random(len(Bd)) - 0.5) * 0.02
    ax.scatter(Bd + jit, Bpred, s=14, alpha=0.5, color="#4c72b0", label="GT-free selector $B(y)$")
    ax.axhline(Bconst[0], color="#c44e52", lw=1.8, ls="--", label=f"constant $B={Bconst[0]:.2f}$")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, label="oracle line")
    rho = np.corrcoef(np.argsort(np.argsort(Bd)), np.argsort(np.argsort(Bpred)))[0, 1]
    ax.set_xlabel("oracle per-image $B^\\star$"); ax.set_ylabel("selected $B$")
    ax.set_title(f"(a) The selector barely tracks the oracle (Spearman {rho:+.2f})")
    ax.legend(fontsize=8); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax = axes[1]
    reg_p, reg_c = l_prd - l_orc, l_cst - l_orc
    ax.hist(reg_c, bins=30, alpha=0.6, color="#c44e52", label=f"constant (mean regret {reg_c.mean():.4f})")
    ax.hist(reg_p, bins=30, alpha=0.6, color="#4c72b0", label=f"selector (mean regret {reg_p.mean():.4f})")
    ax.set_xlabel("LPIPS regret vs per-image oracle"); ax.set_ylabel("images")
    ax.set_title(f"(b) Regret: selector gains {reg_c.mean()-reg_p.mean():.4f} over a constant")
    ax.legend(fontsize=8)
    for a in axes: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    for o in [REPO / "paper", REPO / "outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper"]:
        fig.savefig(o / "NOADAPT_EMPIRICAL.pdf", bbox_inches="tight")
        fig.savefig(o / "NOADAPT_EMPIRICAL.png", dpi=200, bbox_inches="tight")
    print(f"wrote NOADAPT_EMPIRICAL: spearman={rho:+.3f} regret const={reg_c.mean():.4f} selector={reg_p.mean():.4f}")


if __name__ == "__main__":
    main()
