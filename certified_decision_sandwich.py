"""Certified-decision sandwich over the measurement fiber — feasibility prototype.

THE CERTIFICATE (exact wording, per design review): for record y, all prior-slab scene hypotheses
consistent with the measurements — {x_hat + B c : c in box}, each satisfying A x' = y to audited float64
tolerance — receive the same label as the deployed output. This establishes SLAB-RELATIVE DETERMINATION
of the label by the measurements + prior slab, NOT correctness. Disclaimers: the 3891-k unmodeled null
coordinates are frozen at x_hat's values; no correctness transfer to f(x_true).

Structure exploited ("geometry, not compute"): the feasible set is an exact affine fiber; the prior slab
B = top-k eigenvectors of P0 C P0 gives a k-dim verification problem (k=8..32) where standard certified
robustness pays for n=4096. Verification = hand-rolled IBP + backward CROWN (all layers affine except
ReLU; classifier is verification-friendly by construction) + best-first input-split BaB in c-space,
which in k dims is near-complete — the whole point of the low-dim slab.

Sandwich per scene: PGD attack in c-box (lower bound; witness twins) vs CROWN/BaB certification (upper
bound); report CERTIFIED / FLIPPED_PHYSICAL / FLIPPED_UNCLIPPED / GAP.

Subcommands: selftest | prep | gate | grid | dual
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gan_high_quality_gi as hq
import vqgan_detail_fusion as vdf

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = vdf.BASE / "detail_fusion_paper"
SB = OUT / "sandwich"
DATA_ROOT = "E:/GAN_FCC_WORK/datasets"


def log(*a): vdf.log(*a)


# ============================================================================ #
# Verification-friendly classifier: conv-ReLU-avgpool x3 -> global mean -> linear.
# No BN, no maxpool: every layer affine except ReLU => clean hand-rolled CROWN.
# ============================================================================ #
class VerifiableCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.fc = nn.Linear(64, nc)
    def forward(self, x):
        h = F.avg_pool2d(F.relu(self.c1(x)), 2)
        h = F.avg_pool2d(F.relu(self.c2(h)), 2)
        h = F.avg_pool2d(F.relu(self.c3(h)), 2)
        return self.fc(h.mean(dim=(2, 3)))


def train_classifier():
    import torchvision as tv
    from torchvision import transforms
    ckpt = SB / "verifiable_cnn.pt"
    net = VerifiableCNN().to(DEV)
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=DEV)); net.eval(); log("loaded verifiable_cnn.pt"); return net
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(1), transforms.ToTensor()])
    ds = tv.datasets.STL10(root=DATA_ROOT, split="test", download=False, transform=tf)  # disjoint from train+unlabeled
    n_val = 1000
    torch.manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
    dv = torch.utils.data.DataLoader(va, batch_size=256)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(20):
        net.train()
        for x, yb in dl:
            x, yb = x.to(DEV), yb.to(DEV)
            opt.zero_grad(); F.cross_entropy(net(x), yb).backward(); opt.step()
        net.eval(); ok = tot = 0
        with torch.no_grad():
            for x, yb in dv:
                ok += int((net(x.to(DEV)).argmax(1).cpu() == yb).sum()); tot += len(yb)
        if ok / tot > best: best, best_state = ok / tot, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        log(f"  clf epoch {ep}: acc {ok/tot:.3f} (best {best:.3f})")
    net.load_state_dict(best_state); SB.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt); net.eval(); log(f"saved verifiable_cnn.pt (acc {best:.3f})")
    return net


# ============================================================================ #
# Bound machinery. Network as layer list; input = c in [cl, cu] (k-dim), first
# layer x = x_hat + B c folded in as an affine layer.
# Layers: ('affine_in', Bmat[4096,k], xhat[4096]) -> reshape [1,64,64]
#         ('conv', W, b), ('relu',), ('avgpool', 2), ('gmean',), ('linear', W, b)
# ============================================================================ #
def net_layers(net: VerifiableCNN):
    return [("conv", net.c1.weight, net.c1.bias), ("relu",), ("avgpool", 2),
            ("conv", net.c2.weight, net.c2.bias), ("relu",), ("avgpool", 2),
            ("conv", net.c3.weight, net.c3.bias), ("relu",), ("avgpool", 2),
            ("gmean",), ("linear", net.fc.weight, net.fc.bias)]


def ibp_forward(layers, Bmat, xhat_img, cl, cu):
    """Interval propagation. Input box in c-space; first map x = xhat + Bc.
    Returns list of (l,u) pre-activation bounds for each ReLU layer + final logits bounds."""
    mu_c = (cl + cu) / 2; r_c = (cu - cl) / 2
    # affine in: x = xhat + B c  -> per-pixel intervals
    mu = xhat_img + (Bmat @ mu_c).reshape(1, 1, 64, 64)
    rad = (Bmat.abs() @ r_c).reshape(1, 1, 64, 64)
    prebounds = []
    l, u = mu - rad, mu + rad
    for L in layers:
        if L[0] == "conv":
            W, b = L[1], L[2]
            mu_, rad_ = (l + u) / 2, (u - l) / 2
            cmu = F.conv2d(mu_, W, b, padding=1)
            crad = F.conv2d(rad_, W.abs(), None, padding=1)
            l, u = cmu - crad, cmu + crad
        elif L[0] == "relu":
            prebounds.append((l, u))
            l, u = l.clamp(min=0), u.clamp(min=0)
        elif L[0] == "avgpool":
            l, u = F.avg_pool2d(l, L[1]), F.avg_pool2d(u, L[1])
        elif L[0] == "gmean":
            l, u = l.mean(dim=(2, 3)), u.mean(dim=(2, 3))
        elif L[0] == "linear":
            W, b = L[1], L[2]
            mu_, rad_ = (l + u) / 2, (u - l) / 2
            lm = mu_ @ W.T + b; lr = rad_ @ W.abs().T
            l, u = lm - lr, lm + lr
    return prebounds, (l, u)


def crown_lower(layers, Bmat, xhat_img, cl, cu, spec, prebounds):
    """Backward CROWN lower bound of spec @ logits over the c-box.
    spec: [S,10]. Returns (lb [S], A_in [S,k]) — the input-linear coefficients (design-dual signal).
    Linear state: bound_fn(z_layer) >= sum(Lambda * z) + delta, propagated backward."""
    S = spec.shape[0]
    Lambda = None; delta = None
    ridx = len(prebounds)  # walk relu bounds from the back
    for L in reversed(layers):
        if L[0] == "linear":
            W, b = L[1], L[2]
            if Lambda is None:
                Lambda = spec @ W                      # [S, 64]
                delta = spec @ b                        # [S]
            else:
                delta = delta + Lambda @ b
                Lambda = Lambda @ W
        elif L[0] == "gmean":
            # z_out[c] = mean over 8x8 -> distribute coefficient/64 over spatial
            Lambda = Lambda[:, :, None, None].expand(S, Lambda.shape[1], 8, 8) / 64.0
            Lambda = Lambda.contiguous()
        elif L[0] == "avgpool":
            k = L[1]
            # transpose of avg-pool: upsample with 1/k^2 weights
            Lambda = F.interpolate(Lambda, scale_factor=k, mode="nearest") / (k * k)
        elif L[0] == "relu":
            ridx -= 1
            l, u = prebounds[ridx]
            l, u = l.expand_as(Lambda[0:1]).squeeze(0), u.expand_as(Lambda[0:1]).squeeze(0)
            # elementwise relaxation per spec row
            pos = (l >= 0).float(); neg = (u <= 0).float(); amb = 1.0 - pos - neg
            ua = u.clamp(min=1e-12); la = l.clamp(max=-1e-12)
            s_up = torch.where(amb.bool(), u / (u - l + 1e-12), torch.zeros_like(u))     # upper chord slope
            b_up = torch.where(amb.bool(), -u * l / (u - l + 1e-12), torch.zeros_like(u))
            s_lo = (u.abs() >= l.abs()).float()                                          # adaptive lower slope 0/1
            new_L = torch.zeros_like(Lambda); add_d = torch.zeros(S, device=Lambda.device, dtype=Lambda.dtype)
            Lp = Lambda.clamp(min=0); Ln = Lambda.clamp(max=0)
            # lambda>=0 -> use LOWER relaxation (slope s, bias 0); lambda<0 -> UPPER (slope s_up, bias b_up)
            new_L = Lp * (pos + amb * s_lo) + Ln * (pos + amb * s_up)
            add_d = (Ln * (amb * b_up)).sum(dim=tuple(range(1, Lambda.ndim)))
            Lambda = new_L; delta = delta + add_d
        elif L[0] == "conv":
            W, b = L[1], L[2]
            delta = delta + (Lambda.sum(dim=(2, 3)) * b[None, :]).sum(1)
            Lambda = F.conv_transpose2d(Lambda, W, None, padding=1)
    # input affine layer: x = xhat + B c ; bound = Lambda.x + delta -> A_in = Lambda_flat @ B
    Lf = Lambda.reshape(S, -1)
    delta = delta + Lf @ xhat_img.reshape(-1)
    A_in = Lf @ Bmat                                    # [S,k]
    mu_c = (cl + cu) / 2; r_c = (cu - cl) / 2
    lb = delta + A_in @ mu_c - A_in.abs() @ r_c
    return lb, A_in


def verify_margin(layers, Bmat, xhat_img, cl, cu, pred, n_class=10):
    """CROWN margin lower bound + per-dim gap signal. spec rows: e_pred - e_j (j != pred)."""
    spec = torch.zeros(n_class - 1, n_class, device=DEV)
    js = [j for j in range(n_class) if j != pred]
    for i, j in enumerate(js): spec[i, pred] = 1.0; spec[i, j] = -1.0
    prebounds, _ = ibp_forward(layers, Bmat, xhat_img, cl, cu)
    lb, A_in = crown_lower(layers, Bmat, xhat_img, cl, cu, spec, prebounds)
    worst = int(lb.argmin())
    gap_signal = (A_in.abs() * ((cu - cl) / 2)[None, :]).sum(0)   # per-dim contribution to bound width
    return float(lb.min()), gap_signal


def bab_verify(layers, Bmat, xhat_img, cl0, cu0, pred, max_boxes=4096, batch=128):
    """Best-first input-split branch-and-bound in c-space. Returns (certified, worst_lb, n_boxes)."""
    heap = [(cl0.clone(), cu0.clone())]
    lbs = []
    n_done = 0
    while heap and n_done < max_boxes:
        cur = heap[:batch]; heap = heap[batch:]
        keep = []
        for (cl, cu) in cur:
            lb, sig = verify_margin(layers, Bmat, xhat_img, cl, cu, pred)
            n_done += 1
            if lb > 0: continue                    # subbox certified
            # split the dim with max |signal|*width
            width = cu - cl
            j = int((sig * width).argmax())
            mid = (cl[j] + cu[j]) / 2
            a_cl, a_cu = cl.clone(), cu.clone(); a_cu[j] = mid
            b_cl, b_cu = cl.clone(), cu.clone(); b_cl[j] = mid
            keep += [(a_cl, a_cu), (b_cl, b_cu)]
            lbs.append(lb)
        heap += keep
    if not heap: return True, 0.0, n_done
    return False, (min(lbs) if lbs else float("nan")), n_done


def pgd_attack(net, Bmat, xhat_img, cl, cu, pred, steps=200, restarts=5):
    """PGD in c-space to flip argmax. Returns (flipped, c_wit, in_range01)."""
    k = cl.shape[0]
    for r in range(restarts):
        torch.manual_seed(1000 + r)
        c = (cl + (cu - cl) * torch.rand(k, device=DEV)).requires_grad_(True)
        opt = torch.optim.Adam([c], lr=0.05 * float((cu - cl).mean()))
        for _ in range(steps):
            x = xhat_img + (Bmat @ c).reshape(1, 1, 64, 64)
            logits = net(x.float())
            margin = logits[0, pred] - torch.max(logits[0, [j for j in range(10) if j != pred]])
            opt.zero_grad(); margin.backward(); opt.step()
            with torch.no_grad(): c.clamp_(min=cl, max=cu)
            if margin.item() < 0:
                with torch.no_grad():
                    xw = xhat_img + (Bmat @ c).reshape(1, 1, 64, 64)
                    if net(xw.float()).argmax(1).item() != pred:
                        in01 = bool((xw >= -1e-6).all() and (xw <= 1 + 1e-6).all())
                        return True, c.detach(), in01
    return False, None, None


# ============================================================================ #
# Slab construction + scene prep
# ============================================================================ #
def build_slab(kmax=32):
    """B = top-k right singular vectors of the null-projected scaled train matrix; sigma = singular values."""
    cache = SB / "slab_basis.pt"
    if cache.exists():
        d = torch.load(cache, map_location=DEV); log("loaded slab_basis.pt"); return d["B"], d["sigma"], d["audit"]
    cfg = vdf.load_cfg(0); cfg["data"]["dataset_root"] = DATA_ROOT
    sub = vdf.Substrate(cfg, DEV)
    Z = torch.from_numpy(sub.lmmse.z_scaled).to(DEV).double()      # [N,4096], centered/scaled
    Zn = sub.projector.null_project_flat(Z)                         # rows -> null space
    _, S, Vh = torch.linalg.svd(Zn, full_matrices=False)
    B = Vh[:kmax].T.contiguous()                                    # [4096,kmax]
    sig = S[:kmax].contiguous()                                     # prior std per direction
    # hygiene: re-project + renormalize, audit ||A b_j||_inf
    B = sub.projector.null_project_flat(B.T).T
    B = B / B.norm(dim=0, keepdim=True)
    audit = float(sub.projector.A_forward(B.T).abs().max())
    SB.mkdir(parents=True, exist_ok=True)
    torch.save({"B": B, "sigma": sig, "audit": audit}, cache)
    log(f"slab built: k={kmax}, ||A b_j||_inf = {audit:.2e}, sigma[0..4]={[round(float(s),3) for s in sig[:5]]}")
    return B, sig, audit


def prep_scenes(n=128):
    """Deployed recon x_hat (fusion B=0.55, audited) + truth + labels for dev scenes."""
    cfg = vdf.load_cfg(0)
    meas, proj = vdf.build_meas(cfg, DEV)
    pre = vdf.prep_residuals(vdf.load_pack(0, "dev", DEV), meas, proj)
    xh = vdf.fuse(("scalar", 0.55), pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], meas, proj, [])
    return xh[:n].double(), pre["truth"][:n], pre["y"][:n], meas, proj


# ============================================================================ #
def cmd_selftest():
    """Soundness: CROWN/IBP lower bound <= empirical min margin over dense samples, on a tiny random net."""
    torch.manual_seed(0)
    net = VerifiableCNN().to(DEV)
    layers = net_layers(net)
    k = 6
    Bmat = torch.randn(4096, k, device=DEV); Bmat /= Bmat.norm(dim=0, keepdim=True)
    xhat = torch.rand(1, 1, 64, 64, device=DEV) * 0.5 + 0.25
    for w in [0.01, 0.05, 0.2]:
        cl = -w * torch.ones(k, device=DEV); cu = w * torch.ones(k, device=DEV)
        with torch.no_grad(): pred = int(net(xhat.float()).argmax(1))
        lb, _ = verify_margin(layers, Bmat.float(), xhat.float(), cl, cu, pred)
        # empirical min margin over samples (batched)
        emp = float("inf")
        with torch.no_grad():
            for _ in range(10):
                C = cl + (cu - cl) * torch.rand(2000, k, device=DEV)
                X = xhat.reshape(1, -1) + C @ Bmat.T.float()
                logits = net(X.reshape(-1, 1, 64, 64).float())
                m = logits[torch.arange(len(C)), pred][:, None] - logits[:, [j for j in range(10) if j != pred]]
                emp = min(emp, float(m.min()))
        ok = lb <= emp + 1e-4
        log(f"  selftest w={w}: CROWN lb={lb:.4f} <= empirical min={emp:.4f} : {'OK' if ok else 'VIOLATION'}")
        assert ok, "soundness violation"
    log("selftest PASSED")


def cmd_gate(k=8, n_scenes=16):
    """Feasibility gate: PGD + CROWN + small BaB at (k, w in {1,2}) on n scenes."""
    net = train_classifier(); layers = net_layers(net)
    Ball, sig, audit = build_slab(32)
    Bk = Ball[:, :k].float(); sk = sig[:k].float()
    xh, truth, y, meas, proj = prep_scenes(n_scenes)
    res = {}
    for w in [1.0, 2.0]:
        rows = []
        t0 = time.time()
        for i in range(n_scenes):
            xi = xh[i:i + 1].float()
            with torch.no_grad(): pred = int(net(xi).argmax(1))
            cl, cu = -w * sk, w * sk
            flip, cwit, in01 = pgd_attack(net, Bk, xi, cl, cu, pred)
            if flip:
                rows.append({"scene": i, "status": "FLIPPED_PHYSICAL" if in01 else "FLIPPED_UNCLIPPED"}); continue
            lb, _ = verify_margin(layers, Bk, xi, cl, cu, pred)
            if lb > 0:
                rows.append({"scene": i, "status": "CERTIFIED", "how": "CROWN", "lb": lb}); continue
            cert, worst, nb = bab_verify(layers, Bk, xi, cl, cu, pred, max_boxes=2048)
            rows.append({"scene": i, "status": "CERTIFIED" if cert else "GAP",
                         "how": f"BaB({nb})", "lb": worst})
        cnt = {}
        for r in rows: cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        log(f"GATE k={k} w={w}: {cnt}  ({time.time()-t0:.0f}s, audit ||A b||={audit:.1e})")
        res[f"w{w}"] = {"counts": cnt, "rows": rows}
    (SB / f"gate_k{k}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"wrote gate_k{k}.json")


def cmd_diag(k=8, n_scenes=64):
    """Fast regime finder: (1) empirical containment — does slab sigma overstate real images' null spread?
    (2) CROWN-only CDR + PGD-flip vs w (no BaB). Tells us if a viable certification regime exists at all."""
    net = train_classifier(); layers = net_layers(net)
    Ball, sig, audit = build_slab(32)
    Bk = Ball[:, :k].float(); sk = sig[:k].float()
    xh, truth, y, meas, proj = prep_scenes(n_scenes)
    # containment: e = B^T (x_true - x_hat) per scene, k-dim
    E = (truth[:n_scenes].reshape(n_scenes, -1).double() - xh.reshape(n_scenes, -1)) @ Ball[:, :k].double()  # [n,k]
    e_std = E.std(0).float()                      # empirical per-direction spread of real images
    ratio = (e_std / sk).cpu().numpy()            # how real spread compares to prior sigma
    log(f"containment k={k}: real-image std / prior-sigma per dim = {[round(float(r),2) for r in ratio]}")
    for w in [0.25, 0.5, 1.0, 2.0]:
        inbox = float(((E.abs() <= (w * sk).double()[None, :]).all(1)).float().mean())
        log(f"  w={w}: fraction of real scenes with B^T(x_true-x_hat) inside box = {inbox:.3f}")
    # w-sweep: CROWN-only CDR + PGD physical-flip rate
    res = {"containment_ratio": ratio.tolist(), "sweep": {}}
    for w in [0.1, 0.25, 0.5, 1.0]:
        cert = flipP = flipU = unk = 0; t0 = time.time()
        for i in range(n_scenes):
            xi = xh[i:i + 1].float()
            with torch.no_grad(): pred = int(net(xi).argmax(1))
            cl, cu = -w * sk, w * sk
            lb, _ = verify_margin(layers, Bk, xi, cl, cu, pred)
            if lb > 0: cert += 1; continue
            flip, cwit, in01 = pgd_attack(net, Bk, xi, cl, cu, pred, steps=100, restarts=2)
            if flip and in01: flipP += 1
            elif flip: flipU += 1
            else: unk += 1
        res["sweep"][f"w{w}"] = {"crown_certified": cert, "flipped_physical": flipP,
                                 "flipped_unclipped": flipU, "unknown": unk, "n": n_scenes}
        log(f"SWEEP w={w}: CROWN-cert {cert}/{n_scenes}  flip_phys {flipP}  flip_unclip {flipU}  unknown(gap) {unk}  ({time.time()-t0:.0f}s)")
    (SB / f"diag_k{k}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"wrote diag_k{k}.json")


def cmd_risk(k=8, n_scenes=128):
    """The cheap win test: does the certified decision radius r* (largest w whose decision is CROWN-certified,
    in units of prior sigma) predict DECISION FIDELITY DF = 1[argmax f(x_hat) == argmax f(truth)]?
    If certified-robust decisions match the true scene's decision more often, r* is a GT-free trust score:
    trust measurement-determined decisions, abstain on prior-dependent ones. ~4 CROWN passes/scene."""
    net = train_classifier(); layers = net_layers(net)
    Ball, sig, audit = build_slab(32)
    Bk = Ball[:, :k].float(); sk = sig[:k].float()
    xh, truth, y, meas, proj = prep_scenes(n_scenes)
    grid = [0.1, 0.25, 0.5, 1.0]
    rows = []
    # finer radius grid for a continuous r*; confidence = softmax top1-top2 margin at x_hat
    fgrid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    with torch.no_grad():
        lo_hat = net(xh.float()); pred_hat = lo_hat.argmax(1).cpu()
        pred_tru = net(truth[:n_scenes].float()).argmax(1).cpu()
        sm = torch.softmax(lo_hat, 1).cpu()
        top2 = sm.topk(2, 1).values
        conf = (top2[:, 0] - top2[:, 1]).numpy()   # softmax margin (standard confidence for selective pred)
    t0 = time.time()
    for i in range(n_scenes):
        xi = xh[i:i + 1].float(); pred = int(pred_hat[i])
        rstar = 0.0
        for w in fgrid:
            lb, _ = verify_margin(layers, Bk, xi, -w * sk, w * sk, pred)
            if lb > 0: rstar = w
            else: break
        rows.append({"scene": i, "rstar": rstar, "conf": float(conf[i]), "DF": int(pred_hat[i] == pred_tru[i])})
    dt = time.time() - t0
    r = np.array([x["rstar"] for x in rows]); df = np.array([x["DF"] for x in rows]); cf = np.array([x["conf"] for x in rows])
    base_df = float(df.mean())

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean(); d = ra.std() * rb.std()
        return float((ra * rb).mean() / d) if d > 0 else 0.0

    def sel_auc(score):
        """area under DF-vs-coverage as we admit scenes in decreasing score (selective-prediction AUC)."""
        order = np.argsort(-score); dfo = df[order]; cum = np.cumsum(dfo) / (np.arange(len(dfo)) + 1)
        return float(cum.mean())

    log(f"RISK k={k}: n={n_scenes}  base DF={base_df:.3f}  ({dt:.0f}s, {1000*dt/n_scenes:.0f}ms/scene)")
    log(f"  Spearman(r*,DF)={spearman(r,df):+.3f}  Spearman(conf,DF)={spearman(cf,df):+.3f}  Spearman(r*,conf)={spearman(r,cf):+.3f}")
    log(f"  selective-pred AUC:  r*={sel_auc(r):.3f}   conf={sel_auc(cf):.3f}   combined(r*+z*conf)={sel_auc(r/ (r.std()+1e-9) + cf/(cf.std()+1e-9)):.3f}")
    # THE novelty test: high-confidence but LOW r* -> measurement-aware catches confident-but-fragile decisions
    hi_conf = cf >= np.median(cf)
    lo_r = r <= np.median(r)
    hc_lr = hi_conf & lo_r          # confident yet not certified (prior-dependent)
    hc_hr = hi_conf & ~lo_r         # confident and certified
    log(f"  among HIGH-confidence scenes: DF(certified r*>med)={df[hc_hr].mean() if hc_hr.any() else float('nan'):.3f} (n={int(hc_hr.sum())})  "
        f"vs DF(uncertified r*<=med)={df[hc_lr].mean() if hc_lr.any() else float('nan'):.3f} (n={int(hc_lr.sum())})")
    out = {"k": k, "n": n_scenes, "base_DF": base_df, "fgrid": fgrid,
           "spearman": {"rstar_DF": spearman(r, df), "conf_DF": spearman(cf, df), "rstar_conf": spearman(r, cf)},
           "sel_auc": {"rstar": sel_auc(r), "conf": sel_auc(cf)},
           "high_conf_split": {"DF_certified": float(df[hc_hr].mean()) if hc_hr.any() else None, "n_cert": int(hc_hr.sum()),
                               "DF_uncertified": float(df[hc_lr].mean()) if hc_lr.any() else None, "n_uncert": int(hc_lr.sum())},
           "coverage": {}}
    for w in fgrid:
        m = r >= w
        if not m.any(): continue
        out["coverage"][f"rstar>={w}"] = {"coverage": float(m.mean()), "DF_certified": float(df[m].mean()),
                                          "DF_rest": float(df[~m].mean()) if (~m).any() else None,
                                          "lift": float(df[m].mean() - base_df)}
        log(f"  r* >= {w}: cov {m.mean():.2f}  DF(cert) {df[m].mean():.3f}  DF(rest) {df[~m].mean() if (~m).any() else float('nan'):.3f}  lift {df[m].mean()-base_df:+.3f}")
    (SB / f"risk_k{k}.json").write_text(json.dumps({**out, "rows": rows}, indent=2, default=float))
    log(f"wrote risk_k{k}.json")


# ============================================================================ #
# Stage 2: stronger verifiable classifier (train WITH BN, fold exactly at eval)
# + real-label scenes from the classifier's heldout split (double-clean:
# classifier never trained on them; refiners never saw STL10 'test' at all).
# ============================================================================ #
class VerifiableCNN2(nn.Module):
    """conv-BN-ReLU-avgpool x3 + GAP + linear. BN folds exactly into conv at eval."""
    def __init__(self, nc=10, ch=(32, 64, 128)):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch[0], 3, padding=1); self.b1 = nn.BatchNorm2d(ch[0])
        self.c2 = nn.Conv2d(ch[0], ch[1], 3, padding=1); self.b2 = nn.BatchNorm2d(ch[1])
        self.c3 = nn.Conv2d(ch[1], ch[2], 3, padding=1); self.b3 = nn.BatchNorm2d(ch[2])
        self.fc = nn.Linear(ch[2], nc)
    def forward(self, x):
        h = F.avg_pool2d(F.relu(self.b1(self.c1(x))), 2)
        h = F.avg_pool2d(F.relu(self.b2(self.c2(h))), 2)
        h = F.avg_pool2d(F.relu(self.b3(self.c3(h))), 2)
        return self.fc(h.mean(dim=(2, 3)))


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Exact eval-mode fold: W' = W * g/sqrt(v+eps) per out-channel; b' = beta + (b - mean)*g/sqrt(v+eps)."""
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    out = nn.Conv2d(conv.in_channels, conv.out_channels, 3, padding=1)
    out.weight = nn.Parameter((conv.weight.detach() * s[:, None, None, None]).clone())
    b = conv.bias.detach() if conv.bias is not None else torch.zeros_like(s)
    out.bias = nn.Parameter((bn.bias.detach() + (b - bn.running_mean.detach()) * s).clone())
    return out


class FoldedCNN(nn.Module):
    def __init__(self, net2: VerifiableCNN2):
        super().__init__()
        self.c1 = fold_bn(net2.c1, net2.b1); self.c2 = fold_bn(net2.c2, net2.b2)
        self.c3 = fold_bn(net2.c3, net2.b3); self.fc = net2.fc
    def forward(self, x):
        h = F.avg_pool2d(F.relu(self.c1(x)), 2)
        h = F.avg_pool2d(F.relu(self.c2(h)), 2)
        h = F.avg_pool2d(F.relu(self.c3(h)), 2)
        return self.fc(h.mean(dim=(2, 3)))


def _stl10_test_split():
    """Reproduce the exact train/heldout split used for classifier training (seed 0)."""
    import torchvision as tv
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(1), transforms.ToTensor()])
    ds = tv.datasets.STL10(root=DATA_ROOT, split="test", download=False, transform=tf)
    torch.manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - 1000, 1000])
    return ds, tr, va


def train_classifier2():
    ckpt = SB / "verifiable_cnn2_folded.pt"
    if ckpt.exists():
        net = FoldedCNN(VerifiableCNN2()).to(DEV)
        net.load_state_dict(torch.load(ckpt, map_location=DEV)); net.eval()
        log("loaded verifiable_cnn2_folded.pt"); return net
    from torchvision import transforms
    ds, tr, va = _stl10_test_split()
    aug = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(64, padding=4)])
    net2 = VerifiableCNN2().to(DEV)
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
    dv = torch.utils.data.DataLoader(va, batch_size=256)
    opt = torch.optim.AdamW(net2.parameters(), lr=2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    best, best_state = -1.0, None
    for ep in range(40):
        net2.train()
        for x, yb in dl:
            x, yb = aug(x.to(DEV)), yb.to(DEV)
            opt.zero_grad(); F.cross_entropy(net2(x), yb).backward(); opt.step()
        sched.step()
        net2.eval(); ok = tot = 0
        with torch.no_grad():
            for x, yb in dv:
                ok += int((net2(x.to(DEV)).argmax(1).cpu() == yb).sum()); tot += len(yb)
        if ok / tot > best: best, best_state = ok / tot, {k: v.detach().cpu().clone() for k, v in net2.state_dict().items()}
        if ep % 5 == 0 or ep == 39: log(f"  clf2 epoch {ep}: acc {ok/tot:.3f} (best {best:.3f})")
    net2.load_state_dict(best_state); net2.eval()
    folded = FoldedCNN(net2).to(DEV).eval()
    # fold audit: mathematically exact; numerically float32 accumulation differs -> check
    # relative error + 100% argmax agreement (downstream uses ONLY the folded net, consistently)
    with torch.no_grad():
        xs = torch.rand(256, 1, 64, 64, device=DEV)
        a, b = net2(xs), folded(xs)
        d = (a - b).abs().max().item(); rel = d / a.abs().max().clamp(min=1e-9).item()
        agree = float((a.argmax(1) == b.argmax(1)).float().mean())
    assert rel < 5e-3 and agree == 1.0, f"BN fold mismatch rel={rel:.1e} agree={agree}"
    torch.save(folded.state_dict(), ckpt)
    log(f"saved verifiable_cnn2_folded.pt (heldout acc {best:.3f}, fold rel|diff|={rel:.1e}, argmax agree={agree:.3f})")
    return folded


def prep_labeled_scenes(n=256):
    """Reconstruct the classifier's heldout STL10-test images through the deployed GI pipeline.
    Double-clean labels: classifier never trained on them; refiners/priors never saw split='test'.
    (Heldout was used for classifier early stopping -- mild selection leakage, disclosed.)"""
    cache = SB / f"labeled_scenes_{n}.pt"
    if cache.exists():
        d = torch.load(cache, map_location=DEV); log(f"loaded labeled_scenes_{n}.pt"); return d["xhat"], d["truth"], d["labels"]
    import anchor_initialized_vqgan_inversion as ai
    ds, tr, va = _stl10_test_split()
    xs, ys = [], []
    for i in range(n):
        x, lab = va[i]; xs.append(x); ys.append(lab)
    X = torch.stack(xs).to(DEV); labels = torch.tensor(ys)
    cfg = vdf.load_cfg(0); cfg["data"]["dataset_root"] = DATA_ROOT
    sub = vdf.Substrate(cfg, DEV)
    meas, proj = sub.measurement, sub.projector
    priors = {ai.VQAE: ai.load_prior(ai.VQAE, vdf.ROOT / cfg["priors"]["vqae_checkpoint"], cfg, DEV),
              ai.VQGAN: ai.load_prior(ai.VQGAN, vdf.ROOT / cfg["priors"]["vqgan_checkpoint"], cfg, DEV)}
    refs = {ai.VQAE: ai.load_refiner_checkpoint(vdf.refiner_ckpt(0, ai.VQAE), cfg, DEV),
            ai.VQGAN: ai.load_refiner_checkpoint(vdf.refiner_ckpt(0, ai.VQGAN), cfg, DEV)}
    dt = float(cfg["training"].get("distance_temperature", 1.0)); st = float(cfg["training"].get("soft_temperature", 1.0))
    @torch.no_grad()
    def refine(kind, x0, unc):
        p = priors[kind]; z0 = p.model.encode(x0)
        dz, dl = refs[kind](x0, unc, z0)
        logits = ai.logits_from_latent(z0 + dz, p, distance_temperature=dt) + dl
        zq, _, _ = ai.quantize_from_logits(p, logits, soft_temperature=st, straight_through=False)
        return p.model.decode_embeddings(zq)
    xh_all = []
    with torch.no_grad():
        for i in range(0, n, 32):
            xb = X[i:i + 32]
            y = meas.A_forward(meas.flatten_img(xb))
            x0 = meas.unflatten_img(sub.lmmse.anchor(y, meas, device=DEV))
            unc = sub.lmmse.uncertainty_map(img_size=64, device=DEV, batch_size=xb.shape[0], dtype=xb.dtype)
            xA = refine(ai.VQAE, x0, unc); xG = refine(ai.VQGAN, x0, unc)
            pre = vdf.prep_residuals({"x0": x0, "x_A": xA, "x_G": xG, "y": y, "truth": xb,
                                      "source_index": torch.arange(xb.shape[0]), "label": labels[i:i + 32]}, meas, proj)
            xh = vdf.fuse(("scalar", 0.55), pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], meas, proj, [])
            xh_all.append(xh.double())
    xhat = torch.cat(xh_all)
    torch.save({"xhat": xhat.cpu(), "truth": X.cpu(), "labels": labels}, cache)
    log(f"cached labeled_scenes_{n}.pt")
    return xhat.to(DEV), X, labels


def cmd_risk2(k=16, n_scenes=256):
    """Paper-grade pass: strong folded classifier + REAL labels + the headline catch metric.
    Reports: DF (decision fidelity) AND label-correctness; r* vs confidence global ranking;
    matched-budget catch comparison INSIDE the high-confidence set."""
    net = train_classifier2()
    layers = net_layers_folded(net)
    Ball, sig, _ = build_slab(32)
    Bk = Ball[:, :k].float(); sk = sig[:k].float()
    xhat, truth, labels = prep_labeled_scenes(n_scenes)
    labels = labels.cpu(); truth = truth.to(DEV)
    fgrid = [0.02, 0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    with torch.no_grad():
        lo_hat = net(xhat.float()); pred_hat = lo_hat.argmax(1).cpu()
        pred_tru = net(truth.float()).argmax(1).cpu()
        sm = torch.softmax(lo_hat, 1).cpu(); top2 = sm.topk(2, 1).values
        conf = (top2[:, 0] - top2[:, 1]).numpy()
    acc_truth = float((pred_tru == labels).float().mean())

    # FRAGILITY PROBE: is r*=0 genuine (PGD finds flips) or CROWN looseness (GAP)? subsample.
    nfp = min(64, n_scenes)
    for w in [0.05, 0.1]:
        cert = flip = gap = 0
        for i in range(nfp):
            xi = xhat[i:i + 1].float(); pred = int(pred_hat[i])
            lb, _ = verify_margin(layers, Bk, xi, -w * sk, w * sk, pred)
            if lb > 0: cert += 1; continue
            fl, _, in01 = pgd_attack(net, Bk, xi, -w * sk, w * sk, pred, steps=150, restarts=3)
            if fl and in01: flip += 1
            else: gap += 1
        log(f"  FRAGILITY w={w} (n={nfp}): CROWN-cert {cert}  PGD-flip(physical) {flip}  GAP {gap}  "
            f"-> {'GENUINELY FRAGILE' if flip>gap else 'CROWN-LOOSE(GAP-dominated)'}")
    rows = []
    t0 = time.time()
    for i in range(n_scenes):
        xi = xhat[i:i + 1].float(); pred = int(pred_hat[i])
        rstar = 0.0
        for w in fgrid:
            lb, _ = verify_margin(layers, Bk, xi, -w * sk, w * sk, pred)
            if lb > 0: rstar = w
            else: break
        rows.append({"scene": i, "rstar": rstar, "conf": float(conf[i]),
                     "DF": int(pred_hat[i] == pred_tru[i]), "correct": int(pred_hat[i] == labels[i])})
    dt = time.time() - t0
    r = np.array([x["rstar"] for x in rows]); cf = np.array([x["conf"] for x in rows])
    df = np.array([x["DF"] for x in rows]); cor = np.array([x["correct"] for x in rows])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean(); d = ra.std() * rb.std()
        return float((ra * rb).mean() / d) if d > 0 else 0.0

    def sel_auc(score, target):
        order = np.argsort(-score); t = target[order]
        cum = np.cumsum(t) / (np.arange(len(t)) + 1)
        return float(cum.mean())

    log(f"RISK2 k={k}: n={n_scenes}  clf-acc-on-truth={acc_truth:.3f}  base DF={df.mean():.3f}  base correct={cor.mean():.3f}  ({dt:.0f}s)")
    for name, tgt in [("DF", df), ("correct", cor)]:
        log(f"  [{name}] Spearman r*={spearman(r, tgt):+.3f} conf={spearman(cf, tgt):+.3f} | sel-AUC r*={sel_auc(r, tgt):.3f} conf={sel_auc(cf, tgt):.3f}")
    # HEADLINE: matched-budget catch inside the high-confidence half
    hi = cf >= np.median(cf)
    out_catch = {}
    for name, tgt in [("DF", df), ("correct", cor)]:
        err = (1 - tgt).astype(bool)
        n_err_hi = int(err[hi].sum())
        # r*-flag: lowest-r* scenes within hi-conf, budget = 25% of hi-conf set
        budget = max(1, int(0.25 * hi.sum()))
        idx_hi = np.where(hi)[0]
        flag_r = idx_hi[np.argsort(r[hi])[:budget]]
        flag_c = idx_hi[np.argsort(cf[hi])[:budget]]          # confidence's own ranking, same budget
        caught_r = int(err[flag_r].sum()); caught_c = int(err[flag_c].sum())
        out_catch[name] = {"n_hi": int(hi.sum()), "n_err_in_hi": n_err_hi, "budget": budget,
                           "caught_by_rstar": caught_r, "caught_by_conf": caught_c,
                           "recall_rstar": caught_r / max(n_err_hi, 1), "recall_conf": caught_c / max(n_err_hi, 1),
                           "precision_rstar": caught_r / budget, "precision_conf": caught_c / budget}
        log(f"  [CATCH {name}] hi-conf n={hi.sum()} errors={n_err_hi} budget={budget}: "
            f"r* catches {caught_r} (recall {caught_r/max(n_err_hi,1):.2f}, prec {caught_r/budget:.2f}) "
            f"vs conf catches {caught_c} (recall {caught_c/max(n_err_hi,1):.2f}, prec {caught_c/budget:.2f})")
    out = {"k": k, "n": n_scenes, "clf_acc_on_truth": acc_truth,
           "base_DF": float(df.mean()), "base_correct": float(cor.mean()),
           "spearman": {"rstar_DF": spearman(r, df), "conf_DF": spearman(cf, df),
                        "rstar_correct": spearman(r, cor), "conf_correct": spearman(cf, cor),
                        "rstar_conf": spearman(r, cf)},
           "sel_auc": {"rstar_DF": sel_auc(r, df), "conf_DF": sel_auc(cf, df),
                       "rstar_correct": sel_auc(r, cor), "conf_correct": sel_auc(cf, cor)},
           "catch": out_catch, "fgrid": fgrid, "rows": rows,
           "leakage_note": "scenes = classifier's heldout STL10-test subset: classifier never trained on them "
                           "(used only for early stopping -- disclosed); refiners/priors never saw split=test."}
    (SB / f"risk2_k{k}.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"wrote risk2_k{k}.json")


def net_layers_folded(net: FoldedCNN):
    return [("conv", net.c1.weight, net.c1.bias), ("relu",), ("avgpool", 2),
            ("conv", net.c2.weight, net.c2.bias), ("relu",), ("avgpool", 2),
            ("conv", net.c3.weight, net.c3.bias), ("relu",), ("avgpool", 2),
            ("gmean",), ("linear", net.fc.weight, net.fc.bias)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["selftest", "prep", "gate", "diag", "risk", "risk2"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--scenes", type=int, default=16)
    a = ap.parse_args()
    log("device =", DEV)
    SB.mkdir(parents=True, exist_ok=True)
    if a.command == "selftest": cmd_selftest()
    elif a.command == "prep": train_classifier(); build_slab(32)
    elif a.command == "gate": cmd_gate(a.k, a.scenes)
    elif a.command == "diag": cmd_diag(a.k, a.scenes)
    elif a.command == "risk": cmd_risk(a.k, a.scenes)
    elif a.command == "risk2": cmd_risk2(a.k, a.scenes)


if __name__ == "__main__":
    main()
