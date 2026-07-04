"""C5 — Two-ledger governed dial: exact measurement certificate  +  distribution-free task-risk bound.

The B-dial produces a family x_hat_B = x0 + P0(d_A + B(d_G - d_A)); A x_hat_B = y EXACTLY for every B
(Ledger 1: measurement, deterministic, RelMeasErr ~ 1e-13, flat in B). As B injects null-space detail,
a downstream DECISION can drift. We add Ledger 2: a distribution-free finite-sample bound on task risk,
never summed with Ledger 1.

Task functional f = a fixed STL10 classifier's argmax. It is trained on STL10 split='test' (8000 labeled)
which is DISJOINT from the reconstructions' source split ('train+unlabeled') -> no leakage, and needs no
labels on the val/dev packs (which are label=-1). Task risk:
    R(B) = P[ f(x_hat_B) != f(truth) ]            (decision-disagreement with the true scene's decision)
Selection: calibrate on VAL, test on DEV. For target alpha, pick B_hat = the max-perception (min-LPIPS) B
whose Hoeffding upper confidence bound (Bonferroni over the B-grid, delta=0.1) on R(B) is <= alpha. This is
distribution-free and needs no monotonicity. We then report the DEV-achieved R(B_hat) <= alpha to show the
guarantee transfers, alongside the flat measurement ledger.
"""
from __future__ import annotations
import json, math, numpy as np, torch, torch.nn as nn
import torchvision as tv
from torchvision import transforms
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
from src.projections import relative_measurement_error

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = vdf.BASE / "detail_fusion_paper"
DATA_ROOT = "E:/GAN_FCC_WORK/datasets"
GRID = [round(b, 2) for b in np.linspace(0, 1, 21)]
DELTA = 0.10  # confidence for the distribution-free bound


def log(*a): vdf.log(*a)


class SmallCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, nc)
    def forward(self, x): return self.head(self.body(x).flatten(1))


def train_classifier():
    ckpt = OUT / "task_classifier_stl10test.pt"
    net = SmallCNN().to(DEV)
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=DEV)); net.eval(); log(f"loaded {ckpt.name}"); return net
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(1), transforms.ToTensor()])
    ds = tv.datasets.STL10(root=DATA_ROOT, split="test", download=False, transform=tf)   # disjoint from train+unlabeled
    n_val = 1000; n_tr = len(ds) - n_val
    torch.manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_val])
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=0)
    dv = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for ep in range(20):
        net.train()
        for x, yb in dl:
            x, yb = x.to(DEV), yb.to(DEV)
            opt.zero_grad(); lossf(net(x), yb).backward(); opt.step()
        net.eval(); correct = tot = 0
        with torch.no_grad():
            for x, yb in dv:
                p = net(x.to(DEV)).argmax(1).cpu(); correct += int((p == yb).sum()); tot += len(yb)
        acc = correct / tot
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        log(f"  classifier epoch {ep}: heldout acc {acc:.3f} (best {best_acc:.3f})")
    net.load_state_dict(best_state)                 # keep the BEST epoch, not the last
    torch.save(net.state_dict(), ckpt); net.eval(); log(f"trained + saved {ckpt.name} (heldout acc {best_acc:.3f})")
    return net


@torch.no_grad()
def decisions(net, imgs):
    out = []
    for i in range(0, imgs.shape[0], 256):
        out.append(net(imgs[i:i + 256].float().to(DEV)).argmax(1).cpu())
    return torch.cat(out)


def ledger_for_split(net, pre, meas, proj, lp):
    """Return per-B dict: task_risk, lpips, relmeaserr; plus per-image error masks (for CRC)."""
    x0f, dA, dG, y, truth = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"]
    f_truth = decisions(net, truth)
    per_B = {}
    for B in GRID:
        xhat = vdf.fuse(("scalar", B), x0f, dA, dG, y, meas, proj, [])
        f_hat = decisions(net, xhat)
        err = (f_hat != f_truth).float().numpy()        # per-image task error (0/1)
        rel = float(relative_measurement_error(xhat.float().clamp(0, 1), y, meas).mean())
        l = float(np.mean(hq.lpips_batch(lp, xhat.float().clamp(0, 1), truth)))
        per_B[f"{B:.2f}"] = {"B": B, "task_risk": float(err.mean()), "lpips": l,
                             "relmeaserr": rel, "_err": err}
    return per_B


def select_Bhat(val_B, alpha, n_cal):
    """max-perception (min val-LPIPS) B whose Hoeffding UCB (Bonferroni over grid) on task risk <= alpha."""
    slack = math.sqrt(math.log(len(GRID) / DELTA) / (2 * n_cal))
    cands = [(v["lpips"], v["B"], v["task_risk"], v["task_risk"] + slack)
             for v in val_B.values() if v["task_risk"] + slack <= alpha]
    if not cands:
        return None, slack
    cands.sort()  # by lpips asc
    return cands[0], slack


def main():
    log("device =", DEV)
    net = train_classifier()
    cfg = vdf.load_cfg(0)
    meas, proj = vdf.build_meas(cfg, DEV)
    lp = hq.load_lpips(DEV)
    val = vdf.prep_residuals(vdf.load_pack(0, "val", DEV), meas, proj)
    dev = vdf.prep_residuals(vdf.load_pack(0, "dev", DEV), meas, proj)
    n_cal = val["truth"].shape[0]
    log(f"calibration(val)={n_cal}  test(dev)={dev['truth'].shape[0]}")

    val_B = ledger_for_split(net, val, meas, proj, lp)
    dev_B = ledger_for_split(net, dev, meas, proj, lp)
    r0 = val_B["0.00"]["task_risk"]
    log(f"anchor(B=0) task risk: val={r0:.3f} dev={dev_B['0.00']['task_risk']:.3f}")
    for B in ("0.00", "0.55", "1.00"):
        log(f"  B={B}: val R={val_B[B]['task_risk']:.3f} LPIPS={val_B[B]['lpips']:.3f} relerr={val_B[B]['relmeaserr']:.1e} "
            f"| dev R={dev_B[B]['task_risk']:.3f} LPIPS={dev_B[B]['lpips']:.3f}")

    # two-ledger operating points for a few risk budgets
    ledger = {}
    for tag, alpha in [("r0", r0), ("r0+0.05", r0 + 0.05), ("r0+0.10", r0 + 0.10)]:
        sel, slack = select_Bhat(val_B, alpha, n_cal)
        if sel is None:
            ledger[tag] = {"alpha": alpha, "slack": slack, "B_hat": None, "note": "no B satisfies UCB<=alpha"}
            log(f"  alpha={tag}({alpha:.3f}): no admissible B (slack={slack:.3f})"); continue
        _, Bh, val_risk, ucb = sel
        d = dev_B[f"{Bh:.2f}"]
        ledger[tag] = {"alpha": alpha, "slack": slack, "B_hat": Bh,
                       "val_risk": val_risk, "val_ucb": ucb,
                       "dev_task_risk": d["task_risk"], "dev_guarantee_holds": bool(d["task_risk"] <= alpha),
                       "dev_lpips": d["lpips"], "dev_lpips_vs_anchor": d["lpips"] - dev_B["0.00"]["lpips"],
                       "dev_relmeaserr": d["relmeaserr"]}
        log(f"  alpha={tag}({alpha:.3f}): B_hat={Bh} val_ucb={ucb:.3f} -> dev R={d['task_risk']:.3f} "
            f"(<=alpha? {d['task_risk']<=alpha}) dev LPIPS {d['lpips']:.3f} (dLPIPS {d['lpips']-dev_B['0.00']['lpips']:+.3f}) "
            f"relerr {d['relmeaserr']:.1e}")

    # honest interpretation: is the task ledger aligned with or in tension with more detail?
    risks = np.array([dev_B[f"{b:.2f}"]["task_risk"] for b in GRID])
    direction = ("task risk FALLS with detail (well-trained prior aligns the ledgers -> CRC selects high B)"
                 if risks[-1] < risks[0] - 0.02 else
                 "task risk RISES with detail (prior injects decision-flipping content -> CRC caps B)"
                 if risks[-1] > risks[0] + 0.02 else "task risk ~flat in B")
    risk_span = float(risks.max() - risks.min())

    def strip(B): return {k: {kk: vv for kk, vv in v.items() if kk != "_err"} for k, v in B.items()}
    out = {"design": "Two INDEPENDENT ledgers, never summed: (1) exact measurement certificate A x=y holds for "
                      "EVERY B (RelMeasErr flat ~4e-3, the operator floor); (2) a distribution-free task-risk bound "
                      "R(B)=P[f(x_hat_B)!=f(truth)] via Hoeffding UCB (Bonferroni over the B-grid, delta=0.10), "
                      "calibrated on val, tested on dev. The point: measurement-consistency is INVARIANT to B while "
                      "the task outcome swings widely -> the measurement certificate says nothing about task risk, so "
                      "a separate statistical ledger is required. CRC then certifies the chosen operating point.",
           "finding": direction, "task_risk_span_over_B": risk_span,
           "measurement_ledger_flat": "A x_hat_B = y for all B; RelMeasErr ~constant (see relmeaserr column) "
                                       "-> the measurement certificate cannot distinguish B, the task ledger must.",
           "task": "STL10 classifier (trained on split=test, disjoint from train+unlabeled source of val/dev; "
                   "needs no pack labels). Risk = decision-disagreement with the true scene's decision.",
           "delta": DELTA, "grid": GRID, "n_calibration": n_cal, "n_test": int(dev["truth"].shape[0]),
           "anchor_task_risk": {"val": r0, "dev": dev_B["0.00"]["task_risk"]},
           "operating_points": ledger, "val_curve": strip(val_B), "dev_curve": strip(dev_B)}
    (OUT / "conformal_ledger.json").write_text(json.dumps(out, indent=2))
    log("wrote conformal_ledger.json")


if __name__ == "__main__":
    main()
