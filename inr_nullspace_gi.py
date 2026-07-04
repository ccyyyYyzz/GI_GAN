"""Exploration (beyond GAN): a coordinate-INR / SIREN reconstructor for ghost imaging.
Per-scene test-time fit (no training set, DIP-style): min_theta || A f_theta - y ||^2 + tv*TV(f_theta),
then optionally rectify onto {x: A x = y} with the exact projector. Compare to the LMMSE anchor x0 and
the current VQGAN recon x_G on the same cached images/measurements (seed0 dev). Honest probe of whether a
novel architecture (SIREN) beats the linear anchor at 5% GI.
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
from src.projections import get_exact_projector

def log(*a): vdf.log(*a)

class Siren(nn.Module):
    def __init__(self, w=256, depth=4, w0=30.0):
        super().__init__()
        layers=[nn.Linear(2,w)]; self.w0=w0
        for _ in range(depth-1): layers.append(nn.Linear(w,w))
        self.hidden=nn.ModuleList(layers); self.out=nn.Linear(w,1)
        with torch.no_grad():
            self.hidden[0].weight.uniform_(-1/2, 1/2)
            for l in self.hidden[1:]: l.weight.uniform_(-np.sqrt(6/w)/w0, np.sqrt(6/w)/w0)
    def forward(self, c):
        h=torch.sin(self.w0*self.hidden[0](c))
        for l in self.hidden[1:]: h=torch.sin(self.w0*l(h))
        return torch.sigmoid(self.out(h))

def tv(img):
    return (img[...,1:,:]-img[...,:-1,:]).abs().mean()+(img[...,:,1:]-img[...,:,:-1]).abs().mean()

def met(pred, truth, lp):
    pred=pred.clamp(0,1)
    rmse=hq.full_rmse_torch(pred, truth); psnr=-20.0*np.log10(max(float(rmse),1e-12))
    l=hq.lpips_batch(lp, pred, truth)
    return psnr, (float(np.mean(l)) if l is not None else float("nan"))

def main():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=vdf.load_cfg(0)
    rows_np,_=hq.build_structured_operator_rows(img_size=64,total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]),hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]),seed=int(cfg["operator"]["seed"]))
    meas=hq.make_measurement_operator(rows_np,img_size=64,device=dev,lambda_solver=float(cfg["operator"]["lambda_solver"]))
    proj=get_exact_projector(meas,dtype=torch.float64,device=dev)
    pack=vdf.load_pack(0,"dev",device=dev)
    truth,y,x0,xG=pack["truth"].to(dev),pack["y"].to(dev),pack["x0"].to(dev),pack["x_G"].to(dev)
    lp=hq.load_lpips(dev)
    K=6
    gy,gx=torch.meshgrid(torch.linspace(-1,1,64,device=dev),torch.linspace(-1,1,64,device=dev),indexing="ij")
    coords=torch.stack([gx.reshape(-1),gy.reshape(-1)],1)  # (4096,2)
    res={"inr":[], "inr_audit":[], "lmmse":[], "vqgan":[]}
    for k in range(K):
        yk=y[k:k+1].float(); tk=truth[k:k+1]
        torch.manual_seed(k); f=Siren().to(dev); opt=torch.optim.Adam(f.parameters(),lr=1e-4)
        for step in range(1500):
            img=f(coords).reshape(1,1,64,64)
            pred_y=meas.A_forward(meas.flatten_img(img))
            loss=((pred_y-yk)**2).mean()+0.02*tv(img)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            img=f(coords).reshape(1,1,64,64)
            audit=meas.unflatten_img(proj.audit_flat(meas.flatten_img(img).double(), yk.double())).float()
        for nm,p in [("inr",img),("inr_audit",audit),("lmmse",x0[k:k+1]),("vqgan",xG[k:k+1])]:
            ps,l=met(p,tk,lp); res[nm].append((ps,l))
        log(f"  img{k}: INR {res['inr'][-1][0]:.2f}dB/{res['inr'][-1][1]:.3f}lp | "
            f"INR+audit {res['inr_audit'][-1][0]:.2f}/{res['inr_audit'][-1][1]:.3f} | "
            f"LMMSE {res['lmmse'][-1][0]:.2f}/{res['lmmse'][-1][1]:.3f} | VQGAN {res['vqgan'][-1][0]:.2f}/{res['vqgan'][-1][1]:.3f}")
    def mean(nm,i): return float(np.mean([r[i] for r in res[nm]]))
    summ={nm:{"psnr":mean(nm,0),"lpips":mean(nm,1)} for nm in res}
    log("MEAN over %d imgs:"%K)
    for nm in ["inr","inr_audit","lmmse","vqgan"]:
        log(f"  {nm:10s} PSNR={summ[nm]['psnr']:.2f}  LPIPS={summ[nm]['lpips']:.3f}")
    (vdf.BASE/"detail_fusion_paper"/"inr_gi_probe.json").write_text(json.dumps(summ,indent=2))
    log("wrote inr_gi_probe.json")

if __name__=="__main__":
    main()
