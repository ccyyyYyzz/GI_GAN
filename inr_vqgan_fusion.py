"""Follow-up 1 (fixed): INR (high-PSNR, training-free) as the STRUCTURE branch vs VQAE, each fused with
VQGAN detail via the exact null-space dial. Balanced point = min-LPIPS subject to PSNR >= (structure-only
PSNR) - 0.5 (the paper's gate). Caches d_INR so re-scoring is instant.
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
from src.projections import get_exact_projector

def log(*a): vdf.log(*a)
class Siren(nn.Module):
    def __init__(self, w=256, depth=4, w0=30.0):
        super().__init__(); self.w0=w0
        self.hid=nn.ModuleList([nn.Linear(2,w)]+[nn.Linear(w,w) for _ in range(depth-1)]); self.out=nn.Linear(w,1)
        with torch.no_grad():
            self.hid[0].weight.uniform_(-1/2,1/2)
            for l in self.hid[1:]: l.weight.uniform_(-np.sqrt(6/w)/w0, np.sqrt(6/w)/w0)
    def forward(self,c):
        h=torch.sin(self.w0*self.hid[0](c))
        for l in self.hid[1:]: h=torch.sin(self.w0*l(h))
        return torch.sigmoid(self.out(h))
def tv(img): return (img[...,1:,:]-img[...,:-1,:]).abs().mean()+(img[...,:,1:]-img[...,:,:-1]).abs().mean()
def fit_inr(coords, yk, meas, dev, steps=1200):
    torch.manual_seed(0); f=Siren().to(dev); opt=torch.optim.Adam(f.parameters(),lr=1e-4)
    for _ in range(steps):
        img=f(coords).reshape(1,1,64,64)
        loss=((meas.A_forward(meas.flatten_img(img))-yk)**2).mean()+0.02*tv(img)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return f(coords).reshape(1,1,64,64)

def main():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=vdf.load_cfg(0)
    rows_np,_=hq.build_structured_operator_rows(img_size=64,total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]),hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]),seed=int(cfg["operator"]["seed"]))
    meas=hq.make_measurement_operator(rows_np,img_size=64,device=dev,lambda_solver=float(cfg["operator"]["lambda_solver"]))
    proj=get_exact_projector(meas,dtype=torch.float64,device=dev)
    pack=vdf.load_pack(0,"dev",device=dev); pre=vdf.prep_residuals(pack,meas,proj)
    x0f,dA,dG,y,truth=pre["x0f"],pre["d_A"],pre["d_G"],pre["y"],pre["truth"]
    lp=hq.load_lpips(dev); K=64
    x0K,dAK,dGK,yK,tK=x0f[:K],dA[:K],dG[:K],y[:K],truth[:K]
    cache=vdf.BASE/"detail_fusion_paper"/f"dINR_seed0_dev_K{K}.pt"
    if cache.exists():
        dINR=torch.load(cache).to(dev); log(f"loaded cached d_INR {cache.name}")
    else:
        gy,gx=torch.meshgrid(torch.linspace(-1,1,64,device=dev),torch.linspace(-1,1,64,device=dev),indexing="ij")
        coords=torch.stack([gx.reshape(-1),gy.reshape(-1)],1)
        log(f"fitting INR for K={K} images ...")
        dl=[proj.null_project_flat(meas.flatten_img(fit_inr(coords,y[k:k+1].float(),meas,dev)).double()-x0f[k:k+1]) for k in range(K)]
        dINR=torch.cat(dl,0); torch.save(dINR.cpu(), cache); log("cached d_INR")
    def rec(d):
        pred=meas.unflatten_img(proj.audit_flat(x0K+proj.null_project_flat(d),yK)).float().clamp(0,1)
        r=float(torch.as_tensor(hq.full_rmse_torch(pred,tK)).mean())
        return -20*np.log10(max(r,1e-12)), float(np.mean(hq.lpips_batch(lp,pred,tK)))
    inr_only=rec(dINR); vqae_only=rec(dAK); vqgan_only=rec(dGK)
    def score(dS, ref):
        best=None
        for B in [round(b,2) for b in np.linspace(0,1,21)]:
            pred=meas.unflatten_img(proj.audit_flat(x0K+proj.null_project_flat(dS+B*(dGK-dS)),yK)).float().clamp(0,1)
            r=float(torch.as_tensor(hq.full_rmse_torch(pred,tK)).mean()); ps=-20*np.log10(max(r,1e-12))
            l=float(np.mean(hq.lpips_batch(lp,pred,tK)))
            if ps>=ref-0.5 and (best is None or l<best[2]): best=(B,ps,l)
        return best if best else (None,0,9)
    b_vqae=score(dAK, vqae_only[0]); b_inr=score(dINR, inr_only[0])
    log(f"K={K}:")
    log(f"  INR only:          PSNR={inr_only[0]:.2f} LPIPS={inr_only[1]:.3f}")
    log(f"  VQAE only:         PSNR={vqae_only[0]:.2f} LPIPS={vqae_only[1]:.3f}")
    log(f"  VQGAN only:        PSNR={vqgan_only[0]:.2f} LPIPS={vqgan_only[1]:.3f}")
    log(f"  VQAE+VQGAN (bal):  B={b_vqae[0]} PSNR={b_vqae[1]:.2f} LPIPS={b_vqae[2]:.3f}  (ref PSNR>= {vqae_only[0]-0.5:.2f})")
    log(f"  INR +VQGAN (bal):  B={b_inr[0]} PSNR={b_inr[1]:.2f} LPIPS={b_inr[2]:.3f}  (ref PSNR>= {inr_only[0]-0.5:.2f})")
    log(f"  --> INR-structure balanced LPIPS delta vs VQAE-structure: {b_inr[2]-b_vqae[2]:+.4f}")
    out={"K":K,"inr_only":inr_only,"vqae_only":vqae_only,"vqgan_only":vqgan_only,
         "vqae_vqgan_balanced":b_vqae,"inr_vqgan_balanced":b_inr,"inr_vs_vqae_balanced_lpips":b_inr[2]-b_vqae[2]}
    (vdf.BASE/"detail_fusion_paper"/"inr_vqgan_fusion.json").write_text(json.dumps(out,indent=2))
    log("wrote inr_vqgan_fusion.json")

if __name__=="__main__": main()
