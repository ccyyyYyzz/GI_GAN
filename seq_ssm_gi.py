"""Follow-up 3 (Prototype C): the sequence-native angle. Read the raw 1-D bucket stream y_t, each step
tagged with a LEARNED embedding of its operator row a_t, with a diagonal state-space model (S4D-style,
the liquid/SSM family, pure-PyTorch scan -> no CUDA-kernel friction), readout to a 64x64 image, then
project onto {x: A x = y} with the exact projector. Trained from scratch on STL10; a PoC of the
sequence+projector wedge (honest: small net, expect modest quality, novelty is the framing).
"""
from __future__ import annotations
import json, numpy as np, torch, torch.nn as nn
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
from src.projections import get_exact_projector

def log(*a): vdf.log(*a)

class DiagSSM(nn.Module):
    """Minimal S4D-style diagonal real SSM over the length-m bucket sequence."""
    def __init__(self, m, emb=32, H=192, npix=4096):
        super().__init__()
        self.row_emb=nn.Parameter(0.02*torch.randn(m, emb))      # per-operator-row learned embedding
        self.inp=nn.Linear(emb+1, H)
        self.logA=nn.Parameter(torch.log(torch.linspace(0.5,0.99,H)))  # decay per channel in (0,1)
        self.B=nn.Parameter(0.1*torch.randn(H)); self.C=nn.Parameter(0.1*torch.randn(H))
        self.readout=nn.Sequential(nn.Linear(H,512), nn.GELU(), nn.Linear(512,npix))
    def forward(self, y):                                        # y: (batch, m)
        b, m = y.shape
        u=torch.cat([self.row_emb.unsqueeze(0).expand(b,-1,-1), y.unsqueeze(-1)], -1)  # (b,m,emb+1)
        u=self.inp(u)                                            # (b,m,H)
        a=torch.sigmoid(self.logA)                               # (H,) decay
        h=torch.zeros(b, u.shape[-1], device=y.device); pooled=0.0
        for t in range(m):
            h=a*h + self.B*u[:,t,:]
            pooled=pooled + (self.C*h)
        img=self.readout(pooled/m).reshape(b,1,64,64)
        return torch.sigmoid(img)

def main():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=vdf.load_cfg(0); cfg["data"]["dataset_root"]="E:/GAN_FCC_WORK/datasets"
    sub=vdf.Substrate(cfg, dev); meas, proj = sub.measurement, sub.projector
    lp=hq.load_lpips(dev)
    tr=hq.build_loader(sub.train_ds, batch_size=64, workers=0, shuffle=True, seed=0, device=dev)
    dvl=hq.build_loader(sub.dev_ds, batch_size=64, workers=0, shuffle=False, seed=7, device=dev)
    m=int(cfg["operator"]["total_m"])
    net=DiagSSM(m).to(dev); opt=torch.optim.Adam(net.parameters(), lr=3e-4)
    log(f"training DiagSSM (m={m}) ...")
    it=iter(tr); STEPS=1500
    for step in range(STEPS):
        try: x,_,_=next(it)
        except StopIteration: it=iter(tr); x,_,_=next(it)
        x=x.to(dev); y=meas.A_forward(meas.flatten_img(x))
        pred=net(y)
        loss=((pred-x)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step%300==0: log(f"  step {step}: mse={loss.item():.4f}")
    # eval on dev (project onto {Ax=y})
    net.eval(); ps_raw=[]; lp_raw=[]; ps_pr=[]; lp_pr=[]
    with torch.no_grad():
        for x,_,_ in dvl:
            x=x.to(dev); y=meas.A_forward(meas.flatten_img(x)); pred=net(y)
            projd=meas.unflatten_img(proj.audit_flat(meas.flatten_img(pred).double(), y.double())).float().clamp(0,1)
            for a,b,store_p,store_l in [(pred,x,ps_raw,lp_raw),(projd,x,ps_pr,lp_pr)]:
                r=float(torch.as_tensor(hq.full_rmse_torch(a.clamp(0,1),b)).mean()); store_p.append(-20*np.log10(max(r,1e-12)))
                store_l.append(float(np.mean(hq.lpips_batch(lp,a.clamp(0,1),b))))
            break  # one dev batch (64) is enough for a PoC number
    summ={"ssm_raw":{"psnr":float(np.mean(ps_raw)),"lpips":float(np.mean(lp_raw))},
          "ssm_projected":{"psnr":float(np.mean(ps_pr)),"lpips":float(np.mean(lp_pr))}}
    log(f"  SSM raw:       PSNR={summ['ssm_raw']['psnr']:.2f} LPIPS={summ['ssm_raw']['lpips']:.3f}")
    log(f"  SSM+projector: PSNR={summ['ssm_projected']['psnr']:.2f} LPIPS={summ['ssm_projected']['lpips']:.3f}")
    (vdf.BASE/"detail_fusion_paper"/"seq_ssm_gi.json").write_text(json.dumps(summ,indent=2))
    log("wrote seq_ssm_gi.json")

if __name__=="__main__": main()
