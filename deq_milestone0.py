"""Follow-up 2 (DEQ Milestone-0, no training): iterate the frozen VQGAN refiner as a null-space
fixed-point map. x_{k+1} = audit( x0 + P0( refine_VQGAN(x_k) - x0 ) ), starting x_0 = x0 (anchor).
k=1 reproduces the current one-step fusion x_G. Question: does iterating (k=2..K) beat one-step?
If not, that is a rigor result: our one-step null-space fusion is already at/near the fixed point.
"""
from __future__ import annotations
import json, numpy as np, torch, yaml
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
import anchor_initialized_vqgan_inversion as ai
from src.projections import get_exact_projector

def log(*a): vdf.log(*a)

def main():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=vdf.load_cfg(0)
    cfg["data"]["dataset_root"]="E:/GAN_FCC_WORK/datasets"      # relocated STL10
    sub=vdf.Substrate(cfg, dev)
    meas, proj = sub.measurement, sub.projector
    priors={ai.VQAE: ai.load_prior(ai.VQAE, vdf.ROOT/cfg["priors"]["vqae_checkpoint"], cfg, dev),
            ai.VQGAN: ai.load_prior(ai.VQGAN, vdf.ROOT/cfg["priors"]["vqgan_checkpoint"], cfg, dev)}
    refs={ai.VQGAN: ai.load_refiner_checkpoint(vdf.refiner_ckpt(0, ai.VQGAN), cfg, dev)}
    dt=float(cfg["training"].get("distance_temperature",1.0)); st=float(cfg["training"].get("soft_temperature",1.0))
    lp=hq.load_lpips(dev)
    loader=hq.build_loader(sub.dev_ds, batch_size=64, workers=0, shuffle=False, seed=int(cfg["seed"])+7, device=dev)
    x,_,_=next(iter(loader)); x=x.to(dev)                       # 64 truth images
    y=meas.A_forward(meas.flatten_img(x))
    x0=meas.unflatten_img(sub.lmmse.anchor(y, meas, device=dev))
    unc=sub.lmmse.uncertainty_map(img_size=sub.img, device=dev, batch_size=x.shape[0], dtype=x.dtype)
    x0f=meas.flatten_img(x0).double(); yf=y.double()

    def refine_vqgan(anchor):
        p=priors[ai.VQGAN]; z0=p.model.encode(anchor)
        dz,dl=refs[ai.VQGAN](anchor, unc, z0)
        logits=ai.logits_from_latent(z0+dz, p, distance_temperature=dt)+dl
        zq,_,_=ai.quantize_from_logits(p, logits, soft_temperature=st, straight_through=False)
        return p.model.decode_embeddings(zq)

    def score(img):
        img=img.float().clamp(0,1); r=float(torch.as_tensor(hq.full_rmse_torch(img,x)).mean())
        return -20*np.log10(max(r,1e-12)), float(np.mean(hq.lpips_batch(lp,img,x)))

    cur=x0.clone(); rows={}
    with torch.no_grad():
        for k in range(1,21):
            raw=refine_vqgan(cur)
            df=proj.null_project_flat(meas.flatten_img(raw).double()-x0f)
            cur=meas.unflatten_img(proj.audit_flat(x0f+df, yf)).float()
            if k in (1,2,3,5,10,20):
                ps,l=score(cur); rows[k]=(ps,l)
                log(f"  DEQ iter k={k:2d}: PSNR={ps:.2f} LPIPS={l:.3f}")
    b=rows[1]; best_k=min(rows, key=lambda kk: rows[kk][1])
    log(f"one-step (k=1): PSNR={b[0]:.2f} LPIPS={b[1]:.3f} | best k={best_k} LPIPS={rows[best_k][1]:.3f} "
        f"(delta {rows[best_k][1]-b[1]:+.4f})")
    (vdf.BASE/"detail_fusion_paper"/"deq_milestone0.json").write_text(json.dumps({str(k):v for k,v in rows.items()},indent=2))
    log("wrote deq_milestone0.json")

if __name__=="__main__": main()
