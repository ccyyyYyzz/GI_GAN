"""Figure 1 — the impossibility-first three-panel: CANNOT / CAN / THEREFORE.
(a) CANNOT: a feasible-wrong twin — target x_i and a different scene u that reproduce the SAME bucket
    record to ~1e-13 on the 64x64 m=205 fusion operator (consistency != correctness).
(b) CAN: the ground-truth-free audit contracts the measurement residual by orders of magnitude at
    negligible PSNR cost (quality vs accountability separation).
(c) THEREFORE: the metered dial — LMMSE -> VQAE -> balanced -> VQGAN, A x_hat_B = y exact for every B.
Renders METHOD_FIG1.{png,pdf} into the detail_fusion_paper assets.
"""
from __future__ import annotations
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import gan_high_quality_gi as hq, vqgan_detail_fusion as vdf
from src.projections import get_exact_projector

C_BLUE="#3a6ea5"; C_RED="#c0392b"; C_GOLD="#e0a500"; INK="#1c1c1c"

def get_pair():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=vdf.load_cfg(0)
    rows_np,_=hq.build_structured_operator_rows(img_size=int(cfg["data"]["img_size"]),total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]),hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]),seed=int(cfg["operator"]["seed"]))
    meas=hq.make_measurement_operator(rows_np,img_size=int(cfg["data"]["img_size"]),device=dev,lambda_solver=float(cfg["operator"]["lambda_solver"]))
    proj=get_exact_projector(meas,dtype=torch.float64,device=dev)
    pack=vdf.load_pack(0,"dev",device=dev); truth=pack["truth"].to(dev)
    A=torch.as_tensor(rows_np,dtype=torch.float64,device=dev); flat=meas.flatten_img(truth).to(torch.float64); Y=flat@A.T
    rng=np.random.default_rng(7); best=None
    for _ in range(3000):
        i,j=int(rng.integers(0,truth.shape[0])),int(rng.integers(0,truth.shape[0]))
        if i==j or torch.mean((truth[i]-truth[j])**2).item()<8e-3: continue
        u=proj.audit_flat(flat[j:j+1],Y[i:i+1]); rel=(torch.norm(u@A.T-Y[i:i+1])/torch.norm(Y[i:i+1])).item()
        ui=u.reshape(truth[i:i+1].shape).float()
        best=(truth[i,0].cpu().numpy(), truth[j,0].cpu().numpy(), ui[0,0].cpu().numpy(), rel); break
    return best

xi,xj,u,rel = get_pair()
fig=plt.figure(figsize=(12,3.6))
gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1.15],wspace=0.28)

# ---- (a) CANNOT ----
axa=fig.add_subplot(gs[0]); axa.axis("off")
axa.set_title("(a)  Cannot certify null", fontsize=11, color=INK, loc="left", fontweight="bold")
def chip(ax,img,x,y,w,edge,lab):
    a=ax.inset_axes([x,y,w,w]); a.imshow(img,cmap="gray",vmin=0,vmax=1); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_color(edge); s.set_linewidth(2)
    a.set_title(lab,fontsize=8.5,color=edge)
chip(axa,xi,0.02,0.30,0.34,C_BLUE,r"true scene $x_i$")
chip(axa,u,0.02,-0.12,0.34,C_RED,r"feasible-wrong $u$")
axa.annotate("same bucket\n"+r"$A u = y_i$", xy=(0.42,0.36),xytext=(0.42,0.36),fontsize=9,ha="left",va="center")
axa.text(0.42,0.10,r"$\frac{\|A u - y_i\|}{\|y_i\|}\approx 10^{-13}$",fontsize=11,ha="left",color=C_RED)
axa.text(0.42,-0.14,r"(vs noise floor $\sim 10^{-3}$)",fontsize=7.5,ha="left",color="#666")
axa.text(0.0,-0.34,r"consistency $\neq$ correctness",fontsize=9.5,style="italic",color=INK)

# ---- (b) CAN ----
axb=fig.add_subplot(gs[1])
axb.set_title("(b)  Can audit row record", fontsize=11, color=INK, loc="left", fontweight="bold")
labels=["learned","BP","Tikhonov"]; pre=[3.68e-2,2.1e-2,1.5e-2]; post=[1.90e-6,3.0e-6,4.0e-6]
xpos=np.arange(len(labels))
axb.bar(xpos-0.18,pre,0.34,color="#bbb",label="pre-audit")
axb.bar(xpos+0.18,post,0.34,color=C_BLUE,label="post-audit")
axb.set_yscale("log"); axb.set_ylabel("RelMeasErr (log)",fontsize=9)
axb.set_xticks(xpos); axb.set_xticklabels(labels,fontsize=8.5); axb.legend(fontsize=7.5,loc="upper right")
axb.text(0.02,0.02,r"exact per-mode contraction $\frac{\lambda}{\lambda+\sigma_i^2}$",
         transform=axb.transAxes,fontsize=9,color=INK)
for sp in ("top","right"): axb.spines[sp].set_visible(False)

# ---- (c) THEREFORE ----
axc=fig.add_subplot(gs[2])
axc.set_title("(c)  Therefore govern prior detail", fontsize=11, color=INK, loc="left", fontweight="bold")
B=[0,0.55,0.72,1.0]; lp=[0.300,0.202,0.182,0.172]; names=["VQAE","balanced","q-lite","VQGAN"]
cols=[C_BLUE,C_GOLD,"#8a7", C_RED]
axc.plot(B,lp,"-",color="#999",zorder=1)
for b,l,nm,cc in zip(B,lp,names,cols):
    axc.scatter([b],[l],s=70,color=cc,zorder=3,edgecolor="white")
    axc.annotate(nm,(b,l),textcoords="offset points",xytext=(0,8),fontsize=8,color=cc,ha="center")
axc.set_xlabel(r"dial $B$ (null-space only)",fontsize=9); axc.set_ylabel("LPIPS $\downarrow$",fontsize=9)
axc.text(0.02,0.06,r"$A\hat{x}_B=y$ exactly for every $B$",transform=axc.transAxes,fontsize=9,color=INK)
for sp in ("top","right"): axc.spines[sp].set_visible(False)

fig.suptitle("Certify what you measure, govern what you cannot", fontsize=13, fontweight="bold", y=1.02)
out=vdf.BASE/"detail_fusion_paper"
for ext in ("png","pdf"): fig.savefig(out/f"METHOD_FIG1.{ext}",dpi=200,bbox_inches="tight")
plt.close(fig); print("wrote METHOD_FIG1.png/pdf ; feasible-wrong rel=%.2e"%rel)
