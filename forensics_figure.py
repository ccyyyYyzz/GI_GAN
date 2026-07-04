"""Cross-target saturation figure — the forensic case study's headline.

Three published DL-GI pipelines (pretrained+fine-tune / untrained DIP / self-supervised), each on its own
released or measured operator, all terminate in the same place: the row (certifiable) ledger is repaired,
then exhausted, and the headline PSNR is paid 3-8 dB above the operator's range ceiling from null-space
(uncertifiable) content, with terminal error 93-97% null.

Panels: (a) PEDL fine-tuning trajectory (the exemplar: row repair -> null-limited plateau vs its ceiling);
(b) MSE-improvement attribution, row vs null, per target; (c) headline dB above own range ceiling +
terminal null share.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"E:\ns_mc_gan_gi_code_fcc_phase1")
OUT = REPO / "outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper"

pedl = json.loads((OUT / "forensics_pedl_stl10.json").read_text())
gidb = json.loads((OUT / "forensics_gidc_partB.json").read_text())
n2g = json.loads((OUT / "forensics_n2g.json").read_text())
traj = np.load(OUT / "forensics_pedl_trajectory.npy")   # [300, 7]: psnr,mse_row,mse_null,null_share,align,halluc,consist

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

# ---- (a) PEDL trajectory ----
ax = axes[0]
steps = np.arange(traj.shape[0])
ceil = pedl["stage2_trajectory"]["summary"]["range_ceiling_psnr_minnorm"]
ax.plot(steps, traj[:, 0], color="#4c72b0", lw=1.8, label="PSNR")
ax.axhline(ceil, color="#c44e52", ls="--", lw=1.4, label=f"range ceiling ({ceil:.1f} dB)")
ax.fill_between(steps, ceil, traj[:, 0], where=traj[:, 0] > ceil, color="#c44e52", alpha=0.12)
ax.set_xlabel("fine-tuning step"); ax.set_ylabel("PSNR (dB)")
ax.annotate("paid from the\nnull ledger", xy=(220, (ceil + traj[220, 0]) / 2), fontsize=8.5,
            ha="center", color="#c44e52")
ax2 = ax.twinx()
ax2.plot(steps, 100 * traj[:, 3], color="#55a868", lw=1.2, alpha=0.85)
ax2.set_ylabel("null share of error (%)", color="#55a868"); ax2.tick_params(axis="y", colors="#55a868")
ax2.set_ylim(0, 100)
ax.set_title("(a) PEDL: their fine-tuning, decomposed")
ax.legend(fontsize=8, loc="lower right")

# ---- (b) gain attribution ----
ax = axes[1]
def attribution(s0_row, s0_null, s1_row, s1_null):
    tot = (s0_row + s0_null) - (s1_row + s1_null)
    return (s0_row - s1_row) / tot, (s0_null - s1_null) / tot
p0 = pedl["stage2_trajectory"]["key_steps"]["0"]; pE = pedl["stage2_trajectory"]["key_steps"]["299"]
g0 = gidb["GIDC_step0"]; gE = gidb["GIDC_step200"]
nL = n2g["decompositions"]["LS"]; nE = n2g["decompositions"]["N2G"]
bars = [("PEDL\n(fine-tune)", *attribution(p0["mse_row"], p0["mse_null"], pE["mse_row"], pE["mse_null"])),
        ("GIDC Part B\n(DIP steps)", *attribution(g0["mse_row"], g0["mse_null"], gE["mse_row"], gE["mse_null"])),
        ("Noise2Ghost\n(vs LS)", *attribution(nL["mse_row"], nL["mse_null"], nE["mse_row"], nE["mse_null"]))]
ys = np.arange(len(bars))
row_sh = [b[1] for b in bars]; null_sh = [b[2] for b in bars]
ax.barh(ys, row_sh, color="#4c72b0", label="row (certifiable) repair")
ax.barh(ys, null_sh, left=row_sh, color="#c44e52", label="null (uncertifiable) injection")
for y, (nm, r, nl) in zip(ys, bars):
    ax.text(r / 2, y, f"{100*r:.0f}%", ha="center", va="center", color="w", fontsize=9, fontweight="bold")
    ax.text(r + nl / 2, y, f"{100*nl:.0f}%", ha="center", va="center", color="w", fontsize=9, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_xlabel("share of MSE improvement"); ax.set_xlim(0, 1)
ax.set_title("(b) Where each method's gain lives")
ax.legend(fontsize=8, loc="lower right")

# ---- (c) headline vs ceiling ----
ax = axes[2]
items = [("PEDL", pE["psnr"] - ceil, pE["null_share"]),
         ("GIDC\nPart B", gE["psnr"] - gidb["range_ceiling_psnr"], gE["null_share_of_error"]),
         ("Noise2\nGhost", nE["psnr"] - n2g["range_ceilings"]["psnr_Aplus_y"], nE["null_share_of_error"])]
xs = np.arange(len(items))
b = ax.bar(xs, [i[1] for i in items], color="#c44e52", width=0.55)
for xi, (nm, db, nsh) in zip(xs, items):
    ax.text(xi, db + 0.12, f"+{db:.1f} dB", ha="center", fontsize=9, fontweight="bold")
    ax.text(xi, 0.25, f"terminal error\n{100*nsh:.0f}% null", ha="center", fontsize=7.5, color="w")
ax.set_xticks(xs); ax.set_xticklabels([i[0] for i in items], fontsize=8.5)
ax.set_ylabel("headline PSNR above own range ceiling (dB)")
ax.set_title("(c) The uncertifiable margin")
ax.axhline(0, color="k", lw=0.8)

for a in axes:
    a.spines["top"].set_visible(False)
    if a is not axes[0]: a.spines["right"].set_visible(False)
fig.suptitle("Projector forensics of published DL-GI pipelines, on their own operators: "
             "gains saturate the certifiable ledger and are paid from the null space", fontsize=10.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"FORENSICS_CROSS_TARGET.{ext}", dpi=200, bbox_inches="tight")
    fig.savefig(REPO / "paper" / f"FORENSICS_CROSS_TARGET.{ext}", dpi=200, bbox_inches="tight")
print("wrote FORENSICS_CROSS_TARGET.png/pdf (outputs + paper/)")
