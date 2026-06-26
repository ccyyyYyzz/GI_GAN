"""Paper assembly: audit + facts + main table + reproducibility manifest for the
LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED result. No new experiments — assembly only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import vqgan_detail_fusion as vdf
import anchor_initialized_vqgan_inversion as ai

ROOT = vdf.ROOT
BASE = vdf.BASE
DEV = BASE / "detail_fusion"
LOCK = BASE / "detail_fusion_locked"
PAPER = BASE / "detail_fusion_paper"
SEEDS = [0, 1, 2]
TABLE_METRICS = ["lpips", "psnr", "full_rmse", "ssim", "rapsd", "relmeaserr"]
ARM_ORDER = ["lmmse_anchor", "vqae", "fusion_balanced", "fusion_quality_lite", "vqgan"]
ARM_LABEL = {"lmmse_anchor": "LMMSE anchor", "vqae": "VQAE", "fusion_balanced": "Balanced fusion",
             "fusion_quality_lite": "Quality-lite fusion", "vqgan": "Full VQGAN"}
LOWER_BETTER = {"lpips", "full_rmse", "rapsd", "relmeaserr"}


def sha256_file(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled(csv_path: Path, metrics):
    by = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for m in metrics:
                try:
                    by[r["method"]][m].append(float(r[m]))
                except (TypeError, ValueError):
                    pass
    return {a: {m: float(np.mean(v)) for m, v in d.items()} for a, d in by.items()}


def run():
    PAPER.mkdir(parents=True, exist_ok=True)
    blockers = []

    # ---- read frozen artifacts ----
    gate = json.loads((LOCK / "LOCKED_GATE_REPORT.json").read_text())
    prereg = json.loads((LOCK / "PREREGISTRATION.json").read_text())
    lman = json.loads((LOCK / "reports" / "locked_split_manifest.json").read_text())
    devgate = json.loads((DEV / "MULTISEED_FUSION_GATE_REPORT.json").read_text())

    # ---- audit checks ----
    audit = {}
    audit["locked_hash_audit_pass"] = bool(lman["dedup_audit"]["hash_audit_pass"])
    audit["locked_overlap_with_consumed"] = int(lman["dedup_audit"]["overlap_with_consumed"])
    audit["locked_n"] = int(lman["n_images"])
    audit["consumed_union_size"] = int(lman["consumed_provenance"]["union_size"])
    audit["consumed_union_sha256"] = lman["consumed_provenance"]["union_sha256"]
    audit["locked_split_source_indices_sha256"] = lman["source_indices_sha256"]
    audit["classification"] = gate["classification"]
    audit["gate_conditions_balanced"] = gate["conditions"]["fusion_balanced"]
    audit["balanced_conditions_passed"] = sum(1 for v in gate["conditions"]["fusion_balanced"].values() if v)
    audit["balanced_conditions_total"] = len(gate["conditions"]["fusion_balanced"])
    audit["relmeaserr_max_balanced"] = gate["arms"]["fusion_balanced"]["method_relmeaserr_max"]

    # B provenance: must come from val selection JSONs, exact per seed
    bal, ql = {}, {}
    for s in SEEDS:
        selp = DEV / "canary" / f"seed{s}_selection.json"
        sel = json.loads(selp.read_text())["selected_specs"]
        bal[s] = float(sel["balanced"][1][0])
        ql[s] = float(sel["quality_lite"][1][0])
    audit["B_source"] = "val-frozen development selection (detail_fusion/canary/seed*_selection.json)"
    audit["frozen_B_balanced"] = bal
    audit["frozen_B_quality_lite"] = ql
    if {int(k): v for k, v in gate["frozen_B_balanced"].items()} != bal:
        blockers.append("frozen_B_balanced in gate report disagrees with val selection JSONs")

    # operator identity across seeds
    op_sha = set()
    for s in SEEDS:
        m = json.loads((BASE / f"anchor_multiseed_hashclean_seed{s}/reports/operator_manifest.json").read_text())
        op_sha.add(m.get("rows_sha256"))
    audit["operator_rows_sha256"] = sorted(op_sha)
    if len(op_sha) != 1:
        blockers.append(f"operator rows_sha256 not identical across seeds: {op_sha}")

    # ---- artifact hashes ----
    ckpt_hashes = {}
    for s in SEEDS:
        cfg = vdf.load_cfg(s)
        files = {
            "vqae_prior": ROOT / cfg["priors"]["vqae_checkpoint"],
            "vqgan_prior": ROOT / cfg["priors"]["vqgan_checkpoint"],
            "vqae_refiner": vdf.refiner_ckpt(s, ai.VQAE),
            "vqgan_refiner": vdf.refiner_ckpt(s, ai.VQGAN),
        }
        ckpt_hashes[f"seed{s}"] = {}
        for k, p in files.items():
            h = sha256_file(Path(p))
            ckpt_hashes[f"seed{s}"][k] = h
            if h == "MISSING":
                blockers.append(f"checkpoint missing: seed{s}/{k} -> {p}")
    script_hashes = {p.name: sha256_file(ROOT / p.name) for p in [
        Path("vqgan_detail_fusion.py"), Path("vqgan_detail_fusion_locked.py"),
        Path("vqgan_detail_fusion_locked_figs.py")]}
    package_hashes = {
        "locked_package": sha256_file(LOCK / "VQGAN_DETAIL_FUSION_LOCKED_CONFIRMATION_PACKAGE.zip"),
        "dev_package": sha256_file(DEV / "VQGAN_DETAIL_FUSION_CONFIRMATION_PACKAGE.zip"),
    }

    # ---- pooled absolute method means (LOCKED + dev) ----
    locked_means = pooled(LOCK / "locked_per_image_rows.csv", TABLE_METRICS)
    dev_means = pooled(DEV / "canary" / "seed0_dev_rows.csv", TABLE_METRICS)  # seed0 dev arms (no lmmse_anchor)
    # n rows sanity
    with open(LOCK / "locked_per_image_rows.csv") as f:
        n_locked_rows = sum(1 for _ in f) - 1
    audit["locked_per_image_rows"] = n_locked_rows
    if n_locked_rows != audit["locked_n"] * len(SEEDS) * len(ARM_ORDER):
        blockers.append(f"locked per-image row count {n_locked_rows} != 512*3*5")

    # ---- FACTS ----
    bal_arm = gate["arms"]["fusion_balanced"]
    facts = {
        "classification": gate["classification"],
        "audit": audit,
        "blockers": blockers,
        "frozen_B": {"balanced": bal, "quality_lite": ql, "source": audit["B_source"]},
        "locked_balanced_vs_vqae": {
            "delta_lpips": bal_arm["metrics"]["lpips"]["mean"],
            "lpips_ci": [bal_arm["metrics"]["lpips"]["ci_low"], bal_arm["metrics"]["lpips"]["ci_high"]],
            "lpips_relative_gain": bal_arm["lpips_relative_gain"],
            "delta_psnr": bal_arm["metrics"]["psnr"]["mean"],
            "delta_full_rmse": bal_arm["metrics"]["full_rmse"]["mean"],
            "delta_ssim": bal_arm["metrics"]["ssim"]["mean"],
            "delta_rapsd": bal_arm["metrics"]["rapsd"]["mean"],
            "relmeaserr_mean": bal_arm["method_relmeaserr_mean"],
            "relmeaserr_max": bal_arm["method_relmeaserr_max"],
            "seeds_same_direction": bal_arm["lpips_same_direction_neg"],
            "per_seed_lpips": bal_arm["per_seed"]["lpips"],
        },
        "dev_balanced_vs_vqae": {
            "delta_lpips": devgate["arms"]["fusion_balanced"]["metrics"]["lpips"]["mean_delta"],
            "lpips_relative_gain": devgate["arms"]["fusion_balanced"]["lpips_relative_gain"],
            "delta_psnr": devgate["arms"]["fusion_balanced"]["metrics"]["psnr"]["mean_delta"],
        },
        "locked_method_means": locked_means,
        "dev_method_means": dev_means,
        "ckpt_hashes": ckpt_hashes,
        "script_hashes": script_hashes,
        "package_hashes": package_hashes,
        "locked_split": {"n": audit["locked_n"], "source_indices_sha256": audit["locked_split_source_indices_sha256"],
                         "consumed_union_size": audit["consumed_union_size"]},
    }
    (PAPER / "FACTS.json").write_text(json.dumps(facts, indent=2))

    # ---- MAIN_TABLE (locked absolute means) ----
    rows = []
    for a in ARM_ORDER:
        m = locked_means.get(a, {})
        rows.append({"method": ARM_LABEL[a], **{k: m.get(k, float("nan")) for k in TABLE_METRICS}})
    with open(PAPER / "MAIN_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + TABLE_METRICS)
        w.writeheader()
        for r in rows:
            w.writerow({"method": r["method"], **{k: f"{r[k]:.6f}" for k in TABLE_METRICS}})

    # best per column (for bolding in LaTeX)
    best = {}
    for k in TABLE_METRICS:
        vals = {a: locked_means.get(a, {}).get(k, np.nan) for a in ARM_ORDER}
        best[k] = (min if k in LOWER_BETTER else max)(vals, key=lambda a: vals[a])
    hdr = {"lpips": "LPIPS$\\downarrow$", "psnr": "PSNR$\\uparrow$", "full_rmse": "RMSE$\\downarrow$",
           "ssim": "SSIM$\\uparrow$", "rapsd": "RAPSD$\\downarrow$", "relmeaserr": "RelMeasErr$\\downarrow$"}
    tex = [r"\begin{tabular}{l" + "c" * len(TABLE_METRICS) + "}", r"\toprule",
           "Method & " + " & ".join(hdr[k] for k in TABLE_METRICS) + r" \\", r"\midrule"]
    for a in ARM_ORDER:
        m = locked_means.get(a, {})
        cells = []
        for k in TABLE_METRICS:
            v = m.get(k, float("nan"))
            s = f"{v:.2e}" if k == "relmeaserr" else f"{v:.4f}"
            cells.append(f"\\textbf{{{s}}}" if best[k] == a else s)
        tex.append(f"{ARM_LABEL[a]} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "MAIN_TABLE.tex").write_text("\n".join(tex))

    # ---- REPRODUCIBILITY_MANIFEST ----
    repro = {
        "result": gate["classification"],
        "method": "x_hat = x0 + P0( d_A + B (d_G - d_A) ); d_A=P0(x_A-x0), d_G=P0(x_G-x0); exact measurement audit",
        "operator": {"rows_sha256": audit["operator_rows_sha256"][0], "total_m": 205, "img_size": 64,
                     "seed": 772001, "lambda_solver": 1e-6, "lmmse_lambda": 1e-3, "n": 4096},
        "frozen_B": {"balanced": bal, "quality_lite": ql, "source": audit["B_source"]},
        "splits": {"hash_clean": True, "train": 20000, "val": 512, "dev": 512, "locked": audit["locked_n"],
                   "locked_source_indices_sha256": audit["locked_split_source_indices_sha256"],
                   "consumed_union_size": audit["consumed_union_size"],
                   "consumed_union_sha256": audit["consumed_union_sha256"]},
        "checkpoints_sha256": ckpt_hashes,
        "scripts_sha256": script_hashes,
        "packages_sha256": package_hashes,
        "scoring": {"per_image_seed_averaged_delta": True, "image_bootstrap_reps": 2000, "bootstrap_seed": 20260626,
                    "one_shot": True},
        "environment": {"python": "3.11", "torch": "2.2.1+cu121", "device": "RTX 4060 Laptop",
                        "interpreter": "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe"},
        "entrypoints": ["vqgan_detail_fusion.py (regen|canary|gate)",
                        "vqgan_detail_fusion_locked.py (build|regen|score|report)",
                        "vqgan_detail_fusion_locked_figs.py"],
    }
    (PAPER / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(repro, indent=2))

    # ---- console audit summary ----
    print("=== AUDIT ===")
    print(f"classification: {audit['classification']}")
    print(f"locked hash_audit_pass={audit['locked_hash_audit_pass']} overlap_with_consumed={audit['locked_overlap_with_consumed']} n={audit['locked_n']}")
    print(f"consumed union size={audit['consumed_union_size']}")
    print(f"balanced conditions: {audit['balanced_conditions_passed']}/{audit['balanced_conditions_total']}")
    print(f"relmeaserr max (balanced)={audit['relmeaserr_max_balanced']:.2e}")
    print(f"frozen B balanced={bal} quality_lite={ql}")
    print(f"operator rows_sha256 unique across seeds: {len(op_sha)==1} ({audit['operator_rows_sha256']})")
    print(f"locked per-image rows={n_locked_rows} (expected {512*3*5})")
    print("\n=== LOCKED method means ===")
    for a in ARM_ORDER:
        m = locked_means.get(a, {})
        print(f"  {ARM_LABEL[a]:<22} " + " ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in TABLE_METRICS))
    print(f"\nBLOCKERS: {blockers if blockers else 'NONE'}")
    print("wrote FACTS.json, MAIN_TABLE.csv, MAIN_TABLE.tex, REPRODUCIBILITY_MANIFEST.json")


if __name__ == "__main__":
    run()
