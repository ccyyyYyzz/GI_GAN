"""Structure-Detail FCC diagnostic.

A variant of the FCC row-null diagnostic on the *already-confirmed* VQAE/VQGAN
detail-fusion data (BALANCED_VQGAN_FUSION_CONFIRMED). Instead of (r=P_R x, n=P_0 x)
of a natural image, the pair is:

    structure  s = x_A          (VQAE refiner reconstruction)
    detail     d = P0(x_G - x_A) = d_G - d_A   (the GAN null-space detail the
               fusion dial moves along)

Real pair (s_i, d_i); mismatch (s_i, d_j). Because d is EXACTLY in null(A),
A(s_i + d_j) = A(s_i + d_i) for all j -> every counterfactual is measurement
equivalent (exact feasibility), exactly as in row-null FCC.

Question (diagnostic, NOT reconstruction): does the GAN-generated detail d carry
a learnable, structure-conditioned compatibility that EXCEEDS deployable nuisance
baselines? This is texture-structure compatibility -- plausibly more hopeful than
random row-null FCC, because d is generator output, not the null of a natural image.

Data: detail_fusion/cache/seed{0,1,2}_{val,dev}.pt (x0,x_A,x_G,y,source_index,...),
operator = structured 5% (m=205, rows_sha256 8a16664e...). Reuses the FCC critic /
deployable controls / classifier from fcc_diagnostic_canary + src.fcc_canary.

Splits per fusion seed (firewalled by STRUCTURE source-index, single seed -> no
detail-realization confound):
    train = val-split structures (512)
    val   = dev-split structures[:128]   (checkpoint selection)
    dev   = dev-split structures[128:]   (scoring, 384)

Run:  python structure_detail_fcc.py --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from src.projections import get_exact_projector
from src.compatibility_data import SplitComponents, compute_train_normalization, tensor_sha256, save_json, write_csv
from src.phase1_1_controls import nuisance_balanced_derangement, random_derangement
from src import fcc_canary as fc
import fcc_diagnostic_canary as drv

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/compatibility/measurement_conditioned_vqgan"
CACHE = BASE / "detail_fusion/cache"
OUT_ROOT = ROOT / "outputs/compatibility/structure_detail_fcc"
FROZEN_ROWS_SHA256 = "8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628"


def build_operator(device: torch.device):
    cfg = yaml.safe_load((BASE / "anchor_multiseed_hashclean_seed0/config_used.yaml").read_text())
    img = int(cfg["data"]["img_size"])
    rows_np, op_meta = hq.build_structured_operator_rows(
        img_size=img, total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]), hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]), seed=int(cfg["operator"]["seed"]),
    )
    measurement = hq.make_measurement_operator(rows_np, img_size=img, device=device,
                                               lambda_solver=float(cfg["operator"]["lambda_solver"]))
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    return measurement, projector, op_meta, img


@torch.no_grad()
def make_components(name: str, pack: dict, sel: np.ndarray, measurement, projector, device) -> SplitComponents:
    """s = x_A, d = P0(x_G - x_A); exact float64 null projection of the detail."""
    sel = np.asarray(sel, dtype=int)
    xA = pack["x_A"][sel].to(device)
    xG = pack["x_G"][sel].to(device)
    truth = pack["truth"][sel].float().cpu()
    y = pack["y"][sel].float().cpu()
    s_flat = measurement.flatten_img(xA).double()
    d_flat = projector.null_project_flat(measurement.flatten_img(xG - xA).double())
    return SplitComponents(
        name=name,
        x=truth,                                  # natural image, for dup detection / reference
        r=s_flat.cpu().float(),                   # structure
        n=d_flat.cpu().float(),                   # GAN detail (exact null)
        y=y,
        labels=pack["label"][sel].long(),
        source_indices=pack["source_index"][sel].long(),
        projector_info=projector.info_dict(),
    )


@torch.no_grad()
def sd_geometry(comp: SplitComponents, measurement, projector, device, count=64) -> dict[str, Any]:
    c = min(count, comp.size)
    s = comp.r[:c].to(device).double()
    d = comp.n[:c].to(device).double()
    y = comp.y[:c].to(device).double()
    denom = torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    detail_null_rel = torch.linalg.norm(projector.A_forward(d), dim=1) / denom   # A d ~ 0 (exact, float64)
    struct_cons_rel = torch.linalg.norm(projector.A_forward(s) - y, dim=1) / denom  # A s ~ y (refiner fidelity)
    return {
        "sample_count": int(c),
        "float64_A_P0_rel_max": float(detail_null_rel.max().item()),    # detail null-ness (exact)
        "reconstruction_rel_max": float(struct_cons_rel.max().item()),  # structure consistency (cached recon)
        "detail_is_exact_null": bool(detail_null_rel.max() < 1e-6),
        "null_tol": 1e-6,  # cached recons are float32; ~1e-9 achieved, 1e-6 is the realistic pass floor
        "struct_consistency_note": "A x_A != y (~17%): the cached VQAE recon is the raw refiner output, not bucket-audited. Irrelevant to FCC feasibility, which only needs the swapped detail to be null.",
        "pass": bool(detail_null_rel.max() < 1e-6),  # feasibility hinges on detail being null; structure fidelity is informational
    }


@torch.no_grad()
def sd_feasibility(comp: SplitComponents, measurement, projector, donors: np.ndarray, device, max_pairs=256) -> dict[str, Any]:
    """Counterfactual s_i + d_j is measurement-equivalent to s_i + d_i iff A d_j = A d_i (both ~0).
    We report the exact detail null-ness of the swapped detail."""
    donors = np.asarray(donors, dtype=int)
    c = min(comp.size, max_pairs)
    d_swap = comp.n[torch.as_tensor(donors[:c], dtype=torch.long)].to(device).double()
    s = comp.r[:c].to(device).double()
    y = comp.y[:c].to(device).double()
    denom = torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    detail_null_rel = torch.linalg.norm(projector.A_forward(d_swap), dim=1) / denom
    # measurement equivalence of counterfactual vs true pair: A(s+d_j) - A(s+d_i) = A d_j - A d_i
    d_self = comp.n[:c].to(device).double()
    equiv_rel = torch.linalg.norm(projector.A_forward(d_swap) - projector.A_forward(d_self), dim=1) / denom
    return {
        "pairs_checked": int(c),
        "u_rel_max": float(equiv_rel.max().item()),            # measurement-equivalence true<->counterfactual
        "donor_null_rel_max": float(detail_null_rel.max().item()),
        "pass_float32_proxy": bool(detail_null_rel.max() < 1e-6 and equiv_rel.max() < 1e-6),
    }


def build_seed(fusion_seed: int, cfg: dict, device: torch.device) -> dict[str, Any]:
    measurement, projector, op_meta, img = build_operator(device)
    a_sha = tensor_sha256(measurement.A)
    rows_match = bool(op_meta.get("rows_sha256") == FROZEN_ROWS_SHA256)
    out = OUT_ROOT / f"seed{fusion_seed}"
    (out / "reports").mkdir(parents=True, exist_ok=True)

    val_pack = torch.load(CACHE / f"seed{fusion_seed}_val.pt", map_location="cpu", weights_only=False)
    dev_pack = torch.load(CACHE / f"seed{fusion_seed}_dev.pt", map_location="cpu", weights_only=False)

    # Partition by structure source-index (deterministic). train=val-split; val/dev split the dev-split structures.
    dev_order = np.argsort(dev_pack["source_index"].numpy())
    train_sel = np.arange(val_pack["x_A"].shape[0])
    val_sel = dev_order[:128]
    dev_sel = dev_order[128:]

    splits = {
        "train": make_components("train", val_pack, train_sel, measurement, projector, device),
        "val": make_components("val", dev_pack, val_sel, measurement, projector, device),
        "dev": make_components("dev", dev_pack, dev_sel, measurement, projector, device),
    }

    # firewall: structure source-index disjoint across splits
    sidx = {k: set(int(i) for i in v.source_indices.tolist()) for k, v in splits.items()}
    overlap = {
        "train__val": len(sidx["train"] & sidx["val"]),
        "train__dev": len(sidx["train"] & sidx["dev"]),
        "val__dev": len(sidx["val"] & sidx["dev"]),
    }

    geom = sd_geometry(splits["train"], measurement, projector, device)
    rseed = int(cfg.get("seed", 0))
    rand_d = random_derangement(splits["train"].size, seed=rseed + 11)
    feas_rand = sd_feasibility(splits["train"], measurement, projector, rand_d, device)
    bal_d, bal_rep = nuisance_balanced_derangement(splits["train"], seed=rseed + 12)
    feas_bal = sd_feasibility(splits["train"], measurement, projector, bal_d, device)

    audit_rows = []
    for name, comp in splits.items():
        for k in range(comp.size):
            audit_rows.append({"split": name, "source_index": int(comp.source_indices[k].item()),
                               "truth_sha256": tensor_sha256(comp.x[k])})
    write_csv(out / "reports" / "sample_hash_audit.csv", audit_rows)

    manifest = {
        "phase": "structure_detail_fcc_build", "fusion_seed": fusion_seed, "created_at": drv.now_iso(),
        "git_commit": drv.git_commit(), "img_size": img,
        "pairing": {"structure": "x_A (VQAE recon)", "detail": "P0(x_G - x_A) = d_G - d_A (GAN null-space detail)"},
        "operator": {"a_sha256": a_sha, "m": int(measurement.m), "n": int(measurement.n),
                     "pattern_type": "structured_dct_hadamard_random_5pct", "matrix_normalization": "structured",
                     "rows_sha256": op_meta.get("rows_sha256"), "rows_sha256_matches_frozen": rows_match},
        "splits": {k: {"count": v.size, "source_index_min": int(min(sidx[k])), "source_index_max": int(max(sidx[k])),
                       "source_indices_sha256": tensor_sha256(v.source_indices)} for k, v in splits.items()},
        "cross_split_structure_source_index_overlap": overlap,
        "geometry_checks": geom,
        "feasibility": {"random": feas_rand, "nuisance_balanced": feas_bal, "balance_report": bal_rep},
        "data_provenance": "Reuses BALANCED_VQGAN_FUSION_CONFIRMED detail_fusion cache (hash-clean val/dev). "
                           "Authorized development reuse; NOT a locked test.",
    }
    save_json(out / "reports" / "build_manifest.json", manifest)

    cache_dir = out / "counterfactual_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, comp in splits.items():
        torch.save(asdict(comp), cache_dir / f"{name}_components.pt")
    save_json(out / "normalization_train_only.json", compute_train_normalization(splits["train"]))

    print(f"[sd-build seed{fusion_seed}] rows_match={rows_match} m={measurement.m} "
          f"geom detail_null={geom['float64_A_P0_rel_max']:.2e} struct_cons={geom['reconstruction_rel_max']:.2e} "
          f"feas_rand={feas_rand['pass_float32_proxy']} feas_bal={feas_bal['pass_float32_proxy']} "
          f"balance_smd_max={bal_rep['feature_smd_max']:.3f} overlap={overlap}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Structure-Detail FCC diagnostic on VQAE/VQGAN fusion data.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="configs/compatibility/structure_detail_fcc.yaml")
    args = ap.parse_args()
    device = drv.resolve_device(args.device)
    base_cfg = drv.load_config(args.config)

    results = {}
    for fs in args.seeds:
        cfg = dict(base_cfg)
        cfg["output_dir"] = str(OUT_ROOT / f"seed{fs}")
        cfg["device"] = args.device
        out = drv.out_dir(cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        with (out / "config_used.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"\n===== Structure-Detail FCC: fusion seed {fs} =====")
        build_seed(fs, cfg, device)
        drv.cmd_train(cfg, device)
        drv.cmd_eval(cfg, device)
        res = drv.cmd_classify(cfg, device)
        results[f"seed{fs}"] = res["classification"]

    print("\n===== Structure-Detail FCC summary =====")
    for k, v in results.items():
        print(f"  {k}: {v}")
    save_json(OUT_ROOT / "MULTISEED_CLASSIFICATION_SUMMARY.json", {"per_seed": results, "created_at": drv.now_iso()})


if __name__ == "__main__":
    main()
