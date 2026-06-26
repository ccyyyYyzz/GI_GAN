"""One-shot LOCKED confirmatory test for the VQGAN detail-fusion balanced operating point.

See locked_bundle/CODEX_NEXT_GOAL_VQGAN_DETAIL_FUSION_LOCKED.md. The method is FROZEN from
development: no retraining, no learned gate, no B re-selection. The per-seed blend scalar B is
read verbatim from the val-frozen development selection.

Pipeline (firewall-respecting): build -> regen -> score -> report.
  build : audit ALL historically consumed STL10 samples, construct a brand-new hash-clean LOCKED
          split disjoint from every consumed set (raw-SHA256 dedup), write the pre-registration.
  regen : regenerate x0 / x_A / x_G for the locked images with the frozen priors+refiners.
  score : apply the frozen per-seed B, score all arms, run the per-image seed-averaged-delta image
          bootstrap, emit the mechanical classification. ONCE.
  report: Pareto (dev vs locked), claim-evidence ledger, reproducible summary.

Statistics (per brief): for image i, seed s, Delta_{i,s} = m_fusion(i,s) - m_vqae(i,s);
primary statistic averages over seeds first (Delta_bar_i = mean_s Delta_{i,s}) then bootstraps
over images, so image x seed repeats are not treated as independent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

import gan_high_quality_gi as hq
import anchor_initialized_vqgan_inversion as ai
import vqgan_detail_fusion as vdf
from src.projections import relative_measurement_error

ROOT = vdf.ROOT
BASE = vdf.BASE
DEV = BASE / "detail_fusion"
LOCK = BASE / "detail_fusion_locked"
CACHE = LOCK / "cache"
SEEDS = [0, 1, 2]
METRIC_COLS = vdf.METRIC_COLS
LOCKED_N = 512
CERT_POOL = Path("E:/ns_mc_gan_gi/results/cert_package_20260612/cache/split_train_indices_stl10_train_unlabeled.npy")
BOOT_REPS = 2000
BOOT_SEED = 20260626


def log(*a):
    vdf.log(*a)


def make_transform(img: int = 64):
    return transforms.Compose([transforms.Resize((img, img)), transforms.Grayscale(1), transforms.ToTensor()])


def frozen_B() -> tuple[dict, dict]:
    """Exact per-seed B, read verbatim from the val-frozen development selection (no approximation)."""
    bal, ql = {}, {}
    for s in SEEDS:
        sel = json.loads((DEV / "canary" / f"seed{s}_selection.json").read_text())["selected_specs"]
        bal[s] = float(sel["balanced"][1][0])
        ql[s] = float(sel["quality_lite"][1][0])
    return bal, ql


# --------------------------------------------------------------------------- #
# Consumed-sample audit: union of raw STL10 image SHA256 over every consumed line.
# --------------------------------------------------------------------------- #
def build_consumed_raw(base_tu, base_test, transform):
    probe_tu = hq.IndexedTensorDataset(base_tu, [], transform)
    probe_test = hq.IndexedTensorDataset(base_test, [], transform)
    consumed: set[str] = set()
    prov: dict = {}

    # A: every per-image hash audit recorded in the repo (fusion, GAN-HQ, ga_nsgan, dev/val, ...)
    csvs = sorted(Path("outputs/compatibility").glob("**/reports/sample_hash_audit.csv"))
    nA = 0
    for c in csvs:
        with open(c, newline="") as f:
            for r in csv.DictReader(f):
                h = r.get("raw_sha256")
                if h:
                    consumed.add(h)
                    nA += 1
    prov["A_sample_hash_audit_csv"] = {"files": len(csvs), "rows_seen": nA}

    # B: Phase2 add-on locked manifests (source_index on train+unlabeled, no recorded hash)
    nB = 0
    for p in ["outputs/compatibility/phase2_addon_locked_test/lock_v1/reports/locked_split_manifest.json",
              "outputs/compatibility/phase2_addon_locked_test_external/lock_v1/reports/locked_split_manifest.json"]:
        pp = Path(p)
        if pp.exists():
            m = json.loads(pp.read_text())
            for r in (m.get("rows") or []):
                si = r.get("source_index")
                if si is not None and 0 <= int(si) < len(base_tu):
                    consumed.add(probe_tu.raw_hash(int(si)))
                    nB += 1
    prov["B_phase2_locked"] = {"resolved": nB}

    # C: final-v4 / phase1_x locked indices on STL10 TEST (defensive — physically disjoint split)
    nC = 0
    for f in ["phase1_4ir_incident_recovery/manifests/final_locked_test_64_v4_indices.npy",
              "phase1_4a_final_freeze_and_blind/manifests/final_locked_test_64_v3_indices.npy",
              "phase1_3r_recovery_and_relock/manifests/final_locked_test_64_v2_indices.npy",
              "phase1_1_corrected_rad5/reports/final_locked_test_indices.npy"]:
        pp = Path("outputs/compatibility") / f
        if pp.exists():
            for si in np.load(pp).astype(int).tolist():
                if 0 <= si < len(base_test):
                    consumed.add(probe_test.raw_hash(int(si)))
                    nC += 1
    prov["C_finalv4_test"] = {"resolved": nC}

    # D: the ENTIRE cert-pool permutation SPLIT_TRAIN — superset of every Phase2/gauge/phase69-81
    #    consumer (their artifacts are partly off-repo), so excluding the whole pool needs no per-config offsets.
    nD = 0
    pool_n = 0
    if CERT_POOL.exists():
        pool = np.load(CERT_POOL).astype(int)
        pool_n = int(len(pool))
        for si in pool.tolist():
            if 0 <= si < len(base_tu):
                consumed.add(probe_tu.raw_hash(int(si)))
                nD += 1
    prov["D_split_train_pool"] = {"path": str(CERT_POOL), "exists": CERT_POOL.exists(), "pool_n": pool_n, "resolved": nD}

    union_sha = hashlib.sha256("".join(sorted(consumed)).encode()).hexdigest()
    prov["union_size"] = len(consumed)
    prov["union_sha256"] = union_sha
    return consumed, prov


def build_locked_split(base_tu, transform, consumed_raw, n):
    probe = hq.IndexedTensorDataset(base_tu, [], transform)
    picked, picked_hashes = [], set()
    skipped_consumed = skipped_dup = 0
    scanned_until = -1
    for si in range(len(base_tu)):
        scanned_until = si
        h = probe.raw_hash(si)
        if h in consumed_raw:
            skipped_consumed += 1
            continue
        if h in picked_hashes:
            skipped_dup += 1
            continue
        picked_hashes.add(h)
        picked.append(si)
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise RuntimeError(f"LOCKED_SPLIT_TOO_SMALL:{len(picked)}:{n}")
    ds = hq.IndexedTensorDataset(base_tu, picked, transform)
    scan = {"scanned_until": scanned_until, "skipped_consumed": skipped_consumed,
            "skipped_intra_dup": skipped_dup, "n": len(picked)}
    return ds, picked, scan


def cmd_build():
    LOCK.mkdir(parents=True, exist_ok=True)
    cfg = vdf.load_cfg(0)
    root = cfg["data"]["dataset_root"]
    transform = make_transform(int(cfg["data"]["img_size"]))
    log("loading STL10 train+unlabeled + test bases ...")
    base_tu = datasets.STL10(root=root, split="train+unlabeled", download=True)
    base_test = datasets.STL10(root=root, split="test", download=True)
    log(f"train+unlabeled len={len(base_tu)}, test len={len(base_test)}")

    log("building consumed raw-hash union (A csv / B phase2 / C final-v4 / D cert-pool) ...")
    consumed, prov = build_consumed_raw(base_tu, base_test, transform)
    log("consumed union:", json.dumps(prov))

    log(f"scanning for {LOCKED_N} fresh hash-clean locked images ...")
    locked_ds, picked, scan = build_locked_split(base_tu, transform, consumed, LOCKED_N)
    log("locked scan:", json.dumps(scan))

    # dedup audit (raw + transformed) over the locked split
    reports = LOCK / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = hq.save_split_hash_audit(reports / "sample_hash_audit.csv", {"locked": locked_ds})
    # belt-and-suspenders: locked raw hashes disjoint from the consumed union
    locked_raw = set()
    with open(reports / "sample_hash_audit.csv", newline="") as f:
        for r in csv.DictReader(f):
            locked_raw.add(r["raw_sha256"])
    overlap = len(locked_raw & consumed)
    hash_audit_pass = (len(audit.get("raw_duplicates", [])) == 0
                       and len(audit.get("transformed_duplicates", [])) == 0
                       and overlap == 0
                       and len(locked_raw) == LOCKED_N)
    log(f"dedup audit: raw_dups={len(audit.get('raw_duplicates', []))} "
        f"transformed_dups={len(audit.get('transformed_duplicates', []))} "
        f"overlap_with_consumed={overlap} unique_locked={len(locked_raw)} PASS={hash_audit_pass}")

    idx = np.asarray(picked, dtype=np.int64)
    manifest = {
        "split": "locked",
        "dataset_name": "STL10",
        "source_split": "train+unlabeled",
        "dataset_root": root,
        "img_size": int(cfg["data"]["img_size"]),
        "hash_clean": True,
        "n_images": LOCKED_N,
        "source_index_min": int(idx.min()),
        "source_index_max": int(idx.max()),
        "source_indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "scan": scan,
        "consumed_provenance": prov,
        "dedup_audit": {"raw_duplicates": len(audit.get("raw_duplicates", [])),
                        "transformed_duplicates": len(audit.get("transformed_duplicates", [])),
                        "overlap_with_consumed": overlap, "hash_audit_pass": hash_audit_pass},
    }
    (reports / "locked_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(reports / "locked_source_indices.npy", idx)

    # pre-registration (written BEFORE any locked score exists)
    bal, ql = frozen_B()
    op_rows_sha = None
    try:
        op_rows_sha = json.loads((BASE / "anchor_multiseed_hashclean_seed0/reports/operator_manifest.json").read_text()).get("rows_sha256")
    except Exception:
        pass
    prereg = {
        "title": "VQGAN detail-fusion LOCKED test pre-registration",
        "frozen_method": "no retraining, no learned gate, no B re-selection; null-space fusion x_hat=x0+P0(d_A+B(d_G-d_A)); exact measurement audit",
        "frozen_B_balanced": bal,
        "frozen_B_quality_lite": ql,
        "B_source": "val-frozen development selection (detail_fusion/canary/seed*_selection.json)",
        "operator_rows_sha256": op_rows_sha,
        "operator_seed": int(cfg["operator"]["seed"]),
        "lmmse_lambda": float(cfg["operator"]["lmmse_lambda"]),
        "priors": {s: {"vqae": str(vdf.load_cfg(s)["priors"]["vqae_checkpoint"]),
                       "vqgan": str(vdf.load_cfg(s)["priors"]["vqgan_checkpoint"]),
                       "vqae_refiner": str(vdf.refiner_ckpt(s, ai.VQAE)),
                       "vqgan_refiner": str(vdf.refiner_ckpt(s, ai.VQGAN))} for s in SEEDS},
        "locked_split": {"n": LOCKED_N, "source_indices_sha256": manifest["source_indices_sha256"],
                         "dedup": manifest["dedup_audit"]},
        "primary": "Balanced mode LPIPS vs VQAE (per-image seed-averaged delta, image bootstrap, CI upper<0)",
        "balanced_success_conditions": {
            "lpips_rel_gain_ge_0.05": True, "lpips_ci_upper_lt_0": True,
            "psnr_drop_le_0.5db": True, "rmse_increase_le_0.005": True,
            "rapsd_not_worse": True, "ge_2_of_3_seeds_same_direction": True,
            "relmeaserr_le_1e-5": True, "hash_audit_pass": True,
        },
        "quality_lite_success": "same but psnr drop <= 1.0 dB",
        "statistics": "Delta_bar_i = mean_s(m_fusion(i,s)-m_vqae(i,s)); bootstrap over images; report per-seed too",
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED},
        "classifications": ["LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED", "LOCKED_QUALITY_LITE_ONLY",
                            "LOCKED_VQGAN_TRADEOFF_ONLY", "LOCKED_DEV_NOT_REPLICATED", "INVALID_LOCKED_EXPERIMENT"],
        "one_shot": "score executed once; refuses to overwrite without --force",
    }
    (LOCK / "PREREGISTRATION.json").write_text(json.dumps(prereg, indent=2))
    log("wrote PREREGISTRATION.json + reports/{locked_split_manifest.json, locked_source_indices.npy, sample_hash_audit.csv}")
    if not hash_audit_pass:
        log("WARNING: hash audit did NOT pass — scoring would be INVALID_LOCKED_EXPERIMENT")


# --------------------------------------------------------------------------- #
# Regenerate locked recons with the frozen priors+refiners.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def recon_locked(seed, sub, locked_ds, device):
    cfg = sub.cfg
    dt = float(cfg["training"].get("distance_temperature", 1.0))
    st = float(cfg["training"].get("soft_temperature", 1.0))
    priors = {ai.VQAE: ai.load_prior(ai.VQAE, ROOT / cfg["priors"]["vqae_checkpoint"], cfg, device),
              ai.VQGAN: ai.load_prior(ai.VQGAN, ROOT / cfg["priors"]["vqgan_checkpoint"], cfg, device)}
    refs = {ai.VQAE: ai.load_refiner_checkpoint(vdf.refiner_ckpt(seed, ai.VQAE), cfg, device),
            ai.VQGAN: ai.load_refiner_checkpoint(vdf.refiner_ckpt(seed, ai.VQGAN), cfg, device)}

    def refine(kind, x0, unc):
        p = priors[kind]
        z0 = p.model.encode(x0)
        dz, dl = refs[kind](x0, unc, z0)
        logits = ai.logits_from_latent(z0 + dz, p, distance_temperature=dt) + dl
        zq, _, _ = ai.quantize_from_logits(p, logits, soft_temperature=st, straight_through=False)
        return p.model.decode_embeddings(zq)

    loader = hq.build_loader(locked_ds, batch_size=int(cfg["data"].get("eval_batch_size", 16)),
                             workers=0, shuffle=False, seed=int(cfg["seed"]) + 99, device=device)
    acc = defaultdict(list)
    for x, label, idx in loader:
        x = x.to(device)
        flat = sub.measurement.flatten_img(x)
        y = sub.measurement.A_forward(flat)
        x0 = sub.measurement.unflatten_img(sub.lmmse.anchor(y, sub.measurement, device=device))
        unc = sub.lmmse.uncertainty_map(img_size=sub.img, device=device, batch_size=x.shape[0], dtype=x.dtype)
        acc["x0"].append(x0.cpu()); acc["x_A"].append(refine(ai.VQAE, x0, unc).cpu())
        acc["x_G"].append(refine(ai.VQGAN, x0, unc).cpu())
        acc["y"].append(y.cpu()); acc["truth"].append(x.cpu())
        acc["source_index"].append(idx.cpu()); acc["label"].append(label.cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def load_locked_ds(sub):
    idx = np.load(LOCK / "reports" / "locked_source_indices.npy").astype(int).tolist()
    transform = make_transform(sub.img)
    base = sub.train_ds.base
    return hq.IndexedTensorDataset(base, idx, transform)


def cmd_regen(device):
    cfg = vdf.load_cfg(0)
    sub = vdf.Substrate(cfg, device)
    locked_ds = load_locked_ds(sub)
    CACHE.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        sub.cfg = vdf.load_cfg(seed)
        pack = recon_locked(seed, sub, locked_ds, device)
        torch.save(pack, CACHE / f"locked_seed{seed}.pt")
        log(f"locked seed{seed}: cached {pack['x0'].shape[0]} recons")


# --------------------------------------------------------------------------- #
# One-shot scoring + classification.
# --------------------------------------------------------------------------- #
def per_image_metrics(pred, truth, y, measurement, lpips_fn) -> dict:
    clip = pred.clamp(0, 1)
    rmse = hq.full_rmse_torch(clip, truth)
    crmse = hq.centered_rmse_torch(clip, truth)
    psnr = -20.0 * np.log10(np.maximum(rmse, 1e-12))
    lp = hq.lpips_batch(lpips_fn, clip, truth)
    rel = relative_measurement_error(pred, y, measurement).detach().cpu().numpy()
    p = clip[:, 0].detach().cpu().numpy(); t = truth.clamp(0, 1)[:, 0].detach().cpu().numpy()
    rapsd = np.array([float(np.linalg.norm(hq.rapsd_np(p[i]) - hq.rapsd_np(t[i]))) for i in range(p.shape[0])])
    ssim = np.array([float(hq.ssim_metric(clip[i:i + 1], truth[i:i + 1])) for i in range(clip.shape[0])])
    return {"full_rmse": np.asarray(rmse), "centered_rmse": np.asarray(crmse), "psnr": np.asarray(psnr),
            "ssim": ssim, "lpips": np.asarray(lp, dtype=float) if lp is not None else np.full(len(rmse), np.nan),
            "rapsd": rapsd, "relmeaserr": rel}


def image_bootstrap(delta, reps=BOOT_REPS, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    obs = float(np.mean(delta))
    boots = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(reps)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": obs, "ci_low": float(lo), "ci_high": float(hi), "n_images": int(len(delta))}


def cmd_score(device, force=False):
    out = LOCK / "LOCKED_GATE_REPORT.json"
    if out.exists() and not force:
        log("LOCKED_GATE_REPORT.json already exists — one-shot guard. Use --force only if you must re-run.")
        return
    bal, ql = frozen_B()
    cfg0 = vdf.load_cfg(0)
    measurement, projector = vdf.build_meas(cfg0, device)
    lpips_fn = hq.load_lpips(device)
    arms = {"lmmse_anchor": None, "vqae": {s: 0.0 for s in SEEDS}, "vqgan": {s: 1.0 for s in SEEDS},
            "fusion_balanced": bal, "fusion_quality_lite": ql}
    # M[arm][metric] = dict source_index -> list of per-seed values
    M = {a: {m: defaultdict(list) for m in METRIC_COLS} for a in arms}
    all_rows = []
    for seed in SEEDS:
        pk = torch.load(CACHE / f"locked_seed{seed}.pt", map_location=device)
        pre = vdf.prep_residuals(pk, measurement, projector)
        x0f, d_A, d_G, y, truth, idx = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"], pre["idx"]
        sidx = idx.cpu().numpy().tolist()
        for arm, bmap in arms.items():
            if arm == "lmmse_anchor":
                xhat = measurement.unflatten_img(x0f.to(torch.float32))
                B = 0.0
            else:
                B = float(bmap[seed])
                xhat = vdf.fuse(("scalar", B), x0f, d_A, d_G, y, measurement, projector, [])
            pim = per_image_metrics(xhat, truth, y, measurement, lpips_fn)
            for j, si in enumerate(sidx):
                for m in METRIC_COLS:
                    M[arm][m][si].append(float(pim[m][j]))
            rows = vdf.score_method(xhat, truth, y, measurement, lpips_fn, arm, B, idx, pre["label"], seed, "locked")
            all_rows.extend(rows)
        log(f"locked seed{seed}: scored {len(sidx)} images x {len(arms)} arms")

    # primary stats: per-image seed-averaged delta (arm - vqae), image bootstrap
    images = sorted(M["vqae"]["lpips"].keys())

    def block(arm):
        res = {"metrics": {}, "per_seed": {}}
        for m in METRIC_COLS:
            dbar = np.array([np.mean([M[arm][m][i][s] - M["vqae"][m][i][s] for s in range(len(SEEDS))]) for i in images])
            res["metrics"][m] = image_bootstrap(dbar)
        # per-seed mean delta (lpips) + direction
        for m in ["lpips", "psnr", "rapsd", "full_rmse"]:
            res["per_seed"][m] = {s: float(np.mean([M[arm][m][i][s] - M["vqae"][m][i][s] for i in images])) for s in range(len(SEEDS))}
        res["lpips_same_direction_neg"] = int(sum(1 for s in range(len(SEEDS)) if res["per_seed"]["lpips"][s] < 0))
        vqae_lpips_mean = float(np.mean([M["vqae"]["lpips"][i][s] for i in images for s in range(len(SEEDS))]))
        res["lpips_relative_gain"] = -res["metrics"]["lpips"]["mean"] / max(abs(vqae_lpips_mean), 1e-12)
        res["method_relmeaserr_mean"] = float(np.mean([M[arm]["relmeaserr"][i][s] for i in images for s in range(len(SEEDS))]))
        res["method_relmeaserr_max"] = float(np.max([M[arm]["relmeaserr"][i][s] for i in images for s in range(len(SEEDS))]))
        return res

    blocks = {a: block(a) for a in ["fusion_balanced", "fusion_quality_lite", "vqgan"]}
    manifest = json.loads((LOCK / "reports" / "locked_split_manifest.json").read_text())
    hash_pass = bool(manifest["dedup_audit"]["hash_audit_pass"])

    def cond(b, psnr_tol):
        lp = b["metrics"]["lpips"]
        return {
            "lpips_rel_gain_ge_0.05": bool(b["lpips_relative_gain"] >= 0.05),
            "lpips_ci_upper_lt_0": bool(lp["ci_high"] < 0),
            "lpips_2_of_3_seeds_same_direction": bool(b["lpips_same_direction_neg"] >= 2),
            "psnr_drop_within_tol": bool(b["metrics"]["psnr"]["mean"] >= -psnr_tol),
            "rmse_increase_le_0.005": bool(b["metrics"]["full_rmse"]["mean"] <= 0.005),
            "rapsd_not_worse": bool(b["metrics"]["rapsd"]["mean"] <= 0.0),
            "relmeaserr_le_1e-5": bool(b["method_relmeaserr_mean"] <= 1e-5),
            "hash_audit_pass": hash_pass,
        }

    bal_c = cond(blocks["fusion_balanced"], 0.5)
    ql_c = cond(blocks["fusion_quality_lite"], 1.0)
    vqgan_c = cond(blocks["vqgan"], 2.5)
    balanced_pass = all(bal_c.values())
    quality_pass = all(ql_c[k] for k in ["lpips_rel_gain_ge_0.05", "lpips_ci_upper_lt_0",
                                         "lpips_2_of_3_seeds_same_direction", "psnr_drop_within_tol",
                                         "relmeaserr_le_1e-5", "hash_audit_pass"])
    vqgan_quality = all(vqgan_c[k] for k in ["lpips_rel_gain_ge_0.05", "lpips_ci_upper_lt_0",
                                             "lpips_2_of_3_seeds_same_direction", "relmeaserr_le_1e-5", "hash_audit_pass"])

    if not hash_pass or not bal_c["relmeaserr_le_1e-5"]:
        classification = "INVALID_LOCKED_EXPERIMENT"
    elif balanced_pass:
        classification = "LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED"
    elif quality_pass:
        classification = "LOCKED_QUALITY_LITE_ONLY"
    elif vqgan_quality:
        classification = "LOCKED_VQGAN_TRADEOFF_ONLY"
    else:
        classification = "LOCKED_DEV_NOT_REPLICATED"

    report = {
        "classification": classification,
        "balanced_locked_passed": balanced_pass,
        "quality_lite_locked_passed": quality_pass,
        "full_vqgan_quality_passed": vqgan_quality,
        "hash_audit_pass": hash_pass,
        "frozen_B_balanced": bal, "frozen_B_quality_lite": ql,
        "locked_n": LOCKED_N, "seeds": SEEDS, "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED},
        "conditions": {"fusion_balanced": bal_c, "fusion_quality_lite": ql_c, "vqgan": vqgan_c},
        "arms": blocks,
        "locked_split_sha256": manifest["source_indices_sha256"],
    }
    out.write_text(json.dumps(report, indent=2))
    cols = list(all_rows[0].keys())
    with open(LOCK / "locked_per_image_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)
    log("CLASSIFICATION:", classification)
    for arm in ["fusion_balanced", "fusion_quality_lite", "vqgan"]:
        b = blocks[arm]
        lp, ps, rp = b["metrics"]["lpips"], b["metrics"]["psnr"], b["metrics"]["rapsd"]
        log(f"  {arm}: dLPIPS={lp['mean']:.4f}[{lp['ci_low']:.4f},{lp['ci_high']:.4f}] "
            f"relgain={b['lpips_relative_gain']:.3f} dPSNR={ps['mean']:.2f} dRMSE={b['metrics']['full_rmse']['mean']:+.4f} "
            f"dRAPSD={rp['mean']:+.5f} seedsneg={b['lpips_same_direction_neg']}/3 rel={b['method_relmeaserr_mean']:.1e}")


def cmd_report():
    rep = json.loads((LOCK / "LOCKED_GATE_REPORT.json").read_text())
    dev = json.loads((DEV / "MULTISEED_FUSION_GATE_REPORT.json").read_text())
    lines = [
        "# VQGAN Detail-Fusion LOCKED Confirmatory Test",
        "",
        f"Classification: `{rep['classification']}`",
        f"- balanced_locked_passed: `{rep['balanced_locked_passed']}`",
        f"- quality_lite_locked_passed: `{rep['quality_lite_locked_passed']}`",
        f"- hash_audit_pass: `{rep['hash_audit_pass']}`  | locked_n: {rep['locked_n']} | seeds: {rep['seeds']}",
        f"- frozen B balanced: {rep['frozen_B_balanced']} | quality-lite: {rep['frozen_B_quality_lite']}",
        f"- locked split source_indices_sha256: `{rep['locked_split_sha256']}`",
        "",
        "Method frozen from development (no retraining / no gate / no B re-selection). Brand-new hash-clean",
        "locked split, disjoint from every consumed STL10 set (raw-SHA256 dedup). Statistics: per-image",
        "seed-averaged delta then image bootstrap.",
        "",
        "## Fusion-vs-VQAE on the locked split (per-image seed-averaged delta, image bootstrap)",
        "",
        "| arm | ΔLPIPS [CI] | rel.gain | ΔPSNR | ΔRMSE | ΔRAPSD | seeds neg | relmeaserr |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ["fusion_balanced", "fusion_quality_lite", "vqgan"]:
        b = rep["arms"][arm]
        lp = b["metrics"]["lpips"]
        lines.append(
            f"| {arm} | {lp['mean']:.4f} [{lp['ci_low']:.4f}, {lp['ci_high']:.4f}] | {b['lpips_relative_gain']:.3f} | "
            f"{b['metrics']['psnr']['mean']:.2f} | {b['metrics']['full_rmse']['mean']:+.4f} | "
            f"{b['metrics']['rapsd']['mean']:+.5f} | {b['lpips_same_direction_neg']}/3 | {b['method_relmeaserr_mean']:.1e} |")
    lines += ["", "## Conditions (fusion_balanced)", "", f"`{json.dumps(rep['conditions']['fusion_balanced'])}`", ""]
    dev_bal = dev["arms"]["fusion_balanced"]
    dev_lpips_mean = dev_bal["metrics"]["lpips"].get("mean_delta", dev_bal["metrics"]["lpips"].get("mean"))
    lines += ["## Dev vs Locked (balanced, ΔLPIPS vs VQAE)", "",
              f"- development: {dev_lpips_mean:.4f} (rel.gain {dev_bal['lpips_relative_gain']:.3f})",
              f"- locked:      {rep['arms']['fusion_balanced']['metrics']['lpips']['mean']:.4f} "
              f"(rel.gain {rep['arms']['fusion_balanced']['lpips_relative_gain']:.3f})", ""]
    (LOCK / "LOCKED_CONFIRMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    led = ["# Claim-Evidence Ledger — VQGAN Detail Fusion LOCKED", "",
           f"Mechanical classification: `{rep['classification']}`.", "",
           "| claim | evidence | status |", "|---|---|---|",
           f"| Locked split is brand-new & unconsumed | raw-SHA256 dedup vs all consumed sets, overlap=0 | "
           f"{'PASS' if rep['hash_audit_pass'] else 'FAIL'} |",
           f"| B frozen from validation (no re-selection) | frozen_B read from dev selection JSONs | PASS |",
           f"| Balanced fusion generalizes (LPIPS↓, distortion in tol) | locked balanced conditions | "
           f"{'CONFIRMED' if rep['balanced_locked_passed'] else 'NOT CONFIRMED'} |",
           f"| Exact measurement consistency | locked relmeaserr mean ~{rep['arms']['fusion_balanced']['method_relmeaserr_mean']:.1e} | "
           f"{'PASS' if rep['conditions']['fusion_balanced']['relmeaserr_le_1e-5'] else 'FAIL'} |"]
    (LOCK / "LOCKED_CLAIM_EVIDENCE_LEDGER.md").write_text("\n".join(led), encoding="utf-8")
    log("wrote LOCKED_CONFIRMATION_REPORT.md + LOCKED_CLAIM_EVIDENCE_LEDGER.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "regen", "score", "report", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log("device =", device)
    if args.command in {"build", "all"}:
        cmd_build()
    if args.command in {"regen", "all"}:
        cmd_regen(device)
    if args.command in {"score", "all"}:
        cmd_score(device, force=args.force)
    if args.command in {"report", "all"}:
        cmd_report()


if __name__ == "__main__":
    main()
