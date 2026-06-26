from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "compatibility" / "measurement_conditioned_vqgan"
AGG_ROOT = OUT_ROOT / "multiseed_pareto_confirmation"


METRICS_LOWER = {"lpips", "rapsd", "kid", "full_rmse", "centered_rmse", "relmeaserr"}
METRICS_HIGHER = {"psnr", "ssim", "edge_sharpness"}
QUALITY_METRICS = ["lpips", "rapsd", "psnr", "ssim", "full_rmse", "centered_rmse", "relmeaserr"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_dir(seed: int) -> Path:
    return OUT_ROOT / f"anchor_multiseed_hashclean_seed{seed}"


def load_seed(seed: int) -> dict[str, Any]:
    out = seed_dir(seed)
    reports = out / "reports"
    required = [
        reports / "gate_report.json",
        reports / "stage0_decision.json",
        reports / "refiner_manifests.json",
        reports / "duplicate_audit.json",
        reports / "final_dev_per_image.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Seed {seed} missing required outputs: {missing}")
    gate = read_json(reports / "gate_report.json")
    stage0 = read_json(reports / "stage0_decision.json")
    manifests = json.loads((reports / "refiner_manifests.json").read_text(encoding="utf-8"))
    duplicate = read_json(reports / "duplicate_audit.json")
    final = pd.read_csv(reports / "final_dev_per_image.csv")
    return {
        "seed": seed,
        "out": out,
        "reports": reports,
        "gate": gate,
        "stage0": stage0,
        "manifests": manifests,
        "duplicate": duplicate,
        "final": final,
    }


def selected_quality(seed_payload: dict[str, Any]) -> dict[str, Any]:
    gate = seed_payload["gate"]
    vqae = gate["selected_betas"]["vqae"]["selected"]
    vqgan = gate["selected_betas"]["vqgan"]["selected"]
    return {
        "vqae_beta": float(vqae["beta"]),
        "vqgan_beta": float(vqgan["beta"]),
        "vqae_val": vqae,
        "vqgan_val": vqgan,
    }


def manifest_for(manifests: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    for item in manifests:
        if item.get("kind") == kind:
            return item
    raise KeyError(kind)


def select_balanced(seed_payload: dict[str, Any], psnr_tol: float, rmse_tol: float) -> dict[str, Any]:
    manifests = seed_payload["manifests"]
    vqae_manifest = manifest_for(manifests, "vqae")
    vqgan_manifest = manifest_for(manifests, "vqgan")
    vqae_sel = vqae_manifest["best"]["selection"]["selected"]
    vqae_beta = float(vqae_sel["beta"])
    best_step = int(vqgan_manifest["best"]["step"])
    val_path = seed_payload["out"] / "runs" / f"seed{seed_payload['seed']}" / "vqgan_refiner" / f"val_step{best_step:06d}_method_metrics.csv"
    if not val_path.exists():
        raise FileNotFoundError(val_path)
    val = pd.read_csv(val_path)
    val = val[val["method"] == "vqgan_refiner_nullblend"].copy()
    val["beta"] = val["beta"].astype(float)
    feasible = val[
        (val["psnr_mean"] >= float(vqae_sel["psnr_mean"]) - psnr_tol)
        & (val["full_rmse_mean"] <= float(vqae_sel["full_rmse_mean"]) + rmse_tol)
    ].copy()
    if feasible.empty:
        return {
            "status": "NO_FEASIBLE_BALANCED_BETA",
            "vqae_beta": vqae_beta,
            "vqgan_beta": None,
            "vqae_val": vqae_sel,
            "vqgan_val": None,
            "best_step": best_step,
            "candidate_count": int(len(val)),
        }
    best = feasible.sort_values(["lpips_mean", "full_rmse_mean"], ascending=[True, True]).iloc[0].to_dict()
    return {
        "status": "PASS",
        "vqae_beta": vqae_beta,
        "vqgan_beta": float(best["beta"]),
        "vqae_val": vqae_sel,
        "vqgan_val": best,
        "best_step": best_step,
        "candidate_count": int(len(val)),
    }


def rows_for_selection(seed_payload: dict[str, Any], selection: dict[str, Any]) -> pd.DataFrame:
    final = seed_payload["final"].copy()
    seed = int(seed_payload["seed"])
    vqae_beta = float(selection["vqae_beta"])
    vqgan_beta = selection["vqgan_beta"]
    if vqgan_beta is None:
        raise ValueError(f"Seed {seed} has no vqgan beta for selection")
    vqgan_beta = float(vqgan_beta)
    vqae = final[(final["method"] == "vqae_refiner_nullblend") & (np.isclose(final["beta"].astype(float), vqae_beta))].copy()
    vqgan = final[(final["method"] == "vqgan_refiner_nullblend") & (np.isclose(final["beta"].astype(float), vqgan_beta))].copy()
    merged = vqgan.merge(vqae, on="source_index", suffixes=("_vqgan", "_vqae"))
    if merged.empty:
        raise ValueError(f"Seed {seed} selection produced no paired rows")
    merged["seed"] = seed
    merged["vqae_beta"] = vqae_beta
    merged["vqgan_beta"] = vqgan_beta
    return merged


def metric_delta_frame(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": paired["seed"].to_numpy(),
            "source_index": paired["source_index"].to_numpy(),
            "delta": paired[f"{metric}_vqgan"].to_numpy(dtype=float) - paired[f"{metric}_vqae"].to_numpy(dtype=float),
        }
    )


def clustered_bootstrap(delta_df: pd.DataFrame, reps: int, rng: np.random.Generator) -> dict[str, Any]:
    seeds = sorted(delta_df["seed"].unique().tolist())
    by_seed = {seed: delta_df[delta_df["seed"] == seed]["delta"].to_numpy(dtype=float) for seed in seeds}
    observed = float(delta_df["delta"].mean())
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        vals: list[np.ndarray] = []
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        for seed in chosen:
            arr = by_seed[int(seed)]
            vals.append(rng.choice(arr, size=len(arr), replace=True))
        boot[i] = float(np.concatenate(vals).mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    per_seed = {str(seed): float(by_seed[seed].mean()) for seed in seeds}
    return {
        "mean_delta": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "per_seed_mean_delta": per_seed,
        "same_direction_negative_count": int(sum(v < 0 for v in per_seed.values())),
        "same_direction_positive_count": int(sum(v > 0 for v in per_seed.values())),
        "n_images_total": int(len(delta_df)),
        "n_seeds": int(len(seeds)),
    }


def summarize_mode(name: str, seed_payloads: list[dict[str, Any]], selections: dict[int, dict[str, Any]], reps: int) -> dict[str, Any]:
    paired_rows = []
    for payload in seed_payloads:
        seed = int(payload["seed"])
        if selections[seed].get("vqgan_beta") is None:
            continue
        paired_rows.append(rows_for_selection(payload, selections[seed]))
    if not paired_rows:
        return {"mode": name, "status": "NO_PAIRED_ROWS"}
    paired = pd.concat(paired_rows, ignore_index=True)
    AGG_ROOT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(AGG_ROOT / f"{name}_paired_per_image.csv", index=False)
    rng = np.random.default_rng(20260626)
    metrics = {}
    for metric in QUALITY_METRICS:
        metrics[metric] = clustered_bootstrap(metric_delta_frame(paired, metric), reps=reps, rng=rng)
    method_means = {}
    for method in ["vqae", "vqgan"]:
        suffix = f"_{method}"
        method_means[method] = {
            metric: float(paired[f"{metric}{suffix}"].mean())
            for metric in QUALITY_METRICS
            if f"{metric}{suffix}" in paired.columns
        }
    lpips = metrics["lpips"]
    rapsd = metrics["rapsd"]
    psnr = metrics["psnr"]
    rel = method_means["vqgan"].get("relmeaserr", math.inf)
    return {
        "mode": name,
        "status": "READY",
        "selections": selections,
        "metrics": metrics,
        "method_means": method_means,
        "conditions": {
            "lpips_gain_ge_5pct_ci_upper_lt0": bool(
                lpips["ci_high"] < 0
                and (-lpips["mean_delta"] / max(method_means["vqae"]["lpips"], 1e-12)) >= 0.05
            ),
            "lpips_2_of_3_seeds_same_direction": bool(lpips["same_direction_negative_count"] >= 2),
            "rapsd_same_direction": bool(rapsd["mean_delta"] < 0),
            "psnr_drop_within_2p5db": bool(psnr["mean_delta"] >= -2.5),
            "relmeaserr_ok": bool(rel <= 1e-5),
        },
    }


def stage0_summary(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for payload in seed_payloads:
        seed = int(payload["seed"])
        st = payload["stage0"]
        vqae = st.get("vqae_teacher_best", {})
        vqgan = st.get("vqgan_teacher_best", {})
        rows.append(
            {
                "seed": seed,
                "decision": st.get("decision"),
                "vqae_lpips": vqae.get("lpips_mean"),
                "vqgan_lpips": vqgan.get("lpips_mean"),
                "vqae_kid": vqae.get("kid"),
                "vqgan_kid": vqgan.get("kid"),
                "vqae_rapsd": vqae.get("rapsd_mean"),
                "vqgan_rapsd": vqgan.get("rapsd_mean"),
                "vqgan_better_lpips": (vqgan.get("lpips_mean", math.inf) < vqae.get("lpips_mean", -math.inf)),
                "vqgan_better_kid": (vqgan.get("kid", math.inf) < vqae.get("kid", -math.inf)),
                "vqgan_better_rapsd": (vqgan.get("rapsd_mean", math.inf) < vqae.get("rapsd_mean", -math.inf)),
            }
        )
    return {
        "rows": rows,
        "all_pass_transfer_headroom": all(r["decision"] == "PASS_TRANSFER_HEADROOM" for r in rows),
        "vqgan_teacher_lpips_better_count": int(sum(bool(r["vqgan_better_lpips"]) for r in rows)),
        "vqgan_teacher_kid_better_count": int(sum(bool(r["vqgan_better_kid"]) for r in rows)),
        "vqgan_teacher_rapsd_better_count": int(sum(bool(r["vqgan_better_rapsd"]) for r in rows)),
    }


def duplicate_summary(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ok = True
    for payload in seed_payloads:
        dup = payload["duplicate"]
        raw = dup.get("raw_duplicates", [])
        transformed = dup.get("transformed_duplicates", [])
        clean = len(raw) == 0 and len(transformed) == 0
        ok = ok and clean
        rows.append({"seed": int(payload["seed"]), "clean": clean, "raw_duplicates": len(raw), "transformed_duplicates": len(transformed)})
    return {"clean_all_seeds": ok, "rows": rows}


def write_report(summary: dict[str, Any]) -> None:
    quality = summary["quality"]
    balanced = summary["balanced"]
    lines = [
        "# VQGAN Multi-Seed Pareto Confirmation",
        "",
        f"Classification: `{summary['classification']}`",
        f"Development gate passed: `{summary['development_gate_passed']}`",
        "",
        "## Quality Mode",
    ]
    if quality["status"] == "READY":
        for metric in ["lpips", "rapsd", "psnr", "full_rmse", "centered_rmse", "relmeaserr"]:
            m = quality["metrics"][metric]
            lines.append(f"- {metric}: mean delta VQGAN-VQAE = {m['mean_delta']:.6g}, 95% cluster CI [{m['ci_low']:.6g}, {m['ci_high']:.6g}], per-seed {m['per_seed_mean_delta']}")
        lines.append(f"- method means: {json.dumps(quality['method_means'], indent=2)}")
    lines += ["", "## Balanced Mode"]
    if balanced["status"] == "READY":
        for metric in ["lpips", "rapsd", "psnr", "full_rmse", "centered_rmse", "relmeaserr"]:
            m = balanced["metrics"][metric]
            lines.append(f"- {metric}: mean delta VQGAN-VQAE = {m['mean_delta']:.6g}, 95% cluster CI [{m['ci_low']:.6g}, {m['ci_high']:.6g}], per-seed {m['per_seed_mean_delta']}")
    else:
        lines.append(f"- {balanced['status']}")
    lines += [
        "",
        "## Transfer Ceiling",
        f"- all stage0 transfer headroom pass: `{summary['stage0']['all_pass_transfer_headroom']}`",
        f"- VQGAN teacher LPIPS better seeds: {summary['stage0']['vqgan_teacher_lpips_better_count']}/3",
        f"- VQGAN teacher KID better seeds: {summary['stage0']['vqgan_teacher_kid_better_count']}/3",
        f"- VQGAN teacher RAPSD better seeds: {summary['stage0']['vqgan_teacher_rapsd_better_count']}/3",
        "",
        "## Hash Discipline",
        f"- duplicate-clean all seeds: `{summary['duplicates']['clean_all_seeds']}`",
        "",
        "## Decision",
        summary["decision_rationale"],
    ]
    (AGG_ROOT / "MULTISEED_PARETO_CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_outputs() -> dict[str, Any]:
    zip_path = OUT_ROOT / "VQGAN_MULTI_SEED_PARETO_CONFIRMATION_PACKAGE.zip"
    if zip_path.exists():
        zip_path.unlink()
    include = [
        AGG_ROOT,
        ROOT / "configs" / "compatibility",
        ROOT / "scripts" / "aggregate_vqgan_multiseed_pareto.py",
        ROOT / "scripts" / "run_vqgan_multiseed_local.py",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in include:
            if item.is_file():
                zf.write(item, item.relative_to(ROOT))
            elif item.is_dir():
                for path in item.rglob("*"):
                    if path.is_file() and path.suffix.lower() not in {".pt", ".pth"}:
                        zf.write(path, path.relative_to(ROOT))
        for seed in [0, 1, 2]:
            reports = seed_dir(seed) / "reports"
            if reports.exists():
                for path in reports.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(ROOT))
    return {"zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256_file(zip_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--balanced-psnr-drop-tolerance-db", type=float, default=0.5)
    parser.add_argument("--balanced-rmse-increase-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    AGG_ROOT.mkdir(parents=True, exist_ok=True)
    seed_payloads = [load_seed(seed) for seed in args.seeds]
    quality_sel = {int(p["seed"]): selected_quality(p) for p in seed_payloads}
    balanced_sel = {
        int(p["seed"]): select_balanced(p, args.balanced_psnr_drop_tolerance_db, args.balanced_rmse_increase_tolerance)
        for p in seed_payloads
    }
    quality = summarize_mode("quality", seed_payloads, quality_sel, reps=args.bootstrap_reps)
    balanced = summarize_mode("balanced", seed_payloads, balanced_sel, reps=args.bootstrap_reps)
    stage0 = stage0_summary(seed_payloads)
    duplicates = duplicate_summary(seed_payloads)
    qcond = quality.get("conditions", {})
    bcond = balanced.get("conditions", {})
    if not duplicates["clean_all_seeds"]:
        classification = "INVALID_EXPERIMENT"
        passed = False
        rationale = "Hash duplicate audit failed in at least one seed."
    elif not stage0["all_pass_transfer_headroom"] or stage0["vqgan_teacher_lpips_better_count"] < len(args.seeds):
        classification = "VQGAN_PRIOR_UNSTABLE_ACROSS_SEEDS"
        passed = False
        rationale = "At least one seed lacks VQGAN teacher transfer headroom versus VQAE."
    elif all(qcond.get(k, False) for k in ["lpips_gain_ge_5pct_ci_upper_lt0", "lpips_2_of_3_seeds_same_direction", "rapsd_same_direction", "psnr_drop_within_2p5db", "relmeaserr_ok"]):
        classification = "VQGAN_PRIOR_TRANSFER_CONFIRMED_MULTI_SEED"
        passed = True
        rationale = "Quality-mode VQGAN improves LPIPS by the preregistered margin with cluster CI excluding zero, RAPSD same direction, measurement consistency passing, and PSNR within 2.5 dB tolerance."
    elif qcond.get("lpips_gain_ge_5pct_ci_upper_lt0", False) and qcond.get("relmeaserr_ok", False):
        classification = "VQGAN_TRANSFER_WITH_DISTORTION_TRADEOFF"
        passed = False
        rationale = "LPIPS improves, but distortion/Pareto constraints fail."
    else:
        classification = "VQGAN_PRIOR_TRANSFER_NOT_CONFIRMED_MULTI_SEED"
        passed = False
        rationale = "Quality-mode LPIPS/RAPSD/seed-direction gates did not jointly pass."
    balanced_gate = bool(
        balanced.get("status") == "READY"
        and all(bcond.get(k, False) for k in ["lpips_gain_ge_5pct_ci_upper_lt0", "lpips_2_of_3_seeds_same_direction", "rapsd_same_direction", "relmeaserr_ok"])
        and balanced["metrics"]["psnr"]["mean_delta"] >= -args.balanced_psnr_drop_tolerance_db
        and balanced["metrics"]["full_rmse"]["mean_delta"] <= args.balanced_rmse_increase_tolerance
    )
    summary = {
        "classification": classification,
        "development_gate_passed": passed,
        "balanced_gate_passed": balanced_gate,
        "decision_rationale": rationale,
        "quality": quality,
        "balanced": balanced,
        "stage0": stage0,
        "duplicates": duplicates,
        "seeds": args.seeds,
        "bootstrap_reps": args.bootstrap_reps,
        "balanced_tolerances": {
            "psnr_drop_db": args.balanced_psnr_drop_tolerance_db,
            "full_rmse_increase": args.balanced_rmse_increase_tolerance,
        },
    }
    (AGG_ROOT / "multiseed_gate_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(stage0["rows"]).to_csv(AGG_ROOT / "stage0_transfer_ceiling_by_seed.csv", index=False)
    pd.DataFrame(duplicates["rows"]).to_csv(AGG_ROOT / "duplicate_audit_by_seed.csv", index=False)
    write_report(summary)
    ledger = [
        "# Claim-Evidence Ledger",
        "",
        f"- Claim: multi-seed VQGAN prior transfer. Evidence: `{AGG_ROOT / 'multiseed_gate_report.json'}` classification `{classification}`.",
        f"- Claim: hash-clean split discipline. Evidence: duplicate-clean all seeds = `{duplicates['clean_all_seeds']}`.",
        f"- Claim: teacher-level VQGAN transfer headroom. Evidence: VQGAN teacher LPIPS better {stage0['vqgan_teacher_lpips_better_count']}/3 seeds.",
        f"- Claim: balanced Pareto confirmation. Evidence: balanced_gate_passed = `{balanced_gate}`.",
    ]
    (AGG_ROOT / "CLAIM_EVIDENCE_LEDGER.md").write_text("\n".join(ledger) + "\n", encoding="utf-8")
    package = package_outputs()
    summary["package"] = package
    (AGG_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"classification": classification, "development_gate_passed": passed, "balanced_gate_passed": balanced_gate, "package": package}, indent=2))


if __name__ == "__main__":
    main()
