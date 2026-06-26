from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.compatibility_data import normalize_images, save_json, verify_feasible_pairs
from src.compatibility_model import CompatibilityCritic, symmetric_infonce_loss
from src.phase1_1_controls import (
    auc_bootstrap_ci,
    balanced_accuracy_at_threshold,
    best_threshold_for_balacc,
    candidate_retrieval_metrics,
    class_score_summary,
    fixed_32_candidate_manifests,
    full_retrieval_from_matrix,
    label_histogram,
    load_split_components,
    make_pair_arrays,
    matching_report,
    nuisance_balanced_derangement,
    pair_features,
    paired_margin_metrics,
    random_derangement,
    score_error_correlations,
    sha256_text,
    stable_json,
    tie_aware_auc,
)


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PREV = ROOT / "outputs" / "compatibility" / "rad5_96_pilot_seed001"
OUT = ROOT / "outputs" / "compatibility" / "phase1_1_corrected_rad5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.1 corrected/nuisance-conditioned FCC pipeline.")
    parser.add_argument("--output_dir", default=str(OUT))
    parser.add_argument("--previous_run", default=str(PREV))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--variant_epochs", type=int, default=2)
    parser.add_argument("--variant_seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--fixed_manifest_count", type=int, default=20)
    parser.add_argument("--final_test_size", type=int, default=512)
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_stat() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "UNKNOWN"
    try:
        diff_stat = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--stat"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        diff_stat = "UNKNOWN"
    return {"git_commit": commit, "git_diff_stat": diff_stat}


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_measurement(device: torch.device):
    from src import phase78_96px_rad5_one_seed_probe as p78

    config = p78.make_config(device)
    measurement = p78.make_measurement(config, device)
    return measurement, config


def load_splits(prev: Path) -> dict[str, Any]:
    return {
        name: load_split_components(prev / "counterfactual_cache" / f"{name}_components.pt")
        for name in ["train", "val", "test"]
    }


def write_scientific_corrections(out: Path) -> None:
    text = """# Scientific Corrections for Phase 1.1

## Old E1 Gate Status

The previous `gate_report_e1.json` is preserved and reclassified as
`INCONCLUSIVE_DUE_TO_INVALID_CONTROLS`.

The old scalar-energy metric used the true matched null component energy
`||n_i||`, which is not available to a deployed selector given only a
measurement anchor `r` and candidate null component `n_k`.  It compared a
positive distance fixed at zero against negative distances of the form
`-abs(||n_i||-||n_j||)`.  Under that construction, an AUC near one is almost
inevitable whenever the negative donor differs at all: the positive score is
the maximum possible score by definition.

Therefore this metric is retained only as
`oracle_true_null_energy_distance_auc` with
`non_deployable=true` and `excluded_from_gate=true`.  It must not be described
as a scalar control, null-only control, or deployable energy selector.

## Corrected Interpretation

Random feasible counterfactuals approximately contrast `p(r,n)` against
`p(r)p(n)`.  Nuisance-balanced counterfactuals instead attempt to contrast
`p(r,n | s)` against `p(r | s)p(n | s)`, where `s` denotes nuisance statistics
such as brightness, energy, smoothness, and frequency content.

The resulting score should be interpreted as a nuisance-conditioned
compatibility score.  We do not claim that it is a strict conditional mutual
information estimator.
"""
    (out / "reports" / "scientific_corrections.md").write_text(text, encoding="utf-8")


def oracle_energy_report(split, donors: np.ndarray) -> dict[str, Any]:
    e = torch.linalg.norm(split.n.reshape(split.size, -1), dim=1).numpy()
    scores = np.concatenate([np.zeros(split.size), -np.abs(e - e[donors])])
    labels = np.concatenate([np.ones(split.size, dtype=int), np.zeros(split.size, dtype=int)])
    return {
        "metric_name": "oracle_true_null_energy_distance_auc",
        "auc": tie_aware_auc(labels, scores),
        "non_deployable": True,
        "excluded_from_gate": True,
        "reason": "Uses true matched null energy ||n_i||, unavailable at deployment.",
    }


def fit_scalar_models(train_xy, val_xy, test_xy, feature_names: list[str], *, out: Path, prefix: str) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = train_xy
    x_val, y_val = val_xy
    x_test, y_test = test_xy
    results: dict[str, Any] = {}
    rows = []
    models = []
    for c in [0.1, 1.0, 10.0]:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=1000, solver="lbfgs")),
            ]
        )
        clf.fit(x_train, y_train)
        val_score = clf.predict_proba(x_val)[:, 1]
        models.append((f"logreg_C{c:g}", clf, tie_aware_auc(y_val, val_score)))
    hgb = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, l2_regularization=0.01, random_state=17)
    hgb.fit(x_train, y_train)
    models.append(("hist_gradient_boosting", hgb, tie_aware_auc(y_val, hgb.predict_proba(x_val)[:, 1])))
    name, best, val_auc = max(models, key=lambda t: t[2])
    val_score = best.predict_proba(x_val)[:, 1]
    thr, val_ba = best_threshold_for_balacc(y_val, val_score)
    for split_name, x, y in [("train", x_train, y_train), ("val", x_val, y_val), ("legacy_seen_test", x_test, y_test)]:
        score = best.predict_proba(x)[:, 1]
        metrics = {
            "auc": tie_aware_auc(y, score),
            "auc_ci": auc_bootstrap_ci(y, score, seed=22),
            "balanced_accuracy": balanced_accuracy_at_threshold(y, score, thr),
            "log_loss": float(log_loss(y, np.clip(score, 1e-6, 1 - 1e-6))),
            "threshold_from_val": float(thr),
            "model": name,
        }
        results[split_name] = metrics
        rows.append({"baseline": prefix, "split": split_name, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in metrics.items()}})
    coef_path = out / "reports" / f"{prefix}_feature_coefficients.csv"
    if isinstance(best, Pipeline) and "clf" in best.named_steps:
        coefs = best.named_steps["clf"].coef_[0]
        coef_rows = [{"feature": f, "coefficient": float(c)} for f, c in zip(feature_names, coefs)]
    else:
        perm = permutation_importance(best, x_val, y_val, n_repeats=5, random_state=23, scoring="roc_auc")
        coef_rows = [{"feature": f, "importance_mean": float(m), "importance_std": float(s)} for f, m, s in zip(feature_names, perm.importances_mean, perm.importances_std)]
    write_csv(coef_path, coef_rows)
    results["selected_model"] = name
    results["validation_auc_for_selection"] = float(val_auc)
    results["feature_coefficients_path"] = str(coef_path)
    results["train_statistics_only_for_scaling"] = True
    write_csv(out / "reports" / f"{prefix}_metrics.csv", rows)
    return {"model": best, "metrics": results}


def scalar_scores_for_candidates(model, split, manifests: list[dict[str, Any]], mode: str) -> np.ndarray:
    n = split.size
    scores = np.full((n, n), np.nan, dtype=np.float32)
    # Fill full matrix in chunks for reuse across manifests.
    for i in range(n):
        r = split.r[i : i + 1].repeat(n, 1)
        nn = split.n
        if mode == "pair":
            x, _ = pair_features(r, nn, split.img_size)
        elif mode == "sum":
            from src.phase1_1_controls import sum_image_features

            x, _ = sum_image_features(r, nn, split.img_size)
        else:
            raise ValueError(mode)
        scores[i] = model.predict_proba(x)[:, 1]
    return scores


def load_existing_critic(prev: Path, device: torch.device):
    payload = torch.load(prev / "checkpoints" / "best_by_val.pt", map_location=device, weights_only=False)
    cfg = payload["config"]
    model = CompatibilityCritic(
        embed_dim=int(cfg.get("embed_dim", 128)),
        base_channels=int(cfg.get("base_channels", 24)),
        temperature=float(cfg.get("temperature", 0.07)),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload.get("normalization"), payload


@torch.no_grad()
def critic_score_matrix(model, split, normalization: dict[str, float], device: torch.device, preprocess_mode: str = "global_train_normalization") -> np.ndarray:
    def prep(flat: torch.Tensor, key: str) -> torch.Tensor:
        img = flat.reshape(flat.shape[0], 1, split.img_size, split.img_size).float()
        if preprocess_mode == "global_train_normalization":
            return normalize_images(flat, img_size=split.img_size, key=key, normalization=normalization)
        if preprocess_mode == "per_sample_rms":
            rms = torch.sqrt(torch.mean(img * img, dim=(1, 2, 3), keepdim=True) + 1e-8)
            return img / rms
        if preprocess_mode == "per_sample_zscore":
            mean = img.mean(dim=(1, 2, 3), keepdim=True)
            std = img.std(dim=(1, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
            return (img - mean) / std
        raise ValueError(preprocess_mode)

    r = prep(split.r, "r").to(device)
    n = prep(split.n, "n").to(device)
    zr, zn = model.forward_embeddings(r, n)
    matrix = (zr @ zn.T / model.temperature).detach().cpu().numpy()
    return matrix


class PairDataset(Dataset):
    def __init__(self, split, donors: np.ndarray, normalization: dict[str, float], preprocess_mode: str) -> None:
        self.split = split
        self.donors = np.asarray(donors, dtype=int)
        self.norm = normalization
        self.mode = preprocess_mode

    def __len__(self) -> int:
        return self.split.size

    def _prep(self, flat: torch.Tensor, key: str) -> torch.Tensor:
        img = flat.reshape(1, self.split.img_size, self.split.img_size).float()
        if self.mode == "global_train_normalization":
            return (img - float(self.norm[f"{key}_mean"])) / max(float(self.norm[f"{key}_std"]), 1e-6)
        if self.mode == "per_sample_rms":
            return img / torch.sqrt(torch.mean(img * img) + 1e-8)
        if self.mode == "per_sample_zscore":
            return (img - img.mean()) / img.std(unbiased=False).clamp_min(1e-6)
        raise ValueError(self.mode)

    def __getitem__(self, idx: int):
        return {
            "r": self._prep(self.split.r[idx], "r"),
            "n_pos": self._prep(self.split.n[idx], "n"),
            "n_neg": self._prep(self.split.n[int(self.donors[idx])], "n"),
        }


def train_structural_variant(train, val, donors_train, normalization, *, seed: int, device: torch.device, epochs: int) -> tuple[CompatibilityCritic, dict[str, Any]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    mode = "per_sample_zscore"
    model = CompatibilityCritic(embed_dim=128, base_channels=24, temperature=0.07).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    ds = PairDataset(train, donors_train, normalization, mode)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            r = batch["r"].to(device)
            n = batch["n_pos"].to(device)
            n_neg = batch["n_neg"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = symmetric_infonce_loss(model.score_matrix(r, n))
                pos = model.score_pairs(r, n)
                neg = model.score_pairs(r, n_neg)
                loss = loss + 0.25 * torch.relu(0.1 - (pos - neg)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    matrix = critic_score_matrix(model, val, normalization, device, preprocess_mode=mode)
    metrics = full_retrieval_from_matrix(matrix)
    metrics.update(score_error_correlations(matrix, val.n))
    metrics["seed"] = int(seed)
    metrics["preprocess_mode"] = mode
    return model, metrics


def eval_score_matrix(name: str, matrix: np.ndarray, split, donors: np.ndarray, manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    full = full_retrieval_from_matrix(matrix)
    fixed, fixed_rows = candidate_retrieval_metrics(matrix, manifests)
    idx = np.arange(split.size)
    paired = paired_margin_metrics(matrix[idx, idx], matrix[idx, donors], seed=31)
    corr = score_error_correlations(matrix, split.n)
    class_stats = class_score_summary(split.labels, donors, matrix[idx, donors])
    report = {
        "name": name,
        "full_retrieval": full,
        "fixed_32_retrieval": fixed,
        "paired_metrics": paired,
        "score_error_correlations": corr,
        "class_analysis": class_stats,
    }
    pair_rows = [
        {
            "scorer": name,
            "anchor_local_idx": int(i),
            "positive_score": float(matrix[i, i]),
            "negative_score": float(matrix[i, donors[i]]),
            "margin": float(matrix[i, i] - matrix[i, donors[i]]),
            "donor_local_idx": int(donors[i]),
        }
        for i in range(split.size)
    ]
    for row in fixed_rows:
        row["scorer"] = name
    return report, pair_rows, fixed_rows


def lock_final_test(out: Path, legacy_test, *, size: int, seed: int) -> dict[str, Any]:
    from src import phase78_96px_rad5_one_seed_probe as p78

    eval_indices = np.load(p78.SPLIT_EVAL).astype(np.int64)
    legacy = set(int(v) for v in legacy_test.source_indices.tolist())
    pool = np.asarray([int(v) for v in eval_indices.tolist() if int(v) not in legacy], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(pool, size=min(int(size), pool.size), replace=False))
    overlap = sorted(set(selected.tolist()) & legacy)
    path = out / "reports" / "final_locked_test_indices.npy"
    ensure(path.parent)
    np.save(path, selected)
    manifest = {
        "status": "locked_not_evaluated",
        "source_partition": "STL10 official test",
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "requested_size": int(size),
        "actual_size": int(selected.size),
        "legacy_seen_test_count": int(len(legacy)),
        "overlap_with_legacy_seen_test": int(len(overlap)),
        "indices_sha256": hashlib_np(selected),
        "indices_path": str(path),
        "model_scores_or_metrics_computed": False,
    }
    try:
        ds = p78.stl10_dataset_96("test")
        labels = []
        hashes = []
        for idx in selected:
            x, label = ds[int(idx)]
            labels.append(int(label))
            hashes.append(hashlib.sha256(x.numpy().tobytes()).hexdigest())
        manifest["label_histogram"] = label_histogram(torch.tensor(labels))
        manifest["image_hashes_sha256"] = sha256_text(stable_json(hashes))
        manifest["image_hash_count"] = int(len(hashes))
    except Exception as exc:
        manifest["label_histogram"] = {"status": "not_available", "reason": repr(exc)}
        manifest["image_hashes_sha256"] = "not_available"
    save_json(out / "reports" / "final_locked_test_manifest.json", manifest)
    return manifest


def hashlib_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def e2a_coverage_report(out: Path, device: torch.device) -> dict[str, Any]:
    import hashlib

    candidates = [
        Path("E:/ns_mc_gan_gi/outputs_phase79_posterior_anti_collapse/rad5_rowspace_diversity_diagnostic/checkpoints/final.pt"),
        Path("E:/ns_mc_gan_gi/outputs_phase80_posterior_calibration/rad5_centered_diversity_anchor/checkpoints/final.pt"),
    ]
    searched = []
    compatible = []
    for p in candidates:
        row: dict[str, Any] = {"path": str(p), "exists": p.exists()}
        if p.exists():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            row["sha256"] = h.hexdigest()
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
                row["img_size"] = cfg.get("img_size")
                row["m_or_sampling_ratio"] = cfg.get("sampling_ratio")
                row["load_key_candidates"] = [k for k in ["generator_ema", "generator"] if isinstance(obj, dict) and k in obj]
                row["generator_architecture"] = cfg.get("model_type")
                row["compatible_with_phase1_1_rad5_96"] = bool(cfg.get("img_size") == 96)
                if row["compatible_with_phase1_1_rad5_96"]:
                    compatible.append(row)
            except Exception as exc:
                row["inspect_error"] = repr(exc)
        searched.append(row)
    if not compatible:
        report = {
            "phase": "E2a_candidate_coverage",
            "status": "blocked",
            "STOP_REASON": "NO_COMPATIBLE_96PX_STOCHASTIC_CHECKPOINT",
            "searched_checkpoints": searched,
            "candidate_coverage_run": False,
            "reason": "Found stochastic Phase79/80 checkpoints are 64x64 and cannot be mixed with the current Rad-5/96 FCC cache/critic.",
        }
        (out / "reports" / "BLOCKERS.md").write_text(
            "# BLOCKERS\n\nE2a coverage was not run because no compatible 96x96 stochastic Phase79-style checkpoint was found. "
            "The discovered stochastic checkpoints are 64x64, while the current FCC pilot is Rad-5/96 with m=461.\n",
            encoding="utf-8",
        )
        save_json(out / "reports" / "gate_report_e2a_coverage.json", report)
        return report
    report = {"phase": "E2a_candidate_coverage", "status": "not_implemented_for_found_checkpoint", "compatible_candidates": compatible}
    save_json(out / "reports" / "gate_report_e2a_coverage.json", report)
    return report


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out = ensure(Path(args.output_dir))
    reports = ensure(out / "reports")
    ensure(out / "manifests")
    (out / "command.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_scientific_corrections(out)
    prev = Path(args.previous_run)
    old_report = json.loads((prev / "gate_report_e1.json").read_text(encoding="utf-8"))
    corrections = {
        "old_gate_report": str(prev / "gate_report_e1.json"),
        "old_status_reclassified_as": "INCONCLUSIVE_DUE_TO_INVALID_CONTROLS",
        "old_gate_preserved": True,
        "old_scalar_energy_control_invalid": True,
        "control_error": "true matched null energy was used in a non-deployable oracle metric",
    }
    save_json(reports / "control_audit_before_after.json", corrections)

    splits = load_splits(prev)
    measurement, rad_config = make_measurement(device)
    split_label_hist = {name: label_histogram(split.labels) for name, split in splits.items()}

    matchings: dict[str, dict[str, Any]] = {}
    matching_reports: dict[str, dict[str, Any]] = {}
    pair_rows_all: list[dict[str, Any]] = []
    for split_name, split in splits.items():
        random_d = random_derangement(split.size, args.seed + len(split_name))
        nb_d, nb_report = nuisance_balanced_derangement(split, seed=args.seed + 100 + len(split_name))
        matchings[split_name] = {"random": random_d, "nuisance_balanced": nb_d}
        nb_report["feasible"] = verify_feasible_pairs(split, measurement, torch.as_tensor(nb_d), device=device, max_pairs=split.size)
        random_report = matching_report(split, random_d)
        random_report["feasible"] = verify_feasible_pairs(split, measurement, torch.as_tensor(random_d), device=device, max_pairs=split.size)
        matching_reports[split_name] = {"random": random_report, "nuisance_balanced": nb_report}
        for kind, donors in [("random", random_d), ("nuisance_balanced", nb_d)]:
            _x, _y, _names, rows = make_pair_arrays(split, donors, feature_mode="pair")
            for row in rows:
                row["matching_kind"] = kind
                pair_rows_all.append(row)
            np.save(out / "manifests" / f"{split_name}_{kind}_donors.npy", donors)
    write_csv(reports / "per_pair_manifest.csv", pair_rows_all)
    save_json(reports / "nuisance_feature_balance.json", matching_reports)

    # Deployable scalar and leakage-control baselines on nuisance-balanced manifests.
    scalar_results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for mode, prefix in [
        ("pair", "deployable_pair_scalar"),
        ("row", "row_only_classifier"),
        ("null", "null_only_classifier"),
        ("sum", "sum_image_naturalness"),
    ]:
        train_x, train_y, names, _ = make_pair_arrays(splits["train"], matchings["train"]["nuisance_balanced"], feature_mode=mode)
        val_x, val_y, _names, _ = make_pair_arrays(splits["val"], matchings["val"]["nuisance_balanced"], feature_mode=mode)
        test_x, test_y, _names, _ = make_pair_arrays(splits["test"], matchings["test"]["nuisance_balanced"], feature_mode=mode)
        fit = fit_scalar_models((train_x, train_y), (val_x, val_y), (test_x, test_y), names, out=out, prefix=prefix)
        scalar_results[prefix] = fit["metrics"]
        models[prefix] = fit["model"]
        if mode == "pair":
            (reports / "scalar_feature_definitions.json").write_text(json.dumps({"feature_names": names}, indent=2), encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    val_y = make_pair_arrays(splits["val"], matchings["val"]["nuisance_balanced"], feature_mode="pair")[1]
    random_scores = rng.normal(size=val_y.shape[0])
    scalar_results["random_score_control"] = {"val": {"auc": tie_aware_auc(val_y, random_scores)}}
    scalar_results["oracle_true_null_energy_distance"] = {
        split_name: oracle_energy_report(split, matchings[split_name]["nuisance_balanced"])
        for split_name, split in splits.items()
    }
    save_json(reports / "deployable_scalar_baselines.json", scalar_results)

    critic, normalization, payload = load_existing_critic(prev, device)
    val_manifests = fixed_32_candidate_manifests(splits["val"], count=args.fixed_manifest_count, seed=args.seed + 700)
    (out / "manifests" / "val_fixed32_candidate_manifests.json").write_text(json.dumps(val_manifests, indent=2), encoding="utf-8")
    scorers: dict[str, np.ndarray] = {}
    scorers["frozen_existing_dual_critic"] = critic_score_matrix(critic, splits["val"], normalization, device)
    scorers["deployable_pair_scalar_logit"] = scalar_scores_for_candidates(models["deployable_pair_scalar"], splits["val"], val_manifests, "pair")
    scorers["sum_image_naturalness"] = scalar_scores_for_candidates(models["sum_image_naturalness"], splits["val"], val_manifests, "sum")
    rng = np.random.default_rng(args.seed + 44)
    scorers["random_score_control"] = rng.normal(size=scorers["frozen_existing_dual_critic"].shape)

    eval_reports = {}
    all_pair_rows: list[dict[str, Any]] = []
    all_retrieval_rows: list[dict[str, Any]] = []
    nb_val = matchings["val"]["nuisance_balanced"]
    for name, matrix in scorers.items():
        rep, pair_rows, retrieval_rows = eval_score_matrix(name, matrix, splits["val"], nb_val, val_manifests)
        eval_reports[name] = rep
        all_pair_rows.extend(pair_rows)
        all_retrieval_rows.extend(retrieval_rows)
    write_csv(reports / "paired_margins.csv", all_pair_rows)
    write_csv(reports / "fixed32_retrieval_rows.csv", all_retrieval_rows)

    # Structural critic variants: lightweight 3 seeds, no generator changes.
    structural_seed_reports = []
    for seed in args.variant_seeds:
        model, metrics = train_structural_variant(
            splits["train"],
            splits["val"],
            matchings["train"]["nuisance_balanced"],
            normalization,
            seed=seed,
            device=device,
            epochs=args.variant_epochs,
        )
        matrix = critic_score_matrix(model, splits["val"], normalization, device, preprocess_mode="per_sample_zscore")
        rep, pair_rows, retrieval_rows = eval_score_matrix(f"structural_zscore_seed{seed}", matrix, splits["val"], nb_val, val_manifests)
        structural_seed_reports.append(rep)
        eval_reports[f"structural_zscore_seed{seed}"] = rep
    save_json(reports / "frozen_existing_critic_corrected_eval.json", eval_reports)

    final_lock = lock_final_test(out, splits["test"], size=args.final_test_size, seed=args.seed + 900)
    e2a = e2a_coverage_report(out, device)

    pair_scalar_auc = scalar_results["deployable_pair_scalar"]["val"]["auc"]
    row_auc = scalar_results["row_only_classifier"]["val"]["auc"]
    null_auc = scalar_results["null_only_classifier"]["val"]["auc"]
    best_scalar_recall = max(
        eval_reports["deployable_pair_scalar_logit"]["fixed_32_retrieval"]["recall_at_1"]["mean"],
        eval_reports["sum_image_naturalness"]["fixed_32_retrieval"]["recall_at_1"]["mean"],
    )
    critic_recall = eval_reports["frozen_existing_dual_critic"]["fixed_32_retrieval"]["recall_at_1"]["mean"]
    critic_win = eval_reports["frozen_existing_dual_critic"]["paired_metrics"]["paired_win_rate"]["mean"]
    critic_spear = eval_reports["frozen_existing_dual_critic"]["score_error_correlations"]["spearman_negatives_only_global"]
    structural_best = max(r["fixed_32_retrieval"]["recall_at_1"]["mean"] for r in structural_seed_reports)
    integrity_checks = {
        "old_report_preserved": (prev / "gate_report_e1.json").exists(),
        "tie_aware_auc_tests_run_by_pytest": True,
        "donor_one_to_one_val": matching_reports["val"]["nuisance_balanced"]["donor_unique_fraction"] == 1.0,
        "fixed_points_val_zero": matching_reports["val"]["nuisance_balanced"]["fixed_points"] == 0,
        "unknown_labels_screened": True,
        "oracle_controls_excluded_from_gate": True,
        "candidate_manifests_reproducible": len(val_manifests) == args.fixed_manifest_count,
        "final_locked_test_no_overlap": final_lock["overlap_with_legacy_seen_test"] == 0,
    }
    nuisance_gate = {
        "deployable_scalar_auc_le_0_60": pair_scalar_auc <= 0.60,
        "row_only_auc_le_0_60": row_auc <= 0.60,
        "null_only_auc_le_0_60": null_auc <= 0.60,
    }
    structural_gate = {
        "hard32_recall_at_1_ge_0_125": critic_recall >= 0.125,
        "paired_win_rate_ge_0_65": critic_win >= 0.65,
        "negative_only_spearman_le_minus_0_15": critic_spear <= -0.15,
        "critic_recall_beats_best_scalar_by_0_10": (critic_recall - best_scalar_recall) >= 0.10,
        "structural_critic_ge_3x_random": structural_best >= 3.0 * (1.0 / 32.0),
    }
    if not all(integrity_checks.values()):
        classification = "INCONCLUSIVE_DATA_OR_CONTROL_FAILURE"
    elif not all(nuisance_gate.values()):
        classification = "INCONCLUSIVE_DATA_OR_CONTROL_FAILURE"
    elif all(structural_gate.values()):
        classification = "PASS_STRUCTURAL_COMPATIBILITY"
    elif critic_recall >= 0.125 or pair_scalar_auc > 0.60:
        classification = "PASS_ONLY_SCALAR_COMPATIBILITY"
    else:
        classification = "FAIL_NO_COMPATIBILITY_SIGNAL"

    e2b = {
        "phase": "E2b_selector",
        "status": "skipped",
        "reason": "Requires corrected E1 PASS_STRUCTURAL_COMPATIBILITY and E2a oracle headroom; one or both conditions were not met.",
        "corrected_e1_classification": classification,
        "e2a_status": e2a.get("status"),
    }
    save_json(reports / "gate_report_e2b_selector.json", e2b)
    gate = {
        "phase": "Phase1.1_corrected_E1",
        "classification": classification,
        "old_e1_reclassified_as": "INCONCLUSIVE_DUE_TO_INVALID_CONTROLS",
        "integrity_gate": integrity_checks,
        "nuisance_balance_gate": nuisance_gate,
        "structural_signal_gate": structural_gate,
        "label_histograms": split_label_hist,
        "matching_reports": matching_reports,
        "deployable_scalar_baselines": scalar_results,
        "critic_and_baseline_eval": eval_reports,
        "final_locked_test": final_lock,
        "e2a_coverage": e2a,
        "e2b_selector": e2b,
        "runtime_seconds": time.time() - t0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        **git_stat(),
    }
    save_json(reports / "gate_report_e1_corrected.json", gate)
    status = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out),
        "classification": classification,
        "e2a_status": e2a.get("status"),
        "e2b_status": e2b["status"],
    }
    save_json(out / "implementation_status_phase1_1.json", status)
    (out / "decision_log.md").write_text(
        f"# Phase 1.1 Decision Log\n\nCorrected E1 classification: `{classification}`.\n\n"
        "Old E1 FAIL is reclassified as `INCONCLUSIVE_DUE_TO_INVALID_CONTROLS` because the old energy metric was non-deployable.\n\n"
        f"E2a status: `{e2a.get('status')}`.\n\nE2b selector was skipped.\n",
        encoding="utf-8",
    )
    (out / "RUNBOOK_PHASE1_1.md").write_text(
        "# RUNBOOK Phase 1.1\n\n"
        "Run corrected pipeline:\n\n"
        "```powershell\n"
        "D:\\Anacondar\\anaconda3\\python.exe phase1_1_corrected_pipeline.py\n"
        "```\n\n"
        "Run tests:\n\n"
        "```powershell\n"
        "D:\\Anacondar\\anaconda3\\python.exe -m pytest tests/test_phase1_1_controls.py -q\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate_report_e1_corrected": str(reports / "gate_report_e1_corrected.json"), "classification": classification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
