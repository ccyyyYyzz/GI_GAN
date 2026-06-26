from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .compatibility_data import SplitComponents


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tie_aware_auc(labels: Iterable[int] | np.ndarray, scores: Iterable[float] | np.ndarray) -> float:
    labels = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels).astype(int)
    scores = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores).astype(float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        wins = 0.0
        total = float(pos.size * neg.size)
        for p in pos:
            wins += float(np.sum(p > neg))
            wins += 0.5 * float(np.sum(p == neg))
        return wins / total


def balanced_accuracy_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = labels.astype(int)
    pred = (scores >= float(threshold)).astype(int)
    pos = labels == 1
    neg = labels == 0
    tpr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return 0.5 * (tpr + tnr)


def best_threshold_for_balacc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    values = np.unique(scores.astype(float))
    if values.size == 0:
        return 0.0, float("nan")
    if values.size > 256:
        values = np.quantile(values, np.linspace(0.0, 1.0, 256))
    best_t = float(values[0])
    best_ba = -1.0
    for t in values:
        ba = balanced_accuracy_at_threshold(labels, scores, float(t))
        if ba > best_ba:
            best_t = float(t)
            best_ba = float(ba)
    return best_t, best_ba


def bootstrap_ci(values: np.ndarray, *, seed: int = 0, n_boot: int = 500, fn=np.mean) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    stats = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, values.size, size=values.size)
        stats.append(float(fn(values[idx])))
    return {
        "mean": float(fn(values)),
        "lo": float(np.quantile(stats, 0.025)),
        "hi": float(np.quantile(stats, 0.975)),
        "n": int(values.size),
    }


def auc_bootstrap_ci(labels: np.ndarray, scores: np.ndarray, *, seed: int = 0, n_boot: int = 300) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, labels.size, size=labels.size)
        if np.unique(labels[idx]).size < 2:
            continue
        vals.append(tie_aware_auc(labels[idx], scores[idx]))
    return {
        "auc": tie_aware_auc(labels, scores),
        "lo": float(np.quantile(vals, 0.025)) if vals else float("nan"),
        "hi": float(np.quantile(vals, 0.975)) if vals else float("nan"),
        "n_boot_used": int(len(vals)),
    }


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size != x.size:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).correlation)
    except Exception:
        xr = _average_ranks(x)
        yr = _average_ranks(y)
        xr = xr - xr.mean()
        yr = yr - yr.mean()
        denom = np.linalg.norm(xr) * np.linalg.norm(yr)
        return float(np.dot(xr, yr) / denom) if denom > 0 else float("nan")


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def load_split_components(path: str | Path) -> SplitComponents:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return SplitComponents(
        name=str(obj["name"]),
        x=obj["x"].float(),
        r=obj["r"].float(),
        n=obj["n"].float(),
        y=obj["y"].float(),
        labels=obj["labels"].long(),
        source_indices=obj["source_indices"].long(),
        projector_info=dict(obj.get("projector_info", {})),
    )


def _flat_to_img(x: torch.Tensor, img_size: int) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, 0]
    return x.reshape(x.shape[0], img_size, img_size)


def component_features(x: torch.Tensor, img_size: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    img = _flat_to_img(x.float(), img_size)
    flat = img.reshape(img.shape[0], -1)
    n_pix = float(flat.shape[1])
    eps = 1e-12
    feats: list[torch.Tensor] = []
    names: list[str] = []

    def add(name: str, v: torch.Tensor) -> None:
        feats.append(v.reshape(-1).float())
        names.append(f"{prefix}_{name}")

    add("mean", flat.mean(dim=1))
    add("std", flat.std(dim=1, unbiased=False))
    add("rms", torch.sqrt(torch.mean(flat * flat, dim=1) + eps))
    add("l1_sqrtN", flat.abs().sum(dim=1) / math.sqrt(n_pix))
    add("log_l2", torch.log(torch.linalg.norm(flat, dim=1).clamp_min(eps)))
    add("min", flat.min(dim=1).values)
    add("max", flat.max(dim=1).values)
    for q in [0.01, 0.05, 0.5, 0.95, 0.99]:
        add(f"q{int(q*100):02d}", torch.quantile(flat, q, dim=1))
    dx = img[:, :, 1:] - img[:, :, :-1]
    dy = img[:, 1:, :] - img[:, :-1, :]
    add("tv", (dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))))
    add("grad_rms", torch.sqrt((dx * dx).mean(dim=(1, 2)) + (dy * dy).mean(dim=(1, 2)) + eps))
    fft = torch.fft.fft2(img)
    power = (fft.real * fft.real + fft.imag * fft.imag).float()
    h, w = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    rr = torch.sqrt((yy.float() - h / 2.0) ** 2 + (xx.float() - w / 2.0) ** 2)
    rr = torch.fft.fftshift(rr).to(power.device)
    maxr = float(rr.max().item())
    total = power.reshape(power.shape[0], -1).sum(dim=1).clamp_min(eps)
    for name, lo, hi in [("freq_low", 0.0, 0.15), ("freq_mid", 0.15, 0.35), ("freq_high", 0.35, 1.01)]:
        mask = ((rr / maxr) >= lo) & ((rr / maxr) < hi)
        add(name, power[:, mask].sum(dim=1) / total)
    add("spectral_centroid", (power * (rr / maxr)).reshape(power.shape[0], -1).sum(dim=1) / total)
    return torch.stack(feats, dim=1).cpu().numpy(), names


def pair_features(r: torch.Tensor, n: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    rf, rn = component_features(r, img_size, "r")
    nf, nn = component_features(n, img_size, "n")
    u = r.reshape(r.shape[0], -1) + n.reshape(n.shape[0], -1)
    uf, un = component_features(u, img_size, "u")
    features = [rf, nf, uf]
    names = rn + nn + un
    eps = 1e-8
    # Interactions for common component feature names.
    base_r = {name[2:]: idx for idx, name in enumerate(rn)}
    base_n = {name[2:]: idx for idx, name in enumerate(nn)}
    inter_cols = []
    inter_names = []
    for key in sorted(set(base_r) & set(base_n)):
        a = rf[:, base_r[key]]
        b = nf[:, base_n[key]]
        inter_cols.append(np.abs(a - b))
        inter_names.append(f"absdiff_{key}")
        inter_cols.append(a * b)
        inter_names.append(f"product_{key}")
        if key in {"rms", "std", "l1_sqrtN", "log_l2"}:
            inter_cols.append(a / (np.abs(b) + eps))
            inter_names.append(f"ratio_r_over_n_{key}")
    if inter_cols:
        features.append(np.stack(inter_cols, axis=1))
        names += inter_names
    below = (u < 0).float().mean(dim=1).cpu().numpy()
    above = (u > 1).float().mean(dim=1).cpu().numpy()
    features.append(np.stack([below, above], axis=1))
    names += ["u_fraction_below_0", "u_fraction_above_1"]
    return np.concatenate(features, axis=1).astype(np.float32), names


def row_only_features(r: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    return component_features(r, img_size, "r")


def null_only_features(n: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    return component_features(n, img_size, "n")


def sum_image_features(r: torch.Tensor, n: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    return component_features(r.reshape(r.shape[0], -1) + n.reshape(n.shape[0], -1), img_size, "u")


def make_pair_arrays(
    split: SplitComponents,
    donors: np.ndarray,
    *,
    feature_mode: str = "pair",
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    idx = np.arange(split.size)
    donors_t = torch.as_tensor(donors, dtype=torch.long)
    r_pos = split.r
    n_pos = split.n
    r_neg = split.r
    n_neg = split.n[donors_t]
    if feature_mode == "pair":
        x_pos, names = pair_features(r_pos, n_pos, split.img_size)
        x_neg, _ = pair_features(r_neg, n_neg, split.img_size)
    elif feature_mode == "row":
        x_pos, names = row_only_features(r_pos, split.img_size)
        x_neg, _ = row_only_features(r_neg, split.img_size)
    elif feature_mode == "null":
        x_pos, names = null_only_features(n_pos, split.img_size)
        x_neg, _ = null_only_features(n_neg, split.img_size)
    elif feature_mode == "sum":
        x_pos, names = sum_image_features(r_pos, n_pos, split.img_size)
        x_neg, _ = sum_image_features(r_neg, n_neg, split.img_size)
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode}")
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones(split.size, dtype=int), np.zeros(split.size, dtype=int)])
    rows = []
    for label, anchors, nulls in [(1, idx, idx), (0, idx, donors)]:
        for a, nidx in zip(anchors, nulls):
            rows.append(
                {
                    "split": split.name,
                    "label": int(label),
                    "anchor_local_idx": int(a),
                    "null_local_idx": int(nidx),
                    "anchor_source_index": int(split.source_indices[int(a)].item()),
                    "null_source_index": int(split.source_indices[int(nidx)].item()),
                }
            )
    return x, y, names, rows


def cheap_component_features_for_matching(x: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    full, names = component_features(x, img_size, "q")
    keep = [
        i
        for i, name in enumerate(names)
        if any(token in name for token in ["mean", "std", "rms", "l1_sqrtN", "log_l2", "tv", "grad_rms", "freq_low", "freq_mid", "freq_high", "spectral_centroid"])
    ]
    return full[:, keep], [names[i].replace("q_", "") for i in keep]


def cheap_pair_features_from_components(rf: np.ndarray, nf: np.ndarray, component_names: list[str]) -> tuple[np.ndarray, list[str]]:
    eps = 1e-8
    cols = [rf, nf, np.abs(rf - nf), rf * nf, rf / (np.abs(nf) + eps)]
    names = (
        [f"r_{n}" for n in component_names]
        + [f"n_{n}" for n in component_names]
        + [f"absdiff_{n}" for n in component_names]
        + [f"product_{n}" for n in component_names]
        + [f"ratio_{n}" for n in component_names]
    )
    return np.concatenate(cols, axis=1).astype(np.float32), names


def cheap_u_features_for_matching(r_flat: torch.Tensor, n_flat: torch.Tensor, img_size: int) -> tuple[np.ndarray, list[str]]:
    u = (r_flat.reshape(r_flat.shape[0], -1) + n_flat.reshape(n_flat.shape[0], -1)).float()
    img = u.reshape(u.shape[0], img_size, img_size)
    eps = 1e-12
    cols = [
        u.mean(dim=1),
        u.std(dim=1, unbiased=False),
        torch.sqrt(torch.mean(u * u, dim=1) + eps),
        u.abs().sum(dim=1) / math.sqrt(float(u.shape[1])),
        torch.log(torch.linalg.norm(u, dim=1).clamp_min(eps)),
        u.min(dim=1).values,
        u.max(dim=1).values,
        torch.quantile(u, 0.05, dim=1),
        torch.quantile(u, 0.95, dim=1),
        (u < 0).float().mean(dim=1),
        (u > 1).float().mean(dim=1),
    ]
    dx = img[:, :, 1:] - img[:, :, :-1]
    dy = img[:, 1:, :] - img[:, :-1, :]
    cols.append(dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2)))
    cols.append(torch.sqrt((dx * dx).mean(dim=(1, 2)) + (dy * dy).mean(dim=(1, 2)) + eps))
    names = [
        "u_mean",
        "u_std",
        "u_rms",
        "u_l1_sqrtN",
        "u_log_l2",
        "u_min",
        "u_max",
        "u_q05",
        "u_q95",
        "u_fraction_below_0",
        "u_fraction_above_1",
        "u_tv",
        "u_grad_rms",
    ]
    return torch.stack(cols, dim=1).cpu().numpy().astype(np.float32), names


def random_derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    for _ in range(200):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def nuisance_balanced_derangement(split: SplitComponents, *, seed: int = 0, jitter: float = 1e-6) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    rf, names = cheap_component_features_for_matching(split.r, split.img_size)
    nf, _ = cheap_component_features_for_matching(split.n, split.img_size)
    pos_base, feat_names_base = cheap_pair_features_from_components(rf, nf, names)
    pos_u, feat_names_u = cheap_u_features_for_matching(split.r, split.n, split.img_size)
    pos = np.concatenate([pos_base, pos_u], axis=1)
    feat_names = feat_names_base + feat_names_u
    scale = pos.std(axis=0)
    scale[scale < 1e-6] = 1.0
    rng = np.random.default_rng(int(seed))
    n = split.size
    cost = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        rr = np.repeat(rf[i : i + 1], n, axis=0)
        cand_base, _ = cheap_pair_features_from_components(rr, nf, names)
        r_flat = split.r[i : i + 1].repeat(n, 1)
        cand_u, _ = cheap_u_features_for_matching(r_flat, split.n, split.img_size)
        cand = np.concatenate([cand_base, cand_u], axis=1)
        diff = (cand - pos[i : i + 1]) / scale
        cost[i] = np.mean(diff * diff, axis=1)
    np.fill_diagonal(cost, np.inf)
    finite = np.isfinite(cost)
    max_finite = float(cost[finite].max()) if finite.any() else 1e9
    cost[~finite] = max_finite + 1e6
    cost += rng.normal(0.0, float(jitter), size=cost.shape)
    row, col = linear_sum_assignment(cost)
    donors = np.empty(n, dtype=int)
    donors[row] = col
    # Rare fallback if jitter/finite replacement produced a fixed point.
    fixed = np.where(donors == np.arange(n))[0]
    for i in fixed:
        j = (i + 1) % n
        donors[i], donors[j] = donors[j], donors[i]
    report = matching_report(split, donors, feature_names_for_balance=feat_names)
    report.update({"matching_seed": int(seed), "matching_cost_mean": float(cost[np.arange(n), donors].mean())})
    return donors, report


def matching_report(split: SplitComponents, donors: np.ndarray, *, feature_names_for_balance: list[str] | None = None) -> dict[str, Any]:
    donors = np.asarray(donors, dtype=int)
    n = split.size
    x, y, names, _rows = make_pair_arrays(split, donors, feature_mode="pair")
    pos = x[y == 1]
    neg = x[y == 0]
    pooled_std = x.std(axis=0)
    pooled_std[pooled_std < 1e-6] = 1.0
    smd = np.abs((pos.mean(axis=0) - neg.mean(axis=0)) / pooled_std)
    try:
        from scipy.stats import ks_2samp

        ks = np.array([ks_2samp(pos[:, i], neg[:, i]).statistic for i in range(x.shape[1])])
    except Exception:
        ks = np.full(x.shape[1], np.nan)
    n_energy = np.linalg.norm(split.n.reshape(n, -1).numpy(), axis=1)
    rel_ediff = np.abs(n_energy - n_energy[donors]) / np.maximum(n_energy, 1e-12)
    pair_hashes = [tensor_hash(split.x[i]) for i in range(n)]
    dup = sum(1 for i, j in enumerate(donors) if pair_hashes[i] == pair_hashes[int(j)])
    xflat = split.x.reshape(n, -1).numpy()
    near = 0
    for i, j in enumerate(donors):
        d = np.linalg.norm(xflat[i] - xflat[int(j)]) / max(np.linalg.norm(xflat[i]), 1e-12)
        if d < 1e-4:
            near += 1
    return {
        "split": split.name,
        "count": int(n),
        "fixed_points": int(np.sum(donors == np.arange(n))),
        "donor_unique_fraction": float(np.unique(donors).size / max(1, n)),
        "positive_negative_n_marginal_same_multiset": bool(np.array_equal(np.sort(donors), np.arange(n))),
        "feature_smd_mean": float(np.nanmean(smd)),
        "feature_smd_max": float(np.nanmax(smd)),
        "feature_ks_mean": float(np.nanmean(ks)),
        "feature_ks_max": float(np.nanmax(ks)),
        "worst_smd_features": [
            {"feature": names[int(i)], "smd": float(smd[int(i)])}
            for i in np.argsort(-smd)[:10]
        ],
        "energy_relative_difference_mean": float(rel_ediff.mean()),
        "energy_relative_difference_max": float(rel_ediff.max()),
        "exact_duplicate_pairs": int(dup),
        "near_duplicate_pairs": int(near),
        "manifest_hash": sha256_text(stable_json({"split": split.name, "donors": donors.tolist()})),
    }


def label_histogram(labels: torch.Tensor) -> dict[str, Any]:
    arr = labels.detach().cpu().numpy().astype(int)
    valid = arr >= 0
    hist = {str(int(k)): int(v) for k, v in zip(*np.unique(arr[valid], return_counts=True))} if valid.any() else {}
    return {
        "count": int(arr.size),
        "labeled_count": int(valid.sum()),
        "unlabeled_count": int((~valid).sum()),
        "histogram_labeled_only": hist,
        "class_analysis_applicable": bool(valid.sum() >= 2 and len(hist) >= 2),
    }


def class_score_summary(labels: torch.Tensor, donors: np.ndarray, neg_scores: np.ndarray) -> dict[str, Any]:
    arr = labels.detach().cpu().numpy().astype(int)
    valid = (arr >= 0) & (arr[donors] >= 0)
    if valid.sum() < 2:
        return {"status": "not_applicable", "valid_pair_count": int(valid.sum())}
    same = arr[valid] == arr[donors][valid]
    if same.sum() == 0 or (~same).sum() == 0:
        return {"status": "not_applicable", "valid_pair_count": int(valid.sum()), "same_count": int(same.sum()), "cross_count": int((~same).sum())}
    vals = np.asarray(neg_scores)[valid]
    return {
        "status": "ok",
        "valid_pair_count": int(valid.sum()),
        "same_count": int(same.sum()),
        "cross_count": int((~same).sum()),
        "same_score_mean": float(vals[same].mean()),
        "cross_score_mean": float(vals[~same].mean()),
    }


def full_retrieval_from_matrix(score_matrix: np.ndarray) -> dict[str, float]:
    n = score_matrix.shape[0]
    ranks = []
    for i in range(n):
        order = np.argsort(-score_matrix[i])
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks = np.asarray(ranks)
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def fixed_32_candidate_manifests(
    split: SplitComponents,
    *,
    count: int = 20,
    donors_per_anchor: int = 32,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rf, names = cheap_component_features_for_matching(split.r, split.img_size)
    nf, _ = cheap_component_features_for_matching(split.n, split.img_size)
    pos, _ = cheap_pair_features_from_components(rf, nf, names)
    scale = pos.std(axis=0)
    scale[scale < 1e-6] = 1.0
    rng = np.random.default_rng(int(seed))
    manifests = []
    n = split.size
    for m in range(int(count)):
        candidates = []
        for i in range(n):
            rr = np.repeat(rf[i : i + 1], n, axis=0)
            cand_feat, _ = cheap_pair_features_from_components(rr, nf, names)
            diff = (cand_feat - pos[i : i + 1]) / scale
            cost = np.mean(diff * diff, axis=1)
            cost[i] = -np.inf
            # Draw from the nearest half to avoid identical manifests.
            order = np.argsort(cost)
            near_pool = order[: max(donors_per_anchor * 3, donors_per_anchor)]
            neg = rng.choice(near_pool[near_pool != i], size=min(donors_per_anchor - 1, n - 1), replace=False)
            cand = np.concatenate([[i], neg.astype(int)])
            candidates.append(cand.tolist())
        payload = {"manifest_id": f"fixed32_{m:02d}", "seed": int(seed + m), "candidates": candidates}
        payload["manifest_hash"] = sha256_text(stable_json(payload))
        manifests.append(payload)
    return manifests


def candidate_retrieval_metrics(score_matrix: np.ndarray, manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    recalls1 = []
    recalls5 = []
    mrrs = []
    for manifest in manifests:
        ranks = []
        for i, cand in enumerate(manifest["candidates"]):
            cand = np.asarray(cand, dtype=int)
            scores = score_matrix[i, cand]
            order = np.argsort(-scores)
            rank = int(np.where(cand[order] == i)[0][0]) + 1
            ranks.append(rank)
            rows.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "manifest_hash": manifest["manifest_hash"],
                    "anchor_local_idx": i,
                    "rank": rank,
                    "candidate_count": int(cand.size),
                    "top_candidate": int(cand[order[0]]),
                    "positive_score": float(score_matrix[i, i]),
                    "top_score": float(scores[order[0]]),
                }
            )
        ranks = np.asarray(ranks)
        recalls1.append(float(np.mean(ranks <= 1)))
        recalls5.append(float(np.mean(ranks <= 5)))
        mrrs.append(float(np.mean(1.0 / ranks)))
    return {
        "recall_at_1": bootstrap_ci(np.asarray(recalls1), seed=11),
        "recall_at_5": bootstrap_ci(np.asarray(recalls5), seed=12),
        "mrr": bootstrap_ci(np.asarray(mrrs), seed=13),
        "manifest_hashes": [m["manifest_hash"] for m in manifests],
        "manifest_count": int(len(manifests)),
    }, rows


def paired_margin_metrics(pos: np.ndarray, neg: np.ndarray, *, seed: int = 0) -> dict[str, Any]:
    margin = np.asarray(pos) - np.asarray(neg)
    wins = (margin > 0).astype(float) + 0.5 * (margin == 0).astype(float)
    try:
        from scipy.stats import binomtest

        p_value = float(binomtest(int(np.sum(margin > 0)), n=margin.size, p=0.5, alternative="greater").pvalue)
    except Exception:
        p_value = float("nan")
    return {
        "paired_win_rate": bootstrap_ci(wins, seed=seed),
        "median_margin": float(np.median(margin)),
        "median_margin_ci": bootstrap_ci(margin, seed=seed + 1, fn=np.median),
        "sign_test_p_value_greater": p_value,
    }


def score_error_correlations(score_matrix: np.ndarray, nulls: torch.Tensor) -> dict[str, float]:
    n = score_matrix.shape[0]
    nf = nulls.reshape(n, -1).numpy()
    scores_all = []
    errors_all = []
    scores_neg = []
    errors_neg = []
    per_anchor = []
    for i in range(n):
        err = np.linalg.norm(nf - nf[i : i + 1], axis=1)
        sc = score_matrix[i]
        scores_all.append(sc)
        errors_all.append(err)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scores_neg.append(sc[mask])
        errors_neg.append(err[mask])
        per_anchor.append(spearman_np(sc[mask], err[mask]))
    return {
        "spearman_including_positive": spearman_np(np.concatenate(scores_all), np.concatenate(errors_all)),
        "spearman_negatives_only_global": spearman_np(np.concatenate(scores_neg), np.concatenate(errors_neg)),
        "mean_per_anchor_spearman_on_negatives": float(np.nanmean(per_anchor)),
    }


def k_prefix_indices(kmax: int, ks: Iterable[int]) -> dict[int, list[int]]:
    base = list(range(int(kmax)))
    out: dict[int, list[int]] = {}
    for k in ks:
        if int(k) > int(kmax):
            raise ValueError("K cannot exceed Kmax.")
        out[int(k)] = base[: int(k)]
    return out


def overlap_count(a: Iterable[int], b: Iterable[int]) -> int:
    return len(set(int(v) for v in a) & set(int(v) for v in b))
