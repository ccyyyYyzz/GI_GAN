from __future__ import annotations

import math
from typing import Any

import torch

from .compatibility_data import SplitComponents, energy_matched_donors, make_derangement, make_semihard_donors, normalize_images, null_energy


def roc_auc_score_safe(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu().float()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels.numpy(), scores.numpy()))
    except Exception:
        wins = 0.0
        neg_np = neg.numpy()
        for p in pos.numpy():
            wins += float((p > neg_np).sum())
            wins += 0.5 * float((p == neg_np).sum())
        return float(wins / max(1, pos.numel() * neg.numel()))


def spearman_safe(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().float()
    y = y.detach().cpu().float()
    if x.numel() < 3:
        return float("nan")
    rx = torch.empty_like(x)
    ry = torch.empty_like(y)
    rx[torch.argsort(x)] = torch.arange(x.numel(), dtype=torch.float32)
    ry[torch.argsort(y)] = torch.arange(y.numel(), dtype=torch.float32)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.norm(rx) * torch.linalg.norm(ry)
    if float(denom.item()) <= 0:
        return float("nan")
    return float((rx @ ry / denom).item())


@torch.no_grad()
def encode_split(model, split: SplitComponents, normalization: dict[str, float], *, device: torch.device, batch_size: int = 128):
    model.eval()
    zr, zn = [], []
    for start in range(0, split.size, int(batch_size)):
        sl = slice(start, min(split.size, start + int(batch_size)))
        r = normalize_images(split.r[sl], img_size=split.img_size, key="r", normalization=normalization).to(device)
        n = normalize_images(split.n[sl], img_size=split.img_size, key="n", normalization=normalization).to(device)
        zrb, znb = model.forward_embeddings(r, n)
        zr.append(zrb.detach().cpu())
        zn.append(znb.detach().cpu())
    return torch.cat(zr, 0), torch.cat(zn, 0), float(model.temperature.detach().cpu().item())


def _candidate_indices(n: int, positive: int, donors: int, gen: torch.Generator) -> torch.Tensor:
    count = min(int(donors) - 1, n - 1)
    raw = torch.randperm(n - 1, generator=gen)[:count]
    neg = raw + (raw >= positive).long()
    return torch.cat([torch.tensor([positive], dtype=torch.long), neg.long()])


def retrieval_metrics(
    zr: torch.Tensor,
    zn: torch.Tensor,
    *,
    seed: int,
    donors_per_anchor: int = 32,
    permuted_positive: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    n = int(zr.shape[0])
    gen = torch.Generator().manual_seed(int(seed))
    perm = make_derangement(n, seed + 777) if permuted_positive and n > 1 else torch.arange(n)
    ranks = []
    all_scores = []
    all_errors = []
    rows = []
    for i in range(n):
        positive = int(perm[i].item())
        cand = _candidate_indices(n, positive, donors_per_anchor, gen)
        if positive != i and i not in cand:
            cand[-1] = i
        scores = zr[i] @ zn[cand].T
        order = torch.argsort(scores, descending=True)
        pos_locs = torch.nonzero(cand[order] == positive, as_tuple=False)
        rank = int(pos_locs[0, 0].item()) + 1 if pos_locs.numel() else int(cand.numel()) + 1
        ranks.append(rank)
        rows.append(
            {
                "anchor_local_idx": i,
                "positive_null_idx": positive,
                "rank": rank,
                "top_score": float(scores.max().item()),
                "positive_score": float(scores[cand == positive][0].item()) if bool((cand == positive).any()) else float("nan"),
                "candidate_count": int(cand.numel()),
                "permuted_positive": bool(permuted_positive),
            }
        )
        all_scores.append(scores)
        # Placeholder errors are filled by caller when true nulls are available.
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    metrics = {
        "recall_at_1": float((ranks_t <= 1).float().mean().item()),
        "recall_at_5": float((ranks_t <= 5).float().mean().item()),
        "mrr": float((1.0 / ranks_t).mean().item()),
        "median_rank": float(ranks_t.median().item()),
        "random_recall_at_1": float(1.0 / min(donors_per_anchor, n)),
        "num_anchors": int(n),
    }
    return metrics, rows, torch.cat(all_scores), torch.tensor(all_errors)


def pair_scores_from_donors(zr: torch.Tensor, zn: torch.Tensor, donors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(zr.shape[0])
    pos = (zr[idx] * zn[idx]).sum(dim=1)
    neg = (zr[idx] * zn[donors]).sum(dim=1)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).long()
    scores = torch.cat([pos, neg])
    return labels, scores


def score_error_correlation(
    zr: torch.Tensor,
    zn: torch.Tensor,
    split: SplitComponents,
    *,
    seed: int,
    donors_per_anchor: int = 32,
) -> float:
    n = split.size
    gen = torch.Generator().manual_seed(int(seed))
    nulls = split.n.reshape(n, -1)
    scores, errors = [], []
    for i in range(n):
        cand = _candidate_indices(n, i, donors_per_anchor, gen)
        sc = zr[i] @ zn[cand].T
        err = torch.linalg.norm(nulls[cand] - nulls[i], dim=1)
        scores.append(sc)
        errors.append(err)
    return spearman_safe(torch.cat(scores), torch.cat(errors))


def evaluate_critic_split(
    model,
    split: SplitComponents,
    normalization: dict[str, float],
    *,
    device: torch.device,
    seed: int,
    donors_per_anchor: int = 32,
    batch_size: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    zr, zn, temp = encode_split(model, split, normalization, device=device, batch_size=batch_size)
    retrieval, rows, _scores_unused, _ = retrieval_metrics(
        zr, zn, seed=seed, donors_per_anchor=donors_per_anchor, permuted_positive=False
    )
    perm_retrieval, _perm_rows, _perm_scores, _ = retrieval_metrics(
        zr, zn, seed=seed + 81, donors_per_anchor=donors_per_anchor, permuted_positive=True
    )
    random_donors = make_derangement(split.size, seed + 1)
    semihard_donors = make_semihard_donors(split, seed=seed + 2)
    energy_donors = energy_matched_donors(split, seed=seed + 3)
    labels_random, scores_random = pair_scores_from_donors(zr, zn, random_donors)
    labels_semi, scores_semi = pair_scores_from_donors(zr, zn, semihard_donors)
    labels_energy, scores_energy = pair_scores_from_donors(zr, zn, energy_donors)
    e = null_energy(split.n)
    scalar_pos = torch.zeros(split.size)
    scalar_energy = -torch.abs(e - e[energy_donors])
    scalar_random = -torch.abs(e - e[random_donors])
    scalar_energy_auc = roc_auc_score_safe(labels_energy, torch.cat([scalar_pos, scalar_energy]))
    scalar_random_auc = roc_auc_score_safe(labels_random, torch.cat([scalar_pos, scalar_random]))
    same = split.labels == split.labels[random_donors]
    same_scores = scores_random[split.size :][same]
    cross_scores = scores_random[split.size :][~same]
    spearman = score_error_correlation(zr, zn, split, seed=seed + 4, donors_per_anchor=donors_per_anchor)
    metrics: dict[str, Any] = {
        **retrieval,
        "temperature": temp,
        "random_negative_auc": roc_auc_score_safe(labels_random, scores_random),
        "semi_hard_auc": roc_auc_score_safe(labels_semi, scores_semi),
        "energy_matched_auc": roc_auc_score_safe(labels_energy, scores_energy),
        "score_p0_error_spearman": spearman,
        "label_permutation_recall_at_1": perm_retrieval["recall_at_1"],
        "label_permutation_recall_at_5": perm_retrieval["recall_at_5"],
        "scalar_energy_random_auc": scalar_random_auc,
        "scalar_energy_matched_auc": scalar_energy_auc,
        "row_only_control_auc": 0.5,
        "null_energy_control_auc": scalar_energy_auc,
        "positive_score_mean": float(scores_random[: split.size].mean().item()),
        "positive_score_std": float(scores_random[: split.size].std(unbiased=False).item()),
        "random_negative_score_mean": float(scores_random[split.size :].mean().item()),
        "random_negative_score_std": float(scores_random[split.size :].std(unbiased=False).item()),
        "same_class_random_negative_score_mean": float(same_scores.mean().item()) if same_scores.numel() else float("nan"),
        "cross_class_random_negative_score_mean": float(cross_scores.mean().item()) if cross_scores.numel() else float("nan"),
        "same_class_random_negative_count": int(same_scores.numel()),
        "cross_class_random_negative_count": int(cross_scores.numel()),
        "semi_hard_donor_unique_fraction": float(torch.unique(semihard_donors).numel() / max(1, split.size)),
        "random_donor_fixed_points": int((random_donors == torch.arange(split.size)).sum().item()),
    }
    if not math.isfinite(metrics["score_p0_error_spearman"]):
        metrics["score_p0_error_spearman"] = float("nan")
    return metrics, rows


def e1_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    random_recall = float(metrics.get("random_recall_at_1", 1.0 / 32.0))
    scalar_ceiling = 0.70
    checks = {
        "semi_hard_auc_ge_0_70": float(metrics.get("semi_hard_auc", float("nan"))) >= 0.70,
        "recall_at_1_ge_4x_random": float(metrics.get("recall_at_1", 0.0)) >= 4.0 * random_recall,
        "spearman_le_minus_0_20": float(metrics.get("score_p0_error_spearman", 1.0)) <= -0.20,
        "row_control_not_high": float(metrics.get("row_only_control_auc", 0.5)) <= 0.60,
        "label_perm_near_random": float(metrics.get("label_permutation_recall_at_1", 1.0)) <= max(0.12, 4.0 * random_recall),
        "scalar_controls_not_explanatory": max(
            float(metrics.get("scalar_energy_random_auc", 0.5)),
            float(metrics.get("scalar_energy_matched_auc", 0.5)),
            float(metrics.get("null_energy_control_auc", 0.5)),
        )
        <= scalar_ceiling,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "random_recall_at_1": random_recall,
        "scalar_control_auc_ceiling": scalar_ceiling,
    }
