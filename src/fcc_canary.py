"""FCC diagnostic canary (64x64) helper logic.

Implements the data construction, *deployable* nuisance controls, and mechanical
classification for a clean re-run of the Feasible Counterfactual Compatibility
(FCC) diagnostic described in
``_inbox/FCC_THEORY_AND_PROMPT_BUNDLE/FCC_THEORY_RANGE_NULL_COMPATIBILITY.md``.

Design notes (why this module exists instead of reusing eval_compatibility):

* The original Phase-1 scalar control compared the *true matched* null energy
  (``-|e_i - e_i| = 0``) against donor energies, so positives scored exactly 0
  and any non-identical donor scored < 0 -> AUC ~ 1.0 for *any* feature. That is
  a non-deployable oracle, not a control. Phase 1.1 already retired it as
  ``oracle_true_null_energy_distance_auc`` (non_deployable=true). This module
  only ever uses *deployable* baselines: a classifier that sees phi(r, n) of a
  candidate pair and never the truth.
* All projections are exact, matrix-free (no dense P0), float64. No clipping is
  applied before projection / feasibility checks.

Heavy primitives (exact projector, phi features, assignment-based
nuisance-balanced derangement, retrieval/auc/bootstrap helpers) are reused from
``src.measurement``, ``src.projections``, ``src.compatibility_data`` and
``src.phase1_1_controls``.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .compatibility_data import SplitComponents, decompose_split, make_derangement
from .phase1_1_controls import (
    bootstrap_ci,
    make_pair_arrays,
    nuisance_balanced_derangement,
    pair_features,
    paired_margin_metrics,
    random_derangement,
    tie_aware_auc,
)


# --------------------------------------------------------------------------- #
# Raw split container consumed by decompose_split (needs .name/.x/.labels/.indices)
# --------------------------------------------------------------------------- #
@dataclass
class RawSplit:
    name: str
    x: torch.Tensor          # [N, 1, H, W] float in [0, 1]
    labels: torch.Tensor     # [N] long (-1 for unlabeled)
    indices: torch.Tensor    # [N] long source indices into the base dataset


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- #
# Consumed-hash exclusion set
# --------------------------------------------------------------------------- #
def _iter_csv_column(path: Path, column: str) -> Iterable[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                return
            for row in reader:
                v = row.get(column)
                if v:
                    yield str(v)
    except Exception:
        return


def collect_consumed_raw_hashes(repo_root: Path, base, *, verbose: bool = True) -> dict[str, Any]:
    """Union of raw STL10 sha256 hashes already consumed by prior runs.

    Sources:
      A. Every ``sample_hash_audit.csv`` under ``outputs/`` (raw_sha256 column).
      B. The certification pool ``split_train_indices_stl10_train_unlabeled.npy``
         (50000 indices -> raw_hash via the base dataset bytes).
    The hashes are ``sha256(STL10.data[i].tobytes())`` and are image-size
    agnostic, so they identify a source image regardless of the 64 vs 96 px
    transform used by a given phase.
    """
    repo_root = Path(repo_root)
    raw_hashes: set[str] = set()
    sources: list[dict[str, Any]] = []

    audit_files = sorted((repo_root / "outputs").rglob("sample_hash_audit*.csv"))
    for path in audit_files:
        before = len(raw_hashes)
        for h in _iter_csv_column(path, "raw_sha256"):
            raw_hashes.add(h)
        sources.append({"source": str(path.relative_to(repo_root)), "added": len(raw_hashes) - before})

    cert_npy = Path("E:/ns_mc_gan_gi/results/cert_package_20260612/cache/split_train_indices_stl10_train_unlabeled.npy")
    cert_added = 0
    cert_present = bool(cert_npy.exists())
    if cert_present:
        idx = np.load(cert_npy).astype(np.int64).tolist()
        before = len(raw_hashes)
        for si in idx:
            try:
                item = base.data[int(si)]
                raw_hashes.add(sha256_bytes(item.tobytes()))
                cert_added += 1
            except Exception:
                continue
        sources.append({"source": str(cert_npy), "present": True, "indices": len(idx), "added": len(raw_hashes) - before})
    else:
        # Portability guard: the cert pool supplies the bulk of the exclusion set.
        # A re-run on a machine without it would silently admit images this run
        # excluded, so make the gap loud and machine-readable instead of silent.
        sources.append({"source": str(cert_npy), "present": False, "indices": 0, "added": 0})
        print(f"[exclusion][WARNING] cert pool NOT found at {cert_npy}; exclusion set is "
              f"sample_hash_audit-only and is NOT reproducible against the certification pool.")

    if verbose:
        print(f"[exclusion] collected {len(raw_hashes)} consumed raw hashes from {len(sources)} sources "
              f"(cert pool present={cert_present}, indices hashed: {cert_added}).")
    return {"raw_hashes": raw_hashes, "sources": sources, "cert_pool_present": cert_present}


# --------------------------------------------------------------------------- #
# Hash-clean 64x64 split builder
# --------------------------------------------------------------------------- #
def build_hash_clean_split(
    base,
    transform,
    *,
    counts: dict[str, int],
    exclude_raw_hashes: set[str],
    scan_start: int = 0,
    seed: int = 0,
) -> tuple[dict[str, RawSplit], dict[str, Any]]:
    """Scan the base dataset for unique, unconsumed raw images and slice splits.

    Dedups by ``raw_hash`` (first occurrence wins), skips any hash in
    ``exclude_raw_hashes``, and collects ``sum(counts.values())`` images. The
    collected unique indices are then sliced into named splits in declaration
    order so the three splits are mutually disjoint by construction.
    """
    total_needed = int(sum(counts.values()))
    n_base = len(base)
    seen: set[str] = set()
    unique_indices: list[int] = []
    unique_hashes: list[str] = []
    skipped_consumed = 0
    skipped_internal_dup = 0

    order = list(range(int(scan_start), n_base)) + list(range(0, int(scan_start)))
    for si in order:
        item = base.data[int(si)]
        h = sha256_bytes(item.tobytes())
        if h in exclude_raw_hashes:
            skipped_consumed += 1
            continue
        if h in seen:
            skipped_internal_dup += 1
            continue
        seen.add(h)
        unique_indices.append(int(si))
        unique_hashes.append(h)
        if len(unique_indices) >= total_needed:
            break
    if len(unique_indices) < total_needed:
        raise RuntimeError(
            f"Only found {len(unique_indices)} unique unconsumed images, need {total_needed}."
        )

    splits: dict[str, RawSplit] = {}
    span_manifest: dict[str, Any] = {}
    cursor = 0
    for name, count in counts.items():
        sel = unique_indices[cursor : cursor + int(count)]
        sel_hashes = unique_hashes[cursor : cursor + int(count)]
        cursor += int(count)
        xs, labels = [], []
        for si in sel:
            img, label = base[int(si)]
            xs.append(transform(img))
            labels.append(int(label) if label is not None and int(label) >= 0 else -1)
        x = torch.stack(xs, 0).float()          # [count, 1, H, W], values in [0,1], NO clipping beyond ToTensor
        if x.ndim == 3:
            x = x.unsqueeze(1)
        splits[name] = RawSplit(
            name=name,
            x=x,
            labels=torch.tensor(labels, dtype=torch.long),
            indices=torch.tensor(sel, dtype=torch.long),
        )
        span_manifest[name] = {
            "count": int(count),
            "source_index_min": int(min(sel)),
            "source_index_max": int(max(sel)),
            "source_indices_sha256": sha256_bytes(np.asarray(sel, dtype=np.int64).tobytes()),
            "raw_hashes_sha256": sha256_bytes("".join(sel_hashes).encode("utf-8")),
        }
    manifest = {
        "counts": dict(counts),
        "scan_start": int(scan_start),
        "scanned_images_for_unique": int(cursor),
        "skipped_consumed": int(skipped_consumed),
        "skipped_internal_duplicates": int(skipped_internal_dup),
        "excluded_hash_pool_size": int(len(exclude_raw_hashes)),
        "spans": span_manifest,
        "note": "Hash-clean: first occurrence of each raw STL10 hash, excluding all consumed-hash sources.",
    }
    return splits, manifest


def cross_split_overlap(splits: dict[str, SplitComponents]) -> dict[str, int]:
    """Raw-index overlap between named splits (should all be 0)."""
    idx = {name: set(int(i) for i in comp.source_indices.tolist()) for name, comp in splits.items()}
    names = list(idx)
    out = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out[f"{names[a]}__{names[b]}"] = len(idx[names[a]] & idx[names[b]])
    return out


# --------------------------------------------------------------------------- #
# Exact geometry / feasibility runtime checks (Task A) -- float64, matrix-free
# --------------------------------------------------------------------------- #
@torch.no_grad()
def geometry_checks(split: SplitComponents, measurement, *, device: torch.device, sample_count: int = 32) -> dict[str, Any]:
    from .projections import get_exact_projector

    count = min(int(sample_count), split.size)
    projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
    flat = split.x[:count].to(device).reshape(count, -1).double()
    y = projector.A_forward(flat)
    r = projector.row_project_flat(flat)
    n = projector.null_project_flat(flat)
    denom_y = torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    denom_x = torch.linalg.norm(flat, dim=1).clamp_min(1e-12)
    ap0_rel = torch.linalg.norm(projector.A_forward(n), dim=1) / denom_y
    recon = torch.linalg.norm((r + n) - flat, dim=1) / denom_x
    dot = torch.sum(r * n, dim=1).abs() / (
        torch.linalg.norm(r, dim=1).clamp_min(1e-12) * torch.linalg.norm(n, dim=1).clamp_min(1e-12)
    )
    return {
        "sample_count": int(count),
        "float64_A_P0_rel_max": float(ap0_rel.max().item()),
        "reconstruction_rel_max": float(recon.max().item()),
        "orthogonality_cos_max": float(dot.max().item()),
        "pass": bool(ap0_rel.max() < 1e-9 and recon.max() < 1e-10 and dot.max() < 1e-8),
        "projector_info": projector.info_dict(),
    }


@torch.no_grad()
def feasibility_check(split: SplitComponents, measurement, donors: np.ndarray, *, device: torch.device, max_pairs: int | None = None) -> dict[str, Any]:
    donors_t = torch.as_tensor(np.asarray(donors), dtype=torch.long)
    count = split.size if max_pairs is None else min(split.size, int(max_pairs))
    r = split.r[:count].to(device)
    n = split.n[donors_t[:count]].to(device)
    y = split.y[:count].to(device)
    u = r + n  # feasible counterfactual; A u = A r + A n = y exactly (A n = 0)
    rel_u = (torch.linalg.norm(measurement.A_forward(u) - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12))
    rel_n = (torch.linalg.norm(measurement.A_forward(n), dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12))
    return {
        "pairs_checked": int(count),
        "u_rel_mean": float(rel_u.mean().item()),
        "u_rel_max": float(rel_u.max().item()),
        "donor_null_rel_max": float(rel_n.max().item()),
        "pass_float32_proxy": bool(rel_u.max() < 1e-5 and rel_n.max() < 1e-5),
    }


# --------------------------------------------------------------------------- #
# Deployable baseline classifiers (pure torch; no sklearn dependency)
# --------------------------------------------------------------------------- #
@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "StandardScaler":
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        return StandardScaler(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        z = (x - self.mean) / self.std
        return np.clip(z, -8.0, 8.0)


class DeployableClassifier:
    """Logistic regression or small MLP fit with full-batch Adam on a deployable
    (truth-blind) phi(r, n) feature vector. Returns calibrated-ish probabilities."""

    def __init__(self, kind: str = "logistic", hidden: int = 64, l2: float = 1e-3, steps: int = 400, lr: float = 5e-2, seed: int = 0):
        self.kind = kind
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.steps = int(steps)
        self.lr = float(lr)
        self.seed = int(seed)
        self.scaler: StandardScaler | None = None
        self.model: torch.nn.Module | None = None

    def _build(self, d: int) -> torch.nn.Module:
        torch.manual_seed(self.seed)
        if self.kind == "logistic":
            return torch.nn.Linear(d, 1)
        return torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden), torch.nn.SiLU(), torch.nn.Linear(self.hidden, 1)
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DeployableClassifier":
        self.scaler = StandardScaler.fit(x)
        xt = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
        self.model = self._build(xt.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
        lossf = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        with torch.enable_grad():  # robust even if called under an outer torch.no_grad()
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.model(xt), yt)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.scaler is not None
        xt = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        return torch.sigmoid(self.model(xt)).reshape(-1).cpu().numpy()


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(tie_aware_auc(labels, scores))


def fit_deployable_baselines(
    train: SplitComponents,
    *,
    feature_modes: tuple[str, ...] = ("pair", "sum", "row", "null"),
    kinds: tuple[str, ...] = ("logistic", "mlp"),
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Fit truth-blind classifiers on TRAIN positives vs random-derangement negatives."""
    donors = random_derangement(train.size, seed=seed + 1)
    out: dict[str, dict[str, Any]] = {}
    for mode in feature_modes:
        x, y, names, _rows = make_pair_arrays(train, donors, feature_mode=mode)
        for kind in kinds:
            clf = DeployableClassifier(kind=kind, seed=seed + 7).fit(x, y)
            train_auc = _safe_auc(y, clf.score(x))
            out[f"{mode}_{kind}"] = {"clf": clf, "mode": mode, "kind": kind, "n_features": int(x.shape[1]), "train_auc": train_auc, "feature_names": names}
    return out


def baseline_pair_auc(
    clf_entry: dict[str, Any],
    split: SplitComponents,
    donors: np.ndarray,
) -> dict[str, float]:
    """AUC of a fitted deployable classifier separating matched vs mismatched on `split`."""
    clf: DeployableClassifier = clf_entry["clf"]
    mode: str = clf_entry["mode"]
    x, y, _names, _rows = make_pair_arrays(split, np.asarray(donors), feature_mode=mode)
    s = clf.score(x)
    return {"auc": _safe_auc(y, s), "n": int(y.shape[0])}


# --------------------------------------------------------------------------- #
# Fixed-32 retrieval shared by FCC and deployable baselines
# --------------------------------------------------------------------------- #
def build_fixed32_manifest(split: SplitComponents, *, donors_per_anchor: int = 32, seed: int = 0, hard: bool = True) -> list[dict[str, Any]]:
    """One candidate set per anchor: [positive=i] + (k-1) donors.

    hard=True draws donors from the nuisance-feature nearest neighbours of the
    true pair (harder negatives); hard=False draws uniformly at random.
    """
    from .phase1_1_controls import cheap_component_features_for_matching, cheap_pair_features_from_components

    n = split.size
    rng = np.random.default_rng(int(seed))
    k = int(donors_per_anchor)
    manifests = []
    if hard:
        rf, names = cheap_component_features_for_matching(split.r, split.img_size)
        nf, _ = cheap_component_features_for_matching(split.n, split.img_size)
        pos, _ = cheap_pair_features_from_components(rf, nf, names)
        scale = pos.std(axis=0)
        scale[scale < 1e-6] = 1.0
    candidates = []
    for i in range(n):
        if hard:
            rr = np.repeat(rf[i : i + 1], n, axis=0)
            cand_feat, _ = cheap_pair_features_from_components(rr, nf, names)
            cost = np.mean(((cand_feat - pos[i : i + 1]) / scale) ** 2, axis=1)
            cost[i] = np.inf
            order = np.argsort(cost)
            pool = order[order != i][: max(k * 3, k)]
            neg = rng.choice(pool, size=min(k - 1, n - 1), replace=False)
        else:
            pool = np.delete(np.arange(n), i)
            neg = rng.choice(pool, size=min(k - 1, n - 1), replace=False)
        candidates.append([i] + neg.astype(int).tolist())
    payload = {
        "manifest_id": f"fixed{k}_{'hard' if hard else 'rand'}_seed{seed}",
        "donors_per_anchor": k,
        "hard": bool(hard),
        "candidates": candidates,
    }
    payload["manifest_hash"] = sha256_bytes(str(candidates).encode("utf-8"))
    manifests.append(payload)
    return manifests


def retrieval_from_score_fn(manifest: dict[str, Any], score_row_fn) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Generic fixed-candidate retrieval. score_row_fn(i, cand)->scores[len(cand)]."""
    ranks = []
    rows = []
    for i, cand in enumerate(manifest["candidates"]):
        cand = np.asarray(cand, dtype=int)
        scores = np.asarray(score_row_fn(i, cand), dtype=float)
        order = np.argsort(-scores)
        rank = int(np.where(cand[order] == i)[0][0]) + 1
        ranks.append(rank)
        rows.append({"anchor_local_idx": int(i), "rank": int(rank), "candidate_count": int(cand.size),
                     "positive_score": float(scores[np.where(cand == i)[0][0]]), "top_score": float(scores[order[0]])})
    ranks_a = np.asarray(ranks)
    metrics = {
        "recall_at_1": float(np.mean(ranks_a <= 1)),
        "recall_at_5": float(np.mean(ranks_a <= 5)),
        "mrr": float(np.mean(1.0 / ranks_a)),
        "median_rank": float(np.median(ranks_a)),
        "random_recall_at_1": float(1.0 / int(manifest["donors_per_anchor"])),
        "num_anchors": int(len(ranks)),
    }
    return metrics, rows


def fcc_score_matrix(zr: torch.Tensor, zn: torch.Tensor) -> np.ndarray:
    return (zr @ zn.T).cpu().numpy()


def baseline_score_rows(clf_entry: dict[str, Any], split: SplitComponents):
    """Return a score_row_fn(i, cand) for a deployable classifier over (r_i, n_cand)."""
    clf: DeployableClassifier = clf_entry["clf"]
    mode: str = clf_entry["mode"]

    def score_row(i: int, cand: np.ndarray) -> np.ndarray:
        cand = np.asarray(cand, dtype=int)
        r = split.r[i : i + 1].repeat(len(cand), 1)
        n = split.n[torch.as_tensor(cand, dtype=torch.long)]
        if mode == "pair":
            x, _ = pair_features(r, n, split.img_size)
        elif mode == "sum":
            from .phase1_1_controls import sum_image_features
            x, _ = sum_image_features(r, n, split.img_size)
        elif mode == "row":
            from .phase1_1_controls import row_only_features
            x, _ = row_only_features(r, split.img_size)
        elif mode == "null":
            from .phase1_1_controls import null_only_features
            x, _ = null_only_features(n, split.img_size)
        else:
            raise ValueError(mode)
        return clf.score(x)

    return score_row


# --------------------------------------------------------------------------- #
# Mechanical classification (Task E)
# --------------------------------------------------------------------------- #
def classify_fcc(evidence: dict[str, Any], *, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    t = {
        "random_recall_mult": 4.0,     # real-pair retrieval must beat 4x random
        "structural_auc_min": 0.70,    # FCC AUC on nuisance-balanced negs
        "deployable_neutered_max": 0.60,  # deployable baseline must drop near chance on balanced negs
        "fcc_minus_deployable_min": 0.05,  # FCC must exceed best deployable by this margin
        "balance_smd_max": 0.25,       # nuisance-balanced negatives are actually balanced
        "label_perm_max_mult": 4.0,    # label-shuffle retrieval stays near random
        "deployable_signal_floor": 0.60,  # a deployable baseline above this counts as "real-pair signal exists"
    }
    if thresholds:
        t.update(thresholds)

    geom_ok = bool(evidence["geometry"]["pass"]) and bool(evidence["feasibility"]["nuisance_balanced"]["pass_float32_proxy"]) and bool(evidence["feasibility"]["random"]["pass_float32_proxy"])
    if not geom_ok:
        return {"classification": "INVALID_EXPERIMENT", "reason": "projector/feasibility checks failed",
                "thresholds": t, "geometry_ok": geom_ok, "checks": {"geometry_ok": geom_ok}, "key_values": {}}

    rr1 = float(evidence["layer_a"]["recall_at_1"])
    rand_rr1 = float(evidence["layer_a"]["random_recall_at_1"])
    # "real-pair signal exists" if EITHER the trained critic OR a deployable baseline separates real pairs
    # above chance. (A weak critic alone must not be read as "no signal" when a scalar baseline clearly separates.)
    critic_real_pair_signal = rr1 >= t["random_recall_mult"] * rand_rr1
    best_deploy_rand_auc = float(evidence["layer_b"].get("best_deployable_random_auc", 0.5))
    deployable_separates = best_deploy_rand_auc >= t["deployable_signal_floor"]
    real_pair_signal = bool(critic_real_pair_signal or deployable_separates)

    label_perm_rr1 = float(evidence["layer_a"].get("label_permutation_recall_at_1", 1.0))
    label_shuffle_ok = label_perm_rr1 <= max(t["label_perm_max_mult"] * rand_rr1, 0.10)

    fcc_bal_auc = float(evidence["layer_b"]["fcc"]["balanced_auc"])
    best_deploy_bal_auc = float(evidence["layer_b"]["best_deployable_balanced_auc"])
    smd_max = float(evidence["layer_b"]["balance"]["feature_smd_max"])
    balanced_well = smd_max <= t["balance_smd_max"]

    deployable_neutered = best_deploy_bal_auc <= t["deployable_neutered_max"]
    fcc_strong = fcc_bal_auc >= t["structural_auc_min"]
    fcc_exceeds = (fcc_bal_auc - best_deploy_bal_auc) >= t["fcc_minus_deployable_min"]

    # "Structural" is a CRITIC claim: the trained critic must itself retrieve real pairs AND exceed
    # deployable baselines on neutralised nuisance-balanced negatives. A deployable-only separation is never structural.
    # The distortion (balanced-AUC) arm is GATED on the negatives being genuinely balanced: if `balanced_well`
    # is false the arm is inconclusive-by-construction and may not count toward a structural claim.
    structural_balanced_arm = bool(balanced_well and fcc_strong and deployable_neutered and fcc_exceeds)
    structural = bool(critic_real_pair_signal and label_shuffle_ok and structural_balanced_arm)

    if structural:
        transfer = evidence.get("layer_c", {}).get("transfer_confirmed", None)
        if transfer is True:
            classification = "STRUCTURAL_COMPATIBILITY_CONFIRMED"
        else:
            classification = "REAL_PAIR_SIGNAL_BUT_NO_GENERATED_TRANSFER"
    elif not real_pair_signal:
        classification = "NO_COMPATIBILITY_SIGNAL"
    else:
        # Real-pair retrieval exists but deployable nuisance baselines explain it
        # (not neutered) or FCC does not exceed them on balanced negatives.
        classification = "ONLY_SCALAR_OR_ARTIFACT_SIGNAL"

    return {
        "classification": classification,
        "thresholds": t,
        "checks": {
            "geometry_ok": geom_ok,
            "real_pair_signal": bool(real_pair_signal),
            "critic_real_pair_signal": bool(critic_real_pair_signal),
            "deployable_separates_real_pairs": bool(deployable_separates),
            "label_shuffle_near_random": bool(label_shuffle_ok),
            "fcc_balanced_auc_ge_min": bool(fcc_strong),
            "deployable_neutered_on_balanced": bool(deployable_neutered),
            "fcc_exceeds_deployable": bool(fcc_exceeds),
            "negatives_well_balanced": bool(balanced_well),
            "structural_balanced_arm": bool(structural_balanced_arm),
        },
        "key_values": {
            "recall_at_1": rr1,
            "random_recall_at_1": rand_rr1,
            "recall_at_1_over_random": rr1 / max(rand_rr1, 1e-12),
            "label_permutation_recall_at_1": label_perm_rr1,
            "fcc_balanced_auc": fcc_bal_auc,
            "best_deployable_random_auc": best_deploy_rand_auc,
            "best_deployable_balanced_auc": best_deploy_bal_auc,
            "fcc_minus_deployable_balanced_auc": fcc_bal_auc - best_deploy_bal_auc,
            "balance_feature_smd_max": smd_max,
        },
    }
