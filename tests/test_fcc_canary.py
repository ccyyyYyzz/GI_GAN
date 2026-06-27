from __future__ import annotations

import numpy as np
import torch

from src.compatibility_data import decompose_split
from src.measurement import GhostMeasurementOperator
from src import fcc_canary as fc


def _toy_split(name="dev", n=24, img=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, img, img, generator=g)
    labels = torch.full((n,), -1, dtype=torch.long)
    indices = torch.arange(n, dtype=torch.long)
    rs = fc.RawSplit(name=name, x=x, labels=labels, indices=indices)
    op = GhostMeasurementOperator(img_size=img, sampling_ratio=0.25, pattern_type="rademacher",
                                  noise_std=0.0, matrix_normalization="legacy_sqrt_m", device="cpu", seed=123)
    comp = decompose_split(rs, op, device=torch.device("cpu"), batch_size=8, dtype=torch.float64)
    return comp, op


def test_exact_geometry_float64():
    comp, op = _toy_split()
    g = fc.geometry_checks(comp, op, device=torch.device("cpu"), sample_count=comp.size)
    assert g["pass"], g
    assert g["float64_A_P0_rel_max"] < 1e-9
    assert g["reconstruction_rel_max"] < 1e-10


def test_feasible_counterfactual_is_measurement_equivalent():
    comp, op = _toy_split()
    donors = fc.random_derangement(comp.size, seed=1)
    feas = fc.feasibility_check(comp, op, donors, device=torch.device("cpu"))
    assert feas["pass_float32_proxy"], feas
    assert feas["u_rel_max"] < 1e-5


def test_nuisance_balanced_derangement_is_one_to_one_no_fixed_points():
    comp, _ = _toy_split(n=20)
    donors, report = fc.nuisance_balanced_derangement(comp, seed=3)
    assert donors.shape[0] == comp.size
    assert int(np.sum(donors == np.arange(comp.size))) == 0
    assert np.unique(donors).size == comp.size
    assert report["fixed_points"] == 0


def test_standard_scaler_handles_nonfinite():
    x = np.array([[1.0, np.inf], [2.0, np.nan], [3.0, -np.inf]])
    sc = fc.StandardScaler.fit(x)
    z = sc.transform(x)
    assert np.isfinite(z).all()
    assert (np.abs(z) <= 8.0 + 1e-9).all()


def test_deployable_classifier_learns_separable_and_fails_noise():
    rng = np.random.default_rng(0)
    # separable
    xpos = rng.normal(2.0, 0.3, size=(200, 4))
    xneg = rng.normal(-2.0, 0.3, size=(200, 4))
    x = np.concatenate([xpos, xneg]); y = np.concatenate([np.ones(200), np.zeros(200)])
    clf = fc.DeployableClassifier(kind="logistic", steps=300).fit(x, y)
    auc = fc._safe_auc(y, clf.score(x))
    assert auc > 0.95
    # pure noise -> ~chance
    xn = rng.normal(0, 1, size=(400, 4)); yn = (rng.random(400) > 0.5).astype(float)
    clfn = fc.DeployableClassifier(kind="logistic", steps=300).fit(xn, yn)
    aucn = fc._safe_auc(yn, clfn.score(xn))
    assert 0.35 < aucn < 0.75


def test_classify_invalid_when_geometry_fails():
    ev = {
        "geometry": {"pass": False},
        "feasibility": {"random": {"pass_float32_proxy": True}, "nuisance_balanced": {"pass_float32_proxy": True}},
        "layer_a": {"recall_at_1": 0.5, "random_recall_at_1": 0.03125, "label_permutation_recall_at_1": 0.03},
        "layer_b": {"fcc": {"balanced_auc": 0.9}, "best_deployable_balanced_auc": 0.5, "balance": {"feature_smd_max": 0.1}},
    }
    assert fc.classify_fcc(ev)["classification"] == "INVALID_EXPERIMENT"


def test_classify_only_scalar_when_deployable_not_neutered():
    ev = {
        "geometry": {"pass": True},
        "feasibility": {"random": {"pass_float32_proxy": True}, "nuisance_balanced": {"pass_float32_proxy": True}},
        "layer_a": {"recall_at_1": 0.75, "random_recall_at_1": 0.03125, "label_permutation_recall_at_1": 0.03},
        "layer_b": {"fcc": {"balanced_auc": 0.95}, "best_deployable_balanced_auc": 0.99, "balance": {"feature_smd_max": 0.1}},
    }
    assert fc.classify_fcc(ev)["classification"] == "ONLY_SCALAR_OR_ARTIFACT_SIGNAL"


def test_classify_structural_when_all_pass_but_no_transfer():
    ev = {
        "geometry": {"pass": True},
        "feasibility": {"random": {"pass_float32_proxy": True}, "nuisance_balanced": {"pass_float32_proxy": True}},
        "layer_a": {"recall_at_1": 0.75, "random_recall_at_1": 0.03125, "label_permutation_recall_at_1": 0.03},
        "layer_b": {"fcc": {"balanced_auc": 0.82}, "best_deployable_balanced_auc": 0.55, "balance": {"feature_smd_max": 0.1}},
        "layer_c": {"transfer_confirmed": None},
    }
    assert fc.classify_fcc(ev)["classification"] == "REAL_PAIR_SIGNAL_BUT_NO_GENERATED_TRANSFER"


def test_classify_no_signal_when_retrieval_near_random():
    ev = {
        "geometry": {"pass": True},
        "feasibility": {"random": {"pass_float32_proxy": True}, "nuisance_balanced": {"pass_float32_proxy": True}},
        "layer_a": {"recall_at_1": 0.04, "random_recall_at_1": 0.03125, "label_permutation_recall_at_1": 0.03},
        "layer_b": {"fcc": {"balanced_auc": 0.51}, "best_deployable_random_auc": 0.52,
                    "best_deployable_balanced_auc": 0.5, "balance": {"feature_smd_max": 0.1}},
    }
    assert fc.classify_fcc(ev)["classification"] == "NO_COMPATIBILITY_SIGNAL"


def test_classify_only_scalar_when_critic_weak_but_deployable_separates():
    # Critic retrieval is near random, but a deployable baseline strongly separates real pairs
    # AND is not neutralised on balanced negs -> the signal is deployable/nuisance, not "no signal".
    ev = {
        "geometry": {"pass": True},
        "feasibility": {"random": {"pass_float32_proxy": True}, "nuisance_balanced": {"pass_float32_proxy": True}},
        "layer_a": {"recall_at_1": 0.10, "random_recall_at_1": 0.03125, "label_permutation_recall_at_1": 0.03},
        "layer_b": {"fcc": {"balanced_auc": 0.63}, "best_deployable_random_auc": 0.996,
                    "best_deployable_balanced_auc": 0.974, "balance": {"feature_smd_max": 0.55}},
    }
    out = fc.classify_fcc(ev)
    assert out["classification"] == "ONLY_SCALAR_OR_ARTIFACT_SIGNAL"
    assert out["checks"]["critic_real_pair_signal"] is False
    assert out["checks"]["deployable_separates_real_pairs"] is True
