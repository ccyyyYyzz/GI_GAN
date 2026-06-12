from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .phase56_common import (
    ExactNullCriticSmall,
    NEGATIVE_TYPES,
    PHASE56_ROOT,
    PairMLP,
    SPLIT_MODES,
    add_args,
    binary_metrics,
    fit_gradient_linear,
    fit_ridge,
    max_value,
    pair_tabular_features,
    read_csv_rows,
    resolve_device,
    save_bar,
    save_histogram,
    score_linear,
    standardize_apply,
    standardize_fit,
    to_float,
    train_torch_model,
    transform_pair_tabular,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 group-split exact-null critic training/evaluation.")
    add_args(parser)
    return parser.parse_args()


def load_pair_tensors(feature: dict, pair_rows: list[dict[str, str]]):
    p0 = feature["p0"].float()
    anchor = feature["anchor"].float()
    xs = feature["x"].float()
    null_imgs = []
    anchor_imgs = []
    labels = []
    splits = []
    ids = []
    for row in pair_rows:
        ai = int(row["anchor_local_idx"])
        ni = int(row["null_local_idx"])
        alpha = row.get("alpha", "")
        if alpha != "":
            a = float(alpha)
            null = a * p0[ai] + (1.0 - a) * p0[ni]
        else:
            null = p0[ni]
        null_imgs.append(null)
        anchor_imgs.append(anchor[ai])
        labels.append(float(row["label"]))
        splits.append(row["split"])
        ids.append(row["pair_id"])
    return {
        "p0": torch.stack(null_imgs),
        "anchor": torch.stack(anchor_imgs),
        "label": torch.tensor(labels).float(),
        "split": splits,
        "pair_id": ids,
    }


def subset(data: dict, split: str):
    idx = [i for i, s in enumerate(data["split"]) if s == split]
    return {
        "p0": data["p0"][idx],
        "anchor": data["anchor"][idx],
        "label": data["label"][idx],
        "pair_id": [data["pair_id"][i] for i in idx],
    }


def handcrafted_scores(p0: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    p = p0.flatten(1).float()
    a = anchor.flatten(1).float()
    p = p - p.mean(1, keepdim=True)
    a = a - a.mean(1, keepdim=True)
    return torch.nn.functional.cosine_similarity(torch.abs(p), torch.abs(a), dim=1)


def eval_linear_model(model_name: str, train: dict, val: dict, test: dict, seed: int):
    set_seed(seed)
    if model_name == "anchor_only_baseline":
        train_x, pca = pair_tabular_features(torch.zeros_like(train["p0"]), train["anchor"], dim=96)
        val_x = transform_pair_tabular(torch.zeros_like(val["p0"]), val["anchor"], pca)
        test_x = transform_pair_tabular(torch.zeros_like(test["p0"]), test["anchor"], pca)
    elif model_name == "p0_only_condition_ignored":
        train_x, pca = pair_tabular_features(train["p0"], torch.zeros_like(train["anchor"]), dim=96)
        val_x = transform_pair_tabular(val["p0"], torch.zeros_like(val["anchor"]), pca)
        test_x = transform_pair_tabular(test["p0"], torch.zeros_like(test["anchor"]), pca)
    elif model_name == "random_anchor_baseline":
        rolled = torch.roll(train["anchor"], shifts=1, dims=0)
        train_x, pca = pair_tabular_features(train["p0"], rolled, dim=96)
        val_x = transform_pair_tabular(val["p0"], torch.roll(val["anchor"], shifts=1, dims=0), pca)
        test_x = transform_pair_tabular(test["p0"], torch.roll(test["anchor"], shifts=1, dims=0), pca)
    else:
        train_x, pca = pair_tabular_features(train["p0"], train["anchor"], dim=128)
        val_x = transform_pair_tabular(val["p0"], val["anchor"], pca)
        test_x = transform_pair_tabular(test["p0"], test["anchor"], pca)
    mean, std = standardize_fit(train_x)
    tr = standardize_apply(train_x, mean, std)
    te = standardize_apply(test_x, mean, std)
    y_train = train["label"].clone()
    y_test = test["label"]
    if model_name == "shuffled_label_baseline":
        gen = torch.Generator().manual_seed(seed)
        y_train = y_train[torch.randperm(y_train.numel(), generator=gen)]
    if model_name in {"ridge_classifier", "anchor_only_baseline", "p0_only_condition_ignored", "random_anchor_baseline", "shuffled_label_baseline"}:
        w = fit_ridge(tr, y_train)
    elif model_name == "pca_logistic":
        w = fit_gradient_linear(tr, y_train, "logistic")
    elif model_name == "pca_linear_svm":
        w = fit_gradient_linear(tr, y_train, "linear_svm")
    else:
        raise ValueError(model_name)
    scores = score_linear(te, w)
    return binary_metrics(y_test, scores), y_test, scores


def eval_deep(train: dict, val: dict, test: dict, seed: int, device: torch.device, epochs: int, kind: str):
    set_seed(seed)
    x_train_img = torch.cat([train["p0"], train["anchor"]], dim=1)
    x_val_img = torch.cat([val["p0"], val["anchor"]], dim=1)
    x_test_img = torch.cat([test["p0"], test["anchor"]], dim=1)
    if kind == "deep_exact_null_critic":
        model = ExactNullCriticSmall(base=16)
        model, _val_metrics = train_torch_model(model, x_train_img, train["label"], x_val_img, val["label"], device=device, epochs=epochs)
        model = model.to(device).eval()
        with torch.no_grad():
            scores = model(x_test_img.to(device)).detach().cpu()
        return binary_metrics(test["label"], scores), test["label"], scores
    train_x, pca = pair_tabular_features(train["p0"], train["anchor"], dim=96)
    val_x = transform_pair_tabular(val["p0"], val["anchor"], pca)
    test_x = transform_pair_tabular(test["p0"], test["anchor"], pca)
    mean, std = standardize_fit(train_x)
    train_x = standardize_apply(train_x, mean, std)
    val_x = standardize_apply(val_x, mean, std)
    test_x = standardize_apply(test_x, mean, std)
    model = PairMLP(train_x.shape[1])
    model, _val_metrics = train_torch_model(model, train_x, train["label"], val_x, val["label"], device=device, epochs=epochs, lr=5e-4)
    model = model.to(device).eval()
    with torch.no_grad():
        scores = model(test_x.to(device)).detach().cpu()
    return binary_metrics(test["label"], scores), test["label"], scores


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    device = resolve_device(args.device)
    results = []
    hist_saved = set()
    deep_allowed = {("strict_both_group_split", "random"), ("pair_split_reproduction", "random")}
    light_models = [
        "ridge_classifier",
        "pca_logistic",
        "pca_linear_svm",
        "anchor_only_baseline",
        "p0_only_condition_ignored",
        "random_anchor_baseline",
        "shuffled_label_baseline",
    ]
    for task in args.tasks:
        feature = torch.load(root / "features" / f"{task}_features.pt", map_location="cpu")
        for split_mode in SPLIT_MODES:
            for neg in NEGATIVE_TYPES:
                pair_path = root / "pairs" / f"{task}_{split_mode}_{neg}_pairs.csv"
                if not pair_path.exists():
                    continue
                pair_data = load_pair_tensors(feature, read_csv_rows(pair_path))
                train = subset(pair_data, "train")
                val = subset(pair_data, "val")
                test = subset(pair_data, "test")
                if train["label"].unique().numel() < 2 or test["label"].unique().numel() < 2:
                    continue
                models = list(light_models)
                if (split_mode, neg) in deep_allowed:
                    models = ["deep_exact_null_critic", "small_mlp_baseline"] + models
                if neg == "alpha_chimera":
                    models = ["ridge_classifier", "pca_logistic", "p0_only_condition_ignored", "anchor_only_baseline"]
                for seed in args.seeds:
                    for model_name in models:
                        try:
                            if model_name in {"deep_exact_null_critic", "small_mlp_baseline"}:
                                metrics, labels, scores = eval_deep(train, val, test, seed, device, args.critic_epochs, model_name)
                            elif model_name == "handcrafted_baseline":
                                scores = handcrafted_scores(test["p0"], test["anchor"])
                                labels = test["label"]
                                metrics = binary_metrics(labels, scores)
                            else:
                                metrics, labels, scores = eval_linear_model(model_name, train, val, test, seed)
                            status = "ok"
                        except Exception as exc:
                            metrics = {}
                            labels = torch.tensor([])
                            scores = torch.tensor([])
                            status = f"failed: {exc}"
                        row = {
                            "task": task,
                            "family": feature["family"],
                            "split_mode": split_mode,
                            "negative_type": neg,
                            "model": model_name,
                            "seed": seed,
                            "status": status,
                            "train_pairs": int(train["label"].numel()),
                            "val_pairs": int(val["label"].numel()),
                            "test_pairs": int(test["label"].numel()),
                            **metrics,
                        }
                        results.append(row)
                        key = (task, split_mode, neg, model_name)
                        if status == "ok" and key not in hist_saved and model_name in {"deep_exact_null_critic", "ridge_classifier"}:
                            save_histogram(root / "score_histograms" / f"{task}_{split_mode}_{neg}_{model_name}.png", labels, scores, f"{task} {split_mode} {neg} {model_name}")
                            hist_saved.add(key)
                # Handcrafted once per setting.
                scores = handcrafted_scores(test["p0"], test["anchor"])
                results.append(
                    {
                        "task": task,
                        "family": feature["family"],
                        "split_mode": split_mode,
                        "negative_type": neg,
                        "model": "handcrafted_baseline",
                        "seed": "deterministic",
                        "status": "ok",
                        "train_pairs": int(train["label"].numel()),
                        "val_pairs": int(val["label"].numel()),
                        "test_pairs": int(test["label"].numel()),
                        **binary_metrics(test["label"], scores),
                    }
                )
    write_rows(root, "group_split_critic_results", results, "Phase56 Group Split Critic Results")
    ok_rows = [r for r in results if r.get("status") == "ok"]
    strict_rows = [r for r in ok_rows if r.get("split_mode") == "strict_both_group_split" and r.get("model") == "deep_exact_null_critic"]
    if not strict_rows:
        strict_rows = [r for r in ok_rows if r.get("split_mode") == "strict_both_group_split" and r.get("model") == "ridge_classifier"]
    save_bar(root / "group_split_auc_by_family.png", strict_rows, "task", "auc", "Strict group-split AUC by family", "AUC")
    save_bar(root / "group_split_auc_by_negative_type.png", [r for r in ok_rows if r.get("split_mode") == "strict_both_group_split" and r.get("model") == "ridge_classifier"], "negative_type", "auc", "Strict AUC by negative type", "AUC")
    save_bar(root / "baseline_comparison.png", [r for r in ok_rows if "baseline" in str(r.get("model")) and r.get("split_mode") == "strict_both_group_split"], "model", "auc", "Baseline checks", "AUC")
    save_bar(root / "group_split_mi_by_family.png", strict_rows, "task", "auc", "AUC proxy (MI not computed)", "AUC proxy")
    save_bar(root / "score_histograms_by_family.png", strict_rows, "task", "auc", "Score histogram summary", "AUC")
    print(root / "group_split_critic_results.csv")


if __name__ == "__main__":
    main()

