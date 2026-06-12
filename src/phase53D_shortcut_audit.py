from __future__ import annotations

import argparse
from typing import Any

import torch

from .datasets import get_val_dataloader
from .phase53C_exact_projector import build_rowspace_basis, project_null
from .phase53D_common import (
    add_phase53d_args,
    binary_metrics,
    configure_light_task,
    fit_gradient_classifier,
    fit_ridge_classifier,
    pooled_image_features,
    resolve_device,
    save_bar,
    save_scatter,
    score_linear,
    standardize_apply,
    standardize_fit,
    write_rows,
)
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53D local residual shortcut audit.")
    add_phase53d_args(parser)
    return parser.parse_args()


@torch.no_grad()
def collect_dataset(args, config: dict[str, Any], measurement, mode: str, device: torch.device) -> dict[str, torch.Tensor]:
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 2)),
        limit_val_samples=int(config.get("limit_val_samples", args.limit_samples)),
        seed=int(config.get("seed", 123)),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    us: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        if x.shape[0] < 2:
            continue
        y = measurement.measure(x)
        anchor_flat = measurement.data_solution(y.float(), mode=config.get("backprojection_mode", "ridge_pinv"))
        anchor = measurement.unflatten_img(anchor_flat)
        xj = torch.roll(x, shifts=1, dims=0)
        if mode == "residual_easy_wrong_y":
            neg = xj
        elif mode == "feasible_hallucination":
            neg = measurement.unflatten_img(measurement.dc_project(measurement.flatten_img(xj.float()), y.float())).clamp(0, 1)
        elif mode == "exact_null_independent":
            A = measurement.get_current_A().detach().float().to(device)
            Q = build_rowspace_basis(A)
            neg_flat = project_null(measurement.flatten_img(xj.float()), Q)
            neg = measurement.unflatten_img(neg_flat)
        else:
            raise ValueError(mode)
        us.append(torch.cat([x, neg], dim=0).detach().cpu())
        anchors.append(torch.cat([anchor, anchor], dim=0).detach().cpu())
        ys.append(torch.cat([y, y], dim=0).detach().cpu())
        labels.append(torch.cat([torch.ones(x.shape[0]), torch.zeros(x.shape[0])], dim=0))
    return {"u": torch.cat(us), "anchor": torch.cat(anchors), "y": torch.cat(ys), "label": torch.cat(labels)}


def residual_features(dataset: dict[str, torch.Tensor], measurement, device: torch.device) -> torch.Tensor:
    u = dataset["u"].to(device)
    y = dataset["y"].to(device)
    flat = measurement.flatten_img(u.float())
    residual = measurement.A_forward(flat) - y.float()
    correction = measurement.AT_forward(measurement.solve_K(residual.float()))
    rel = torch.linalg.norm(residual, dim=1, keepdim=True) / torch.linalg.norm(y.float(), dim=1, keepdim=True).clamp_min(1e-12)
    feats = torch.cat(
        [
            rel,
            torch.linalg.norm(residual, dim=1, keepdim=True),
            residual.mean(dim=1, keepdim=True),
            residual.std(dim=1, keepdim=True),
            torch.linalg.norm(correction, dim=1, keepdim=True) / torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12),
            correction.mean(dim=1, keepdim=True),
            correction.std(dim=1, keepdim=True),
        ],
        dim=1,
    )
    return feats.detach().cpu()


def exact_null_features(dataset: dict[str, torch.Tensor], measurement, Q: torch.Tensor, device: torch.device) -> torch.Tensor:
    u = dataset["u"].to(device)
    anchor = dataset["anchor"]
    flat = measurement.flatten_img(u.float())
    p0 = measurement.unflatten_img(project_null(flat, Q.to(device))).detach().cpu()
    return torch.cat([pooled_image_features(p0, pool=8), pooled_image_features(anchor, pool=8)], dim=1)


def full_anchor_features(dataset: dict[str, torch.Tensor], measurement, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            pooled_image_features(dataset["u"], pool=8),
            pooled_image_features(dataset["anchor"], pool=8),
            residual_features(dataset, measurement, device)[:, :1],
        ],
        dim=1,
    )


def train_eval(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor, classifier: str) -> dict[str, float]:
    mean, std = standardize_fit(train_x)
    tr = standardize_apply(train_x, mean, std)
    ev = standardize_apply(eval_x, mean, std)
    if classifier == "ridge":
        w = fit_ridge_classifier(tr, train_y)
    elif classifier == "logistic":
        w = fit_gradient_classifier(tr, train_y, kind="logistic", steps=140)
    else:
        w = fit_gradient_classifier(tr, train_y, kind="linear_svm", steps=140)
    train_scores = score_linear(tr, w)
    eval_scores = score_linear(ev, w)
    train_metrics = binary_metrics(train_y, train_scores)
    eval_metrics = binary_metrics(eval_y, eval_scores)
    return {
        "train_auc": train_metrics["auc"],
        "eval_auc": eval_metrics["auc"],
        "eval_accuracy": eval_metrics["accuracy"],
        "eval_balanced_accuracy": eval_metrics["balanced_accuracy"],
    }


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    task_root = ensure_dir(out / "shortcut_audit")
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows: list[dict[str, Any]] = []
    modes = ["residual_easy_wrong_y", "feasible_hallucination", "exact_null_independent"]
    tests = [
        ("train_wrong_y_test_feasible", "residual_easy_wrong_y", "feasible_hallucination"),
        ("train_feasible_test_wrong_y", "feasible_hallucination", "residual_easy_wrong_y"),
        ("train_feasible_test_feasible", "feasible_hallucination", "feasible_hallucination"),
        ("train_exact_null_test_feasible", "exact_null_independent", "feasible_hallucination"),
    ]
    for task in args.tasks:
        task_out = ensure_dir(task_root / task)
        info, config, measurement, exact_info = configure_light_task(args, task, task_out, device)
        A = measurement.get_current_A().detach().float().to(device)
        Q = build_rowspace_basis(A)
        datasets = {mode: collect_dataset(args, config, measurement, mode, device) for mode in modes}
        feature_sets = {}
        for mode, data in datasets.items():
            feature_sets[(mode, "residual_features")] = residual_features(data, measurement, device)
            feature_sets[(mode, "full_anchor_features")] = full_anchor_features(data, measurement, device)
            feature_sets[(mode, "exact_null_features")] = exact_null_features(data, measurement, Q, device)
        for test_name, train_mode, eval_mode in tests:
            for feat_name in ["residual_features", "full_anchor_features", "exact_null_features"]:
                for clf in ["ridge", "logistic", "linear_svm"]:
                    metrics = train_eval(
                        feature_sets[(train_mode, feat_name)],
                        datasets[train_mode]["label"],
                        feature_sets[(eval_mode, feat_name)],
                        datasets[eval_mode]["label"],
                        clf,
                    )
                    rows.append(
                        {
                            "task": task,
                            "family": info["metadata"]["display"],
                            "sampling_pct": info["metadata"]["sampling_pct"],
                            "test": test_name,
                            "feature_set": feat_name,
                            "classifier": clf,
                            "train_negative": train_mode,
                            "eval_negative": eval_mode,
                            "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                            **metrics,
                        }
                    )
    write_rows(out, "shortcut_audit_results", rows, "Phase53D Shortcut Audit Results")
    save_bar(out / "residual_classifier_auc.png", [r for r in rows if r["feature_set"] == "residual_features"], "test", "eval_auc", "Residual feature shortcut AUC", "eval AUC")
    save_bar(out / "exact_null_vs_full_features.png", [r for r in rows if r["classifier"] == "ridge"], "feature_set", "eval_auc", "Exact-null vs full features", "eval AUC")
    save_scatter(out / "score_vs_relmeaserr.png", rows, "train_auc", "eval_auc", "Shortcut train vs eval AUC", "train AUC", "eval AUC")
    report = [
        "# Phase53D Residual Shortcut Audit",
        "",
        "This local diagnostic trains only lightweight CPU linear classifiers.",
        "Residual features are expected to solve wrong-y negatives but fail on cross-feasible negatives, showing that row-space residuals are shortcuts rather than null-space plausibility.",
        "Exact-null features are the relevant diagnostic channel for feasible hallucinations.",
    ]
    (out / "SHORTCUT_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "SHORTCUT_AUDIT_REPORT.md")


if __name__ == "__main__":
    main()

