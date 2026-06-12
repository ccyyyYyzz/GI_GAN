from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .datasets import get_val_dataloader
from .phase53C_exact_projector import build_rowspace_basis, project_null
from .phase53D_common import (
    add_phase53d_args,
    binary_metrics,
    configure_light_task,
    fit_pca_basis,
    pair_features,
    pooled_image_features,
    project_pca,
    resolve_device,
    run_cpu_classifier,
    save_bar,
    save_histogram,
    save_scatter,
    to_float,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53D local E1-mini anchor/null information pretest.")
    add_phase53d_args(parser)
    parser.add_argument("--pca_dims", nargs="*", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--max_pca_dim", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def collect_anchor_null(args, config: dict[str, Any], measurement, Q: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
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
    xs: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    p0s: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        flat = measurement.flatten_img(x.float())
        anchor_flat = measurement.data_solution(y.float(), mode=config.get("backprojection_mode", "ridge_pinv"))
        p0_flat = project_null(flat, Q.to(device))
        xs.append(x.detach().cpu())
        anchors.append(measurement.unflatten_img(anchor_flat).detach().cpu())
        p0s.append(measurement.unflatten_img(p0_flat).detach().cpu())
        if len(batch) > 1:
            labels.append(torch.as_tensor(batch[1]).detach().cpu())
        else:
            labels.append(torch.full((x.shape[0],), -1, dtype=torch.long))
    return {
        "x": torch.cat(xs, dim=0),
        "anchor": torch.cat(anchors, dim=0),
        "p0": torch.cat(p0s, dim=0),
        "class_label": torch.cat(labels, dim=0),
    }


def roll_indices(n: int) -> torch.Tensor:
    return torch.roll(torch.arange(n), shifts=1)


def same_class_indices(labels: torch.Tensor) -> torch.Tensor:
    n = labels.numel()
    idx = roll_indices(n)
    for cls in labels.unique().tolist():
        members = torch.where(labels == int(cls))[0]
        if members.numel() > 1:
            idx[members] = torch.roll(members, shifts=1)
    return idx


def nearest_anchor_indices(anchor: torch.Tensor) -> torch.Tensor:
    feats = pooled_image_features(anchor, pool=8)
    feats = (feats - feats.mean(0, keepdim=True)) / feats.std(0, keepdim=True).clamp_min(1e-6)
    dist = torch.cdist(feats, feats)
    dist.fill_diagonal_(float("inf"))
    return torch.argmin(dist, dim=1)


def make_pair_set(p0: torch.Tensor, anchor: torch.Tensor, neg_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = p0.shape[0]
    null_flat = p0.flatten(1)
    anchor_flat = anchor.flatten(1)
    null_pair = torch.cat([null_flat, null_flat[neg_idx]], dim=0)
    anchor_pair = torch.cat([anchor_flat, anchor_flat], dim=0)
    labels = torch.cat([torch.ones(n), torch.zeros(n)], dim=0)
    return null_pair, anchor_pair, labels


def handcrafted_scores(null_pair: torch.Tensor, anchor_pair: torch.Tensor) -> torch.Tensor:
    n = F.normalize(torch.abs(null_pair - null_pair.mean(dim=1, keepdim=True)), dim=1)
    a = F.normalize(torch.abs(anchor_pair - anchor_pair.mean(dim=1, keepdim=True)), dim=1)
    return (n * a).sum(dim=1)


def read_metric(info: dict[str, Any], section: str, key: str) -> float:
    path = info.get("metrics_path")
    if not path:
        return float("nan")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return float(data.get(section, {}).get(key, float("nan")))
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    task_root = ensure_dir(out / "anchor_info_pretest")
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for task in args.tasks:
        task_out = ensure_dir(task_root / task)
        info, config, measurement, exact_info = configure_light_task(args, task, task_out, device)
        A = measurement.get_current_A().detach().float().to(device)
        Q = build_rowspace_basis(A)
        data = collect_anchor_null(args, config, measurement, Q, device)
        p0 = data["p0"].float()
        anchor = data["anchor"].float()
        labels = data["class_label"]
        n = p0.shape[0]
        neg_indices = {
            "random": roll_indices(n),
            "same_class": same_class_indices(labels) if (labels >= 0).any() else roll_indices(n),
            "nearest_anchor": nearest_anchor_indices(anchor),
        }
        stacked = torch.cat([p0.flatten(1), anchor.flatten(1)], dim=0)
        pca_mean, pca_basis = fit_pca_basis(stacked, max(args.pca_dims))
        bp_psnr = read_metric(info, "backprojection", "psnr")
        model_psnr = read_metric(info, "model", "psnr")
        rel_meas = read_metric(info, "model", "rel_meas_error")
        for neg_name, neg_idx in neg_indices.items():
            null_pair, anchor_pair, y = make_pair_set(p0, anchor, neg_idx)
            for dim in args.pca_dims:
                null_z = project_pca(null_pair, pca_mean, pca_basis, dim)
                anchor_z = project_pca(anchor_pair, pca_mean, pca_basis, dim)
                pair_x = pair_features(null_z, anchor_z)
                classifiers = ["ridge"]
                if dim in {64, 128}:
                    classifiers.extend(["logistic", "linear_svm"])
                for clf in classifiers:
                    metrics, val_labels, val_scores = run_cpu_classifier(pair_x, y, clf, seed=args.seed + dim)
                    row = {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "sampling_pct": info["metadata"]["sampling_pct"],
                        "negative_type": neg_name,
                        "model": f"{clf}_pair_pca",
                        "pca_dim": dim,
                        "n_pairs": int(y.numel()),
                        "bp_psnr": bp_psnr,
                        "model_psnr": model_psnr,
                        "rel_meas_err": rel_meas,
                        "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                        **metrics,
                    }
                    rows.append(row)
                    if clf == "ridge" and dim == 128 and neg_name == "random":
                        save_histogram(task_out / f"{task}_{clf}_{dim}_score_histograms.png", val_labels, val_scores, f"{task} ridge random scores")
            null_pair, anchor_pair, y = make_pair_set(p0, anchor, neg_idx)
            base_dim = min(128, max(args.pca_dims))
            null_z = project_pca(null_pair, pca_mean, pca_basis, base_dim)
            anchor_z = project_pca(anchor_pair, pca_mean, pca_basis, base_dim)
            for name, feats in [
                ("condition_ignored_p0_only", null_z),
                ("anchor_only_baseline", anchor_z),
            ]:
                metrics, _vl, _vs = run_cpu_classifier(feats, y, "ridge", seed=args.seed + 7)
                rows.append(
                    {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "sampling_pct": info["metadata"]["sampling_pct"],
                        "negative_type": neg_name,
                        "model": name,
                        "pca_dim": base_dim,
                        "n_pairs": int(y.numel()),
                        "bp_psnr": bp_psnr,
                        "model_psnr": model_psnr,
                        "rel_meas_err": rel_meas,
                        "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                        **metrics,
                    }
                )
            scores = handcrafted_scores(null_pair, anchor_pair)
            metrics = binary_metrics(y, scores)
            rows.append(
                {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "sampling_pct": info["metadata"]["sampling_pct"],
                    "negative_type": neg_name,
                    "model": "handcrafted_abs_cosine_baseline",
                    "pca_dim": "raw",
                    "n_pairs": int(y.numel()),
                    "bp_psnr": bp_psnr,
                    "model_psnr": model_psnr,
                    "rel_meas_err": rel_meas,
                    "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                    **metrics,
                }
            )
        random_idx = neg_indices["random"]
        null_pair, anchor_pair, y = make_pair_set(p0, anchor, random_idx)
        null_z = project_pca(null_pair, pca_mean, pca_basis, 128)
        anchor_z = project_pca(anchor_pair, pca_mean, pca_basis, 128)
        train_metrics, _vl, _vs = run_cpu_classifier(pair_features(null_z, anchor_z), y, "ridge", seed=args.seed + 99)
        # Refit a simple ridge on all random pairs for alpha-chimera scoring.
        from .phase53D_common import fit_ridge_classifier, score_linear, standardize_apply, standardize_fit

        feats = pair_features(null_z, anchor_z)
        mean, std = standardize_fit(feats)
        w = fit_ridge_classifier(standardize_apply(feats, mean, std), y)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            chimera_p0 = alpha * p0.flatten(1) + (1.0 - alpha) * p0.flatten(1)[random_idx]
            chimera_z = project_pca(chimera_p0, pca_mean, pca_basis, 128)
            anchor_z_single = project_pca(anchor.flatten(1), pca_mean, pca_basis, 128)
            score = score_linear(standardize_apply(pair_features(chimera_z, anchor_z_single), mean, std), w)
            alpha_rows.append(
                {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "alpha": alpha,
                    "mean_score": float(score.mean()),
                    "std_score": float(score.std()),
                    "training_random_auc": train_metrics["auc"],
                }
            )
        save_json({"n_samples": int(n), "exact_A_info": exact_info}, task_out / "anchor_pretest_manifest.json")
    write_rows(out, "anchor_null_pretest_results", rows, "Phase53D Anchor Null Pretest Results")
    write_rows(out, "alpha_chimera_scores", alpha_rows, "Phase53D Alpha Chimera Scores")
    pair_rows = [r for r in rows if r.get("model") == "ridge_pair_pca" and str(r.get("pca_dim")) == "128"]
    save_bar(out / "auc_by_family.png", pair_rows, "task", "auc", "E1-mini AUC by family", "AUC")
    save_bar(out / "auc_by_negative_type.png", pair_rows, "negative_type", "auc", "AUC by negative type", "AUC")
    save_scatter(out / "auc_vs_anchor_bp_psnr.png", pair_rows, "bp_psnr", "auc", "AUC vs anchor BP PSNR", "BP PSNR", "AUC")
    save_bar(out / "alpha_chimera_scores.png", alpha_rows, "alpha", "mean_score", "Alpha chimera scores", "mean score")
    save_bar(out / "classifier_score_histograms.png", pair_rows, "task", "auc", "Classifier summary AUC", "AUC")
    best_by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "pair_pca" not in str(row.get("model")):
            continue
        task = str(row["task"])
        if task not in best_by_task or to_float(row.get("auc")) > to_float(best_by_task[task].get("auc")):
            best_by_task[task] = row
    report = [
        "# Phase53D E1-mini Anchor/Null Pretest",
        "",
        "This is a local preflight diagnostic. It trains only lightweight CPU linear classifiers on exact-null/anchor features; no generator or GAN is trained.",
        "",
        "Interpretation rule: AUC >= 0.70 with CI lower bound > 0.60 supports continuing exact-null critic evaluation for that family; all AUC < 0.60 argues against GAN follow-up.",
        "",
        "## Best pair-classifier rows",
    ]
    for task, row in best_by_task.items():
        report.append(
            f"- {task}: best AUC={to_float(row.get('auc')):.3f} "
            f"(CI {to_float(row.get('auc_ci_low')):.3f}-{to_float(row.get('auc_ci_high')):.3f}), "
            f"negative={row.get('negative_type')}, model={row.get('model')}, pca_dim={row.get('pca_dim')}."
        )
    (out / "ANCHOR_NULL_PRETEST_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "ANCHOR_NULL_PRETEST_REPORT.md")


if __name__ == "__main__":
    main()

