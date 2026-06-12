from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .phase48_49_common import file_sha256, write_csv, write_markdown_table
from .phase53B_common import (
    BlindCriticSmall,
    FullShortcutCritic,
    ProjectionConditionedCritic,
    add_common_args,
    anchor_from_y,
    configure_task,
    copy_checkpoint_for_manifest,
    eval_binary_critic,
    finalize_session,
    load_generator,
    make_full_shortcut_inputs,
    make_image_inputs,
    make_loader,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    save_image_grid,
    save_score_histogram,
    train_binary_critic,
    write_command_log,
    write_rows,
)
from .phase53C_exact_projector import build_rowspace_basis, project_null, save_basis, verify_projector
from .utils import ensure_dir


def prepare_exact_projector(measurement, output_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    A = measurement.get_current_A().detach().float()
    Q = build_rowspace_basis(A)
    checks = verify_projector(A, Q)
    basis_path = save_basis(Q, output_dir / "Q_exact_null.pt")
    checks["Q_exact_null_path"] = str(basis_path)
    checks["Q_exact_null_sha256"] = file_sha256(basis_path)
    write_csv(output_dir / "exact_projector_checks.csv", [checks])
    write_markdown_table(output_dir / "exact_projector_checks.md", [checks], "Exact Projector Checks")
    return Q.to(A.device), checks


def exact_null_component(measurement, image_or_flat: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    flat = image_or_flat if image_or_flat.ndim == 2 else measurement.flatten_img(image_or_flat.float())
    p0 = project_null(flat.float(), Q.to(flat.device))
    return measurement.unflatten_img(p0)


def exact_null_flat(measurement, image_or_flat: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    flat = image_or_flat if image_or_flat.ndim == 2 else measurement.flatten_img(image_or_flat.float())
    return project_null(flat.float(), Q.to(flat.device))


def hard_audit(measurement, image: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    flat = measurement.flatten_img(image.float())
    corrected = measurement.dc_project(flat, y.float())
    return measurement.unflatten_img(corrected)


def collect_exact_null_pair_dataset(config: dict[str, Any], measurement, Q: torch.Tensor, device: torch.device, negative_mode: str = "random_roll") -> dict[str, torch.Tensor]:
    loader = make_loader(config, device)
    u_images: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    y_refs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        if x.shape[0] < 2:
            continue
        y = measurement.measure(x)
        anchor = anchor_from_y(measurement, y, config)
        if negative_mode == "alpha_chimera":
            xj = torch.roll(x, shifts=1, dims=0)
            neg = 0.5 * x + 0.5 * xj
        else:
            neg = torch.roll(x, shifts=1, dims=0)
        pos_p0 = exact_null_component(measurement, x, Q)
        neg_p0 = exact_null_component(measurement, neg, Q)
        u_images.append(torch.cat([pos_p0, neg_p0], dim=0).detach().cpu())
        anchors.append(torch.cat([anchor, anchor], dim=0).detach().cpu())
        labels.append(torch.cat([torch.ones(x.shape[0], device=device), torch.zeros(x.shape[0], device=device)]).detach().cpu())
        y_refs.append(torch.cat([y, y], dim=0).detach().cpu())
        targets.append(torch.cat([x, x], dim=0).detach().cpu())
    return {
        "u": torch.cat(u_images, dim=0),
        "anchor": torch.cat(anchors, dim=0),
        "label": torch.cat(labels, dim=0),
        "y": torch.cat(y_refs, dim=0),
        "target": torch.cat(targets, dim=0),
    }


def make_exact_null_inputs(dataset: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat([dataset["u"].to(device), dataset["anchor"].to(device)], dim=1)


def make_condition_ignored_inputs(dataset: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    zeros = torch.zeros_like(dataset["anchor"].to(device))
    return torch.cat([dataset["u"].to(device), zeros], dim=1)


def make_anchor_only_inputs(dataset: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    zeros = torch.zeros_like(dataset["u"].to(device))
    return torch.cat([zeros, dataset["anchor"].to(device)], dim=1)


def handcrafted_scores(dataset: dict[str, torch.Tensor]) -> torch.Tensor:
    p0 = dataset["u"].flatten(1).float()
    anchor = dataset["anchor"].flatten(1).float()
    p0 = p0 - p0.mean(dim=1, keepdim=True)
    anchor = anchor - anchor.mean(dim=1, keepdim=True)
    return F.cosine_similarity(torch.abs(p0), torch.abs(anchor), dim=1)


def bootstrap_auc_ci(labels: torch.Tensor, scores: torch.Tensor, n_boot: int = 200) -> tuple[float, float]:
    from .phase53B_common import binary_metrics

    labels = labels.float().cpu()
    scores = scores.float().cpu()
    if labels.numel() < 4:
        return float("nan"), float("nan")
    vals = []
    n = labels.numel()
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,))
        if len(torch.unique(labels[idx])) < 2:
            continue
        vals.append(binary_metrics(labels[idx], scores[idx])["auc"])
    if not vals:
        return float("nan"), float("nan")
    vals_t = torch.tensor(vals)
    return float(torch.quantile(vals_t, 0.025)), float(torch.quantile(vals_t, 0.975))


def info_nce_estimate(model, dataset: dict[str, torch.Tensor], device: torch.device, max_n: int = 64) -> dict[str, float]:
    model.eval()
    pos = dataset["u"][dataset["label"] == 1][:max_n].to(device)
    anchor = dataset["anchor"][dataset["label"] == 1][:max_n].to(device)
    n = min(pos.shape[0], anchor.shape[0])
    if n < 2:
        return {"infoNCE_loss": float("nan"), "infoNCE_mi_lower_nats": float("nan"), "n": n}
    scores = []
    with torch.no_grad():
        for i in range(n):
            anchors = anchor[i : i + 1].expand(n, -1, -1, -1)
            scores.append(model(torch.cat([pos[:n], anchors], dim=1)))
        logits = torch.stack(scores, dim=0)
        target = torch.arange(n, device=device)
        loss = F.cross_entropy(logits, target)
        mi = torch.log(torch.tensor(float(n), device=device)) - loss
    return {"infoNCE_loss": float(loss.detach().cpu()), "infoNCE_mi_lower_nats": float(mi.detach().cpu()), "n": int(n)}


def soft_project_flat(A: torch.Tensor, flat: torch.Tensor, lam: float) -> torch.Tensor:
    A = A.float()
    flat = flat.float()
    K = A @ A.T + float(lam) * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    rhs = (A @ flat.T).contiguous()
    z = torch.linalg.solve(K, rhs).T
    return flat - z @ A


def leakage_probe(measurement, Q: torch.Tensor, images: torch.Tensor, lambdas: list[float]) -> list[dict[str, Any]]:
    A = measurement.get_current_A().detach().float().to(images.device)
    flat = measurement.flatten_img(images.float())
    denom = torch.linalg.norm(A @ flat.T, dim=0).clamp_min(1e-12)
    rows: list[dict[str, Any]] = []
    exact = project_null(flat, Q.to(flat.device))
    exact_ratio = torch.linalg.norm(A @ exact.T, dim=0) / denom
    rows.append({"projection": "exact_P0", "lambda": 0.0, "mean_leakage_ratio": float(exact_ratio.mean()), "max_leakage_ratio": float(exact_ratio.max())})
    for lam in lambdas:
        soft = soft_project_flat(A, flat, lam)
        ratio = torch.linalg.norm(A @ soft.T, dim=0) / denom
        rows.append({"projection": "soft_PN_lambda", "lambda": lam, "mean_leakage_ratio": float(ratio.mean()), "max_leakage_ratio": float(ratio.max())})
    return rows


__all__ = [
    "BlindCriticSmall",
    "FullShortcutCritic",
    "ProjectionConditionedCritic",
    "add_common_args",
    "bootstrap_auc_ci",
    "collect_exact_null_pair_dataset",
    "configure_task",
    "copy_checkpoint_for_manifest",
    "eval_binary_critic",
    "exact_null_component",
    "exact_null_flat",
    "finalize_session",
    "handcrafted_scores",
    "hard_audit",
    "info_nce_estimate",
    "leakage_probe",
    "load_generator",
    "make_anchor_only_inputs",
    "make_condition_ignored_inputs",
    "make_exact_null_inputs",
    "make_full_shortcut_inputs",
    "make_image_inputs",
    "make_loader",
    "prepare_exact_projector",
    "relmeas_tensor",
    "resolve_device",
    "save_bar_plot",
    "save_image_grid",
    "save_score_histogram",
    "train_binary_critic",
    "write_command_log",
    "write_rows",
]

