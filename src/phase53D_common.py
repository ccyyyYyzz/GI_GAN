from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config, torch_load
from .metrics import batch_metrics
from .models import build_generator
from .phase48_49_common import load_bundle_task, save_run_config, write_csv, write_markdown_table
from .utils import apply_experiment_defaults, ensure_dir, reconstruct_from_measurements, save_config, save_json


DEFAULT_BUNDLE_ROOT = "E:/ns_mc_gan_gi/outputs_phase15/imported_noleak"
DEFAULT_DATASET_ROOT = "E:/ns_mc_gan_gi/data"
DEFAULT_OUTPUT_ROOT = "E:/ns_mc_gan_gi/outputs_phase53D_local_preflight"
DEFAULT_PHASE48_ROOT = "E:/ns_mc_gan_gi/outputs_phase48_49_colab_import"
DEFAULT_PHASE51A_ROOT = "E:/ns_mc_gan_gi/outputs_phase51A_colab_import"
TASK_ORDER = ["rad5", "scr5", "rad10", "scr10"]


def add_phase53d_args(parser):
    parser.add_argument("--bundle_root", default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--phase48_root", default=DEFAULT_PHASE48_ROOT)
    parser.add_argument("--phase51A_root", default=DEFAULT_PHASE51A_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("--limit_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    v = to_float(value)
    if math.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def write_rows(root: str | Path, stem: str, rows: list[dict[str, Any]], title: str) -> None:
    root = ensure_dir(root)
    write_csv(root / f"{stem}.csv", rows)
    write_markdown_table(root / f"{stem}.md", rows, title)


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def configure_light_task(args, task_key: str, task_out: str | Path, device: torch.device):
    info = load_bundle_task(args.bundle_root, task_key)
    config = apply_experiment_defaults(info["config"])
    config["dataset_root"] = args.dataset_root
    config["device"] = str(device)
    config["batch_size"] = int(args.batch_size)
    config["num_workers"] = int(args.num_workers)
    config["limit_val_samples"] = int(args.limit_samples)
    config["phase53D_note"] = "Local preflight / diagnostic only; no full neural training."
    if info["exact_A_path"] is not None:
        config["measurement_operator_exact_path"] = str(info["exact_A_path"])
        config["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
    task_out = ensure_dir(task_out)
    save_run_config(config, task_out)
    save_config(config, task_out / "config_used.yaml")
    save_json(
        {
            "task": task_key,
            "config_path": str(info["config_path"]),
            "checkpoint_path": str(info["checkpoint_path"]),
            "metrics_path": str(info["metrics_path"]) if info.get("metrics_path") else "",
            "exact_A_path": str(info["exact_A_path"]) if info.get("exact_A_path") else "",
            "note": "Source files are referenced, not copied, to keep Phase53D lightweight.",
        },
        task_out / "source_paths.json",
    )
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    save_json(exact_info, task_out / "exact_A_info.json")
    return info, config, measurement, exact_info


def load_eval_generator(info: dict[str, Any], config: dict[str, Any], measurement, device: torch.device):
    checkpoint = torch_load(info["checkpoint_path"], map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        merged["dataset_root"] = config["dataset_root"]
        merged["device"] = str(device)
        merged["batch_size"] = config.get("batch_size", 16)
        merged["num_workers"] = config.get("num_workers", 2)
        merged["limit_val_samples"] = config.get("limit_val_samples", 512)
        if info.get("exact_A_path") is not None:
            merged["measurement_operator_exact_path"] = str(info["exact_A_path"])
            merged["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
        config = apply_experiment_defaults(merged)
    generator = build_generator(config, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("generator_ema") or checkpoint.get("generator")
    else:
        state = checkpoint
    if state is None:
        raise RuntimeError(f"No generator state in checkpoint: {info['checkpoint_path']}")
    generator.load_state_dict(state)
    generator.eval()
    return generator


@torch.no_grad()
def reconstruct_no_full_training(generator, measurement, y: torch.Tensor, config: dict[str, Any], *, final_audit: bool = True):
    return reconstruct_from_measurements(
        generator,
        measurement,
        y,
        use_null_project=bool(config.get("use_null_project", True)),
        use_dc_project=bool(config.get("use_dc_project", True)),
        use_final_dc_project=bool(final_audit),
        backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
        enable_refiner=bool(config.get("enable_refiner", True)),
        output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
        return_extras=True,
    )


def save_bar(path: str | Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, ylabel: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(x_key, i)) for i, row in enumerate(rows)]
    values = [to_float(row.get(y_key)) for row in rows]
    plt.figure(figsize=(max(7, 0.55 * len(labels)), 4.2))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_scatter(path: str | Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [to_float(row.get(x_key)) for row in rows]
    ys = [to_float(row.get(y_key)) for row in rows]
    labels = [str(row.get("task", row.get("family", ""))) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 4.5))
    plt.scatter(xs, ys, s=45)
    for x, y, label in zip(xs, ys, labels):
        if not math.isnan(x) and not math.isnan(y):
            plt.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_histogram(path: str | Path, labels: torch.Tensor, scores: torch.Tensor, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels > 0.5].numpy(), bins=30, alpha=0.65, label="positive")
    plt.hist(scores[labels <= 0.5].numpy(), bins=30, alpha=0.65, label="negative")
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_image_grid(path: str | Path, rows: list[list[torch.Tensor]], titles: list[str], *, max_rows: int = 8, pdf: bool = False) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = rows[:max_rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(2.25 * len(titles), 2.15 * len(rows)))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = plt.subplot(len(rows), len(titles), r * len(titles) + c + 1)
            arr = img.detach().cpu().float().squeeze().numpy()
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180 if not pdf else None)
    plt.close()


def binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()
    return float(cmp.mean())


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    probs = torch.sigmoid(scores)
    pred = (probs >= 0.5).float()
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    pos_recall = tp / max(1.0, tp + fn)
    neg_recall = tn / max(1.0, tn + fp)
    return {
        "auc": binary_auc(labels, scores),
        "accuracy": float((pred == labels).float().mean()),
        "balanced_accuracy": 0.5 * (pos_recall + neg_recall),
        "precision": tp / max(1.0, tp + fp),
        "recall": pos_recall,
        "ece_proxy_brier": float(torch.mean((probs - labels) ** 2)),
    }


def bootstrap_auc_ci(labels: torch.Tensor, scores: torch.Tensor, n_boot: int = 120) -> tuple[float, float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    n = labels.numel()
    vals: list[float] = []
    gen = torch.Generator().manual_seed(5300)
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen)
        if labels[idx].unique().numel() < 2:
            continue
        vals.append(binary_auc(labels[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    t = torch.tensor(vals)
    return float(torch.quantile(t, 0.025)), float(torch.quantile(t, 0.975))


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def standardize_apply(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def fit_pca_basis(x: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().float()
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    q = min(max_dim + 8, min(xc.shape) - 1)
    if q < 2:
        raise ValueError("Not enough samples/features for PCA.")
    _u, _s, v = torch.pca_lowrank(xc, q=q, center=False, niter=2)
    return mean.cpu(), v[:, :max_dim].cpu()


def project_pca(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, dim: int) -> torch.Tensor:
    return (x.detach().cpu().float() - mean[:, : x.shape[1]]) @ basis[:, :dim]


def pair_features(null_z: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
    null_z = null_z.float()
    anchor_z = anchor_z.float()
    return torch.cat([null_z, anchor_z, torch.abs(null_z - anchor_z), null_z * anchor_z], dim=1)


def train_eval_split(n: int, seed: int = 123) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    split = max(4, int(0.7 * n))
    return perm[:split], perm[split:]


def fit_ridge_classifier(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float = 1.0) -> torch.Tensor:
    y = y_train.float() * 2.0 - 1.0
    X = torch.cat([x_train.float(), torch.ones(x_train.shape[0], 1)], dim=1)
    eye = torch.eye(X.shape[1], dtype=X.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(X.T @ X + ridge * eye, X.T @ y)


def score_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    X = torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)
    return X @ w


def fit_gradient_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    kind: str,
    steps: int = 160,
    lr: float = 0.04,
    weight_decay: float = 1e-3,
) -> torch.Tensor:
    x_train = x_train.float()
    y_train = y_train.float()
    w = torch.zeros(x_train.shape[1] + 1, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=weight_decay)
    X = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
    signed = y_train * 2.0 - 1.0
    for _ in range(steps):
        score = X @ w
        if kind == "linear_svm":
            loss = torch.clamp(1.0 - signed * score, min=0).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(score, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return w.detach()


def run_cpu_classifier(x: torch.Tensor, y: torch.Tensor, classifier: str, seed: int = 123) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    x = x.detach().cpu().float()
    y = y.detach().cpu().float()
    train_idx, eval_idx = train_eval_split(y.numel(), seed=seed)
    mean, std = standardize_fit(x[train_idx])
    x_train = standardize_apply(x[train_idx], mean, std)
    x_eval = standardize_apply(x[eval_idx], mean, std)
    if classifier == "ridge":
        w = fit_ridge_classifier(x_train, y[train_idx])
    elif classifier == "linear_svm":
        w = fit_gradient_classifier(x_train, y[train_idx], kind="linear_svm")
    elif classifier == "logistic":
        w = fit_gradient_classifier(x_train, y[train_idx], kind="logistic")
    else:
        raise ValueError(classifier)
    scores = score_linear(x_eval, w)
    metrics = binary_metrics(y[eval_idx], scores)
    ci_low, ci_high = bootstrap_auc_ci(y[eval_idx], scores)
    metrics["auc_ci_low"] = ci_low
    metrics["auc_ci_high"] = ci_high
    return metrics, y[eval_idx], scores


def pooled_image_features(images: torch.Tensor, pool: int = 8) -> torch.Tensor:
    x = images.detach().cpu().float()
    pooled = F.adaptive_avg_pool2d(x, (pool, pool)).flatten(1)
    grad_x = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean(dim=(1, 2, 3), keepdim=True).flatten(1)
    grad_y = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean(dim=(1, 2, 3), keepdim=True).flatten(1)
    stats = torch.cat([x.flatten(1).mean(dim=1, keepdim=True), x.flatten(1).std(dim=1, keepdim=True), grad_x, grad_y], dim=1)
    return torch.cat([pooled, stats], dim=1)


def relmeas_from_images(measurement, image: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    flat = measurement.flatten_img(image.float())
    err = measurement.A_forward(flat) - y.float()
    return torch.linalg.norm(err, dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)


def metrics_for_images(pred: torch.Tensor, target: torch.Tensor, measurement, y: torch.Tensor) -> dict[str, float]:
    out = batch_metrics(pred.clamp(0, 1), target.clamp(0, 1), measurement, y)
    out["rel_meas_err"] = float(relmeas_from_images(measurement, pred, y).mean().detach().cpu())
    return out
