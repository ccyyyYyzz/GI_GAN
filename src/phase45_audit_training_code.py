from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .models import build_generator
from .utils import apply_experiment_defaults, load_config


ROOT = Path("E:/ns_mc_gan_gi")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_phase45_math_repro"
IMPORTED = ROOT / "outputs_phase15" / "imported_noleak"

TASKS = [
    ("Rad-5", "rademacher5_hq_noise001_colab"),
    ("Scr-5", "scrambled_hadamard5_hq_noise001_colab"),
    ("Rad-10", "rademacher10_full_noise001_colab"),
    ("Scr-10", "scrambled_hadamard10_full_noise001_colab"),
    ("MNIST-5", "mnist_hadamard5_full_colab"),
    ("Fashion-5", "fashion_hadamard5_full_colab"),
]

LOSS_KEYS = [
    "lambda_l1",
    "lambda_charbonnier",
    "lambda_ssim",
    "lambda_ms_ssim",
    "lambda_edge",
    "lambda_gradient",
    "lambda_frequency",
    "lambda_dc_loss",
    "lambda_tv",
    "lambda_ms_l1",
    "lambda_stage1_aux",
    "lambda_adv",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(safe_text(v) for v in value)
    return str(value)


def latex_escape(value: Any) -> str:
    text = safe_text(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def infer_measurement_family(config: dict[str, Any]) -> str:
    pattern = str(config.get("pattern_type", "missing"))
    if pattern == "rademacher":
        return "Rademacher dense random"
    if pattern == "lowfreq_hadamard":
        row = bool(config.get("hadamard_random_row_permutation", False))
        col = bool(config.get("hadamard_random_column_permutation", False))
        if row or col:
            return "scrambled Hadamard"
        return "low-frequency Hadamard"
    if pattern == "scrambled_hadamard":
        return "scrambled Hadamard"
    return pattern


def split_description(config: dict[str, Any]) -> str:
    dataset = str(config.get("dataset_name", "missing")).lower()
    if dataset == "stl10":
        train_split = "train+unlabeled"
        val_split = "test"
    elif dataset in {"mnist", "fashion_mnist"}:
        train_split = "train"
        val_split = "test"
    else:
        train_split = "train"
        val_split = "test"
    return (
        f"{train_split} limited to {safe_text(config.get('limit_train_samples'))}; "
        f"{val_split} limited to {safe_text(config.get('limit_val_samples'))}; "
        f"subset seed {safe_text(config.get('seed'))} (+1 for eval)"
    )


def count_model_params(config: dict[str, Any]) -> int | str:
    try:
        model = build_generator(config)
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception as exc:  # pragma: no cover - audit should not fail over display metadata
        return f"missing ({exc})"


def task_record(label: str, task: str) -> dict[str, Any]:
    task_dir = IMPORTED / task
    cfg_path = task_dir / "resolved_config.yaml"
    metrics_path = task_dir / "eval_metrics.json"
    exact_path = task_dir / "measurement_operator_exact.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing final resolved config: {cfg_path}")
    config = apply_experiment_defaults(load_config(cfg_path))
    metrics = read_json(metrics_path)
    n = int(config.get("img_size", 64)) ** 2
    m = int(round(float(config.get("sampling_ratio", 0.0)) * n))
    stage = config.get("training_stage") or {}
    loss_weights = {key: config.get(key, 0.0) for key in LOSS_KEYS}
    return {
        "label": label,
        "task": task,
        "config_path": str(cfg_path),
        "metrics_path": str(metrics_path) if metrics_path.exists() else "missing",
        "dataset": config.get("dataset_name", "missing"),
        "image_size": config.get("img_size", "missing"),
        "channels": 1,
        "split": split_description(config),
        "sampling_ratio": config.get("sampling_ratio", "missing"),
        "m": m,
        "n": n,
        "measurement_family": infer_measurement_family(config),
        "pattern_type": config.get("pattern_type", "missing"),
        "exact_A_exported": exact_path.exists(),
        "exact_A_path": str(exact_path) if exact_path.exists() else "none",
        "A_normalization": config.get("matrix_normalization", "missing"),
        "backprojection_mode": config.get("backprojection_mode", "missing"),
        "noise_model": "additive iid Gaussian epsilon with torch.randn_like(y)",
        "noise_std": config.get("noise_std", "missing"),
        "lambda_op": config.get("lambda_solver", "missing"),
        "model_type": config.get("model_type", "missing"),
        "base_channels": config.get("base_channels", "missing"),
        "model_params": count_model_params(config),
        "epochs": config.get("epochs", "missing"),
        "batch_size": config.get("batch_size", "missing"),
        "optimizer": "Adam for generator and discriminator objects in train.py",
        "lr_g": config.get("lr_g", "missing"),
        "lr_d": config.get("lr_d", "missing"),
        "betas": config.get("betas", "missing"),
        "weight_decay": "none observed / not passed to Adam",
        "lr_schedule": "none observed in train.py",
        "loss_weights": loss_weights,
        "stage1_epochs": stage.get("stage1_epochs", "missing"),
        "refiner_start_epoch": stage.get("refiner_start_epoch", "missing"),
        "stage1_supervision": (
            "no weighted supervision; x_stage1 L1 is computed but lambda_stage1_aux=0"
            if float(config.get("lambda_stage1_aux", 0.0)) == 0.0
            else f"yes, lambda_stage1_aux={config.get('lambda_stage1_aux')}"
        ),
        "final_supervision": "yes; final clipped reconstruction enters image-domain losses",
        "measurement_loss": "yes; F.mse_loss(A u, y), averaged over batch and m",
        "adversarial": (
            "disabled"
            if (not bool(config.get("use_adversarial", True)) or float(config.get("lambda_adv", 0.0)) == 0.0)
            else "active"
        ),
        "ema": f"{safe_text(config.get('use_ema'))}, decay={safe_text(config.get('ema_decay'))}",
        "seed": config.get("seed", "missing"),
        "device": config.get("device", "missing"),
        "checkpoint_metric": (
            f"{config.get('checkpoint_metric_mode', 'score')} score: "
            "PSNR + score_ssim_weight*SSIM - score_relmeas_weight*RelMeasErr"
        ),
        "checkpoint_weights": (
            f"score_ssim_weight={safe_text(config.get('score_ssim_weight'))}, "
            f"score_relmeas_weight={safe_text(config.get('score_relmeas_weight'))}"
        ),
        "eval_model_psnr": (metrics.get("model") or {}).get("psnr", "missing"),
        "eval_model_ssim": (metrics.get("model") or {}).get("ssim", "missing"),
        "eval_model_mse": (metrics.get("model") or {}).get("mse", "missing"),
        "eval_rel_meas_err_clamped": (metrics.get("model") or {}).get(
            "rel_meas_err_clamped", (metrics.get("model") or {}).get("rel_meas_error", "missing")
        ),
        "eval_rel_meas_err_unclamped": (metrics.get("model") or {}).get(
            "rel_meas_err_unclamped", "missing"
        ),
        "clipping": (
            "output_range_mode=clamp_eval_only: image losses and PSNR/SSIM use clipped x_hat; "
            "training measurement loss uses x_hat_unclamped; eval reports clamped RelMeasErr and "
            "rel_meas_err_unclamped when present"
        ),
        "use_null_project": config.get("use_null_project", "missing"),
        "use_dc_project": config.get("use_dc_project", "missing"),
        "use_amp": config.get("use_amp", "missing"),
        "use_augmentation": config.get("use_augmentation", "missing"),
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "label",
        "task",
        "dataset",
        "image_size",
        "channels",
        "split",
        "sampling_ratio",
        "m",
        "n",
        "measurement_family",
        "pattern_type",
        "exact_A_exported",
        "A_normalization",
        "noise_std",
        "lambda_op",
        "model_type",
        "model_params",
        "epochs",
        "batch_size",
        "optimizer",
        "lr_g",
        "lr_d",
        "betas",
        "lr_schedule",
        "stage1_supervision",
        "measurement_loss",
        "ema",
        "checkpoint_metric",
        "eval_model_psnr",
        "eval_model_ssim",
        "eval_rel_meas_err_unclamped",
        "seed",
        "device",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: safe_text(record.get(key, "")) for key in fields})


def latex_repro_table(records: list[dict[str, Any]]) -> str:
    rows = [
        ("Dataset", "dataset"),
        ("Image size/channels", lambda r: f"{r['image_size']}x{r['image_size']} / {r['channels']}"),
        ("Split and limits", "split"),
        ("Sampling ratio", "sampling_ratio"),
        ("m,n", lambda r: f"{r['m']}, {r['n']}"),
        ("Measurement family", "measurement_family"),
        ("Exact A exported", "exact_A_exported"),
        ("A normalization", "A_normalization"),
        ("Noise std", "noise_std"),
        (r"$\lambda_{\rm op}$", "lambda_op"),
        ("Model type", "model_type"),
        ("Parameters", "model_params"),
        ("Epochs / batch", lambda r: f"{r['epochs']} / {r['batch_size']}"),
        ("Optimizer", lambda r: f"Adam, lrG={r['lr_g']}, lrD={r['lr_d']}, betas={safe_text(r['betas'])}"),
        ("LR schedule", "lr_schedule"),
        ("Loss weights", lambda r: "; ".join(f"{k}={safe_text(v)}" for k, v in r["loss_weights"].items())),
        ("Stage-1 supervision", "stage1_supervision"),
        ("Measurement loss", "measurement_loss"),
        ("EMA", "ema"),
        ("Checkpoint metric", "checkpoint_metric"),
        ("Final PSNR/SSIM", lambda r: f"{float(r['eval_model_psnr']):.3f} / {float(r['eval_model_ssim']):.3f}"),
        ("Clipping", "clipping"),
        ("Seed", "seed"),
        ("Device", "device"),
    ]
    labels = [record["label"] for record in records]
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Training and evaluation configuration for the leakage-free final runs. Values are extracted from resolved config files, training code, and final evaluation JSON.}",
        r"\label{tab:repro_config_phase45}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{0.18\textwidth}" + "p{0.13\\textwidth}" * len(labels) + "}",
        r"\toprule",
        "Field & " + " & ".join(latex_escape(label) for label in labels) + r" \\",
        r"\midrule",
    ]
    def fmt_title(title: str) -> str:
        if title.startswith("$") and title.endswith("$"):
            return title
        return latex_escape(title)

    for title, key in rows:
        values = []
        for record in records:
            value = key(record) if callable(key) else record.get(key, "missing")
            values.append(latex_escape(value))
        lines.append(fmt_title(title) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_audit_md(records: list[dict[str, Any]], path: Path) -> None:
    code_files = [
        "src/train.py",
        "src/measurement.py",
        "src/models.py",
        "src/eval.py",
        "src/eval_auto.py",
        "src/metrics.py",
        "src/datasets.py",
        "src/exact_measurement.py",
        "configs/phase10/*.yaml",
        "configs/phase11/*.yaml",
        "configs/phase14_colab/*.yaml",
        "configs/colab/*.yaml",
    ]
    lines = [
        "# Phase 45 Training Code Audit",
        "",
        "Status: generated from resolved no-leak configs, code inspection, and final eval JSON.",
        "",
        "## Files audited",
        "",
    ]
    lines.extend(f"- {item}" for item in code_files)
    lines.extend(
        [
            "",
            "## Actual training objective from code",
            "",
            "Training uses `src/train.py` and `src/losses.py`. For final reported configs, adversarial and pattern-learning losses are disabled by config.",
            "",
            "The generator loss is:",
            "",
            "```text",
            "lambda_l1 * L1(x_hat,x)",
            "+ lambda_dc_loss * MSE(A u, y)",
            "+ lambda_tv * TV(x_hat)",
            "+ lambda_charbonnier * Charbonnier(x_hat,x)",
            "+ lambda_edge * SobelL1(x_hat,x)",
            "+ lambda_ms_l1 * multiscale_L1(x_hat,x)",
            "+ lambda_ssim * (1 - differentiable_SSIM(x_hat,x))",
            "+ lambda_ms_ssim * multiscale_SSIM_loss(x_hat,x)",
            "+ lambda_gradient * finite_difference_L1(x_hat,x)",
            "+ lambda_frequency * log-rFFT2_L1(x_hat,x)",
            "+ lambda_stage1_aux * L1(x_stage1,x)",
            "+ adversarial_term",
            "+ pattern_regularization_term",
            "```",
            "",
            "For the final six runs, `use_adversarial=false`, `lambda_adv=0`, `use_learned_patterns=false`, and `lambda_stage1_aux=0`; therefore no adversarial or pattern loss contributes and stage-1 output is not weighted as a supervised target. The final output is supervised through image-domain losses.",
            "",
            "Measurement loss is exactly `torch.nn.functional.mse_loss(measurement.A_forward(flatten(u)), y)`, averaged over batch and measurement dimension. It is not normalized by `||y||` in the training code.",
            "",
            "For `output_range_mode=clamp_eval_only`, image losses use the clipped `x_hat`; training measurement loss uses `x_hat_unclamped`; evaluation reports the default clipped `rel_meas_error` and also `rel_meas_err_unclamped`.",
            "",
            "## Operator implementation",
            "",
            "`lambda_solver` in config is the operator regularizer `lambda_op`. The fixed operator stores `A` and builds `K=A A^T + lambda_op I_m`. Solves use Cholesky when available; exact-A override calls `set_A_override(..., rebuild_cache=True)`, replacing `A` and rebuilding `K` and the Cholesky cache.",
            "",
            "## Final run records",
            "",
        ]
    )
    for record in records:
        lines.extend(
            [
                f"### {record['label']} ({record['task']})",
                "",
                f"- Config: `{record['config_path']}`",
                f"- Metrics: `{record['metrics_path']}`",
                f"- Dataset: {safe_text(record['dataset'])}; split: {record['split']}",
                f"- Image/channels: {record['image_size']}x{record['image_size']} / {record['channels']}",
                f"- Sampling: {safe_text(record['sampling_ratio'])}; m={record['m']}, n={record['n']}",
                f"- Measurement: {record['measurement_family']}; pattern_type={record['pattern_type']}; normalization={record['A_normalization']}",
                f"- Exact A exported: {safe_text(record['exact_A_exported'])}; path: `{record['exact_A_path']}`",
                f"- Noise: {record['noise_model']}; noise_std={safe_text(record['noise_std'])}",
                f"- lambda_op/lambda_solver: {safe_text(record['lambda_op'])}",
                f"- Model: {record['model_type']}, base_channels={record['base_channels']}, params={record['model_params']}",
                f"- Epochs/batch: {record['epochs']} / {record['batch_size']}",
                f"- Optimizer: {record['optimizer']}; lr_g={record['lr_g']}; lr_d={record['lr_d']}; betas={safe_text(record['betas'])}; weight_decay={record['weight_decay']}",
                f"- LR schedule: {record['lr_schedule']}",
                f"- Loss weights: {', '.join(f'{k}={safe_text(v)}' for k, v in record['loss_weights'].items())}",
                f"- Stage-1 supervision: {record['stage1_supervision']}",
                f"- Final output supervision: {record['final_supervision']}",
                f"- Measurement loss: {record['measurement_loss']}",
                f"- Adversarial loss: {record['adversarial']}",
                f"- EMA: {record['ema']}",
                f"- Checkpoint selection: {record['checkpoint_metric']}; {record['checkpoint_weights']}",
                f"- Final metrics: PSNR={float(record['eval_model_psnr']):.6f}, SSIM={float(record['eval_model_ssim']):.6f}, MSE={float(record['eval_model_mse']):.6f}",
                f"- RelMeasErr: clipped={safe_text(record['eval_rel_meas_err_clamped'])}, unclipped={safe_text(record['eval_rel_meas_err_unclamped'])}",
                f"- Clipping: {record['clipping']}",
                f"- Seed/device: {record['seed']} / {record['device']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Missing or non-claimed fields",
            "",
            "- Exact GPU model is not available for every imported Colab run in local metadata; the configs state `device: cuda` only.",
            "- The imported no-leak result folders store resolved configs and final metrics; not every folder stores the full original training log or checkpoint payload locally.",
            "- Dataset subset membership is reproducible from `limit_train_samples`, `limit_val_samples`, and seed in `src/datasets.py`; the paper should state the deterministic limiting rule rather than list all indices.",
            "",
            "## Generated artifacts",
            "",
            f"- CSV table: `{OUT / 'training_config_table.csv'}`",
            f"- LaTeX table: `{OUT / 'tableS9_training_config_phase45.tex'}`",
            f"- JSON records: `{OUT / 'training_code_audit_records.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [task_record(label, task) for label, task in TASKS]
    write_csv(records, OUT / "training_config_table.csv")
    (OUT / "training_code_audit_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (OUT / "tableS9_training_config_phase45.tex").write_text(
        latex_repro_table(records), encoding="utf-8"
    )
    write_audit_md(records, OUT / "training_code_audit.md")
    print(
        {
            "records": len(records),
            "audit": str(OUT / "training_code_audit.md"),
            "table": str(OUT / "tableS9_training_config_phase45.tex"),
        }
    )


if __name__ == "__main__":
    main()
