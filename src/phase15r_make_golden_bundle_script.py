from __future__ import annotations

import json

from .phase15r_common import REPRO_DEBUG
from .phase15_common import ensure_dir, write_json


COLAB_SCRIPT = r'''
from pathlib import Path
import hashlib
import json
import torch

# Run this in Colab from the project root containing src/.
from src.datasets import get_val_dataloader
from src.eval import make_measurement
from src.metrics import batch_metrics
from src.models import build_generator
from src.utils import apply_experiment_defaults, load_config, reconstruct_from_measurements, resolve_device, set_seed

TASKS = [
    {
        "method_id": "rademacher5_hq_noise001_colab",
        "output_dir": Path("/content/drive/MyDrive/ns_mc_gan_gi/noleak_outputs/rademacher5_hq_noise001_colab"),
    },
    {
        "method_id": "rademacher10_full_noise001_colab",
        "output_dir": Path("/content/drive/MyDrive/ns_mc_gan_gi/noleak_outputs/rademacher10_full_noise001_colab"),
    },
]

EXPORT_DIR = Path("/content/rademacher_golden_bundles")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def load_exact_A(path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        if "A" in payload:
            return payload["A"].to(device=device, dtype=torch.float32)
        for value in payload.values():
            if torch.is_tensor(value) and value.ndim == 2:
                return value.to(device=device, dtype=torch.float32)
    if torch.is_tensor(payload):
        return payload.to(device=device, dtype=torch.float32)
    raise RuntimeError("No exact A tensor found.")

def set_A_exact(measurement, A, lambda_dc):
    if hasattr(measurement, "set_A_override"):
        measurement.set_A_override(A, metadata={"source": "golden_bundle_colab"}, rebuild_cache=True)
        return
    measurement.A = A
    measurement.m = int(A.shape[0])
    measurement.n = int(A.shape[1])
    measurement.sampling_ratio = measurement.m / measurement.n
    eye = torch.eye(measurement.m, device=A.device, dtype=A.dtype)
    measurement.K = A @ A.T + float(lambda_dc) * eye
    measurement._chol = None
    measurement._use_cholesky = True
    try:
        measurement._chol = torch.linalg.cholesky(measurement.K)
    except RuntimeError:
        measurement._use_cholesky = False

for task in TASKS:
    out = task["output_dir"]
    cfg = apply_experiment_defaults(load_config(out / "resolved_config.yaml"))
    cfg["dataset_root"] = "/content/ns_mc_gan_gi_data"
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["batch_size"] = 8
    cfg["limit_val_samples"] = 8
    device = resolve_device(cfg["device"])
    set_seed(int(cfg["seed"]))
    checkpoint_path = out / "best_hq.pt"
    if not checkpoint_path.exists():
        checkpoint_path = out / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        merged = dict(cfg)
        merged.update(checkpoint["config"])
        merged["dataset_root"] = "/content/ns_mc_gan_gi_data"
        merged["device"] = cfg["device"]
        merged["batch_size"] = 8
        merged["limit_val_samples"] = 8
        cfg = apply_experiment_defaults(merged)
    measurement = make_measurement(cfg, device)
    A = load_exact_A(out / "measurement_operator_exact.pt", device)
    set_A_exact(measurement, A, cfg.get("lambda_solver", 0.001))
    generator = build_generator(cfg, measurement=measurement).to(device)
    state_key = "generator_ema" if isinstance(checkpoint, dict) and checkpoint.get("generator_ema") is not None else "generator"
    generator.load_state_dict(checkpoint[state_key] if isinstance(checkpoint, dict) else checkpoint)
    generator.eval()
    loader = get_val_dataloader(
        dataset_root=cfg["dataset_root"],
        img_size=cfg["img_size"],
        batch_size=8,
        num_workers=0,
        limit_val_samples=8,
        seed=cfg["seed"],
        val_split="test",
        pin_memory=device.type == "cuda",
        dataset_name=cfg.get("dataset_name", "stl10"),
        class_filter=cfg.get("class_filter"),
    )
    batch = next(iter(loader))
    x = batch[0].to(device)
    with torch.no_grad():
        y = measurement.measure(x)
        x_hat, x_data, extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(cfg["use_null_project"]),
            use_dc_project=bool(cfg["use_dc_project"]),
            backprojection_mode=cfg.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=cfg.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )
    per_sample = []
    for i in range(x.shape[0]):
        metrics = batch_metrics(x_hat[i:i+1], x[i:i+1], measurement, y[i:i+1])
        per_sample.append({k: float(v) for k, v in metrics.items()})
    bundle = {
        "method_id": task["method_id"],
        "split": "test",
        "sample_count": int(x.shape[0]),
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "x_data": x_data.detach().cpu(),
        "x_stage1": extras.get("x_stage1", torch.empty(0)).detach().cpu(),
        "x_hat_unclamped": extras["x_hat_unclamped"].detach().cpu(),
        "x_metric": x_hat.detach().cpu(),
        "per_sample_metrics": per_sample,
        "A": A.detach().cpu(),
        "A_tensor_sha256": tensor_sha256(A),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "state_key": state_key,
        "config": cfg,
    }
    bundle_path = EXPORT_DIR / f"{task['method_id']}_golden_bundle.pt"
    torch.save(bundle, bundle_path)
    print("wrote", bundle_path)
print("Download /content/rademacher_golden_bundles/*.pt and place them in E:/ns_mc_gan_gi/outputs_phase15/repro_debug/golden_bundles")
'''


def main() -> None:
    ensure_dir(REPRO_DEBUG)
    out = REPRO_DEBUG / "colab_export_rademacher_golden_bundle.py"
    out.write_text(COLAB_SCRIPT.strip() + "\n", encoding="utf-8")
    payload = {
        "colab_script": str(out),
        "expected_download_dir": str(REPRO_DEBUG / "golden_bundles"),
        "status": "pending_golden_bundle_until_user_runs_colab_script",
    }
    write_json(REPRO_DEBUG / "golden_bundle_script_manifest.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
