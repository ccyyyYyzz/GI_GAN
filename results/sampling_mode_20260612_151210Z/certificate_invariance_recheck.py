from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets import get_val_dataloader
from src.eval import make_measurement
from src.exact_measurement import apply_measurement_override_from_config
from src.models import build_generator
from src.utils import apply_experiment_defaults, ensure_dir, load_config, reconstruct_from_measurements, set_seed


ROOT = Path(__file__).resolve().parent
MEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/scrambled_hadamard5_hq_noise001_colab")
PILOT_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling")
MEAN_CKPT = MEAN_ROOT / "last.pt"
PILOT_SOURCE_CKPT = PILOT_ROOT / "scr5" / "source_checkpoint.pt"
OPTIONAL_GAN_CSV = PILOT_ROOT / "optional_gan_results.csv"


def load_state(path: Path, config: dict[str, Any], measurement, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    merged = dict(config)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged.update(checkpoint["config"])
        merged["dataset_root"] = config["dataset_root"]
        merged["device"] = str(device)
        merged["batch_size"] = config["batch_size"]
        merged["limit_val_samples"] = config["limit_val_samples"]
        merged["num_workers"] = config["num_workers"]
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("generator_ema") or checkpoint.get("generator")
        key = "generator_ema" if checkpoint.get("generator_ema") is not None else "generator"
    else:
        state = checkpoint
        key = "raw_state_dict"
    if state is None:
        raise KeyError(f"No generator_ema or generator key in {path}")
    generator.load_state_dict(state)
    generator.eval()
    return generator, key


def relmeas_for_checkpoint(path: Path, config: dict[str, Any], measurement, loader, device: torch.device) -> dict[str, Any]:
    generator, key = load_state(path, config, measurement, device)
    set_seed(int(config["seed"]) + 777)
    vals = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"RelMeasErr {path.name}"):
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            xhat, _xdata, _extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config.get("use_null_project", True)),
                use_dc_project=bool(config.get("use_dc_project", True)),
                use_final_dc_project=bool(config.get("use_final_dc_project", True)),
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            flat = measurement.flatten_img(xhat.float())
            err = measurement.A_forward(flat) - y.float()
            rel = torch.linalg.norm(err, dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)
            vals.append(rel.detach().cpu())
    rel_all = torch.cat(vals)
    return {
        "checkpoint": str(path),
        "loaded_key": key,
        "sample_count": int(rel_all.numel()),
        "rel_meas_error_mean": float(rel_all.mean()),
        "rel_meas_error_std": float(rel_all.std(unbiased=False)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = apply_experiment_defaults(load_config(MEAN_ROOT / "resolved_config.yaml"))
    config["dataset_root"] = "E:/ns_mc_gan_gi/data"
    config["device"] = str(device)
    config["batch_size"] = 16
    config["num_workers"] = 0
    config["limit_val_samples"] = 256
    config["use_final_dc_project"] = True
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    loader_kwargs = dict(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        limit_val_samples=int(config["limit_val_samples"]),
        seed=int(config["seed"]),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    mean_loader = get_val_dataloader(**loader_kwargs)
    pilot_source_loader = get_val_dataloader(**loader_kwargs)

    post_gan_candidates = [
        p
        for p in PILOT_ROOT.rglob("*.pt")
        if p.name not in {"source_checkpoint.pt", "source_checkpoint_last.pt", "Q_exact_null.pt", "measurement_operator_exact.pt"}
        and "source_exact_A" not in p.name
    ]
    result = {
        "device": str(device),
        "eval_set": {
            "dataset_class": "torchvision.datasets.STL10",
            "split": "test",
            "limit_val_samples": config["limit_val_samples"],
            "limit_seed": int(config["seed"]) + 1,
        },
        "measurement_exact_info": exact_info,
        "mean": relmeas_for_checkpoint(MEAN_CKPT, config, measurement, mean_loader, device),
        "g1_source_checkpoint": relmeas_for_checkpoint(PILOT_SOURCE_CKPT, config, measurement, pilot_source_loader, device),
        "post_gan_pilot_checkpoint": {
            "status": "not_found",
            "searched_root": str(PILOT_ROOT),
            "candidate_count": len(post_gan_candidates),
            "candidates": [str(p) for p in post_gan_candidates],
            "failure": "The optional GAN pilot updated the generator in memory but did not save a post-GAN/fine-tuned checkpoint; only source_checkpoint.pt is present for Scr-5.",
        },
        "reported_g1_optional_gan_csv": str(OPTIONAL_GAN_CSV),
    }
    result["source_minus_mean_relmeas"] = (
        result["g1_source_checkpoint"]["rel_meas_error_mean"] - result["mean"]["rel_meas_error_mean"]
    )
    out = ROOT / "CERTIFICATE_INVARIANCE_RECHECK.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
