"""Convention bridge for eval checker against the repo's Scr-5 operator.

This script intentionally builds the measurement operator through the training
module (`src.train.make_measurement`) and uses `measurement.flatten_img` /
`measurement.measure` for layout. It then injects fake samples in the saved P0
null space and runs the eval checker on the resulting dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.checker import check_results, pass_fail_table
from src.train import make_measurement
from src.utils import load_config


DEFAULT_CONFIG = Path(r"E:\ns_mc_gan_gi\outputs_phase15\imported_noleak\scrambled_hadamard5_hq_noise001_colab\resolved_config.yaml")
DEFAULT_P0 = Path(r"E:\ns_mc_gan_gi\results\g2r_protocol\p0\p0_scr5.pt")


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_from_payload(payload, preferred_key: str | None = None) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        if preferred_key and torch.is_tensor(payload.get(preferred_key)):
            return payload[preferred_key]
        for key in ("P0", "p0", "A", "matrix", "arr_0"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
        for value in payload.values():
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Could not find tensor in payload type {type(payload).__name__}")


def _make_reference_images(n_images: int, img_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, img_size),
        torch.linspace(0.0, 1.0, img_size),
        indexing="ij",
    )
    images = []
    for idx in range(n_images):
        cx = 0.30 + 0.10 * idx
        cy = 0.35 + 0.05 * (idx % 2)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.025)
        stripe = 0.08 * torch.sin((idx + 2) * np.pi * xx) * torch.cos((idx + 1) * np.pi * yy)
        img = (0.25 + 0.35 * xx + 0.20 * yy + 0.25 * blob + stripe).clamp(0.0, 1.0)
        images.append(img)
    return torch.stack(images, dim=0).unsqueeze(1).to(torch.float32)


def _scale_to_mse(arr: np.ndarray, target_mse: float) -> np.ndarray:
    current = float(np.mean(arr**2))
    if current <= 0:
        raise ValueError("Cannot scale zero array")
    return arr * np.sqrt(target_mse / current)


def build_bridge_dump(config_path: Path, p0_path: Path, out_dir: Path, seed: int = 20260612) -> Path:
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    config = load_config(config_path)
    config = dict(config)
    config["noise_std"] = 0.0
    measurement = make_measurement(config, device)

    p0_payload = _torch_load(p0_path)
    P0 = _tensor_from_payload(p0_payload, "P0").detach().cpu().to(torch.float64).numpy()
    A = measurement.A.detach().cpu().to(torch.float64).numpy()
    if P0.shape != (measurement.n, measurement.n):
        raise ValueError(f"P0 shape {P0.shape} does not match measurement.n={measurement.n}")

    x_img = _make_reference_images(n_images=4, img_size=measurement.img_size)
    x = measurement.flatten_img(x_img).detach().cpu().to(torch.float64).numpy()
    # Keep the training float32 measurement for audit, but feed the checker a
    # float64 y from the exact same A/layout. Otherwise the certificate measures
    # PyTorch float32 matmul roundoff (~1e-7), not vector-layout consistency.
    y_training_float32 = measurement.measure(x_img).detach().cpu().to(torch.float64).numpy()
    y = x @ A.T
    k = 12

    mean_err = rng.normal(size=x.shape) @ P0.T
    mean_err = _scale_to_mse(mean_err, 5e-4)
    sample_mean = x + mean_err
    raw_noise = rng.normal(size=(x.shape[0], k, x.shape[1]))
    raw_noise -= raw_noise.mean(axis=1, keepdims=True)
    noise = raw_noise.reshape(-1, x.shape[1]) @ P0.T
    noise = _scale_to_mse(noise.reshape(x.shape[0], k, x.shape[1]), 5e-4)
    samples = sample_mean[:, None, :] + noise

    baseline_err = rng.normal(size=x.shape) @ P0.T
    baseline = x + _scale_to_mse(baseline_err, 4e-3)
    ref_x = np.vstack([x, np.roll(x, 31, axis=1), np.roll(x, 197, axis=1)])

    out_dir.mkdir(parents=True, exist_ok=True)
    A_path = out_dir / "scr5_bridge_A.npz"
    dump_path = out_dir / "scr5_bridge_dump.npz"
    np.savez(A_path, A=A)
    np.savez(
        dump_path,
        x=x,
        samples=samples,
        samples_unclipped=samples,
        sample_mean=sample_mean,
        baseline=baseline,
        y=y,
        ref_x=ref_x,
        A_path=str(A_path.resolve()),
        P0_path=str(p0_path.resolve()),
        y_training_float32=y_training_float32,
    )
    return dump_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--p0", type=Path, default=DEFAULT_P0)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/bridge_outputs"))
    parser.add_argument("--json-out", type=Path, default=Path("eval/bridge_outputs/scr5_bridge_report.json"))
    args = parser.parse_args(argv)

    dump = build_bridge_dump(args.config, args.p0, args.out_dir)
    report = check_results(dump, perceptual_backend="edge_mse", compute_distributional=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(pass_fail_table(report))
    cert = report["gates"]["G-CERT"]["values"]
    print(
        "G-CERT "
        f"status={report['gates']['G-CERT']['status']} "
        f"median={cert['median_rel_measurement_error']:.6e} "
        f"max={cert['max_rel_measurement_error']:.6e} "
        f"float32_floor_flag={cert['float32_floor_flag']}"
    )
    print(f"dump={dump}")
    print(f"json={args.json_out}")
    return 0 if report["gates"]["G-CERT"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
