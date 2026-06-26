"""Tiny repo-level Colab sanity test for ghost-imaging modules."""

from __future__ import annotations

import os
import platform
import sys


def rel_norm(numer, denom, eps: float = 1e-12) -> float:
    import torch

    return float((torch.linalg.norm(numer) / torch.linalg.norm(denom).clamp_min(eps)).detach().cpu())


def main() -> int:
    print("python_version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())
    print("sys_path_0:", sys.path[0])

    import torch

    from src.measurement import GhostMeasurementOperator
    from src.models import PlainUNetGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("device:", str(device))
    if device.type == "cuda":
        print("cuda_device_name:", torch.cuda.get_device_name(0))

    torch.manual_seed(20260614)
    measurement = GhostMeasurementOperator(
        img_size=8,
        sampling_ratio=0.25,
        pattern_type="rademacher",
        noise_std=0.0,
        lambda_dc=1e-3,
        device=device,
        seed=123,
    )
    print("measurement_shape:", {"n": measurement.n, "m": measurement.m})

    x = torch.rand(1, 1, 8, 8, device=device)
    y = measurement.measure(x)
    x_data_flat = measurement.data_solution(y)
    x_data = measurement.unflatten_img(x_data_flat)
    v = torch.randn(1, measurement.n, device=device)
    v_null = measurement.null_project(v)
    x_dc = measurement.dc_project(v, y)

    print("measurement_y_shape:", list(y.shape))
    print("x_data_shape:", list(x_data.shape))
    print("null_forward_rel_norm:", f"{rel_norm(measurement.A_forward(v_null), v):.6e}")
    print("dc_forward_rel_error:", f"{rel_norm(measurement.A_forward(x_dc) - y, y):.6e}")

    model = PlainUNetGenerator(base_channels=8).to(device)
    model.eval()
    noise_map = torch.zeros_like(x_data)
    with torch.no_grad():
        residual = model(x_data, noise_map)
    print("model:", model.__class__.__name__)
    print("model_output_shape:", list(residual.shape))
    print("model_output_mean:", f"{float(residual.mean().detach().cpu()):.6e}")
    print("repo_sanity_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
