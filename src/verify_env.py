from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import torch

from .utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Verify NS-MC-GAN runtime environment.")
    parser.add_argument("--dataset_root", default="E:/ns_mc_gan_gi/data")
    parser.add_argument("--output_dir", default="E:/ns_mc_gan_gi/outputs")
    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional explicit path for the JSON report. Defaults to output_dir/env_report.json.",
    )
    return parser.parse_args()


def safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"import_failed: {exc}"


def torch_numpy_bridge_status() -> str:
    try:
        torch.zeros(1).cpu().numpy()
        return "ok"
    except Exception as exc:
        return f"failed: {exc}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = ensure_dir(args.output_dir)
    cuda_available = torch.cuda.is_available()
    gpu_name = "none"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"

    report = {
        "python_version": platform.python_version(),
        "python_executable": os.path.abspath(os.sys.executable),
        "torch_version": torch.__version__,
        "torchvision_version": safe_version("torchvision"),
        "numpy_version": safe_version("numpy"),
        "matplotlib_version": safe_version("matplotlib"),
        "tensorboard_version": safe_version("tensorboard"),
        "skimage_version": safe_version("skimage"),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "cwd": os.getcwd(),
        "dataset_root": str(dataset_root),
        "dataset_root_exists": dataset_root.exists(),
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.exists(),
        "torch_numpy_bridge": torch_numpy_bridge_status(),
    }
    for key, value in report.items():
        print(f"{key}: {value}")
    out_path = save_json(report, args.report_path or (output_dir / "env_report.json"))
    print(f"Saved environment report to: {out_path}")


if __name__ == "__main__":
    main()
