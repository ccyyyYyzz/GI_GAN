"""Minimal Google Colab remote-runtime smoke test."""

from __future__ import annotations

import os
import platform
import sys


def main() -> int:
    print("python_version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())

    try:
        import torch
    except Exception as exc:  # pragma: no cover - remote environment probe
        print("torch_import_ok: false")
        print("torch_import_error:", repr(exc))
        return 1

    print("torch_import_ok: true")
    print("torch_version:", torch.__version__)

    cuda_available = torch.cuda.is_available()
    print("cuda_available:", cuda_available)
    if cuda_available:
        device = torch.device("cuda")
        print("cuda_device_count:", torch.cuda.device_count())
        print("cuda_device_name:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("cuda_device_count:", 0)
        print("cuda_device_name:", "none")

    tensor = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    result = (tensor @ tensor.T).detach().cpu()
    print("tensor_device:", str(device))
    print("tensor_result:", result.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
