from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_MODEL_TYPES = {
    "unet",
    "resunet",
    "nafnet_small",
    "unrolled_ista",
    "residual_unet_small",
    "residual_unet_wide",
    "residual_unet_wide_refiner",
    "hq_unet",
    "hq_two_stage",
    "direct_y_to_image",
    "direct_coeff_to_image",
}

REQUIRED_FIELDS = {
    "seed",
    "img_size",
    "dataset_root",
    "output_dir",
    "dataset_name",
    "batch_size",
    "num_workers",
    "epochs",
    "sampling_ratio",
    "pattern_type",
    "noise_std",
    "lambda_solver",
    "model_type",
}

REQUIRED_IMPORTS = [
    "yaml",
    "torch",
    "torchvision",
    "numpy",
    "tqdm",
    "matplotlib",
    "skimage",
    "src.train",
    "src.eval_auto",
    "src.datasets",
    "src.models",
    "src.exact_measurement",
]

SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |DSA |EC )?PRIVATE KEY"),
    "github token": re.compile(r"(?:ghp|github_pat|glpat)_[A-Za-z0-9_]{20,}"),
    "openai-like key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    "credential assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\s*=\s*['\"][^'\"]{8,}['\"]"
    ),
}

WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:[\\/]")
EXECUTION_PATH_KEYS = {
    "dataset_root",
    "output_dir",
    "measurement_operator_exact_path",
    "exact_a_path",
    "load_generator_checkpoint",
    "load_discriminator_checkpoint",
    "load_pattern_checkpoint",
    "resume_checkpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight a publication baseline config before a real Colab training run."
    )
    parser.add_argument("--config", required=True, help="Config path, preferably repo-relative.")
    parser.add_argument("--device", required=True, choices=["cuda", "cpu", "auto"])
    parser.add_argument("--output_dir", required=True, help="Training output directory to validate.")
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to this script's repo.")
    parser.add_argument(
        "--report_dir",
        default=None,
        help="Optional directory for preflight_report.json/md. Defaults to --output_dir.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Publication Baseline Preflight Report",
        "",
        f"- ok: {report['ok']}",
        f"- config_path: {report['config_path']}",
        f"- device_requested: {report['device_requested']}",
        f"- cuda_available: {report['cuda_available']}",
        f"- gpu_name: {report.get('gpu_name') or 'none'}",
        f"- output_dir: {report['output_dir']}",
        f"- disk_free_gb: {report['disk_free_gb']}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {item}" for item in report["errors"] or ["none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"] or ["none"])
    lines.extend(["", "## Paths Checked", ""])
    for item in report["data_paths_checked"]:
        lines.append(
            f"- {item.get('key')}: {item.get('path')} "
            f"({item.get('status')}, required={item.get('required')})"
        )
    lines.extend(["", "## Missing Paths", ""])
    if report["missing_paths"]:
        for item in report["missing_paths"]:
            lines.append(f"- {item.get('key')}: {item.get('path')} ({item.get('reason')})")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a YAML mapping.")
    return data


def iter_strings(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(iter_strings(child, (*prefix, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(iter_strings(child, (*prefix, str(index))))
    elif isinstance(value, str):
        found.append((".".join(prefix), value))
    return found


def is_execution_path_key(key_path: str) -> bool:
    last = key_path.rsplit(".", 1)[-1].lower()
    if last in EXECUTION_PATH_KEYS:
        return True
    return last.endswith(("_path", "_dir", "_root", "_checkpoint"))


def is_deferred_colab_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    return normalized.startswith("/content/") and not Path("/content").exists()


def resolve_repo_root(raw_root: str | None) -> Path:
    root = Path(raw_root).expanduser() if raw_root else REPO_ROOT
    return root.resolve()


def resolve_config_path(repo_root: Path, raw_config: str, errors: list[str], warnings: list[str]) -> Path:
    if WINDOWS_PATH_RE.search(raw_config):
        errors.append(f"Config path is not portable for Colab execution: {raw_config}")
    if "\\" in raw_config:
        warnings.append(f"Config path uses backslashes; prefer POSIX-style repo-relative paths: {raw_config}")
    raw_path = Path(raw_config).expanduser()
    path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    path = path.resolve()
    if raw_path.is_absolute():
        try:
            path.relative_to(repo_root)
            warnings.append("Config path was absolute; repo-relative config paths are safer for Colab uploads.")
        except ValueError:
            errors.append(f"Config path is outside repo_root: {path}")
    return path


def version_of(module: Any) -> str | None:
    return getattr(module, "__version__", None)


def check_imports(repo_root: Path, errors: list[str]) -> dict[str, str | None]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    versions: dict[str, str | None] = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            versions[name] = version_of(module)
        except Exception as exc:
            versions[name] = None
            errors.append(f"Required import failed for {name}: {type(exc).__name__}: {exc}")
    return versions


def apply_defaults_if_available(config: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    try:
        from src.utils import apply_experiment_defaults

        return apply_experiment_defaults(config)
    except Exception as exc:
        warnings.append(f"Could not apply src.utils.apply_experiment_defaults: {type(exc).__name__}: {exc}")
        return dict(config)


def check_secrets_and_windows_paths(config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    for key_path, value in iter_strings(config):
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(value):
                errors.append(f"Potential secret detected in config at {key_path} ({label}).")
        if WINDOWS_PATH_RE.search(value):
            if is_execution_path_key(key_path):
                errors.append(f"Windows path appears in executable config field {key_path}: {value}")
            else:
                warnings.append(f"Windows path appears in non-execution metadata field {key_path}: {value}")


def path_status(path_text: str, *, required: bool, key: str) -> tuple[dict[str, Any], dict[str, str] | None]:
    entry: dict[str, Any] = {
        "key": key,
        "path": path_text,
        "required": required,
        "exists": None,
        "status": "unknown",
    }
    missing: dict[str, str] | None = None
    if not path_text:
        entry["status"] = "empty"
        if required:
            missing = {"key": key, "path": path_text, "reason": "empty required path"}
        return entry, missing
    if is_deferred_colab_path(path_text):
        entry["exists"] = None
        entry["status"] = "deferred_colab_path"
        entry["note"] = "This /content path can only be checked inside Colab."
        return entry, None
    path = Path(path_text).expanduser()
    exists = path.exists()
    entry["exists"] = exists
    entry["status"] = "exists" if exists else "missing"
    if required and not exists:
        missing = {"key": key, "path": path_text, "reason": "required path does not exist"}
    return entry, missing


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current if current.exists() else Path.cwd()


def check_output_dir(path_text: str, errors: list[str], warnings: list[str]) -> tuple[dict[str, Any], float | None]:
    entry: dict[str, Any] = {
        "key": "output_dir",
        "path": path_text,
        "required": True,
        "exists": None,
        "status": "unknown",
    }
    if is_deferred_colab_path(path_text):
        entry["status"] = "deferred_colab_path"
        warnings.append(f"Output directory is a Colab path and was deferred outside Colab: {path_text}")
        usage = shutil.disk_usage(Path.cwd())
        return entry, round(usage.free / (1024**3), 2)
    path = Path(path_text).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        entry["exists"] = True
        entry["status"] = "writable"
        usage = shutil.disk_usage(str(path))
        return entry, round(usage.free / (1024**3), 2)
    except Exception as exc:
        entry["exists"] = path.exists()
        entry["status"] = "not_writable"
        errors.append(f"Output directory is not writable: {path_text}: {type(exc).__name__}: {exc}")
        usage = shutil.disk_usage(str(nearest_existing_parent(path)))
        return entry, round(usage.free / (1024**3), 2)


def collect_path_checks(config: dict[str, Any], output_dir: str, errors: list[str], warnings: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]], float | None]:
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    dataset_root = str(config.get("dataset_root", ""))
    dataset_entry, dataset_missing = path_status(dataset_root, required=False, key="dataset_root")
    if dataset_entry["status"] == "missing":
        warnings.append(
            "dataset_root does not exist in this runtime; torchvision may create/download it, "
            "but Drive must be mounted and writable for a real Colab run."
        )
    checked.append(dataset_entry)
    if dataset_missing:
        missing.append(dataset_missing)

    for key in [
        "measurement_operator_exact_path",
        "exact_A_path",
        "load_generator_checkpoint",
        "load_discriminator_checkpoint",
        "load_pattern_checkpoint",
        "resume_checkpoint",
    ]:
        value = config.get(key)
        if value in {None, "", "null"}:
            continue
        required = key in {"measurement_operator_exact_path", "exact_A_path"} or bool(
            config.get("exact_A_required", False)
        )
        entry, item_missing = path_status(str(value), required=required, key=key)
        checked.append(entry)
        if item_missing:
            missing.append(item_missing)

    lock = config.get("phase25_measurement_lock") or config.get("phase26_measurement_lock") or {}
    if isinstance(lock, dict) and lock.get("exact_A_path"):
        required = bool(config.get("exact_A_required", False) or lock.get("exact_A_required", False))
        entry, item_missing = path_status(str(lock["exact_A_path"]), required=required, key="measurement_lock.exact_A_path")
        checked.append(entry)
        if item_missing:
            missing.append(item_missing)

    config_output_dir = str(config.get("output_dir", ""))
    if config_output_dir and config_output_dir != output_dir:
        entry, item_missing = path_status(config_output_dir, required=True, key="config.output_dir")
        checked.append(entry)
        if item_missing:
            missing.append(item_missing)

    output_entry, disk_free_gb = check_output_dir(output_dir, errors, warnings)
    checked.append(output_entry)
    return checked, missing, disk_free_gb


def check_device(device_requested: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device_requested": device_requested,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "gpu_name": None,
    }
    try:
        import torch
    except Exception as exc:
        errors.append(f"torch is required but could not be imported: {type(exc).__name__}: {exc}")
        return info

    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if info["cuda_available"]:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            warnings.append(f"CUDA is available but GPU name could not be read: {type(exc).__name__}: {exc}")
    if device_requested == "cuda" and not info["cuda_available"]:
        errors.append("Device 'cuda' was requested, but torch.cuda.is_available() is false.")
    if device_requested == "auto":
        info["device_effective"] = "cuda" if info["cuda_available"] else "cpu"
    else:
        info["device_effective"] = device_requested
    return info


def validate_config(config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    missing_fields = sorted(field for field in REQUIRED_FIELDS if field not in config)
    if missing_fields:
        errors.append(f"Config is missing required fields: {', '.join(missing_fields)}")

    model_type = str(config.get("model_type", "")).lower()
    if model_type and model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(f"Unsupported model_type {model_type!r}; supported={sorted(SUPPORTED_MODEL_TYPES)}")

    if bool(config.get("exact_A_required", False)):
        exact_path = (
            config.get("measurement_operator_exact_path")
            or config.get("exact_A_path")
            or (config.get("phase25_measurement_lock") or {}).get("exact_A_path")
            or (config.get("phase26_measurement_lock") or {}).get("exact_A_path")
        )
        if not exact_path:
            errors.append("exact_A_required=true but no exact-A path is configured.")

    if str(config.get("device", "")).lower() == "cuda":
        warnings.append("Config default device is cuda; preflight must pass CUDA availability before real training.")


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    repo_root = resolve_repo_root(args.repo_root)
    config_path = resolve_config_path(repo_root, args.config, errors, warnings)

    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else Path(args.output_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = (repo_root / report_dir).resolve()
    report_json = report_dir / "preflight_report.json"
    report_md = report_dir / "preflight_report.md"

    package_versions = check_imports(repo_root, errors)
    device_info = check_device(args.device, errors, warnings)

    raw_config: dict[str, Any] = {}
    config: dict[str, Any] = {}
    if not config_path.exists():
        errors.append(f"Config file does not exist: {config_path}")
    else:
        try:
            raw_config = load_yaml(config_path)
            config = apply_defaults_if_available(raw_config, warnings)
            check_secrets_and_windows_paths(raw_config, errors, warnings)
            validate_config(config, errors, warnings)
        except Exception as exc:
            errors.append(f"Config could not be parsed or validated: {type(exc).__name__}: {exc}")
            errors.append(traceback.format_exc(limit=3).strip())

    data_paths_checked, missing_paths, disk_free_gb = collect_path_checks(
        config,
        str(Path(args.output_dir).expanduser()),
        errors,
        warnings,
    )
    for item in missing_paths:
        errors.append(f"Missing required path for {item['key']}: {item['path']} ({item['reason']})")

    try:
        from src.run_protocol import enforce_run_protocol

        if config.get("output_dir"):
            enforce_run_protocol(str(config.get("output_dir")), config)
    except Exception as exc:
        errors.append(f"Run protocol check failed: {type(exc).__name__}: {exc}")

    report: dict[str, Any] = {
        "ok": not errors,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "warnings": warnings,
        "errors": errors,
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "config_arg": args.config,
        "config_path_is_repo_relative": not Path(args.config).expanduser().is_absolute(),
        "device_requested": args.device,
        "cuda_available": device_info["cuda_available"],
        "cuda_device_count": device_info["cuda_device_count"],
        "cuda_version": device_info["cuda_version"],
        "gpu_name": device_info["gpu_name"],
        "data_paths_checked": data_paths_checked,
        "missing_paths": missing_paths,
        "output_dir": str(Path(args.output_dir).expanduser()),
        "disk_free_gb": disk_free_gb,
        "package_versions": package_versions,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "report_json": str(report_json),
        "report_md": str(report_md),
    }

    write_json(report_json, report)
    write_markdown(report_md, report)
    print(json.dumps(json_safe(report), indent=2, sort_keys=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
