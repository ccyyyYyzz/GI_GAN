from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "pub_baselines"
MODELS_PATH = REPO_ROOT / "src" / "models.py"

WINDOWS_DATASET_ROOT = r"E:\ns_mc_gan_gi\data"
WINDOWS_OUTPUT_PREFIX = "E:\\ns_mc_gan_gi\\outputs_pub_baselines\\"
COLAB_DATASET_ROOT = "/content/drive/MyDrive/ns_mc_gan_gi/data"
COLAB_OUTPUT_PREFIX = "/content/drive/MyDrive/ns_mc_gan_gi/outputs_pub_baselines/"


class ValidationError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: expected a YAML mapping.")
    return data


def supported_model_types() -> set[str]:
    tree = ast.parse(MODELS_PATH.read_text(encoding="utf-8"))
    supported: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            if isinstance(node.left, ast.Name) and node.left.id == "model_type":
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                        if isinstance(comparator.value, str):
                            supported.add(comparator.value)
                    if isinstance(op, ast.In) and isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                        for element in comparator.elts:
                            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                                supported.add(element.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    if not supported:
        raise ValidationError("Could not infer supported model_type values from src/models.py.")
    return supported


def is_colab_config(path: Path) -> bool:
    return CONFIG_ROOT / "colab" in path.parents


def require_path_format(path: Path, data: dict[str, Any]) -> None:
    dataset_root = str(data.get("dataset_root", ""))
    output_dir = str(data.get("output_dir", ""))
    if is_colab_config(path):
        if dataset_root != COLAB_DATASET_ROOT:
            raise ValidationError(f"{path}: dataset_root is not the expected Colab path: {dataset_root}")
        if not output_dir.startswith(COLAB_OUTPUT_PREFIX):
            raise ValidationError(f"{path}: output_dir is not under the Colab pub output root: {output_dir}")
    else:
        if dataset_root != WINDOWS_DATASET_ROOT:
            raise ValidationError(f"{path}: dataset_root is not the expected Windows path: {dataset_root}")
        if not output_dir.startswith(WINDOWS_OUTPUT_PREFIX):
            raise ValidationError(f"{path}: output_dir is not under the Windows pub output root: {output_dir}")


def require_no_test_training_or_validation(path: Path, data: dict[str, Any]) -> None:
    for key in ("train_split", "val_split"):
        if key not in data:
            raise ValidationError(f"{path}: {key} must be explicit so defaults cannot point at test.")
        if str(data[key]).strip().lower() == "test":
            raise ValidationError(f"{path}: {key} points to the test split.")
    for key, value in data.items():
        lower_key = str(key).lower()
        if "split" in lower_key and ("train" in lower_key or "val" in lower_key or "validation" in lower_key):
            if str(value).strip().lower() == "test":
                raise ValidationError(f"{path}: {key} points to the test split.")


def main() -> int:
    configs = sorted(CONFIG_ROOT.rglob("*.yaml"))
    if not configs:
        raise ValidationError(f"No pub baseline configs found under {CONFIG_ROOT}.")

    supported = supported_model_types()
    output_dirs: dict[str, Path] = {}
    for path in configs:
        data = load_yaml(path)
        model_type = str(data.get("model_type", "")).lower()
        if model_type not in supported:
            raise ValidationError(f"{path}: unsupported model_type {model_type!r}; supported={sorted(supported)}")

        require_path_format(path, data)
        require_no_test_training_or_validation(path, data)

        output_dir = str(data.get("output_dir", ""))
        if output_dir in output_dirs:
            raise ValidationError(f"{path}: duplicate output_dir with {output_dirs[output_dir]}: {output_dir}")
        output_dirs[output_dir] = path

        if bool(data.get("use_adversarial", False)):
            raise ValidationError(f"{path}: use_adversarial must be false.")
        if float(data.get("lambda_adv", 0.0)) != 0.0:
            raise ValidationError(f"{path}: lambda_adv must be 0.0.")

    print(f"Validated {len(configs)} pub baseline configs.")
    print(f"Supported model types checked against src/models.py: {', '.join(sorted(supported))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
