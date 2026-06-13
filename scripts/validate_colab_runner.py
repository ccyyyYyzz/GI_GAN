from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "colab" / "pub_baselines_colab_runner.ipynb"

REQUIRED_MARKERS = {
    "title": "Publication Baseline Colab Runner",
    "user configuration": "User Configuration",
    "drive zip source mode": "drive_zip",
    "upload zip source mode": "upload_zip",
    "git clone source mode": "git_clone",
    "environment check": "Environment Check",
    "dependency installation": "Dependency Installation",
    "config validation": "Config Validation",
    "baseline command discovery": "Baseline Command Discovery",
    "smoke mode": "RUN_MODE = \"smoke\"",
    "single mode": "single",
    "matrix mode": "matrix",
    "logging and artifacts": "Logging And Artifacts",
    "completion notification": "Completion Notification",
    "copy block": "COLAB_SMOKE_COPY_TO_CHATGPT",
}

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

E_DRIVE_RE = re.compile(r"(?i)\bE:[\\/]")


class ValidationError(RuntimeError):
    pass


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def load_notebook(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: notebook root must be a JSON object.")
    return data


def validate_nbformat(nb: dict[str, Any], warnings: list[str]) -> list[str]:
    errors: list[str] = []
    if nb.get("nbformat") != 4:
        errors.append(f"nbformat must be 4, got {nb.get('nbformat')!r}.")
    if not isinstance(nb.get("nbformat_minor"), int):
        errors.append("nbformat_minor must be an integer.")
    if not isinstance(nb.get("metadata", {}), dict):
        errors.append("metadata must be a JSON object.")

    try:
        import nbformat  # type: ignore

        nbformat.validate(nb)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - depends on optional nbformat
        errors.append(f"nbformat package validation failed: {exc}")
    return errors


def validate_cells(nb: dict[str, Any]) -> tuple[list[str], list[str], str, str]:
    errors: list[str] = []
    warnings: list[str] = []
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("Notebook must contain a non-empty cells list.")
        return errors, warnings, "", ""

    all_sources: list[str] = []
    code_sources: list[str] = []
    non_empty_count = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"Cell {index}: cell must be a JSON object.")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code", "raw"}:
            errors.append(f"Cell {index}: unsupported cell_type {cell_type!r}.")
        source = cell_source(cell)
        if source.strip():
            non_empty_count += 1
        else:
            warnings.append(f"Cell {index}: empty source.")
        all_sources.append(source)
        if cell_type == "code":
            code_sources.append(source)
            if E_DRIVE_RE.search(source):
                errors.append(f"Cell {index}: executable cell contains a hard-coded E: Windows path.")
            if not isinstance(cell.get("outputs", []), list):
                errors.append(f"Cell {index}: code cell outputs must be a list.")
            if "execution_count" not in cell:
                errors.append(f"Cell {index}: code cell is missing execution_count.")
        if cell_type == "markdown" and E_DRIVE_RE.search(source):
            lower = source.lower()
            if not (
                "windows local example" in lower
                or "local windows example" in lower
                or "local-only" in lower
                or "windows-only" in lower
            ):
                errors.append(
                    f"Cell {index}: markdown contains an E: path without a clear local Windows example label."
                )

    if non_empty_count == 0:
        errors.append("Notebook cells are all empty.")
    return errors, warnings, "\n".join(all_sources), "\n".join(code_sources)


def validate_markers(all_text: str) -> list[str]:
    errors: list[str] = []
    for label, marker in REQUIRED_MARKERS.items():
        if marker not in all_text:
            errors.append(f"Missing required marker for {label!r}: {marker!r}.")
    return errors


def validate_defaults(code_text: str) -> list[str]:
    errors: list[str] = []
    if not re.search(r"\bRUN_MODE\s*=\s*['\"]smoke['\"]", code_text):
        errors.append('Notebook default must include RUN_MODE = "smoke".')
    if not re.search(r"\bDRY_RUN\s*=\s*True\b", code_text):
        errors.append("Notebook default must include DRY_RUN = True.")
    if re.search(r"\bRUN_MODE\s*=\s*['\"]matrix['\"]", code_text):
        errors.append("Full matrix mode must not be the notebook default.")
    return errors


def validate_secret_patterns(all_text: str) -> list[str]:
    errors: list[str] = []
    for label, pattern in SECRET_PATTERNS.items():
        match = pattern.search(all_text)
        if match:
            snippet = match.group(0)[:80].replace("\n", "\\n")
            errors.append(f"Potential secret detected ({label}): {snippet!r}.")
    return errors


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not NOTEBOOK_PATH.exists():
        print(f"FAIL: missing notebook: {NOTEBOOK_PATH}")
        return 1

    try:
        nb = load_notebook(NOTEBOOK_PATH)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1

    errors.extend(validate_nbformat(nb, warnings))
    cell_errors, cell_warnings, all_text, code_text = validate_cells(nb)
    errors.extend(cell_errors)
    warnings.extend(cell_warnings)
    errors.extend(validate_markers(all_text))
    errors.extend(validate_defaults(code_text))
    errors.extend(validate_secret_patterns(all_text))

    print("Colab runner validation summary")
    print(f"- notebook: {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    print(f"- cells: {len(nb.get('cells', [])) if isinstance(nb.get('cells'), list) else 'invalid'}")
    print(f"- warnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARN: {warning}")

    if errors:
        print(f"- errors: {len(errors)}")
        for error in errors:
            print(f"  FAIL: {error}")
        return 1

    print("- errors: 0")
    print("PASS: Colab runner notebook is structurally valid and uses conservative defaults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
