from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase15_common import read_csv, write_csv, write_json
from .phase16_common import PHASE16, PHASE15_SUPP, ensure_dir


OUT = PHASE16 / "_aggregate"
EXPERIMENTS = [
    ("safe_exactA", PHASE16 / "safe_exactA_verification" / "safe_exactA_verification.json"),
    ("exactA_reeval_audit", PHASE16 / "exactA_reeval" / "exactA_reeval_results.csv"),
    ("attribution", PHASE16 / "attribution" / "attribution_final.csv"),
    ("real_inference_ablation", PHASE16 / "inference_ablation" / "real_inference_ablation_results.csv"),
    ("noise_sweep", PHASE16 / "noise_sweep" / "noise_sweep_results.csv"),
    ("traditional_baselines", PHASE16 / "traditional_baselines" / "tv_pgd_baseline_results.csv"),
    ("dc_row_control", PHASE16 / "dc_row_control" / "dc_row_final.csv"),
    ("statistics_ci", PHASE16 / "statistics" / "statistics_ci.csv"),
    ("stl10_classwise", PHASE16 / "classwise" / "classwise_stl10_metrics.csv"),
    ("measurement_perturbation", PHASE16 / "measurement_perturbation" / "measurement_perturbation.csv"),
    ("runtime_complexity", PHASE16 / "runtime_complexity" / "runtime_complexity.csv"),
]
FIELDS = ["experiment", "path", "exists", "rows", "status", "notes"]


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.lower() == ".csv":
        return len(read_csv(path))
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return 1
    return 1


def file_manifest() -> list[dict[str, Any]]:
    rows = []
    for path in PHASE16.rglob("*"):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(PHASE16)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
        )
    return sorted(rows, key=lambda r: r["relative_path"])


def main() -> None:
    ensure_dir(OUT)
    rows = []
    for name, path in EXPERIMENTS:
        exists = path.exists()
        rows.append(
            {
                "experiment": name,
                "path": str(path),
                "exists": exists,
                "rows": row_count(path),
                "status": "completed" if exists else "missing",
                "notes": "" if exists else "not generated in this run",
            }
        )
    write_csv(OUT / "phase16_experiment_status.csv", rows, FIELDS)
    write_json(OUT / "phase16_experiment_status.json", rows)
    manifest = file_manifest()
    write_csv(OUT / "phase16_file_manifest.csv", manifest, ["relative_path", "size_bytes", "suffix"])
    write_json(OUT / "phase16_file_manifest.json", manifest)
    lines = [
        "# Phase16 supplementary aggregate",
        "",
        f"Primary output root: `{PHASE16}`",
        f"Compatibility report path: `{PHASE15_SUPP}`",
        "",
        "## Experiment status",
        "",
        "|experiment|status|rows|",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"|{row['experiment']}|{row['status']}|{row['rows']}|")
    lines.extend(
        [
            "",
            "## File count",
            "",
            f"Generated files under Phase16: {len(manifest)}",
            "",
        ]
    )
    (OUT / "PHASE16_AGGREGATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status_rows": len(rows), "files": len(manifest), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
