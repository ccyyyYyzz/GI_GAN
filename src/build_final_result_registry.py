from __future__ import annotations

from .phase12_common import EXPERIMENTS, FIELDS, PHASE12, row_from_experiment, write_csv, write_json, write_md_table
from .utils import ensure_dir


def main() -> None:
    ensure_dir(PHASE12)
    rows = []
    for exp in EXPERIMENTS:
        if exp.get("optional") and not exp["path"].exists():
            continue
        rows.append(row_from_experiment(exp))
    write_csv(PHASE12 / "final_result_registry.csv", rows, FIELDS)
    write_json(PHASE12 / "final_result_registry.json", rows)
    write_md_table(PHASE12 / "final_result_registry.md", rows, FIELDS)
    print(PHASE12 / "final_result_registry.csv")


if __name__ == "__main__":
    main()
