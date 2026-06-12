from __future__ import annotations

import json

from .phase15_common import FIELDS_REGISTRY, METHODS, PHASE15, ensure_dir, method_import_dir, registry_row, write_csv, write_json, write_md_table


def main() -> None:
    out_dir = ensure_dir(PHASE15)
    rows = []
    for method in METHODS:
        import_dir = method_import_dir(method)
        if import_dir.exists():
            rows.append(registry_row(method, import_dir))
        else:
            row = {field: "" for field in FIELDS_REGISTRY}
            row.update(
                {
                    "method_id": method["method_id"],
                    "display_name": method["display_name"],
                    "dataset": method["dataset"],
                    "sampling_ratio": method["sampling_ratio"],
                    "measurement_family": method["measurement_family"],
                    "pattern_type": method["pattern_type"],
                    "noise_std": method["noise_std"],
                    "strict_noleak": True,
                    "preferred_for_paper": False,
                    "exclusion_reason": "missing imported no-leak directory",
                    "notes": "Run phase15_import_noleak_results first.",
                }
            )
            rows.append(row)
    write_csv(out_dir / "noleak_registry.csv", rows, FIELDS_REGISTRY)
    write_md_table(out_dir / "noleak_registry.md", rows, FIELDS_REGISTRY)
    write_json(out_dir / "noleak_registry.json", rows)
    print(json.dumps({"registry_rows": len(rows), "registry": str(out_dir / "noleak_registry.csv")}, indent=2))


if __name__ == "__main__":
    main()
