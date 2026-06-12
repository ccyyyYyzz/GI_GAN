from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SESSIONS = [
    "session_01_eval_probes",
    "session_02_rad5_no_gate",
    "session_03_rad5_no_final_audit",
    "session_04_scr5_no_gate",
    "session_05_scr5_no_final_audit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate imported Phase 48/49 Colab outputs.")
    parser.add_argument("--import_root", default="E:/ns_mc_gan_gi/outputs_phase48_49_colab_import")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(prefix: str, obj: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            row.update(flatten(f"{prefix}{key}_", value))
        else:
            row[f"{prefix}{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo imported sessions found.\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(key, "")) for key in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root)
    output_dir = Path(args.output_dir) if args.output_dir else import_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for session in SESSIONS:
        candidates = [import_root / session, import_root / f"{session}_outputs" / session]
        session_dir = next((p for p in candidates if p.exists()), None)
        row: dict[str, Any] = {"session": session, "present": session_dir is not None}
        if session_dir is not None:
            status = load_json(session_dir / "SESSION_STATUS.json")
            manifest = load_json(session_dir / "MANIFEST.json")
            row.update(flatten("status_", status))
            row.update(flatten("manifest_", manifest))
            report = session_dir / "SESSION_REPORT.md"
            row["session_report"] = str(report) if report.exists() else ""
            eval_metrics = session_dir / "eval_final" / "eval_metrics.json"
            if eval_metrics.exists():
                row.update(flatten("eval_", load_json(eval_metrics)))
            summary = session_dir / "mechanistic_probe_summary.csv"
            row["mechanistic_summary"] = str(summary) if summary.exists() else ""
        rows.append(row)

    write_csv(output_dir / "SESSION_SUMMARY_TABLE.csv", rows)
    write_md(output_dir / "SESSION_SUMMARY_TABLE.md", rows, "Phase 48/49 Session Summary")

    report = [
        "# Aggregate Phase 48/49 Report",
        "",
        "This report aggregates locally imported Colab outputs. Phase 48/49 remains exploratory/ablation/diagnostic unless the user explicitly approves migration into the main paper tables.",
        "",
        "## Imported Sessions",
        "",
    ]
    for row in rows:
        report.append(f"- {row['session']}: present={row['present']}")
    report.extend(
        [
            "",
            "## Next Checks",
            "",
            "- Confirm SHA256SUMS.txt from every session before using metrics.",
            "- Confirm Rademacher exact-A files are present and exact_A_loaded=true.",
            "- Use Session 01 results for mechanistic claims only when the relevant CSV/plot supports the statement.",
            "- Use Sessions 02-05 only as train-time ablations until approved for main-table inclusion.",
        ]
    )
    (output_dir / "AGGREGATE_PHASE48_49_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    claims = [
        "# Supported New Claims",
        "",
        "Fill this after reviewing imported CSVs. Do not auto-promote these exploratory diagnostics into main claims.",
        "",
        "- Measurement dependence: supported only if wrong-y/mismatched A,y degrade and paired row shuffle is stable.",
        "- Gate geometry: supported only if ||A P_N r|| / ||A r|| decreases.",
        "- Audit re-legalization: supported only if post-audit RelMeasErr drops relative to pre-audit.",
        "- Train-time necessity: supported only after Sessions 02-05 complete and pass SHA/config checks.",
    ]
    (output_dir / "SUPPORTED_NEW_CLAIMS.md").write_text("\n".join(claims) + "\n", encoding="utf-8")

    risks = [
        "# Remaining Risks",
        "",
        "- Colab sessions may use different Drive accounts; verify imported bundles and SHA files.",
        "- Full train-time ablations are expensive and should be compared against exact matching no-leak baselines.",
        "- Scrambled Hadamard configs use randomized Hadamard rows/columns with hadamard_zero_filled anchor; do not label them as low-frequency primary HQ.",
        "- Rademacher claims require exact_A_loaded=true and safe cache rebuild metadata.",
    ]
    (output_dir / "REMAINING_RISKS.md").write_text("\n".join(risks) + "\n", encoding="utf-8")
    print(f"Aggregated Phase 48/49 outputs in {output_dir}")


if __name__ == "__main__":
    main()
