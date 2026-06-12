from __future__ import annotations

import argparse
from pathlib import Path


NOTEBOOKS = [
    "session_06_rad5_no_gate_no_final_audit.ipynb",
    "session_07_scr5_no_gate_no_final_audit.ipynb",
    "session_08_rad5_no_final_audit_no_meas_loss.ipynb",
    "session_09_scr5_no_final_audit_no_meas_loss.ipynb",
]


def status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def bullet_path(label: str, path: Path) -> str:
    return f"- {label}: `{path}` [{status(path)}]"


def optional_bundle_status(alias_path: Path, fallback_path: Path) -> str:
    if alias_path.exists():
        return "OK"
    if fallback_path.exists():
        return "OPTIONAL; fallback OK"
    return "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase 51A Colab readiness report.")
    parser.add_argument("--repo_root", default=".", help="Repository root.")
    parser.add_argument("--output_dir", default="E:/ns_mc_gan_gi/outputs_phase51A_colab_ready")
    parser.add_argument("--upload_dir", default="E:/ns_mc_gan_gi/colab_upload")
    parser.add_argument("--phase48_root", default="E:/ns_mc_gan_gi/outputs_phase48_49_colab_import")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    upload_dir = Path(args.upload_dir)
    phase48_root = Path(args.phase48_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_zip = upload_dir / "ns_mc_gan_gi_project_phase51A.zip"
    bundle_zip_51a = upload_dir / "noleak_bundle_phase51A.zip"
    bundle_zip_4849 = upload_dir / "noleak_bundle_phase48_49.zip"

    lines = [
        "# Phase 51A Colab Ready Report",
        "",
        "This report checks the local artifacts needed before starting the four Colab Phase 51A sessions.",
        "No local training is started by this check.",
        "",
        "## Colab notebooks",
    ]
    for nb in NOTEBOOKS:
        lines.append(bullet_path(nb, repo / "colab" / "phase51A" / nb))

    lines.extend(
        [
            "",
            "## Local scripts",
            bullet_path("prepare upload bundle", repo / "scripts" / "phase51A" / "phase51A_prepare_upload_bundle.ps1"),
            bullet_path("merge Colab parts", repo / "scripts" / "phase51A" / "phase51A_merge_colab_parts.ps1"),
            bullet_path("import Colab outputs", repo / "scripts" / "phase51A" / "phase51A_import_colab_outputs.ps1"),
            "",
            "## Python modules",
            bullet_path("Phase51A runner", repo / "src" / "phase51A_train_ablation.py"),
            bullet_path("Phase51A aggregator", repo / "src" / "phase51A_aggregate_reports.py"),
            bullet_path("Phase48/49 standardizer", repo / "src" / "phase51A_standardize_phase48_49.py"),
            "",
            "## Colab upload files",
            bullet_path("Phase51A project zip", project_zip),
            f"- Phase51A no-leak bundle alias (optional): `{bundle_zip_51a}` [{optional_bundle_status(bundle_zip_51a, bundle_zip_4849)}]",
            bullet_path("Phase48/49 no-leak bundle fallback", bundle_zip_4849),
            "",
            "Use either the Phase51A no-leak bundle alias or the Phase48/49 no-leak bundle fallback. "
            "The notebooks search both names.",
            "",
            "## Phase48/49 baseline inputs",
            bullet_path("Phase48/49 aggregate report", phase48_root / "AGGREGATE_PHASE48_49_REPORT.md"),
            bullet_path("Phase48/49 summary CSV", phase48_root / "phase48_49_summary.csv"),
            bullet_path("Phase48/49 supported claims", phase48_root / "PHASE48_49_SUPPORTED_CLAIMS.md"),
            bullet_path("Phase48/49 remaining risks", phase48_root / "PHASE48_49_REMAINING_RISKS.md"),
            "",
            "## Four-session assignment",
            "- Session 06: Rad-5, combined `no_gate + no_final_audit`, trains on Colab.",
            "- Session 07: Scr-5, combined `no_gate + no_final_audit`, trains on Colab.",
            "- Session 08: Rad-5, `no_final_audit + no_meas_loss`, trains on Colab.",
            "- Session 09: Scr-5, `no_final_audit + no_meas_loss`, trains on Colab.",
            "",
            "## Expected local import targets",
            "- Downloads backup: `E:/ns_mc_gan_gi/colab_downloads/phase51A`",
            "- Imported outputs: `E:/ns_mc_gan_gi/outputs_phase51A_colab_import`",
            "",
            "## Expected aggregate outputs after import",
            "- `PHASE51A_AGGREGATE_REPORT.md`",
            "- `PHASE51A_SUPPORTED_CLAIMS.md`",
            "- `PHASE51A_REMAINING_RISKS.md`",
            "- `phase51A_summary.csv` and `phase51A_summary.md`",
            "- `phase51A_ablation_matrix.csv` and `phase51A_ablation_matrix.md`",
        ]
    )

    path = out_dir / "PHASE51A_COLAB_READY_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
