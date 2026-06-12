from __future__ import annotations

import json
from pathlib import Path

from .phase15_common import PHASE15, ensure_dir, load_registry, numeric, read_csv, threshold_for, write_json


def status_word(row: dict[str, str]) -> str:
    _, psnr_thr, ssim_thr = threshold_for(row.get("dataset", ""), numeric(row.get("sampling_ratio")))
    passed = numeric(row.get("psnr")) >= psnr_thr and numeric(row.get("ssim")) >= ssim_thr
    return "PASS" if passed else "FAIL"


def table(rows: list[dict[str, str]]) -> str:
    exact_rows = read_csv(PHASE15 / "exactA_reeval" / "exactA_reeval_results.csv")
    exact_status = {row.get("method_id"): row.get("status") for row in exact_rows}
    lines = [
        "| Method | Dataset | Sampling | PSNR | SSIM | Delta PSNR | Threshold | Reeval |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method_id')} | {row.get('dataset')} | {numeric(row.get('sampling_ratio')):.2f} | "
            f"{numeric(row.get('psnr')):.4f} | {numeric(row.get('ssim')):.4f} | "
            f"{numeric(row.get('delta_psnr')):.4f} | {status_word(row)} | "
            f"{exact_status.get(row.get('method_id'), 'not_required')} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dir(PHASE15)
    registry = load_registry()
    audit = read_csv(PHASE15 / "noleak_audit" / "noleak_audit.csv")
    exact = read_csv(PHASE15 / "exactA_reeval" / "exactA_reeval_results.csv")
    safe_count = sum(1 for row in audit if str(row.get("paper_safe", "")).lower() == "true")
    exact_summary = ", ".join(f"{row.get('method_id')}: {row.get('status')}" for row in exact) if exact else "not run"

    report = f"""# Phase 15 Final Lock Report

## Executive Decision

The Phase 15 strict no-leak package is complete enough to enter paper writing, with one caution: the Rademacher imports have exact-A files but local exact-A re-evaluation currently mismatches the Colab final metrics. Use scrambled Hadamard, MNIST, and Fashion-MNIST as the cleanest main quantitative evidence. Treat Rademacher as conditional or supplementary until the mismatch is resolved or explicitly explained. Use Phase 14 rescue outputs only as auxiliary provenance or supplementary context.

## Main Strict No-Leak Results

{table(registry)}

## No-Leak Audit

- Audit rows: {len(audit)}
- Paper-safe rows: {safe_count}/{len(audit)}
- Audit file: `E:/ns_mc_gan_gi/outputs_phase15/noleak_audit/noleak_audit.csv`

## Exact-A Re-Evaluation

Random-measurement runs include exported exact measurement operators. Local exact-A re-evaluation status: {exact_summary}.

If a row is marked `mismatch` or `failed_interface`, the exact A file is still present and SHA-registered, but the local evaluation wrapper did not reproduce the Colab metric path in this environment. Do not claim local Rademacher reproduction until this is resolved.

## Canonical Output Paths

- Imported no-leak runs: `E:/ns_mc_gan_gi/outputs_phase15/imported_noleak`
- Registry: `E:/ns_mc_gan_gi/outputs_phase15/noleak_registry.csv`
- Final tables: `E:/ns_mc_gan_gi/outputs_phase15/paper_tables_final`
- Final figures: `E:/ns_mc_gan_gi/outputs_phase15/paper_figures_final`
- Locked claims: `E:/ns_mc_gan_gi/outputs_phase15/FINAL_CLAIMS_LOCKED.md`
- Manuscript pack: `E:/ns_mc_gan_gi/outputs_phase15/manuscript_update_pack`

## Paper Inclusion Policy

Include:

- STL-10 5% scrambled Hadamard strict no-leak row as clean main evidence.
- STL-10 10% scrambled Hadamard strict no-leak row as clean main evidence.
- Rademacher 5% and 10% rows as conditional evidence pending exact-A reproduction resolution.
- MNIST and Fashion-MNIST 5% low-frequency Hadamard sanity rows.
- Backprojection comparisons from the same strict evaluations.

Exclude from main claims:

- Old test-selected best checkpoints.
- Old random-measurement checkpoints without exact A.
- Local rescue runs with historical test monitoring.
- Any run missing `last.pt`, `eval_metrics.json`, or no-leak config evidence.

## Journal Positioning

The result is paper-viable as a compact proof-of-concept or methods paper, especially if framed around strict protocol, measurement consistency, and low-rate reconstruction. It is not yet a broad benchmark paper because multi-seed uncertainty and a wider external baseline suite are still missing.

## Next Step

Do not start more training yet. First resolve or explain the Rademacher local exact-A mismatch, then move to manuscript assembly with the Phase 15 locked tables, figures, claims, and audit trail.
"""
    out = PHASE15 / "PHASE15_FINAL_LOCK_REPORT.md"
    out.write_text(report, encoding="utf-8")
    payload = {"report": str(out), "strict_rows": len(registry), "audit_rows": len(audit), "paper_safe_rows": safe_count}
    write_json(PHASE15 / "PHASE15_FINAL_LOCK_REPORT.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
