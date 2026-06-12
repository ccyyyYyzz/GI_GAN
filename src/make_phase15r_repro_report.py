from __future__ import annotations

import json
from pathlib import Path

from .phase15r_common import REPRO_DEBUG, summarize_csv
from .phase15_common import ensure_dir, write_json


REPORT = REPRO_DEBUG / "PHASE15R_RADEMACHER_REPRO_REPORT.md"


def fnum(value: str, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value or "")


def rows_for(rows: list[dict[str, str]], method_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("method_id") == method_id]


def first(rows: list[dict[str, str]], **pred) -> dict[str, str] | None:
    for row in rows:
        ok = True
        for key, value in pred.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    return None


def classify(bp_rows: list[dict[str, str]], variants: list[dict[str, str]], ckpt_rows: list[dict[str, str]]) -> str:
    if any(row.get("status") == "reproduced_variant" for row in variants):
        return "reproduced"
    unsafe_ok = [row for row in bp_rows if row.get("override_mode") == "unsafe_old_chol" and row.get("status") == "A_dataset_backproj_reproduced"]
    safe_ok = [row for row in bp_rows if row.get("override_mode") == "safe_rebuild" and row.get("status") == "A_dataset_backproj_reproduced"]
    if safe_ok and not unsafe_ok:
        return "likely_A_scaling_or_cache_bug"
    if safe_ok:
        return "likely_checkpoint_loading_bug"
    if any(row.get("load_strict_possible_ema") == "False" for row in ckpt_rows):
        return "likely_EMA_or_refiner_bug"
    return "unresolved_requires_golden_bundle"


def md_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    if not rows:
        return "missing"
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(REPRO_DEBUG)
    inventory = summarize_csv(REPRO_DEBUG / "artifact_inventory.csv")
    exact = summarize_csv(REPRO_DEBUG / "exactA_inspection.csv")
    bp = summarize_csv(REPRO_DEBUG / "backprojection_test.csv")
    ckpt = summarize_csv(REPRO_DEBUG / "checkpoint_inspection.csv")
    variants = summarize_csv(REPRO_DEBUG / "eval_variants.csv")
    dataset = summarize_csv(REPRO_DEBUG / "dataset_split_audit.csv")
    golden = summarize_csv(REPRO_DEBUG / "golden_bundle_replay_results.csv")
    classification = classify(bp, variants, ckpt)
    reproduced = [row for row in variants if row.get("status") == "reproduced_variant"]

    exact_summary = []
    for row in exact:
        exact_summary.append(
            f"- {row.get('method_id')}: exists={row.get('file_exists')}, shape={row.get('shape')}, "
            f"norm={row.get('inferred_normalization')}, tensor_sha={row.get('tensor_sha256')}"
        )
    bp_summary = []
    for row in bp:
        bp_summary.append(
            f"- {row.get('method_id')} [{row.get('override_mode')}]: Colab BP {fnum(row.get('colab_backproj_psnr'))} / "
            f"local BP {fnum(row.get('local_backproj_psnr'))}, diff {fnum(row.get('diff_backproj_psnr'))}, "
            f"status={row.get('status')}"
        )
    variant_summary = []
    for row in variants:
        if row.get("status") == "reproduced_variant" or row.get("variant") in {
            "exactA_best_hq_rebuiltK",
            "exactA_last_default",
            "generatedA_seed_default",
        }:
            variant_summary.append(
                f"- {row.get('method_id')} {row.get('variant')}: PSNR={fnum(row.get('psnr'))}, "
                f"SSIM={fnum(row.get('ssim'))}, diff=({fnum(row.get('diff_psnr'))}, {fnum(row.get('diff_ssim'))}), "
                f"status={row.get('status')}"
            )

    report = f"""# Phase 15R Rademacher Reproducibility Report

## Classification

`{classification}`

## Direct Answers

1. Exact A files exist: {'yes' if exact and all(row.get('file_exists') == 'True' for row in exact) else 'no/partial'}.
2. Exact A shape / normalization: {'; '.join(exact_summary) if exact_summary else 'missing'}.
3. Exact A all-operator usage: `GhostMeasurementOperator.set_A_override` was added and rebuilds A, m/n metadata, K, and Cholesky cache. Safe Phase15R paths use it.
4. K / Cholesky cache after override: safe path rebuilds it; unsafe old-chol path is retained only as a diagnostic variant.
5. Backprojection Colab vs local:
{chr(10).join(bp_summary) if bp_summary else '- missing'}
6. Checkpoint loading: see `checkpoint_inspection.csv`; EMA and refiner presence are audited for every available checkpoint.
7. EMA consistency: inspected; eval default is EMA if present, raw generator otherwise.
8. Refiner consistency: inspected via model strict load and refiner key counts.
9. Split / transform consistency: local STL-10 test/train split, transform, first-batch hashes, torch and torchvision versions are recorded.
10. Matching Colab eval variant found: {'yes' if reproduced else 'no'}.
11. Matching variant: {md_table(reproduced, ['method_id', 'variant', 'psnr', 'ssim', 'checkpoint_used', 'A_source', 'model_mode', 'refiner']) if reproduced else 'none'}.
12. Most likely root cause if unresolved: see classification. If safe backprojection still mismatches Colab, the leading suspects are split/transform/data-root or Colab-side evaluation artifact mismatch; use golden bundle next.
13. Rademacher primary evidence status: keep `conditional` unless a reproduced variant is found.
14. Scrambled Hadamard primary result status: remains primary; it does not depend on random exact-A replay.
15. Next step: run the generated Colab golden-bundle exporter against the cloud-side results, then place downloaded `.pt` bundles under `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/golden_bundles`.

## Artifact Inventory

Rows: {len(inventory)}  
Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/artifact_inventory.csv`

## Exact-A Inspection

Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/exactA_inspection.csv`

{md_table(exact, ['method_id', 'file_exists', 'loadable', 'shape', 'inferred_normalization', 'row_norm_mean', 'tensor_sha256', 'notes'])}

## Backprojection Test

Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/backprojection_test.csv`

{md_table(bp, ['method_id', 'override_mode', 'colab_backproj_psnr', 'local_backproj_psnr', 'diff_backproj_psnr', 'status', 'likely_issue'])}

## Checkpoint Inspection

Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/checkpoint_inspection.csv`

{md_table(ckpt, ['method_id', 'checkpoint', 'contains_generator_ema', 'contains_refiner', 'epoch', 'load_strict_possible_raw', 'load_strict_possible_ema', 'notes'])}

## Eval Variants

Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/eval_variants.csv`

{chr(10).join(variant_summary) if variant_summary else 'No variant summary rows available.'}

## Dataset / Split Audit

Path: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/dataset_split_audit.csv`

{md_table(dataset, ['method_id', 'split', 'dataset_name', 'sample_count', 'first_batch_shape', 'notes'])}

## Golden Bundle

- Colab exporter: `E:/ns_mc_gan_gi/outputs_phase15/repro_debug/colab_export_rademacher_golden_bundle.py`
- Local replay script: `src/phase15r_replay_golden_bundle.py`
- Replay status: {md_table(golden, ['bundle', 'status', 'notes']) if golden else 'pending_golden_bundle'}

## Recommendation

Do not move Rademacher back into primary claims until either a local reproduced variant exists or the Colab golden bundle identifies the precise local/Colab divergence. Keep Scrambled Hadamard as the primary STL-10 evidence for now.
"""
    REPORT.write_text(report, encoding="utf-8")
    payload = {
        "report": str(REPORT),
        "classification": classification,
        "reproduced_variant_count": len(reproduced),
        "rademacher_primary": bool(reproduced),
    }
    write_json(REPRO_DEBUG / "PHASE15R_RADEMACHER_REPRO_REPORT.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
