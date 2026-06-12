from __future__ import annotations

import json
import shutil

from .phase15_common import E_ROOT, PHASE15, backup_existing, ensure_dir, load_registry, numeric, read_csv, write_json


PHASE13 = E_ROOT / "outputs_phase13"
PACK = PHASE15 / "manuscript_update_pack"


def row(rows: list[dict[str, str]], method_id: str) -> dict[str, str]:
    return next(item for item in rows if item.get("method_id") == method_id)


def m(item: dict[str, str]) -> str:
    return f"{numeric(item.get('psnr')):.2f}/{numeric(item.get('ssim')):.3f}"


def exact_status(method_id: str) -> str:
    for item in read_csv(PHASE15 / "exactA_reeval" / "exactA_reeval_results.csv"):
        if item.get("method_id") == method_id:
            return item.get("status", "unknown")
    return "not_required"


def write(path_name: str, text: str) -> None:
    ensure_dir(PACK)
    (PACK / path_name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(PHASE15)
    ensure_dir(PACK)
    backup_dir = ""
    if PHASE13.exists():
        target = PHASE15 / "phase13_backup_before_phase15_update"
        backup_existing(target)
        shutil.copytree(PHASE13, target)
        backup_dir = str(target)

    rows = load_registry()
    mnist = row(rows, "mnist_hadamard5_full_colab")
    fashion = row(rows, "fashion_hadamard5_full_colab")
    scr5 = row(rows, "scrambled_hadamard5_hq_noise001_colab")
    rad5 = row(rows, "rademacher5_hq_noise001_colab")
    scr10 = row(rows, "scrambled_hadamard10_full_noise001_colab")
    rad10 = row(rows, "rademacher10_full_noise001_colab")
    rad5_status = exact_status("rademacher5_hq_noise001_colab")
    rad10_status = exact_status("rademacher10_full_noise001_colab")

    write(
        "UPDATED_ABSTRACT.md",
        f"""# Updated Abstract

We present a no-leak evaluation of NS-MC-GAN for low-rate image reconstruction from fixed compressive measurements. On STL-10, strict endpoint checkpoints reach {m(scr5)} for scrambled Hadamard at 5% and {m(scr10)} for scrambled Hadamard at 10% sampling. Simple-domain sanity checks reach {m(mnist)} on MNIST and {m(fashion)} on Fashion-MNIST at 5% low-frequency Hadamard sampling. Rademacher imports reach {m(rad5)} at 5% and {m(rad10)} at 10%, with exact measurement operators exported, but local exact-A reproduction is currently {rad5_status}/{rad10_status}; these rows should be presented conditionally until resolved.
""",
    )
    write(
        "UPDATED_RESULTS_SECTION.md",
        f"""# Updated Results Section

The final strict no-leak table should use `E:/ns_mc_gan_gi/outputs_phase15/paper_tables_final/table_main_strict_noleak_results.csv`. The cleanest primary STL-10 results are scrambled Hadamard {m(scr5)} at 5% and scrambled Hadamard {m(scr10)} at 10%. Rademacher reaches {m(rad5)} at 5% and {m(rad10)} at 10% in the Colab no-leak imports, but local exact-A re-evaluation is currently {rad5_status}/{rad10_status}; mark those rows as conditional until the mismatch is resolved.

The MNIST and Fashion-MNIST sanity checks exceed the simple-domain threshold at 5%, with MNIST {m(mnist)} and Fashion-MNIST {m(fashion)}. They should be described as sanity checks rather than the central natural-image claim.

The paper should explicitly state that local rescue and historical best-checkpoint runs are excluded from the main evidence because their selection histories are not as clean as the Phase 15 strict no-leak imports.
""",
    )
    write(
        "UPDATED_CONTRIBUTIONS.md",
        """# Updated Contributions

1. A strict no-leak reconstruction protocol using endpoint checkpoint evaluation.
2. STL-10 recovery at 5% and 10% sampling for structured scrambled Hadamard measurements, with conditional Rademacher evidence pending exact-A local reproduction.
3. Exact measurement-operator preservation for random-measurement auditability.
4. A complete audit trail with SHA256 manifests, no-leak registry, final tables, figures, and claim locks.
""",
    )
    write(
        "UPDATED_LIMITATIONS.md",
        """# Updated Limitations

- The strict main results are single-run endpoint evaluations, not multi-seed statistical estimates.
- Rademacher reproducibility requires the exact exported measurement operator, and the current local exact-A reproduction mismatch must be resolved or explicitly disclosed.
- Local rescue results with historical test monitoring are not used as strict primary evidence.
- The results support a paper-level proof of concept, not deployment-level robustness.
- The comparison set is centered on backprojection and internal measurement-family comparisons; broader classical and learned baselines remain future work.
""",
    )
    write(
        "UPDATED_MANUSCRIPT_SECTIONS.md",
        """# Updated Manuscript Sections

Use the Phase 15 tables and figures as the canonical source for Results, Ablations, Limitations, and Reproducibility. Cite `FINAL_CLAIMS_LOCKED.md` before making any headline claim.

Recommended Results order:

1. Strict no-leak protocol and checkpoint-selection rule.
2. STL-10 5% main reconstruction table, with Rademacher marked conditional.
3. STL-10 10% main reconstruction table, with Rademacher marked conditional.
4. Measurement-family attribution with the exact-A mismatch disclosed.
5. Simple-domain sanity checks.
6. No-leak audit and exclusion policy.
""",
    )
    write(
        "defense_slides_outline.md",
        """# Defense Slides Outline

1. Problem: low-rate image reconstruction from fixed measurements.
2. Method: NS-MC-GAN with physics-consistent measurement projection.
3. No-leak protocol: endpoint checkpoints, post-training evaluation only.
4. STL-10 5% results.
5. STL-10 10% results.
6. Measurement-family comparison.
7. Simple-domain sanity checks.
8. Reproducibility: exact A, SHA manifests, and current Rademacher mismatch.
9. Limitations and next experiments.
""",
    )
    write(
        "reviewer_risk_register.md",
        """# Reviewer Risk Register

| Risk | Mitigation |
|---|---|
| Test leakage concern | Use Phase 15 no-leak audit and endpoint checkpoint policy. |
| Random matrix reproducibility | Provide exported exact A for Rademacher runs and disclose the current local exact-A mismatch. |
| Single-seed limitation | State clearly and avoid statistical superiority claims. |
| Baseline breadth | Frame as proof of concept and include backprojection baseline. |
| Old result contamination | Exclude historical best-checkpoint and no-exact-A runs from main claims. |
""",
    )
    write(
        "figure_table_manifest.md",
        """# Figure And Table Manifest

Tables: `E:/ns_mc_gan_gi/outputs_phase15/paper_tables_final`

Figures: `E:/ns_mc_gan_gi/outputs_phase15/paper_figures_final`

Claims lock: `E:/ns_mc_gan_gi/outputs_phase15/FINAL_CLAIMS_LOCKED.md`

Final report: `E:/ns_mc_gan_gi/outputs_phase15/PHASE15_FINAL_LOCK_REPORT.md`
""",
    )
    write(
        "WRITING_CHECKLIST.md",
        """# Writing Checklist

- Use only Phase 15 strict no-leak rows in the main results table.
- Label MNIST and Fashion-MNIST as sanity checks.
- Put local rescue outputs in supplementary or omit them.
- Mention exact-A export and current local mismatch for Rademacher reproducibility.
- Avoid superiority claims across measurement families without multi-seed evidence.
- Include the no-leak audit table in supplementary material.
""",
    )
    payload = {"pack_dir": str(PACK), "phase13_backup": backup_dir, "files": sorted(p.name for p in PACK.glob("*.md"))}
    write_json(PHASE15 / "manuscript_update_pack_manifest.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
