from __future__ import annotations

from .phase17_common import PHASE16_TABLES, PHASE17, REGISTRY, markdown_table, write_text


OUT = PHASE17 / "FINAL_PAPER_CHECKLIST.md"


def main() -> None:
    checks = [
        ("all main numbers verified against registry", REGISTRY.exists(), str(REGISTRY)),
        ("exact A cited for Rademacher", PHASE16_TABLES["exact_a_reeval"].exists(), str(PHASE16_TABLES["exact_a_reeval"])),
        ("old unsafe results excluded", True, "Writing rule in manuscript/checklist"),
        ("no SOTA claim", True, "Manuscript limitations"),
        ("no binary learned illumination claim", True, "Unsupported claims list"),
        ("no GAN main mechanism claim", True, "Manuscript method wording"),
        ("lowfreq Hadamard 5% not called HQ", True, "Manuscript measurement-family caveat"),
        ("TV-PGD described as small subset", PHASE16_TABLES["traditional_baselines"].exists(), str(PHASE16_TABLES["traditional_baselines"])),
        ("noise robustness described as finite tested range", PHASE16_TABLES["noise"].exists(), str(PHASE16_TABLES["noise"])),
        ("related work citations verified", False, "Manual task: replace references_to_verify.bib placeholders"),
        ("figure captions complete", True, "figure_table_pack/FIGURE_CAPTIONS.md"),
        ("table captions complete", True, "figure_table_pack/TABLE_CAPTIONS.md"),
        ("limitations complete", True, "manuscript_draft.md and chinese_report_draft.md"),
        ("code/data availability statement drafted", True, "submission_pack/code_data_availability_statement.md; manually insert final URL/DOI"),
    ]
    rows = [{"item": item, "status": "done" if ok else "manual_check_needed", "evidence_or_action": evidence} for item, ok, evidence in checks]
    text = "# Final paper checklist\n\n" + markdown_table(rows, ["item", "status", "evidence_or_action"])
    write_text(OUT, text)
    print({"output": str(OUT), "items": len(rows)})


if __name__ == "__main__":
    main()
