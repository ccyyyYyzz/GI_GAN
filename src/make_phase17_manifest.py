from __future__ import annotations

from .phase17_common import PHASE17, output_file_purpose, write_text


def next_step(path_name: str) -> str:
    if path_name.endswith(".bib") or "citations" in path_name:
        return "Manually verify and replace placeholder citations."
    if "manuscript" in path_name:
        return "Manual scientific editing, citation insertion, and figure placement."
    if "supplement" in path_name:
        return "Trim long tables if required by journal format."
    if "figure_table_pack" in path_name:
        return "Use to draw final figures and place tables."
    if "checklist" in path_name.lower():
        return "Resolve manual_check_needed items before submission."
    return "Review and edit for final submission style."


def main() -> None:
    rows = []
    for path in sorted(PHASE17.rglob("*")):
        if path.is_file() and path.name != "PHASE17_MANIFEST.md":
            rel = path.relative_to(PHASE17).as_posix()
            rows.append((rel, output_file_purpose(path), next_step(rel), path.stat().st_size))
    lines = [
        "# Phase17 manifest",
        "",
        f"Output root: `{PHASE17}`",
        "",
        "|file|purpose|next step|size bytes|",
        "|---|---|---|---|",
    ]
    for rel, purpose, step, size in rows:
        lines.append(f"|{rel}|{purpose}|{step}|{size}|")
    lines.extend(
        [
            "",
            "## Overall recommendation",
            "",
            "Stop broad new experiments unless a specific manuscript or reviewer gap appears. Move to manual polishing, citation verification, figure drawing, and journal-format editing.",
        ]
    )
    write_text(PHASE17 / "PHASE17_MANIFEST.md", "\n".join(lines))
    print({"output": str(PHASE17 / "PHASE17_MANIFEST.md"), "files": len(rows)})


if __name__ == "__main__":
    main()
