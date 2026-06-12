from __future__ import annotations

from pathlib import Path

from .phase18b_common import OUT, write_text


LATEX = OUT / "latex_project_v4"


REPLACEMENTS = {
    "Representative imported no-leak reconstruction grid (Rad-5)": "standalone qualitative reconstruction panel",
    "\\section*{Reference Placeholders}": "",
    "rac12": r"\\frac{1}{2}",
}


def polish_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    notes = []
    for old, new in REPLACEMENTS.items():
        if old in text:
            text = text.replace(old, new)
            notes.append(f"Replaced `{old}` in {path.name}.")
    path.write_text(text, encoding="utf-8")
    return notes


def main() -> None:
    notes = []
    for path in [LATEX / "main.tex", *sorted((LATEX / "sections").glob("*.tex")), LATEX / "supplement" / "supplement.tex"]:
        if path.exists():
            notes.extend(polish_file(path))
    if not notes:
        notes.append("No stale temporary wording found; text already uses Phase18B polished figure flow.")
    write_text(OUT / "TEXT_POLISH_REPORT.md", "# Text Polish Report\n\n" + "\n".join(f"- {note}" for note in notes))
    print({"text_polish_report": str(OUT / "TEXT_POLISH_REPORT.md")})


if __name__ == "__main__":
    main()
