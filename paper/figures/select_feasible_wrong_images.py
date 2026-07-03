from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POOL = Path(__file__).resolve().parent / "feasible_wrong_candidate_pool"
OUT = Path(__file__).resolve().parent / "feasible_wrong_selected"
SELECTED_STEMS = [
    "rank011_u_i1789_j1947_cat_clipped",
    "rank118_u_i1789_j567_dog_clipped",
    "rank057_u_i1789_j1953_bird_clipped",
    "rank106_u_i1789_j452_airplane_clipped",
    "rank031_u_i1789_j1050_ship_clipped",
]


def main() -> None:
    wrong_dir = OUT / "wrong_images"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    for name in ["correct_x_i_1789_car.png", "reference_fig1_horse_j935_clipped.png"]:
        src = POOL / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    rows = list(csv.DictReader((POOL / "candidates.csv").open(newline="", encoding="utf-8")))
    by_stem = {Path(row["image"]).stem: row for row in rows}
    selected_rows: list[dict[str, str]] = []
    for stem in SELECTED_STEMS:
        src = POOL / "candidates" / f"{stem}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = wrong_dir / f"{stem}.png"
        shutil.copy2(src, dst)
        row = dict(by_stem[stem])
        row["selected_image"] = str(dst)
        selected_rows.append(row)

    images = [plt.imread(OUT / "correct_x_i_1789_car.png")]
    titles = ["correct\ncar i=1789"]
    for row in selected_rows:
        images.append(plt.imread(row["selected_image"]))
        titles.append(
            f"rank {int(row['rank']):03d} {row['donor_label']} j={row['donor_j']}\n"
            f"Rel={float(row['relmeaserr_uij_vs_yi']):.1e}"
        )

    fig, axes = plt.subplots(1, len(images), figsize=(2.0 * len(images), 2.3), dpi=220)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.35)
    contact = OUT / "selected_contact_sheet.png"
    fig.savefig(contact, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metadata = {
        "source_pool": str(POOL),
        "correct_image": str(OUT / "correct_x_i_1789_car.png"),
        "reference_fig1_horse": str(OUT / "reference_fig1_horse_j935_clipped.png"),
        "selected_contact_sheet": str(contact),
        "selected": selected_rows,
    }
    (OUT / "selected_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (OUT / "selected_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)
    print(json.dumps({"selected_dir": str(OUT), "n_selected": len(selected_rows), "contact_sheet": str(contact)}, indent=2))


if __name__ == "__main__":
    main()
