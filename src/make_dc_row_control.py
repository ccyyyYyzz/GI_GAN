from __future__ import annotations

from pathlib import Path

from .phase12_common import PHASE12, read_csv, write_csv, write_md_table
from .utils import ensure_dir


OUT = PHASE12 / "dc_row_control"


def find_phase9_rows() -> list[dict]:
    candidates = [
        Path("E:/ns_mc_gan_gi/outputs_phase9/sanity_hadamard/backproj_quality_table.csv"),
        Path("E:/ns_mc_gan_gi/outputs_phase9/phase9_backprojection_quality.csv"),
    ]
    for path in candidates:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def main() -> None:
    ensure_dir(OUT)
    rows = find_phase9_rows()
    normalized = []
    if rows:
        for row in rows:
            joined = " ".join(str(v) for v in row.values()).lower()
            if "dc" in joined or "hadamard" in joined:
                normalized.append(row)
    if not normalized:
        normalized = [
            {"setting": "STL-10 10% lowfreq Hadamard include DC", "backproj_psnr": 21.8253, "backproj_ssim": 0.6420, "source": "phase9 recorded evidence"},
            {"setting": "STL-10 10% lowfreq Hadamard skip DC", "backproj_psnr": 6.7351, "backproj_ssim": 0.1498, "source": "phase9 recorded evidence"},
        ]
    fields = sorted({key for row in normalized for key in row})
    write_csv(OUT / "dc_row_results.csv", normalized, fields)
    write_md_table(OUT / "dc_row_results.md", normalized, fields)
    import matplotlib.pyplot as plt

    labels = [str(row.get("setting", row.get("name", row.get("method", i)))) for i, row in enumerate(normalized)]
    psnr = [float(row.get("backproj_psnr", row.get("psnr", 0)) or 0) for row in normalized]
    ssim = [float(row.get("backproj_ssim", row.get("ssim", 0)) or 0) for row in normalized]
    for vals, ylabel, name in [(psnr, "Backprojection PSNR", "dc_row_psnr_bar"), (ssim, "Backprojection SSIM", "dc_row_ssim_bar")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)), labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " with/without DC row")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=180)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
