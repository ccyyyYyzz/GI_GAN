from __future__ import annotations

import csv
from pathlib import Path

from .utils import ensure_dir


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8")

ROWS = [
    {
        "paper": "Lyu et al., Deep-learning-based ghost imaging",
        "year": 2017,
        "task": "ghost imaging reconstruction",
        "dataset_or_target": "reported GI targets",
        "sampling_ratio": "extremely low sampling",
        "reported_psnr": "needs manual check",
        "reported_ssim": "needs manual check",
        "notes": "需要人工核对原文；GIDL improves quality at extremely low sampling compared with CS; not directly same dataset.",
        "is_directly_comparable": False,
    },
    {
        "paper": "Feng et al., High-speed computational ghost imaging based on an auto-encoder network",
        "year": 2021,
        "task": "computational ghost imaging",
        "dataset_or_target": "task not identical",
        "sampling_ratio": "low sampling",
        "reported_psnr": "about 18",
        "reported_ssim": "about 0.7",
        "notes": "需要人工核对原文；abstract reports PSNR up to about 18 and SSIM up to about 0.7 under low sampling; dataset/task not identical.",
        "is_directly_comparable": False,
    },
    {
        "paper": "Feng et al., CGANCGI under low sampling rate",
        "year": 2022,
        "task": "cGAN computational ghost imaging",
        "dataset_or_target": "task not identical",
        "sampling_ratio": "low sampling",
        "reported_psnr": "needs manual check",
        "reported_ssim": "needs manual check",
        "notes": "需要人工核对原文；cGAN-based CGI; reports large PSNR/SSIM gains over original CGI.",
        "is_directly_comparable": False,
    },
    {
        "paper": "Karim and Rahnavard, GAP single-pixel video GAN",
        "year": 2024,
        "task": "single-pixel video reconstruction",
        "dataset_or_target": "video datasets",
        "sampling_ratio": "5%",
        "reported_psnr": 17.92,
        "reported_ssim": 0.487,
        "notes": "需要人工核对原文；reports 17.92 dB PSNR and 0.487 SSIM at 5% sampling on video datasets.",
        "is_directly_comparable": False,
    },
    {
        "paper": "Mao et al., high-quality/high-diversity conditionally generative ghost imaging",
        "year": 2023,
        "task": "color ghost imaging",
        "dataset_or_target": "color GI context",
        "sampling_ratio": "not identical",
        "reported_psnr": 20.055,
        "reported_ssim": 0.723,
        "notes": "需要人工核对原文；reports PSNR 20.055 dB and SSIM 0.723 in color GI context; not directly identical.",
        "is_directly_comparable": False,
    },
    {
        "paper": "Our Phase 7 continuous physical",
        "year": 2026,
        "task": "STL-10 grayscale ghost imaging",
        "dataset_or_target": "STL-10 grayscale 64x64",
        "sampling_ratio": "5%",
        "reported_psnr": 17.6758,
        "reported_ssim": 0.4386,
        "notes": "STL-10 grayscale 64x64, 5%, PSNR 17.6758, SSIM 0.4386.",
        "is_directly_comparable": True,
    },
    {
        "paper": "Our Phase 8 strong baseline",
        "year": 2026,
        "task": "STL-10 grayscale ghost imaging",
        "dataset_or_target": "auto-filled after Phase 8",
        "sampling_ratio": "5%",
        "reported_psnr": "missing",
        "reported_ssim": "missing",
        "notes": "auto-filled after Phase 8.",
        "is_directly_comparable": True,
    },
]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    csv_path = OUTPUT_DIR / "related_work_table.csv"
    md_path = OUTPUT_DIR / "related_work_table.md"
    fields = list(ROWS[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ROWS)
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in ROWS:
        lines.append("|" + "|".join(str(row[field]) for field in fields) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote related work table to {md_path}")


if __name__ == "__main__":
    main()
