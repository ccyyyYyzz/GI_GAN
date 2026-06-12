from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .utils import ensure_dir, load_config, save_config


PHASE4_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase4")

METHOD_TO_CONFIG = {
    "Phase 4 Matched Binary": Path("configs/phase4_matched_binary_5pct.yaml"),
    "Phase 4 Matched Binary Slow": Path("configs/phase4_matched_binary_slow_5pct.yaml"),
    "Phase 4 Matched Binary No Freeze": Path("configs/phase4_matched_binary_no_freeze_5pct.yaml"),
    "Phase 4 Continuous Contrast": Path("configs/phase4_continuous_contrast_5pct.yaml"),
    "Phase 4 Continuous To Binary": Path("configs/phase4_continuous_to_binary_5pct.yaml"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Phase 4 best sweep configs when criteria are met.")
    parser.add_argument("--phase4_dir", default=str(PHASE4_ROOT))
    parser.add_argument("--tuning_csv", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_fixed(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("method") == "Fixed Rademacher" and row.get("status") == "ok":
            return row
    return {}


def eligible(row: dict, fixed: dict) -> bool:
    if row.get("status") != "ok" or not row.get("method", "").startswith("Phase 4"):
        return False
    psnr = float(row["model_psnr"])
    ssim = float(row["model_ssim"])
    score = float(row["score"])
    fixed_psnr = float(fixed["model_psnr"])
    fixed_ssim = float(fixed["model_ssim"])
    fixed_score = float(fixed["score"])
    return psnr > fixed_psnr or (ssim > fixed_ssim and psnr >= fixed_psnr - 0.5) or score > fixed_score


def main() -> None:
    args = parse_args()
    phase4_dir = Path(args.phase4_dir)
    tuning_csv = Path(args.tuning_csv) if args.tuning_csv else phase4_dir / "phase4_tuning_results.csv"
    rows = read_rows(tuning_csv)
    fixed = find_fixed(rows)
    if not fixed:
        print("missing fixed baseline; not preparing best sweep")
        return
    candidates = [row for row in rows if eligible(row, fixed)]
    if not candidates:
        print("No Phase 4 5% configuration met best-sweep criteria.")
        return
    best = max(candidates, key=lambda row: float(row["score"]))
    source_config = METHOD_TO_CONFIG.get(best["method"])
    if source_config is None or not source_config.exists():
        print(f"missing source config for {best['method']}; not preparing best sweep")
        return
    base = load_config(source_config)
    specs = [
        (0.02, "best_2pct", "E:/ns_mc_gan_gi/outputs_clean_phase2/quick_2pct/best_ssim.pt"),
        (0.05, "best_5pct", "E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt"),
        (0.10, "best_10pct", "E:/ns_mc_gan_gi/outputs_clean_phase2/quick_10pct/best_ssim.pt"),
    ]
    ensure_dir("configs")
    for ratio, name, ckpt in specs:
        config = dict(base)
        config["sampling_ratio"] = ratio
        config["output_dir"] = f"E:/ns_mc_gan_gi/outputs_phase4/{name}"
        config["load_generator_checkpoint"] = ckpt
        config["load_discriminator_checkpoint"] = ckpt
        config["pattern_init"] = "fixed_rademacher_match"
        config["pattern_mode"] = config.get("pattern_mode", "learned_balanced_binary_ste")
        if "load_pattern_checkpoint" in config:
            config["load_pattern_checkpoint"] = None
        save_config(config, Path("configs") / f"phase4_{name}.yaml")
    print(f"Prepared best sweep configs from {best['method']}")


if __name__ == "__main__":
    main()
