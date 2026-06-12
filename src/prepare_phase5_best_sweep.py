from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .utils import ensure_dir, load_config, save_config, save_json


PHASE5_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase5")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")
FIXED_5_PSNR = 18.287900
FIXED_5_SSIM = 0.402722
FIXED_5_SCORE = 22.315117
PHASE4_BEST_SCORE = 22.789400


RUN_DIRS = {
    "Phase 5 Exact Binary": "exact_binary_5pct",
    "Phase 5 Exact Binary Slow": "exact_binary_slow_5pct",
    "Phase 5 Exact Binary FreezeG": "exact_binary_freezeG_5pct",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Phase 5 best sweep configs.")
    parser.add_argument("--phase5_dir", default=str(PHASE5_ROOT))
    parser.add_argument("--phase2_clean_dir", default=str(PHASE2_CLEAN_ROOT))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def qualifies(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    try:
        psnr = float(row.get("model_psnr", "nan"))
        ssim = float(row.get("model_ssim", "nan"))
        score = float(row.get("score", "nan"))
    except Exception:
        return False, []
    if psnr > FIXED_5_PSNR:
        reasons.append("psnr_gt_fixed")
    if ssim > FIXED_5_SSIM and psnr >= FIXED_5_PSNR - 0.5:
        reasons.append("ssim_gt_fixed_psnr_within_0p5db")
    if score > FIXED_5_SCORE:
        reasons.append("score_gt_fixed")
    if score > PHASE4_BEST_SCORE:
        reasons.append("score_gt_phase4_best")
    return bool(reasons), reasons


def checkpoint_for_ratio(phase2_dir: Path, ratio: float) -> Path:
    label = {
        0.01: "quick_1pct",
        0.02: "quick_2pct",
        0.05: "quick_5pct",
        0.10: "quick_10pct",
    }[ratio]
    return phase2_dir / label / "best_ssim.pt"


def output_name(ratio: float) -> str:
    return {
        0.01: "best_1pct",
        0.02: "best_2pct",
        0.05: "best_5pct",
        0.10: "best_10pct",
    }[ratio]


def config_name(ratio: float) -> str:
    return {
        0.01: "phase5_best_1pct.yaml",
        0.02: "phase5_best_2pct.yaml",
        0.05: "phase5_best_5pct.yaml",
        0.10: "phase5_best_10pct.yaml",
    }[ratio]


def main() -> None:
    args = parse_args()
    phase5_dir = ensure_dir(args.phase5_dir)
    phase2_dir = Path(args.phase2_clean_dir)
    rows = read_rows(phase5_dir / "phase5_tuning_results.csv")
    candidates = []
    for row in rows:
        if row.get("status") != "ok" or row.get("method") not in RUN_DIRS:
            continue
        ok, reasons = qualifies(row)
        if ok:
            candidates.append((float(row.get("score", 0.0)), row, reasons))
    result = {
        "generated": False,
        "reason": "no qualifying Phase 5 exact 5% configuration",
        "generated_configs": [],
        "skipped_configs": [],
    }
    if not candidates:
        save_json(result, phase5_dir / "prepare_best_sweep.json")
        print(result["reason"])
        return

    _, best_row, reasons = max(candidates, key=lambda item: item[0])
    run_dir = phase5_dir / RUN_DIRS[best_row["method"]]
    source_config_path = run_dir / "resolved_config.yaml"
    if not source_config_path.exists():
        raise RuntimeError(f"Missing source config: {source_config_path}")
    base_config = load_config(source_config_path)
    result.update(
        {
            "generated": True,
            "reason": "qualified: " + ",".join(reasons),
            "source_method": best_row["method"],
            "source_run_dir": str(run_dir),
            "source_score": best_row.get("score"),
        }
    )

    for ratio in [0.01, 0.02, 0.05, 0.10]:
        ckpt = checkpoint_for_ratio(phase2_dir, ratio)
        cfg_path = Path("configs") / config_name(ratio)
        if not ckpt.exists():
            result["skipped_configs"].append(
                {"ratio": ratio, "config": str(cfg_path), "missing_checkpoint": str(ckpt)}
            )
            continue
        cfg = dict(base_config)
        cfg["sampling_ratio"] = ratio
        cfg["output_dir"] = str(phase5_dir / output_name(ratio))
        cfg["load_generator_checkpoint"] = str(ckpt)
        cfg["load_discriminator_checkpoint"] = str(ckpt)
        cfg["load_pattern_checkpoint"] = None
        cfg["resume_checkpoint"] = None
        cfg["pattern_init"] = "fixed_rademacher_match"
        cfg["effective_A_mode"] = "signed_exact_fixed"
        cfg["use_learned_patterns"] = True
        save_config(cfg, cfg_path)
        result["generated_configs"].append(str(cfg_path))

    save_json(result, phase5_dir / "prepare_best_sweep.json")
    print(f"Prepared Phase 5 best sweep configs from {best_row['method']}")


if __name__ == "__main__":
    main()
