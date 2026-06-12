from __future__ import annotations

import argparse

from .phase26_common import drive_root, markdown_table, output_root, read_csv, safe_float, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 26 architecture/limit report.")
    parser.add_argument("--drive_root", default=None)
    return parser.parse_args()


def top_pca(rows):
    winners = []
    for method_id in sorted({row.get("method_id", "") for row in rows}):
        subset = [row for row in rows if row.get("method_id") == method_id and row.get("status") == "ok"]
        if subset:
            winners.append(sorted(subset, key=lambda row: safe_float(row.get("pca_psnr")), reverse=True)[0])
    return winners


def completed_pilots(rows):
    return [row for row in rows if row.get("status") == "complete"]


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    out = output_root(root)
    pca_rows = read_csv(out / "pca_oracle_full" / "pca_oracle_full_results.csv")
    pilot_rows = read_csv(out / "arch_pilot_results.csv")
    gate_path = out / "PHASE26_GATE_DECISION.md"
    gate_text = gate_path.read_text(encoding="utf-8") if gate_path.exists() else "Gate decision has not been generated yet."
    pca_best = top_pca(pca_rows)
    pilot_done = completed_pilots(pilot_rows)

    text = f"""# Phase 26 Limit and Architecture Report

## 1. Why Arbitrary-Image Recovery Is Impossible

For 64 x 64 grayscale images, n = 4096. At 5% sampling there are about 205 measurements, and at 10% there are about 410. Because m < n, the measurement operator has a non-trivial null space. Arbitrary-image recovery is therefore impossible without additional assumptions or priors.

## 2. What Empirical Limit Means

The empirical limit is the quality reachable under a chosen image prior and measurement-consistency rule. It is not a theoretical guarantee for arbitrary images. In this project, the current result should be interpreted as measurement-consistent reconstruction under a learned STL-10 prior.

## 3. Full PCA Oracle Results

Output CSV: `{out / "pca_oracle_full" / "pca_oracle_full_results.csv"}`

{markdown_table(pca_best, ["method_id", "family", "sampling_ratio", "k", "effective_k", "pca_psnr", "pca_ssim", "current_model_psnr", "gap_to_current_psnr", "exact_A_loaded", "status"])}

## 4. PCA vs Current Model

The PCA oracle is a linear-prior oracle / baseline. If the best PCA rows remain far below the current model, the current gains are not explained by a low-dimensional linear prior alone. If PCA approaches the current model, the current gain may be largely explained by low-dimensional structure.

Figure: `{out / "pca_oracle_full" / "pca_vs_current_model.png"}`

## 5. Architecture Pilot Setup

Pilot configs: `{out / "arch_pilot_config_manifest.csv"}`

The pilot uses 20 epochs, STL-10 5%, train_samples=20000, val_samples=1000, batch_size=8, and the same measurement/loss/budget within each family. Rademacher configs require the strict no-leak exact-A file and safe cache rebuild.

## 6. Architecture Pilot Results

Output CSV: `{out / "arch_pilot_results.csv"}`

{markdown_table(pilot_done, ["config_name", "family", "model_type", "epochs_actual", "psnr", "ssim", "mse", "rel_meas_err", "params", "status"])}

## 7. Gate Decision

{gate_text}

## 8. Whether To Run Full Training

Full 80-epoch architecture training should only be run for a pilot that beats current_hq by at least 0.3 dB PSNR or 0.01 SSIM, or optionally for an efficiency-oriented model that is within 0.2 dB while using much fewer parameters/runtime.

## 9. How To Mention This In The Current Paper

- Do not include medium pilot numbers in the main paper as final results.
- PCA oracle full may be used as supplementary linear-prior baseline if stable and reproducible.
- It is acceptable to say the PCA oracle probes whether a simple linear training-set prior can explain the reconstruction quality.
- Do not claim strict SOTA from Phase 26.
- Do not claim the GAN is the final dominant mechanism from these pilots alone.

## 10. What Should Remain Future Work

- Full no-leak architecture training for any pilot that passes the gate.
- Larger PCA budgets, such as train_samples=20000 and eval_samples=1000, if runtime allows.
- Broader architectures only after the six medium pilots justify further exploration.
- Final paper inclusion only after full approved runs, not from medium pilots.

## Status

This report is a planning and gate document. Architecture pilot values remain planning evidence unless full no-leak training is explicitly approved.
"""
    write_text(out / "PHASE26_LIMIT_ARCHITECTURE_REPORT.md", text)
    write_json(
        out / "PHASE26_LIMIT_ARCHITECTURE_REPORT.json",
        {
            "pca_rows": len(pca_rows),
            "pilot_rows": len(pilot_rows),
            "completed_pilots": len(pilot_done),
            "report": str(out / "PHASE26_LIMIT_ARCHITECTURE_REPORT.md"),
        },
    )
    print({"report": str(out / "PHASE26_LIMIT_ARCHITECTURE_REPORT.md")})


if __name__ == "__main__":
    main()
