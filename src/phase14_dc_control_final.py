from __future__ import annotations

from pathlib import Path

from .phase14_ablation_pack_common import PHASE12_DC, out_dir, plot_bar, read_csv, write_rows


def main() -> None:
    rows = []
    for row in read_csv(PHASE12_DC):
        ratio = float(row.get("sampling_ratio") or 0)
        label = row.get("pattern_label", "")
        if row.get("pattern_type") not in {"lowfreq_hadamard", "hadamard"}:
            continue
        if ratio not in {0.05, 0.1}:
            continue
        if "include_dc" not in label and "skip_dc" not in label:
            continue
        rows.append(
            {
                "setting": f"stl10_{ratio:g}_{label}_{row.get('backprojection_mode')}",
                "dataset": "stl10",
                "sampling_ratio": ratio,
                "pattern_type": row.get("pattern_type"),
                "hadamard_include_dc": row.get("hadamard_include_dc"),
                "backprojection_mode": row.get("backprojection_mode"),
                "psnr": row.get("psnr"),
                "ssim": row.get("ssim"),
                "mse": row.get("mse"),
                "rel_meas_error": row.get("rel_meas_error"),
                "status": row.get("status", "from_phase12_dc_row_control"),
            }
        )
    fields = [
        "setting",
        "dataset",
        "sampling_ratio",
        "pattern_type",
        "hadamard_include_dc",
        "backprojection_mode",
        "psnr",
        "ssim",
        "mse",
        "rel_meas_error",
        "status",
    ]
    write_rows("dc_control_results", rows, fields)
    plot_bar(rows, "psnr", out_dir() / "dc_control_psnr.png", "DC row control PSNR", "PSNR", "setting")
    plot_bar(rows, "ssim", out_dir() / "dc_control_ssim.png", "DC row control SSIM", "SSIM", "setting")
    print(f"Wrote DC control finalization with {len(rows)} rows")


if __name__ == "__main__":
    main()
