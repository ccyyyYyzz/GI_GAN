from __future__ import annotations

import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .phase11_common import ROOT11, as_float, ensure_dir, read_json, read_metrics_for_output, threshold_flags, write_csv_rows, write_json, write_md_table
from .utils import load_config


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    return int(subprocess.run(cmd).returncode)


def main() -> None:
    ensure_dir(ROOT11)
    plan = read_json(ROOT11 / "multiseed_plan.json")
    runs = [row for row in plan.get("runs", []) if row.get("should_run")]
    status_rows = []
    metric_rows = []
    if not runs:
        reason = ""
        if plan.get("runs"):
            reason = plan["runs"][0].get("skipped_reason", "no_should_run_runs")
        status_rows.append({"seed": "", "status": "skipped", "reason": reason or "missing_plan_or_no_should_run"})
    for item in runs:
        config_path = item["config_path"]
        config = load_config(config_path)
        output_dir = Path(config["output_dir"])
        row = {"seed": item.get("seed"), "config_path": config_path, "output_dir": str(output_dir), "start_time": now(), "status": "running"}
        codes = []
        for cmd in [
            [sys.executable, "-m", "src.train", "--config", config_path, "--device", "cuda"],
            [sys.executable, "-m", "src.eval_auto", "--output_dir", str(output_dir), "--config", config_path, "--device", "cuda"],
            [sys.executable, "-m", "src.analyze_convergence", "--output_dir", str(output_dir)],
        ]:
            code = run_cmd(cmd)
            codes.append(code)
            if code != 0:
                break
        row["return_codes"] = codes
        row["end_time"] = now()
        row["status"] = "completed" if codes and all(code == 0 for code in codes) else "failed"
        status_rows.append(row)
        metrics = read_metrics_for_output(output_dir)
        model = metrics.get("model", {})
        stl10_10, _, _ = threshold_flags(config, model)
        metric_rows.append(
            {
                "seed": item.get("seed"),
                "model_psnr": model.get("psnr", ""),
                "model_ssim": model.get("ssim", ""),
                "model_mse": model.get("mse", ""),
                "model_rel_meas_err": model.get("rel_meas_error", ""),
                "threshold_success": stl10_10,
                "status": row["status"] if metrics else "missing_metrics",
            }
        )
    write_json({"tasks": status_rows, "updated_at": now()}, ROOT11 / "multiseed_status.json")
    psnrs = [as_float(row.get("model_psnr")) for row in metric_rows]
    ssims = [as_float(row.get("model_ssim")) for row in metric_rows]
    psnrs = [v for v in psnrs if v is not None]
    ssims = [v for v in ssims if v is not None]
    success_count = sum(1 for row in metric_rows if str(row.get("threshold_success")).lower() == "true")
    summary = [
        {
            "mean_psnr": statistics.mean(psnrs) if psnrs else "",
            "std_psnr": statistics.pstdev(psnrs) if len(psnrs) > 1 else 0.0 if psnrs else "",
            "mean_ssim": statistics.mean(ssims) if ssims else "",
            "std_ssim": statistics.pstdev(ssims) if len(ssims) > 1 else 0.0 if ssims else "",
            "success_count": success_count,
            "threshold_success_rate": success_count / len(metric_rows) if metric_rows else "",
            "status": "completed" if metric_rows else "skipped",
        }
    ]
    fields = ["mean_psnr", "std_psnr", "mean_ssim", "std_ssim", "success_count", "threshold_success_rate", "status"]
    write_csv_rows(summary, ROOT11 / "multiseed_summary.csv", fields)
    write_md_table(summary, ROOT11 / "multiseed_summary.md", fields)
    print(f"Multiseed summary written to: {ROOT11 / 'multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
