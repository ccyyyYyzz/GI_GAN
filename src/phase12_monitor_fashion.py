from __future__ import annotations

from pathlib import Path

from .phase12_common import PHASE10, PHASE12, as_float, latest_epoch_row, process_running, read_json, write_json
from .utils import ensure_dir


def main() -> None:
    out = ensure_dir(PHASE12 / "monitoring")
    run_dir = PHASE10 / "fashion_hadamard5_full"
    processes = process_running("python -m src\\.train --config configs/phase10/fashion_hadamard5_full\\.yaml")
    metrics = read_json(run_dir / "val_metrics_latest.json") or read_json(run_dir / "eval_metrics.json")
    latest = latest_epoch_row(run_dir)
    checkpoints = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    current_epoch = latest.get("epoch", "")
    psnr = metrics.get("model", {}).get("psnr", latest.get("val_model_psnr", ""))
    ssim = metrics.get("model", {}).get("ssim", latest.get("val_model_ssim", ""))
    hq_score = latest.get("hq_score", "")
    if not hq_score and as_float(psnr) is not None and as_float(ssim) is not None:
        hq_score = as_float(psnr) + 20.0 * as_float(ssim)
    target_epochs = 40
    try:
        target_epochs = int((run_dir / "resolved_config.yaml").read_text(encoding="utf-8").split("epochs:", 1)[1].splitlines()[0].strip())
    except Exception:
        pass
    try:
        actual_epoch = int(float(current_epoch))
    except Exception:
        actual_epoch = 0
    if processes:
        status = "running"
        recommendation = "Do not start another GPU training process. Keep monitoring until completion."
    elif (run_dir / "best_hq.pt").exists() and (run_dir / "eval_metrics.json").exists():
        if actual_epoch and actual_epoch < target_epochs:
            status = "stopped_incomplete"
            recommendation = "Training was intentionally stopped before the target epoch; use latest validation only as reproducibility evidence, not a completed full run."
        else:
            status = "completed"
            recommendation = "Fashion local appears complete; use eval_metrics.json and convergence plots."
    elif (run_dir / "best_hq.pt").exists():
        status = "stopped_incomplete" if actual_epoch and actual_epoch < target_epochs else "missing_eval_or_analyze"
        recommendation = "Training was stopped before completion; do not label it as full." if status == "stopped_incomplete" else "Run eval_auto and analyze_convergence after confirming no training process is active."
    else:
        status = "missing_checkpoint"
        recommendation = "No usable local Fashion checkpoint was found."
    data = {
        "status": status,
        "output_dir": str(run_dir),
        "current_epoch": current_epoch,
        "latest_psnr": psnr,
        "latest_ssim": ssim,
        "latest_hq_score": hq_score,
        "latest_checkpoint": str(checkpoints[0]) if checkpoints else "",
        "processes": processes,
        "process_ids": [p.get("ProcessId") for p in processes],
        "recommendation": recommendation,
    }
    write_json(out / "fashion_local_status.json", data)
    lines = [
        "# Fashion Local Status",
        "",
        f"- status: {status}",
        f"- output_dir: {run_dir}",
        f"- current_epoch: {current_epoch}",
        f"- latest_psnr: {psnr}",
        f"- latest_ssim: {ssim}",
        f"- latest_hq_score: {hq_score}",
        f"- latest_checkpoint: {data['latest_checkpoint']}",
        f"- process_ids: {', '.join(str(x) for x in data['process_ids'])}",
        f"- recommendation: {recommendation}",
        "",
    ]
    (out / "fashion_local_status.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "fashion_local_status.json")


if __name__ == "__main__":
    main()
