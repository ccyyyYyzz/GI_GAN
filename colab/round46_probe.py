from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'run_fiber_rate_campaign|prepare_fiber_rate_caches|train_fiber_residual_phase_gan|diagnose_fiber_residual_frequency_fusion|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r46_results")
for directory in sorted(root.glob("seed*")) if root.exists() else []:
    print("RESULT_DIR", directory)
    summary = directory / "campaign_summary.json"
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    for log_name in (
        "driver.log",
        "cache_driver.log",
        "rate02_pipeline_driver.log",
        "rate10_pipeline_driver.log",
    ):
        log = directory / log_name
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            print("LOG_TAIL", log_name)
            print("\n".join(lines[-10:]))

