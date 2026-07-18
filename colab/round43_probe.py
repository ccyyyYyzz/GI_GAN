from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'run_fiber_fusion_multiseed_pipeline|train_fiber_residual_phase_gan|diagnose_fiber_residual_frequency_fusion|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r43_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    summary = directory / "pipeline_summary.json"
    log = directory / "pipeline_driver.log"
    print("RESULT_DIR", directory)
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("PIPELINE_LOG_TAIL")
        print("\n".join(lines[-10:]))
    for child_log in ("gan_driver.log", "control_driver.log", "fusion_driver.log"):
        path = directory / child_log
        if path.exists():
            lines = path.read_text(errors="replace").splitlines()
            print(child_log.upper() + "_TAIL")
            print("\n".join(lines[-4:]))
