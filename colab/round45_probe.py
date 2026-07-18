from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'diagnose_fiber_residual_frequency_fusion|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r45_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    summary = directory / "summary.json"
    log = directory / "driver.log"
    print("RESULT_DIR", directory)
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("LOG_TAIL")
        print("\n".join(lines[-12:]))
