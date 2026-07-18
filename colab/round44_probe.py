from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'diagnose_fused_residual_physical_readout|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r44_results/fused_physical_readout")
summary = root / "summary.json"
log = root / "driver.log"
print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
if log.exists():
    lines = log.read_text(errors="replace").splitlines()
    print("LOG_TAIL")
    print("\n".join(lines[-20:]))
