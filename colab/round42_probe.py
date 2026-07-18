from pathlib import Path
import json
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'train_fiber_residual_spectral_fusion|diagnose_fiber_residual_frequency_fusion|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r42_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    summary = directory / "summary.json"
    partial = directory / "partial_results.json"
    log = directory / "driver.log"
    print("RESULT_DIR", directory)
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    if partial.exists():
        try:
            print("PARTIAL_COUNT", len(json.loads(partial.read_text())))
        except Exception as error:
            print("PARTIAL_ERROR", repr(error))
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("LOG_TAIL")
        print("\n".join(lines[-12:]))
