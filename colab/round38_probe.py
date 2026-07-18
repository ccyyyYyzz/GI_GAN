from pathlib import Path
import json
import subprocess

processes = subprocess.run(
    ["bash", "-lc", "ps -eo pid,etime,cmd | grep -E 'train_vqae_centered_residual_adapter|PID' | grep -v grep"],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r38_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    print("RESULT_DIR", directory)
    summary = directory / "summary.json"
    partial = directory / "partial_results.json"
    log = directory / "driver.log"
    print("SUMMARY", summary.exists(), "PARTIAL", partial.exists())
    if partial.exists():
        try:
            print("PARTIAL_COUNT", len(json.loads(partial.read_text())))
        except Exception as error:
            print("PARTIAL_ERROR", repr(error))
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("LOG_TAIL")
        print("\n".join(lines[-12:]))
