from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash", "-lc",
        "ps -eo pid,etime,cmd | grep -E 'run_selected_fohi_extension|train_fiber_residual_phase_gan|diagnose_fiber_orthogonal_highpass_innovation|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r50_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    print("RESULT_DIR", directory)
    summary = directory / "campaign_summary.json"
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    for name in (
        "driver.log",
        "five_percent_adv0_lowpass.log",
        "rate02_train_gan_adv0.log",
        "rate02_adv0_highpass.log",
        "rate10_train_gan_adv0.log",
        "rate10_adv0_highpass.log",
    ):
        path = directory / name
        if path.exists():
            lines = path.read_text(errors="replace").splitlines()
            print("LOG_TAIL", name)
            print("\n".join(lines[-8:]))
