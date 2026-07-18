from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etime,cmd | grep -E 'run_fohi_causal_campaign|train_fiber_residual_phase_gan|diagnose_fiber_orthogonal_highpass_innovation|PID' | grep -v grep",
    ],
    capture_output=True,
    text=True,
)
print("PROCESSES")
print(processes.stdout.strip())
root = Path("/content/gan_r49_results")
for directory in sorted(root.glob("*")) if root.exists() else []:
    print("RESULT_DIR", directory)
    summary = directory / "campaign_summary.json"
    print("SUMMARY", summary.exists(), summary.stat().st_size if summary.exists() else -1)
    for name in (
        "driver.log",
        "train_gan_adv0.log",
        "train_vqae2.log",
        "A_gan_adv_highpass.log",
        "B_gan_adv0_highpass.log",
        "C_vqae2_highpass.log",
        "D_gan_adv_lowpass.log",
    ):
        path = directory / name
        if path.exists():
            lines = path.read_text(errors="replace").splitlines()
            print("LOG_TAIL", name)
            print("\n".join(lines[-8:]))
