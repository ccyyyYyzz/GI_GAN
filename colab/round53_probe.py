from pathlib import Path
import subprocess


processes = subprocess.run(
    [
        "bash",
        "-lc",
        "ps -eo pid,etimes,cmd | grep -E 'run_noisy_fohi_stress|prepare_fiber_rate_caches|diagnose_fiber_orthogonal' | grep -v grep || true",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print("PROCESSES")
print(processes)
root = Path("/content/gan_r53_results")
for result_dir in sorted(root.glob("operator_seed_*")):
    summary = result_dir / "campaign_summary.json"
    print("RESULT_DIR", result_dir)
    print("SUMMARY", summary.is_file(), summary.stat().st_size if summary.is_file() else 0)
    for log in sorted(result_dir.rglob("*.log")):
        print("LOG_TAIL", log.relative_to(result_dir))
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]))
