from pathlib import Path
import subprocess, sys
repo = Path('/content/repo')
print('repo exists', repo.exists())
print('pytest version check')
r = subprocess.run([sys.executable, '-m', 'pytest', '--version'], cwd=str(repo), text=True, capture_output=True)
print('version rc', r.returncode)
print(r.stdout)
print(r.stderr)
cmd=[sys.executable, '-m', 'pytest', '-q', 'tests/test_anchor_initialized_vqgan_inversion.py', 'tests/test_measurement_conditioned_vqgan.py']
r = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True)
print('pytest rc', r.returncode)
print('--- STDOUT ---')
print(r.stdout[-10000:])
print('--- STDERR ---')
print(r.stderr[-10000:])