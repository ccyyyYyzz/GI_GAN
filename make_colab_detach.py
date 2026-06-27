"""Generate DETACHED Colab launchers: each, when exec'd on a session, bootstraps the repo
(gdown bundle + GitHub patch + tol/zip patches) then launches the anchor job as a nohup
background process and returns immediately, so the job is decoupled from the CLI connection.
"""
from __future__ import annotations

from pathlib import Path

GDRIVE_ID = "1HY_fnmpgVOU7j-25ntCGX6DebotbrqL6"
PATCH_URL = "https://raw.githubusercontent.com/ccyyyYyzz/GI_GAN/codex/vqgan-multiseed-handoff/gan_high_quality_gi_matched.py"
COLAB = Path(__file__).resolve().parent / "colab"
RATES = ["02", "10"]
SEEDS = [0, 1, 2]

TEMPLATE = '''import sys, zipfile, subprocess
from pathlib import Path
BUNDLE = Path('/content/vqgan_rate_repo_bundle.zip')
REPO = Path('/content/repo')
if not REPO.exists():
    if not BUNDLE.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
        gdown.download(id='{gid}', output=str(BUNDLE), quiet=False)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(REPO)
    import urllib.request
    urllib.request.urlretrieve('{patch_url}', str(REPO / 'gan_high_quality_gi_matched.py'))
    mp = REPO / 'src' / 'measurement.py'
    mp.write_text(mp.read_text().replace('tol: float = 1e-10', 'tol: float = 1e-8'))
    cj = REPO / 'colab' / 'vqgan_rate_colab_job_common.py'
    cj.write_text(cj.read_text().replace('p.suffix != ".pt"', 'True'))
DONE = Path('/content/vqgan_rate{rate}_seed{seed}_status.json')
if DONE.exists():
    print('ALREADY_DONE', flush=True)
else:
    CODE = ("import sys; sys.path[:0]=['/content/repo/colab','/content/repo']; "
            "import vqgan_rate_colab_job_common as c; c.main(rate='{rate}', seed={seed}, smoke=False)")
    subprocess.Popen(['nohup', sys.executable, '-u', '-c', CODE],
                     stdout=open('/content/job_out.log', 'w'), stderr=subprocess.STDOUT,
                     start_new_session=True, cwd='/content/repo')
    print('DETACHED_LAUNCHED', flush=True)
'''


def main():
    n = 0
    for rate in RATES:
        for seed in SEEDS:
            body = TEMPLATE.format(gid=GDRIVE_ID, patch_url=PATCH_URL, rate=rate, seed=seed)
            (COLAB / f"vqgan_rate{rate}_seed{seed}_detach.py").write_text(body)
            n += 1
    print(f"wrote {n} detached launchers under colab/")


if __name__ == "__main__":
    main()
