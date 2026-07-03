import sys, zipfile, subprocess
from pathlib import Path
BUNDLE = Path('/content/vqgan_rate_repo_bundle.zip')
REPO = Path('/content/repo')
if not REPO.exists():
    if not BUNDLE.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
        gdown.download(id='1HY_fnmpgVOU7j-25ntCGX6DebotbrqL6', output=str(BUNDLE), quiet=False)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(REPO)
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/ccyyyYyzz/GI_GAN/codex/vqgan-multiseed-handoff/gan_high_quality_gi_matched.py', str(REPO / 'gan_high_quality_gi_matched.py'))
    mp = REPO / 'src' / 'measurement.py'
    mp.write_text(mp.read_text().replace('tol: float = 1e-10', 'tol: float = 1e-8'))
    cj = REPO / 'colab' / 'vqgan_rate_colab_job_common.py'
    cj.write_text(cj.read_text().replace('p.suffix != ".pt"', 'True'))
DONE = Path('/content/vqgan_rate02_seed1_status.json')
if DONE.exists():
    print('ALREADY_DONE', flush=True)
else:
    CODE = ("import sys; sys.path[:0]=['/content/repo/colab','/content/repo']; "
            "import vqgan_rate_colab_job_common as c; c.main(rate='02', seed=1, smoke=False)")
    subprocess.Popen(['nohup', sys.executable, '-u', '-c', CODE],
                     stdout=open('/content/job_out.log', 'w'), stderr=subprocess.STDOUT,
                     start_new_session=True, cwd='/content/repo')
    print('DETACHED_LAUNCHED', flush=True)
