import sys, zipfile, subprocess
from pathlib import Path
BIG_GID = '1HY_fnmpgVOU7j-25ntCGX6DebotbrqL6'
PATCH_URL = 'https://raw.githubusercontent.com/ccyyyYyzz/GI_GAN/codex/vqgan-multiseed-handoff/gan_high_quality_gi_matched.py'
BUNDLE = Path('/content/vqgan_rate_repo_bundle.zip')
REPO = Path('/content/repo')
if not REPO.exists():
    if not BUNDLE.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
        gdown.download(id=BIG_GID, output=str(BUNDLE), quiet=False)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(REPO)
    import urllib.request
    urllib.request.urlretrieve(PATCH_URL, str(REPO / 'gan_high_quality_gi_matched.py'))
    mp = REPO / 'src' / 'measurement.py'
    mp.write_text(mp.read_text().replace('tol: float = 1e-10', 'tol: float = 1e-8'))
    cj = REPO / 'colab' / 'vqgan_rate_colab_job_common.py'
    cj.write_text(cj.read_text().replace('p.suffix != \".pt\"', 'True'))
sys.path.insert(0, str(REPO / 'colab'))
sys.path.insert(0, str(REPO))
import vqgan_rate_colab_job_common as common
common.main(rate='10', seed=2, smoke=True)
