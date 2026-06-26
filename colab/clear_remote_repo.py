from pathlib import Path
import shutil
for p in [Path('/content/repo'), Path('/content/vqgan_multiseed_seed0_artifact.zip'), Path('/content/vqgan_multiseed_seed1_artifact.zip'), Path('/content/vqgan_multiseed_seed2_artifact.zip')]:
    if p.is_dir():
        shutil.rmtree(p)
        print('removed dir', p)
    elif p.exists():
        p.unlink()
        print('removed file', p)
print('ready')