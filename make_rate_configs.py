"""Generate new-sampling-rate anchor configs (reusing the rate-agnostic priors) + Colab
wrappers for the cross-sampling-rate generalization study (B-tier).

Priors (mc_vqgan_prior_long_canary) train only the image prior and have NO operator block,
so the existing 3-seed priors are reused verbatim. Only the anchor refiner is retrained at the
new operator (sampling rate). 2% -> m=82, 10% -> m=410 (5% baseline is m=205).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs" / "compatibility"
COLAB = ROOT / "colab"
N = 4096  # 64*64

# rate -> (total_m, dct, hadamard, random); proportions follow the 5% baseline 128:56:20
RATES = {
    "02": (82, 51, 22, 8),    # ~2.0%
    "10": (410, 257, 112, 40),  # ~10.0%
}
SEEDS = [0, 1, 2]
GDRIVE_ID = "1HY_fnmpgVOU7j-25ntCGX6DebotbrqL6"  # user-shared big bundle (priors+code+configs)
# the only module missing from the bundle is fetched from the user's PUBLIC GitHub (byte-identical)
PATCH_URL = "https://raw.githubusercontent.com/ccyyyYyzz/GI_GAN/codex/vqgan-multiseed-handoff/gan_high_quality_gi_matched.py"


def base_anchor(seed: int) -> dict:
    return yaml.safe_load((CFG / f"anchor_vqgan_inversion_multiseed_hashclean_seed{seed}.yaml").read_text())


def write_anchor(rate: str, seed: int, smoke: bool) -> Path:
    m, dct, had, rnd = RATES[rate]
    assert 1 + dct + had + rnd == m, (rate, 1 + dct + had + rnd, m)
    c = copy.deepcopy(base_anchor(seed))
    tag = f"rate{rate}{'_smoke' if smoke else ''}_seed{seed}"
    c["output_dir"] = f"outputs/compatibility/measurement_conditioned_vqgan/anchor_{tag}"
    # reuse the existing rate-agnostic priors verbatim
    c["priors"]["vqae_checkpoint"] = f"outputs/compatibility/measurement_conditioned_vqgan/prior_multiseed_hashclean_seed{seed}/vqae_continuation/checkpoints/vqae_continuation_best_by_lpips.pt"
    c["priors"]["vqgan_checkpoint"] = f"outputs/compatibility/measurement_conditioned_vqgan/prior_multiseed_hashclean_seed{seed}/vqgan_continuation/checkpoints/vqgan_continuation_best_by_lpips.pt"
    c["operator"].update({"total_m": m, "dct_rows": dct, "hadamard_rows": had, "random_rows": rnd})
    # keep operator seed distinct per rate so each rate gets its own random rows but is reproducible
    c["operator"]["seed"] = 772001 + int(rate)
    c["data"]["dataset_root"] = "/content/datasets"
    if smoke:
        c["training"]["refiner_steps"] = 300
        c["training"]["val_interval"] = 150
        c["eval"]["bootstrap_reps"] = 50
    name = f"anchor_vqgan_inversion_{tag}.yaml"
    (CFG / name).write_text(yaml.safe_dump(c, sort_keys=False))
    return CFG / name


def write_wrapper(rate: str, seed: int, smoke: bool) -> Path:
    tag = f"rate{rate}{'_smoke' if smoke else ''}_seed{seed}"
    body = (
        "import sys, zipfile, subprocess\n"
        "from pathlib import Path\n"
        f"BIG_GID = '{GDRIVE_ID}'\n"
        f"PATCH_URL = '{PATCH_URL}'\n"
        "BUNDLE = Path('/content/vqgan_rate_repo_bundle.zip')\n"
        "REPO = Path('/content/repo')\n"
        "if not REPO.exists():\n"
        "    if not BUNDLE.exists():\n"
        "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)\n"
        "        import gdown\n"
        "        gdown.download(id=BIG_GID, output=str(BUNDLE), quiet=False)\n"
        "    with zipfile.ZipFile(BUNDLE) as z:\n"
        "        z.extractall(REPO)\n"
        "    import urllib.request\n"
        "    urllib.request.urlretrieve(PATCH_URL, str(REPO / 'gan_high_quality_gi_matched.py'))\n"
        "    mp = REPO / 'src' / 'measurement.py'\n"
        "    mp.write_text(mp.read_text().replace('tol: float = 1e-10', 'tol: float = 1e-8'))\n"
        "    cj = REPO / 'colab' / 'vqgan_rate_colab_job_common.py'\n"
        "    cj.write_text(cj.read_text().replace('p.suffix != \\\".pt\\\"', 'True'))\n"
        "sys.path.insert(0, str(REPO / 'colab'))\n"
        "sys.path.insert(0, str(REPO))\n"
        "import vqgan_rate_colab_job_common as common\n"
        f"common.main(rate='{rate}', seed={seed}, smoke={smoke})\n"
    )
    p = COLAB / f"vqgan_{tag}_job.py"
    p.write_text(body)
    return p


def main():
    made = []
    for rate in RATES:
        for seed in SEEDS:
            for smoke in (False, True):
                made.append(write_anchor(rate, seed, smoke))
                made.append(write_wrapper(rate, seed, smoke))
    for p in made:
        print("wrote", p.relative_to(ROOT))
    print(f"total {len(made)} files")


if __name__ == "__main__":
    main()
