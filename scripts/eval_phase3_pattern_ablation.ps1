param(
    [string]$EnvPrefix = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_CACHE_DIR = "E:/ns_mc_gan_gi/pip_cache"

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    conda run -p $EnvPrefix python -s @Arguments
}

Invoke-ProjectPython -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs_phase3/binary_5pct_no_secrip/best_ssim.pt --config configs/phase3_binary_5pct_no_secrip.yaml --device cuda
Invoke-ProjectPython -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs_phase3/binary_5pct_no_decorrelation/best_ssim.pt --config configs/phase3_binary_5pct_no_decorrelation.yaml --device cuda
Invoke-ProjectPython -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs_phase3/binary_5pct_no_energy/best_ssim.pt --config configs/phase3_binary_5pct_no_energy.yaml --device cuda

