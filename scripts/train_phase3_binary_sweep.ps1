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

Invoke-ProjectPython -m src.train --config configs/phase3_learned_binary_2pct.yaml --device cuda
Invoke-ProjectPython -m src.train --config configs/phase3_learned_binary_10pct.yaml --device cuda

