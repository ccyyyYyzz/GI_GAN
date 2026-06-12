param(
    [string]$EnvName = "ns_mc_gan_gi_py311",
    [string]$EnvPrefix = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_CACHE_DIR = "E:/ns_mc_gan_gi/pip_cache"

function Get-CondaCommand {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Candidates = @(
        "D:/Anacondar/anaconda3/Scripts/conda.exe",
        "D:/Anacondar/anaconda3/condabin/conda.bat"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) { return $Candidate }
    }
    return $null
}

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    if ($env:NS_MCGAN_PYTHON) {
        & $env:NS_MCGAN_PYTHON @Arguments
    } else {
        $Conda = Get-CondaCommand
        if ($Conda) {
            if (Test-Path (Join-Path $EnvPrefix "python.exe")) {
                & $Conda run -p $EnvPrefix python -s @Arguments
            } else {
                & $Conda run -n $EnvName python -s @Arguments
            }
        } else {
            & python -s @Arguments
        }
    }
}

Invoke-ProjectPython -m src.sanity_physics --config configs/clean_quick_1pct.yaml --device cuda
Invoke-ProjectPython -m src.train --config configs/clean_quick_1pct.yaml --device cuda

Invoke-ProjectPython -m src.sanity_physics --config configs/clean_quick_2pct.yaml --device cuda
Invoke-ProjectPython -m src.train --config configs/clean_quick_2pct.yaml --device cuda

Invoke-ProjectPython -m src.sanity_physics --config configs/clean_quick_5pct.yaml --device cuda
Invoke-ProjectPython -m src.train --config configs/clean_quick_5pct.yaml --device cuda

Invoke-ProjectPython -m src.sanity_physics --config configs/clean_quick_10pct.yaml --device cuda
Invoke-ProjectPython -m src.train --config configs/clean_quick_10pct.yaml --device cuda
