$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$EnvName = "ns_mc_gan_gi_py311"
$EnvPrefix = "E:/ns_mc_gan_gi/conda_envs/$EnvName"
$env:PIP_CACHE_DIR = "E:/ns_mc_gan_gi/pip_cache"
New-Item -ItemType Directory -Force -Path (Split-Path $EnvPrefix) | Out-Null
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

function Get-CondaCommand {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    $Candidates = @(
        "D:/Anacondar/anaconda3/Scripts/conda.exe",
        "D:/Anacondar/anaconda3/condabin/conda.bat",
        "$env:USERPROFILE/anaconda3/Scripts/conda.exe",
        "$env:USERPROFILE/miniconda3/Scripts/conda.exe"
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            return $Candidate
        }
    }
    return $null
}

$Conda = Get-CondaCommand

if (-not $Conda) {
    Write-Host "conda was not found. Recommended venv fallback:"
    Write-Host "  py -3.11 -m venv .venv"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python -m pip install --upgrade pip"
    Write-Host "  python -m pip install `"numpy<2`""
    Write-Host "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    Write-Host "  python -m pip install -r requirements.txt"
    exit 1
}

$PythonInPrefix = Join-Path $EnvPrefix "python.exe"
$EnvExists = Test-Path $PythonInPrefix
if (-not $EnvExists) {
    & $Conda create -p $EnvPrefix python=3.11 -y
} else {
    Write-Host "Environment already exists: $EnvPrefix"
}

& $Conda run -p $EnvPrefix python -s -m pip install --upgrade pip
& $Conda run -p $EnvPrefix python -s -m pip install "numpy<2"
& $Conda run -p $EnvPrefix python -s -m pip install "torch==2.2.1+cu121" "torchvision==0.17.1+cu121" "torchaudio==2.2.1+cu121" --index-url https://download.pytorch.org/whl/cu121
& $Conda run -p $EnvPrefix python -s -m pip install -r requirements.txt
& $Conda run -p $EnvPrefix python -s -m pip install "numpy<2"

Write-Host "Clean environment is ready: $EnvPrefix"
Write-Host "Activate with: conda activate $EnvPrefix"
Write-Host "Verify with: conda run -p $EnvPrefix python -s -m src.verify_env --dataset_root E:/ns_mc_gan_gi/data --output_dir E:/ns_mc_gan_gi/outputs_clean_phase2 --report_path E:/ns_mc_gan_gi/outputs_clean_phase2/env_report_clean.json"
