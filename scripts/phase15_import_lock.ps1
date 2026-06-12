param(
  [string]$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

conda run -p $EnvPath python -m src.phase15_import_noleak_results
conda run -p $EnvPath python -m src.phase15_build_noleak_registry
