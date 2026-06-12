param(
  [string]$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

conda run -p $EnvPath python -m src.phase15_exactA_reeval
conda run -p $EnvPath python -m src.phase15_noleak_audit
