param(
  [string]$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

conda run -p $EnvPath python -m compileall src
& "$PSScriptRoot/phase15_import_lock.ps1" -EnvPath $EnvPath
& "$PSScriptRoot/phase15_reeval_audit.ps1" -EnvPath $EnvPath
& "$PSScriptRoot/phase15_tables_figures.ps1" -EnvPath $EnvPath
& "$PSScriptRoot/phase15_update_manuscript.ps1" -EnvPath $EnvPath
