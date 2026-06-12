param(
    [string]$Python = "conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

Write-Host "[Phase60] G1 provenance audit"
Invoke-Expression "$Python -m src.phase60_g1_provenance_audit"

Write-Host "[Phase60] Prepare controlled G2 config and safety gate"
Invoke-Expression "$Python -m src.phase60_prepare_g2_config"

Write-Host "[Phase60] G2 null-gauge GAN training gate"
Invoke-Expression "$Python -m src.phase60_train_g2_null_gan"

Write-Host "[Phase60] Sampling-mode evaluation"
Invoke-Expression "$Python -m src.phase60_eval_sampling_mode"

Write-Host "[Phase60] Report"
Invoke-Expression "$Python -m src.phase60_make_report"

Write-Host "[Phase60] Done. Output: E:/ns_mc_gan_gi/outputs_phase60_gan_sampling_mode_g2"
