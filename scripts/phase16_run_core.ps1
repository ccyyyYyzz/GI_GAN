$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = "E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$env:PYTHONPATH = $Repo

$Modules = @(
    "src.phase16_verify_safe_exactA",
    "src.phase16_exactA_reeval_audit",
    "src.phase16_attribution_final",
    "src.phase16_real_inference_ablation",
    "src.phase16_noise_sweep",
    "src.phase16_traditional_baselines",
    "src.phase16_dc_row_control_final",
    "src.phase16_statistics_ci"
)

foreach ($Module in $Modules) {
    Write-Host "==> $Module"
    & $Python -m $Module
}
