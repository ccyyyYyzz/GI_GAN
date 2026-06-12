$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = "E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$env:PYTHONPATH = $Repo

$Modules = @(
    "src.phase16_stl10_classwise",
    "src.phase16_measurement_perturbation",
    "src.phase16_runtime_complexity",
    "src.aggregate_phase16_supplementary",
    "src.make_phase16_supplementary_report",
    "src.phase16_update_writing_claims"
)

foreach ($Module in $Modules) {
    Write-Host "==> $Module"
    & $Python -m $Module
}
