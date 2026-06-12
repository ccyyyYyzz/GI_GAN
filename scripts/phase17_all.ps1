$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Python = "E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$env:PYTHONPATH = $Repo

& $Python -m compileall src

$Modules = @(
    "src.phase17_build_evidence_index",
    "src.phase17_write_manuscript",
    "src.phase17_write_chinese_report",
    "src.phase17_write_supplement",
    "src.phase17_make_figure_table_pack",
    "src.phase17_submission_pack",
    "src.phase17_defense_pack",
    "src.phase17_reviewer_risk_register",
    "src.phase17_final_checklist",
    "src.make_phase17_manifest"
)

foreach ($Module in $Modules) {
    Write-Host "==> $Module"
    & $Python -m $Module
}
