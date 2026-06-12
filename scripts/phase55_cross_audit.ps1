$ErrorActionPreference = "Stop"

$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Out = "E:/ns_mc_gan_gi/outputs_phase55_cross_audit"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "==== Phase55: $Name ===="
    & conda run -p $EnvPath python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Phase55 step failed: $Name"
    }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

Run-Step "extract Phase53C" @("-m", "src.phase55_extract_phase53C")
Run-Step "extract Phase53D" @("-m", "src.phase55_extract_phase53D")
Run-Step "cross audit" @("-m", "src.phase55_cross_audit")
Run-Step "paper claims and next action" @("-m", "src.phase55_make_claims")
Run-Step "select figures" @("-m", "src.phase55_select_figures")
Run-Step "final report" @("-m", "src.phase55_make_report")

Write-Host ""
Write-Host "Phase55 cross-audit complete:"
Write-Host $Out
