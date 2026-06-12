$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

& "$PSScriptRoot\phase16_run_core.ps1"
& "$PSScriptRoot\phase16_run_optional.ps1"
