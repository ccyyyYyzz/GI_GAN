param(
  [string]$DownloadDir = "E:\ns_mc_gan_gi\colab_downloads\phase51A",
  [string]$ImportRoot = "E:\ns_mc_gan_gi\outputs_phase51A_colab_import",
  [string]$Phase48Root = "E:\ns_mc_gan_gi\outputs_phase48_49_colab_import",
  [switch]$RemoveZipAfterImport
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ImportRoot | Out-Null

$zips = Get-ChildItem -LiteralPath $DownloadDir -File -Filter "session_0*_outputs.zip" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "^session_0[6-9]_" } |
  Sort-Object Name
if ($zips.Count -eq 0) {
  throw "No Phase 51A session output zips found in $DownloadDir. Run phase51A_merge_colab_parts.ps1 first if you downloaded split parts."
}

foreach ($zip in $zips) {
  Write-Host "Importing $($zip.Name)"
  Expand-Archive -LiteralPath $zip.FullName -DestinationPath $ImportRoot -Force
  if ($RemoveZipAfterImport) {
    Remove-Item -LiteralPath $zip.FullName -Force
  }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path -LiteralPath $bundledPython) {
  $python = $bundledPython
} else {
  $python = "python"
}

Write-Host "Refreshing Phase 48/49 reports with $python"
Push-Location $repoRoot
try {
  & $python -m src.phase51A_standardize_phase48_49 --import_root $Phase48Root
  if ($LASTEXITCODE -ne 0) { throw "Phase 48/49 standardization failed with exit code $LASTEXITCODE" }

  Write-Host "Aggregating Phase 51A reports with $python"
  & $python -m src.phase51A_aggregate_reports --import_root $ImportRoot --phase48_root $Phase48Root
  if ($LASTEXITCODE -ne 0) { throw "Phase 51A aggregation failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}

Write-Host "Imported Phase 51A outputs to $ImportRoot"
