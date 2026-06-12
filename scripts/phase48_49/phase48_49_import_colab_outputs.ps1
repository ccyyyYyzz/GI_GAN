param(
  [string]$DownloadDir = "E:\ns_mc_gan_gi\colab_downloads\phase48_49",
  [string]$ImportRoot = "E:\ns_mc_gan_gi\outputs_phase48_49_colab_import",
  [switch]$RemoveZipAfterImport
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ImportRoot | Out-Null

$zips = Get-ChildItem -LiteralPath $DownloadDir -File -Filter "session_*_outputs.zip" -ErrorAction SilentlyContinue | Sort-Object Name
if ($zips.Count -eq 0) {
  throw "No session output zips found in $DownloadDir. Run phase48_49_merge_colab_parts.ps1 first if you downloaded split parts."
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

Write-Host "Aggregating reports with $python"
Push-Location $repoRoot
try {
  & $python -m src.phase48_49_aggregate_reports --import_root $ImportRoot
} finally {
  Pop-Location
}
if ($LASTEXITCODE -ne 0) {
  throw "Aggregation failed with exit code $LASTEXITCODE"
}

Write-Host "Imported Phase 48/49 outputs to $ImportRoot"
