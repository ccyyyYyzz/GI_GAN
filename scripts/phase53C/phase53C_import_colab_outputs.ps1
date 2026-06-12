param(
  [string]$DownloadDir = "E:\ns_mc_gan_gi\colab_downloads\phase53C",
  [string]$ImportRoot = "E:\ns_mc_gan_gi\outputs_phase53C_exact_null_critic_import",
  [switch]$RemoveZipAfterImport
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ImportRoot | Out-Null

$zips = Get-ChildItem -LiteralPath $DownloadDir -File -Filter "session_2*_outputs.zip" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "^session_2[0-4]_" } |
  Sort-Object Name
if ($zips.Count -eq 0) {
  throw "No Phase53C session output zips found in $DownloadDir. Run phase53C_merge_colab_parts.ps1 first if you downloaded split parts."
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

Write-Host "Aggregating Phase53C reports with $python"
Push-Location $repoRoot
try {
  & $python -m src.phase53C_aggregate --import_root $ImportRoot
  if ($LASTEXITCODE -ne 0) { throw "Phase53C aggregation failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}

Write-Host "Imported Phase53C outputs to $ImportRoot"
