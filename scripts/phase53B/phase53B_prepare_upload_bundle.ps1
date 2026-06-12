param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")),
  [string]$UploadDir = "E:\ns_mc_gan_gi\colab_upload",
  [string]$ExistingNoLeakBundle = "E:\ns_mc_gan_gi\colab_upload\noleak_bundle_phase48_49.zip",
  [switch]$SkipProjectZip,
  [switch]$CreatePhase53BNoLeakAlias
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $UploadDir | Out-Null

function New-ZipFromDirectory {
  param([string]$SourceDir, [string]$ZipPath)
  if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
  }
  [System.IO.Compression.ZipFile]::CreateFromDirectory($SourceDir, $ZipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
}

if (-not $SkipProjectZip) {
  $tmpProject = Join-Path $env:TEMP "ns_mc_gan_gi_project_phase53B"
  if (Test-Path -LiteralPath $tmpProject) { Remove-Item -LiteralPath $tmpProject -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $tmpProject | Out-Null

  $excludeDirs = @(".git", "__pycache__", ".pytest_cache", "outputs", "data", "runs", "tb")
  Get-ChildItem -LiteralPath $RepoRoot -Force | Where-Object {
    $excludeDirs -notcontains $_.Name
  } | ForEach-Object {
    $dest = Join-Path $tmpProject $_.Name
    if ($_.PSIsContainer) {
      Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    } else {
      Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
  }
  Get-ChildItem -LiteralPath $tmpProject -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
  Get-ChildItem -LiteralPath $tmpProject -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

  $projectZip = Join-Path $UploadDir "ns_mc_gan_gi_project_phase53B.zip"
  New-ZipFromDirectory -SourceDir $tmpProject -ZipPath $projectZip
  Write-Host "Project zip: $projectZip"
}

if ($CreatePhase53BNoLeakAlias) {
  if (-not (Test-Path -LiteralPath $ExistingNoLeakBundle)) {
    throw "Existing no-leak bundle not found: $ExistingNoLeakBundle"
  }
  $aliasPath = Join-Path $UploadDir "noleak_bundle_phase53B.zip"
  Copy-Item -LiteralPath $ExistingNoLeakBundle -Destination $aliasPath -Force
  Write-Host "No-leak alias copied: $aliasPath"
} else {
  Write-Host "No-leak bundle: reuse existing noleak_bundle_phase48_49.zip in $UploadDir"
}

Write-Host "Colab upload directory: $UploadDir"
