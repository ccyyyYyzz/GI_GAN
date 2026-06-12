param(
  [string]$SourceDir = "$env:USERPROFILE\Downloads",
  [string]$TargetDir = "E:\ns_mc_gan_gi\colab_downloads\phase53C",
  [switch]$RemoveSource
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host "Collecting Phase53C downloads from $SourceDir"

$zipFiles = Get-ChildItem -LiteralPath $SourceDir -File -Filter "session_*_outputs.zip" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "^session_2[0-4]_" }
foreach ($zip in $zipFiles) {
  $dest = Join-Path $TargetDir $zip.Name
  Copy-Item -LiteralPath $zip.FullName -Destination $dest -Force
  Write-Host "Copied zip: $($zip.Name)"
  if ($RemoveSource) { Remove-Item -LiteralPath $zip.FullName -Force }
}

$downloadManifests = Get-ChildItem -LiteralPath $SourceDir -File -Filter "session_2*_download_manifest.json" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "^session_2[0-4]_" }
foreach ($manifest in $downloadManifests) {
  Copy-Item -LiteralPath $manifest.FullName -Destination (Join-Path $TargetDir $manifest.Name) -Force
  if ($RemoveSource) { Remove-Item -LiteralPath $manifest.FullName -Force }
}

$splitManifests = Get-ChildItem -LiteralPath $SourceDir -File -Filter "session_2*_split_manifest.json" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "^session_2[0-4]_" }
foreach ($manifest in $splitManifests) {
  $manifestDest = Join-Path $TargetDir $manifest.Name
  Copy-Item -LiteralPath $manifest.FullName -Destination $manifestDest -Force
  $json = Get-Content -LiteralPath $manifest.FullName -Raw | ConvertFrom-Json
  $zipName = Split-Path -Leaf $json.zip
  if (-not $zipName) {
    $zipName = $manifest.Name -replace "_split_manifest\.json$", "_outputs.zip"
  }
  $outZip = Join-Path $TargetDir $zipName
  if (Test-Path -LiteralPath $outZip) {
    Write-Host "Merged zip already exists: $zipName"
    continue
  }
  $partNames = @($json.parts)
  if ($partNames.Count -eq 0) {
    $pattern = "$zipName.part_*"
    $partNames = @(Get-ChildItem -LiteralPath $SourceDir -File -Filter $pattern | Sort-Object Name | Select-Object -ExpandProperty Name)
  }
  if ($partNames.Count -eq 0) {
    Write-Warning "No parts found for $zipName"
    continue
  }
  Write-Host "Merging $($partNames.Count) parts -> $outZip"
  $outStream = [System.IO.File]::Open($outZip, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
  try {
    foreach ($partName in $partNames) {
      $partPath = Join-Path $SourceDir $partName
      if (-not (Test-Path -LiteralPath $partPath)) {
        throw "Missing part: $partPath"
      }
      Copy-Item -LiteralPath $partPath -Destination (Join-Path $TargetDir $partName) -Force
      $inStream = [System.IO.File]::OpenRead($partPath)
      try { $inStream.CopyTo($outStream) } finally { $inStream.Dispose() }
      if ($RemoveSource) { Remove-Item -LiteralPath $partPath -Force }
    }
  } finally {
    $outStream.Dispose()
  }
  if ($json.sha256) {
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $outZip).Hash.ToLowerInvariant()
    if ($sha -ne [string]$json.sha256) {
      throw "SHA256 mismatch for $zipName. expected=$($json.sha256) actual=$sha"
    }
    Write-Host "SHA256 OK: $zipName"
  }
  if ($RemoveSource) { Remove-Item -LiteralPath $manifest.FullName -Force }
}

Write-Host "Done. Phase53C downloads are in $TargetDir"
