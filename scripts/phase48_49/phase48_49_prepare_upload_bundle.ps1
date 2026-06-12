param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")),
  [string]$NoLeakRoot = "E:\ns_mc_gan_gi\outputs_phase15\imported_noleak",
  [string]$UploadDir = "E:\ns_mc_gan_gi\colab_upload",
  [switch]$SkipProjectZip,
  [switch]$SkipNoLeakBundle
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

function Split-LargeFile {
  param([string]$Path, [long]$ChunkBytes = [long](1.8 * 1024 * 1024 * 1024))
  $file = Get-Item -LiteralPath $Path
  if ($file.Length -le $ChunkBytes) { return }
  Get-ChildItem -LiteralPath (Split-Path -Parent $Path) -File -Filter "$($file.Name).part_*" -ErrorAction SilentlyContinue | Remove-Item -Force
  $inStream = [System.IO.File]::OpenRead($Path)
  try {
    $buffer = New-Object byte[] (1024 * 1024)
    $index = 0
    while ($inStream.Position -lt $inStream.Length) {
      $partPath = "{0}.part_{1:D3}" -f $Path, $index
      $outStream = [System.IO.File]::Open($partPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
      try {
        $written = [long]0
        while ($written -lt $ChunkBytes) {
          $remaining = [Math]::Min($buffer.Length, $ChunkBytes - $written)
          $read = $inStream.Read($buffer, 0, [int]$remaining)
          if ($read -le 0) { break }
          $outStream.Write($buffer, 0, $read)
          $written += $read
        }
      } finally {
        $outStream.Dispose()
      }
      $index += 1
    }
  } finally {
    $inStream.Dispose()
  }
  Write-Host "Split large input bundle into parts: $Path.part_*"
}

if (-not $SkipProjectZip) {
  $tmpProject = Join-Path $env:TEMP "ns_mc_gan_gi_project_phase48_49"
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
  Get-ChildItem -LiteralPath $tmpProject -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
  Get-ChildItem -LiteralPath $tmpProject -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
  $projectZip = Join-Path $UploadDir "ns_mc_gan_gi_project_phase48_49.zip"
  New-ZipFromDirectory -SourceDir $tmpProject -ZipPath $projectZip
  Write-Host "Project zip: $projectZip"
}

if (-not $SkipNoLeakBundle) {
  $tasks = @(
    "rademacher5_hq_noise001_colab",
    "scrambled_hadamard5_hq_noise001_colab",
    "rademacher10_full_noise001_colab",
    "scrambled_hadamard10_full_noise001_colab"
  )
  $tmpBundle = Join-Path $env:TEMP "noleak_bundle_phase48_49"
  if (Test-Path -LiteralPath $tmpBundle) { Remove-Item -LiteralPath $tmpBundle -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $tmpBundle | Out-Null
  foreach ($task in $tasks) {
    $src = Join-Path $NoLeakRoot $task
    if (-not (Test-Path -LiteralPath $src)) {
      throw "Missing no-leak task directory: $src"
    }
    $dst = Join-Path $tmpBundle $task
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($name in @("resolved_config.yaml", "last.pt", "eval_metrics.json", "per_epoch_metrics.csv", "measurement_operator_exact.pt", "measurement_operator_exact_manifest.json")) {
      $file = Join-Path $src $name
      if (Test-Path -LiteralPath $file) {
        Copy-Item -LiteralPath $file -Destination (Join-Path $dst $name) -Force
      }
    }
  }
  $bundleZip = Join-Path $UploadDir "noleak_bundle_phase48_49.zip"
  New-ZipFromDirectory -SourceDir $tmpBundle -ZipPath $bundleZip
  Split-LargeFile -Path $bundleZip
  Write-Host "No-leak bundle zip: $bundleZip"
}

Write-Host "Upload these files to each Colab session or place them in Drive: $UploadDir"
