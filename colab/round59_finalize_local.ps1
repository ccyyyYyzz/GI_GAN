param(
    [string]$DownloadRoot = "E:\GAN_FCC_WORK\experiments\gan_gi_journal_round59",
    [string]$RepoRoot = "E:\GAN_FCC_WORK\active_code\completion_gan_round18",
    [string]$Python = "D:\Anacondar\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$extractRoot = Join-Path $DownloadRoot "extracted"
$decisionRoot = Join-Path $DownloadRoot "aggregate_raw_y"
New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null

$laneRoots = @()
foreach ($lane in 0, 1, 2) {
    $archive = Join-Path $DownloadRoot "round59_raw_fiber_lane$lane.zip"
    $sidecar = "$archive.sha256"
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "ROUND59_ARCHIVE_MISSING:lane$lane`:$archive"
    }
    if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
        throw "ROUND59_ARCHIVE_SIDECAR_MISSING:lane$lane`:$sidecar"
    }
    $expected = ((Get-Content -LiteralPath $sidecar -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "ROUND59_ARCHIVE_SHA256_MISMATCH:lane$lane`:$actual!=$expected"
    }
    $laneRoot = Join-Path $extractRoot "lane$lane"
    if (Test-Path -LiteralPath $laneRoot) {
        throw "ROUND59_EXTRACTED_LANE_ALREADY_EXISTS:lane$lane`:$laneRoot"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot
    if (-not (Test-Path -LiteralPath (Join-Path $laneRoot "ROUND59_COMPLETE.json") -PathType Leaf)) {
        throw "ROUND59_EXTRACTED_COMPLETE_RECEIPT_MISSING:lane$lane"
    }
    $laneRoots += $laneRoot
    Write-Host "ROUND59_ARCHIVE_EXTRACTED lane$lane $actual"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "ROUND59_PYTHON_MISSING:$Python"
}
& $Python (Join-Path $RepoRoot "aggregate_round59_raw_y.py") `
    --input-dirs $laneRoots[0] $laneRoots[1] $laneRoots[2] `
    --output-dir $decisionRoot
if ($LASTEXITCODE -ne 0) {
    throw "ROUND59_AGGREGATION_FAILED:$LASTEXITCODE"
}

$decisionJson = Join-Path $decisionRoot "round59_raw_y_decision.json"
$decisionMarkdown = Join-Path $decisionRoot "ROUND59_RAW_Y_DECISION.md"
if (-not (Test-Path -LiteralPath $decisionJson -PathType Leaf) -or
    -not (Test-Path -LiteralPath $decisionMarkdown -PathType Leaf)) {
    throw "ROUND59_DECISION_ARTIFACTS_MISSING"
}
Write-Host "ROUND59_LOCAL_FINALIZATION_COMPLETE $decisionRoot"
