param([switch]$RunLocalTraining)
$ErrorActionPreference = "Stop"

if (-not $RunLocalTraining) {
    Write-Host "Local Phase 14 5% training is disabled by default."
    Write-Host "Use colab/phase14_colab_5pct_hq.ipynb instead."
    Write-Host "This script exits without starting training."
    exit 0
}

throw "Local large training is intentionally blocked for Phase 14. Run the Colab notebook first."
