$ErrorActionPreference = "Stop"
powershell -ExecutionPolicy Bypass -File scripts/phase14_colab_prepare.ps1
powershell -ExecutionPolicy Bypass -File scripts/phase14_local_no_training.ps1
Write-Host "Phase 14 local preparation and no-training reports finished. Colab training still needs to be run externally."
