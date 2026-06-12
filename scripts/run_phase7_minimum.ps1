$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

function Invoke-TrainEval($Config, $OutDir) {
  Write-Host "Training $Config"
  conda run -p $EnvPath python -m src.train --config $Config --device cuda
  $Checkpoint = Join-Path $OutDir "best_score.pt"
  if (-not (Test-Path $Checkpoint)) {
    $Checkpoint = Join-Path $OutDir "best_ssim.pt"
  }
  if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint after training: $OutDir"
  }
  Write-Host "Evaluating $Checkpoint"
  conda run -p $EnvPath python -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
}

Invoke-TrainEval "configs/phase7_flipaware_alpha1_5pct.yaml" "E:/ns_mc_gan_gi/outputs_phase7/flipaware_alpha1_5pct"
Invoke-TrainEval "configs/phase7_flipaware_aggressive_5pct.yaml" "E:/ns_mc_gan_gi/outputs_phase7/flipaware_aggressive_5pct"
Invoke-TrainEval "configs/phase7_continuous_g_only_5pct.yaml" "E:/ns_mc_gan_gi/outputs_phase7/continuous_g_only_5pct"
Invoke-TrainEval "configs/phase7_continuous_physical_5pct.yaml" "E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct"

Write-Host "Running pattern swap"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase7_pattern_swap.ps1

Write-Host "Running measurement quality"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase7_measurement_quality.ps1

Write-Host "Aggregating Phase 7"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/aggregate_phase7.ps1
