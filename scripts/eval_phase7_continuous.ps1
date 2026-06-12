$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  @("configs/phase7_continuous_g_only_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase7/continuous_g_only_5pct"),
  @("configs/phase7_continuous_physical_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct"),
  @("configs/phase7_continuous_pattern_only_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase7/continuous_pattern_only_5pct"),
  @("configs/phase7_continuous_long_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase7/continuous_long_5pct")
)

foreach ($Run in $Runs) {
  $Config = $Run[0]
  $OutDir = $Run[1]
  $Checkpoint = Join-Path $OutDir "best_score.pt"
  if (-not (Test-Path $Checkpoint)) {
    $Checkpoint = Join-Path $OutDir "best_ssim.pt"
  }
  if (Test-Path $Checkpoint) {
    Write-Host "Evaluating $Checkpoint"
    conda run -p $EnvPath python -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
  } else {
    Write-Warning "Missing checkpoint for $OutDir"
  }
}
