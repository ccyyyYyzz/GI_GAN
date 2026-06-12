$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  @("configs/phase4_matched_binary_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/matched_binary_5pct"),
  @("configs/phase4_matched_binary_slow_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/matched_binary_slow_5pct"),
  @("configs/phase4_matched_binary_no_freeze_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/matched_binary_no_freeze_5pct"),
  @("configs/phase4_continuous_contrast_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/continuous_contrast_5pct")
)
foreach ($Run in $Runs) {
  $Config = $Run[0]
  $Out = $Run[1]
  $Checkpoint = "$Out/best_score.pt"
  if (-not (Test-Path -LiteralPath $Checkpoint)) { $Checkpoint = "$Out/best_ssim.pt" }
  if (Test-Path -LiteralPath $Checkpoint) {
    conda run -p $EnvPath python -s -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
  } else {
    Write-Host "missing checkpoint for $Config"
  }
}
