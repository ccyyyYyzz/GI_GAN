$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  @("configs/phase4_best_2pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/best_2pct"),
  @("configs/phase4_best_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/best_5pct"),
  @("configs/phase4_best_10pct.yaml", "E:/ns_mc_gan_gi/outputs_phase4/best_10pct")
)
foreach ($Run in $Runs) {
  $Config = $Run[0]
  $Out = $Run[1]
  if (-not (Test-Path -LiteralPath $Config)) {
    Write-Host "missing $Config"
    continue
  }
  $Checkpoint = "$Out/best_score.pt"
  if (-not (Test-Path -LiteralPath $Checkpoint)) { $Checkpoint = "$Out/best_ssim.pt" }
  if (Test-Path -LiteralPath $Checkpoint) {
    conda run -p $EnvPath python -s -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
  } else {
    Write-Host "missing checkpoint for $Config"
  }
}
