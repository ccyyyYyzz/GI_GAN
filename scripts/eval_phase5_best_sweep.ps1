$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$runs = @(
  @{ Config = "configs/phase5_best_1pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/best_1pct" },
  @{ Config = "configs/phase5_best_2pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/best_2pct" },
  @{ Config = "configs/phase5_best_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/best_5pct" },
  @{ Config = "configs/phase5_best_10pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/best_10pct" }
)

foreach ($run in $runs) {
  if (-not (Test-Path $run.Config)) {
    Write-Host "missing config $($run.Config)"
    continue
  }
  $checkpoint = Join-Path $run.Dir "best_score.pt"
  if (-not (Test-Path $checkpoint)) {
    $checkpoint = Join-Path $run.Dir "best_ssim.pt"
  }
  if (Test-Path $checkpoint) {
    conda run -p $PY python -s -m src.eval --checkpoint $checkpoint --config $run.Config --device cuda
  } else {
    Write-Host "missing checkpoint for $($run.Dir)"
  }
}
