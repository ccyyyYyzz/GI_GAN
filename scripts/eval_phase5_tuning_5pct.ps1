$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$runs = @(
  @{ Config = "configs/phase5_exact_binary_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/exact_binary_5pct" },
  @{ Config = "configs/phase5_exact_binary_slow_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/exact_binary_slow_5pct" },
  @{ Config = "configs/phase5_exact_binary_freezeG_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/exact_binary_freezeG_5pct" },
  @{ Config = "configs/phase5_centered_vs_exact_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase5/centered_vs_exact_5pct" }
)

foreach ($run in $runs) {
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
