$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$runs = @(
  @{ Config = "configs/phase6_g_only_finetune_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/g_only_finetune_5pct" },
  @{ Config = "configs/phase6_pattern_trainable_alpha6_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha6_5pct" },
  @{ Config = "configs/phase6_pattern_trainable_alpha2_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha2_5pct" },
  @{ Config = "configs/phase6_pattern_trainable_alpha1_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha1_5pct" },
  @{ Config = "configs/phase6_pattern_trainable_alpha0p5_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha0p5_5pct" },
  @{ Config = "configs/phase6_pattern_only_alpha1_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/pattern_only_alpha1_5pct" },
  @{ Config = "configs/phase6_soft_signed_train_5pct.yaml"; Dir = "E:/ns_mc_gan_gi/outputs_phase6/soft_signed_train_5pct" }
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
