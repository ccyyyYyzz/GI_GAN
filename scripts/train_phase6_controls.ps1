$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$configs = @(
  "configs/phase6_g_only_finetune_5pct.yaml",
  "configs/phase6_pattern_trainable_alpha6_5pct.yaml",
  "configs/phase6_pattern_trainable_alpha2_5pct.yaml",
  "configs/phase6_pattern_trainable_alpha1_5pct.yaml",
  "configs/phase6_pattern_trainable_alpha0p5_5pct.yaml",
  "configs/phase6_pattern_only_alpha1_5pct.yaml",
  "configs/phase6_soft_signed_train_5pct.yaml"
)

foreach ($config in $configs) {
  if (Test-Path $config) {
    conda run -p $PY python -s -m src.train --config $config --device cuda
  } else {
    Write-Host "missing config $config"
  }
}
