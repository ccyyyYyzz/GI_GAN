$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$configs = @(
  "configs/phase6_fixed_seed123_5pct.yaml",
  "configs/phase6_fixed_seed2026_5pct.yaml",
  "configs/phase6_best_seed42_5pct.yaml",
  "configs/phase6_best_seed123_5pct.yaml",
  "configs/phase6_best_seed2026_5pct.yaml"
)

foreach ($config in $configs) {
  if (Test-Path $config) {
    conda run -p $PY python -s -m src.train --config $config --device cuda
  } else {
    Write-Host "missing config $config"
  }
}
