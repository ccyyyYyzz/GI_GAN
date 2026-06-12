$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$configs = @(
  "configs/phase5_best_1pct.yaml",
  "configs/phase5_best_2pct.yaml",
  "configs/phase5_best_5pct.yaml",
  "configs/phase5_best_10pct.yaml"
)

foreach ($config in $configs) {
  if (Test-Path $config) {
    conda run -p $PY python -s -m src.train --config $config --device cuda
  } else {
    Write-Host "missing config $config"
  }
}
