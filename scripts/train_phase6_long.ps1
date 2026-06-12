$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$config = "configs/phase6_best_long_5pct.yaml"
if (Test-Path $config) {
  conda run -p $PY python -s -m src.train --config $config --device cuda
} else {
  Write-Host "missing config $config"
}
