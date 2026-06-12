$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

$configs = @(
  "configs/phase5_exact_binary_5pct.yaml",
  "configs/phase5_exact_binary_slow_5pct.yaml",
  "configs/phase5_exact_binary_freezeG_5pct.yaml",
  "configs/phase5_centered_vs_exact_5pct.yaml"
)

foreach ($config in $configs) {
  conda run -p $PY python -s -m src.train --config $config --device cuda
}
