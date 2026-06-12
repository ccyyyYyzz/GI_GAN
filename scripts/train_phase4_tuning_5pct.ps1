$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Configs = @(
  "configs/phase4_matched_binary_5pct.yaml",
  "configs/phase4_matched_binary_slow_5pct.yaml",
  "configs/phase4_matched_binary_no_freeze_5pct.yaml",
  "configs/phase4_continuous_contrast_5pct.yaml"
)
foreach ($Config in $Configs) {
  conda run -p $EnvPath python -s -m src.train --config $Config --device cuda
}
