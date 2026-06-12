$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Configs = @(
  "configs/phase7_continuous_g_only_5pct.yaml",
  "configs/phase7_continuous_physical_5pct.yaml",
  "configs/phase7_continuous_pattern_only_5pct.yaml"
)

foreach ($Config in $Configs) {
  Write-Host "Training $Config"
  conda run -p $EnvPath python -m src.train --config $Config --device cuda
}
