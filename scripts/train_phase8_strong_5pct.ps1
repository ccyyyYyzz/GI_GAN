$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$configs = @(
  "configs/phase8_fixed_wide_5pct.yaml",
  "configs/phase8_fixed_wide_refiner_5pct.yaml",
  "configs/phase8_continuous_g_only_wide_5pct.yaml",
  "configs/phase8_continuous_physical_wide_5pct.yaml",
  "configs/phase8_direct_y_fixed_5pct.yaml"
)
foreach ($cfg in $configs) {
  conda run -p $envPath python -m src.train --config $cfg --device cuda
}
