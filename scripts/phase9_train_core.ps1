$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Configs = @(
  "configs/phase9/hadamard10_probe_nonoise.yaml",
  "configs/phase9/hadamard10_probe_noise001.yaml",
  "configs/phase9/rademacher10_probe_noise001.yaml",
  "configs/phase9/scrambled_hadamard10_probe_noise001.yaml",
  "configs/phase9/hadamard5_probe_noise001.yaml"
)
foreach ($Config in $Configs) {
  conda run -p $EnvPath python -m src.train --config $Config --device cuda
}
