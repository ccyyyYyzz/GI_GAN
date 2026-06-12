$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Root = if ($env:NS_MC_GAN_GI_ROOT) { $env:NS_MC_GAN_GI_ROOT } else { "E:/ns_mc_gan_gi" }
conda run -p $EnvPath python -m src.aggregate_phase26_arch_pilot --drive_root $Root
conda run -p $EnvPath python -m src.phase26_gate_decision --drive_root $Root
conda run -p $EnvPath python -m src.make_phase26_limit_arch_report --drive_root $Root
