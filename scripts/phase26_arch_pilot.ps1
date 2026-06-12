$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Root = if ($env:NS_MC_GAN_GI_ROOT) { $env:NS_MC_GAN_GI_ROOT } else { "E:/ns_mc_gan_gi" }
$Configs = if ($env:CONFIGS) { $env:CONFIGS } else { "current_hq_rad5_pilot,nafnet_small_rad5_pilot,unrolled_ista_rad5_pilot,current_hq_scr5_pilot,nafnet_small_scr5_pilot,unrolled_ista_scr5_pilot" }
conda run -p $EnvPath python -m src.phase26_prepare_arch_pilot --drive_root $Root --device cuda
conda run -p $EnvPath python -m src.run_phase26_arch_pilot --drive_root $Root --device cuda --configs $Configs --skip_existing
