$ErrorActionPreference = "Stop"
$repo = "E:\ns_mc_gan_gi_code"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
Set-Location $repo
conda run -p $envPath python -m src.phase71_gauge_cgan_paired_seeds
