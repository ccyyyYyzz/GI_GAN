$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.overfit_hq --config configs/phase9/overfit_hadamard_10pct.yaml --device cuda
