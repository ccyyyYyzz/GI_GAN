$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.aggregate_phase9
conda run -p $EnvPath python -m src.make_phase9_report
