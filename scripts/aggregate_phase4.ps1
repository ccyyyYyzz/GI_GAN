$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -s -m src.aggregate_phase4
conda run -p $EnvPath python -s -m src.make_phase4_report
