$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.aggregate_phase10
conda run -p $EnvPath python -m src.export_phase10_examples
conda run -p $EnvPath python -m src.make_phase10_report
