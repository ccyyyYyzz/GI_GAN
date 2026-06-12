$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.aggregate_phase10
conda run -p $EnvPath python -m src.export_phase10_examples
conda run -p $EnvPath python -m src.make_phase10_report
conda run -p $EnvPath python -m src.phase11_attribution
conda run -p $EnvPath python -m src.aggregate_phase11
conda run -p $EnvPath python -m src.make_phase11_report
