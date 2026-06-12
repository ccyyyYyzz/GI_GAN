$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.phase11_adaptive_continue
conda run -p $EnvPath python -m src.phase11_run_adaptive_continue
conda run -p $EnvPath python -m src.aggregate_phase11
conda run -p $EnvPath python -m src.make_phase11_report
