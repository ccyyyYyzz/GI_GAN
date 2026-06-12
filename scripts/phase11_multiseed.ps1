$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.prepare_phase11_multiseed
conda run -p $EnvPath python -m src.run_phase11_multiseed
conda run -p $EnvPath python -m src.aggregate_phase11
conda run -p $EnvPath python -m src.make_phase11_report
