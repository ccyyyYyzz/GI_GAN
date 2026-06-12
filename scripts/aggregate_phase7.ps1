$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

conda run -p $EnvPath python -m src.aggregate_phase7
conda run -p $EnvPath python -m src.make_phase7_report
conda run -p $EnvPath python -m src.export_phase7_paper_assets
