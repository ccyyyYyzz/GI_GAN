$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.import_phase14_colab_results
conda run -p $EnvPath python -m src.aggregate_phase14
conda run -p $EnvPath python -m src.make_phase14_report
conda run -p $EnvPath python -m src.update_phase13_after_phase14
Write-Host "Phase 14 Colab results imported and reported."
