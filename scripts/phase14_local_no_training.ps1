$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.phase14_unified_eval
conda run -p $EnvPath python -m src.phase14_traditional_baselines
conda run -p $EnvPath python -m src.phase14_checkpoint_ablation
conda run -p $EnvPath python -m src.phase14_noise_sweep
conda run -p $EnvPath python -m src.aggregate_phase14
conda run -p $EnvPath python -m src.make_phase14_report
Write-Host "Phase 14 local no-training reports finished."
