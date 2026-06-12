$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Config = "configs/phase4_debug_matched_binary_5pct.yaml"
$Out = "E:/ns_mc_gan_gi/outputs_phase4/debug_matched_binary_5pct"

conda run -p $EnvPath python -s -m src.sanity_learnable_patterns --config $Config --device cuda
conda run -p $EnvPath python -s -m src.train --config $Config --device cuda
$Checkpoint = "$Out/best_score.pt"
if (-not (Test-Path -LiteralPath $Checkpoint)) { $Checkpoint = "$Out/best_ssim.pt" }
conda run -p $EnvPath python -s -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
