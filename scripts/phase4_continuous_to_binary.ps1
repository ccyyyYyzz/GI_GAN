$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$ContinuousOut = "E:/ns_mc_gan_gi/outputs_phase4/continuous_contrast_5pct"
$Source = "$ContinuousOut/best_score.pt"
if (-not (Test-Path -LiteralPath $Source)) { $Source = "$ContinuousOut/best_ssim.pt" }
$Converted = "E:/ns_mc_gan_gi/outputs_phase4/continuous_to_binary_5pct/init_from_continuous.pt"
conda run -p $EnvPath python -s -m src.convert_continuous_to_binary_checkpoint --continuous_checkpoint $Source --output_checkpoint $Converted --target_mode learned_balanced_binary_ste --target_transmission 0.5 --logit_abs_scale 2.0
conda run -p $EnvPath python -s -m src.train --config configs/phase4_continuous_to_binary_5pct.yaml --device cuda
$Out = "E:/ns_mc_gan_gi/outputs_phase4/continuous_to_binary_5pct"
$Checkpoint = "$Out/best_score.pt"
if (-not (Test-Path -LiteralPath $Checkpoint)) { $Checkpoint = "$Out/best_ssim.pt" }
conda run -p $EnvPath python -s -m src.eval --checkpoint $Checkpoint --config configs/phase4_continuous_to_binary_5pct.yaml --device cuda
