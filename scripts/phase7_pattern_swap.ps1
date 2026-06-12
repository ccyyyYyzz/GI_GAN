$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$FixedCheckpoint = "E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt"
$FixedConfig = "configs/clean_quick_5pct.yaml"

$FlipCheckpoint = "E:/ns_mc_gan_gi/outputs_phase7/flipaware_alpha1_5pct/best_score.pt"
if (Test-Path $FlipCheckpoint) {
  conda run -p $EnvPath python -m src.eval_pattern_swap --fixed_checkpoint $FixedCheckpoint --fixed_config $FixedConfig --learned_checkpoint $FlipCheckpoint --learned_config configs/phase7_flipaware_alpha1_5pct.yaml --output_dir E:/ns_mc_gan_gi/outputs_phase7/pattern_swap/flipaware_alpha1_5pct --device cuda
} else {
  Write-Warning "Missing flip-aware checkpoint for pattern swap"
}

$ContinuousCheckpoint = "E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct/best_score.pt"
if (Test-Path $ContinuousCheckpoint) {
  conda run -p $EnvPath python -m src.eval_pattern_swap --fixed_checkpoint $FixedCheckpoint --fixed_config $FixedConfig --learned_checkpoint $ContinuousCheckpoint --learned_config configs/phase7_continuous_physical_5pct.yaml --output_dir E:/ns_mc_gan_gi/outputs_phase7/pattern_swap/continuous_physical_5pct --device cuda
} else {
  Write-Warning "Missing continuous checkpoint for pattern swap"
}
