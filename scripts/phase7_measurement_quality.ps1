$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

conda run -p $EnvPath python -m src.measurement_quality --config configs/clean_quick_5pct.yaml --checkpoint E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt --output_dir E:/ns_mc_gan_gi/outputs_phase7/measurement_quality/fixed_5pct --num_batches 10 --device cuda
conda run -p $EnvPath python -m src.measurement_quality --config configs/phase5_exact_binary_slow_5pct.yaml --checkpoint E:/ns_mc_gan_gi/outputs_phase5/exact_binary_slow_5pct/best_score.pt --output_dir E:/ns_mc_gan_gi/outputs_phase7/measurement_quality/phase5_exact_binary_slow_5pct --num_batches 10 --device cuda

$FlipCheckpoint = "E:/ns_mc_gan_gi/outputs_phase7/flipaware_alpha1_5pct/best_score.pt"
if (Test-Path $FlipCheckpoint) {
  conda run -p $EnvPath python -m src.measurement_quality --config configs/phase7_flipaware_alpha1_5pct.yaml --checkpoint $FlipCheckpoint --output_dir E:/ns_mc_gan_gi/outputs_phase7/measurement_quality/flipaware_alpha1_5pct --num_batches 10 --device cuda
} else {
  Write-Warning "Missing flip-aware checkpoint for measurement quality"
}

$ContinuousCheckpoint = "E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct/best_score.pt"
if (Test-Path $ContinuousCheckpoint) {
  conda run -p $EnvPath python -m src.measurement_quality --config configs/phase7_continuous_physical_5pct.yaml --checkpoint $ContinuousCheckpoint --output_dir E:/ns_mc_gan_gi/outputs_phase7/measurement_quality/continuous_physical_5pct --num_batches 10 --device cuda
} else {
  Write-Warning "Missing continuous checkpoint for measurement quality"
}
