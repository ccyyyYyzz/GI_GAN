$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$levels = "0.0,0.005,0.01,0.02,0.05"

conda run -p $PY python -s -m src.eval_noise_sweep `
  --checkpoint E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt `
  --config configs/clean_quick_5pct.yaml `
  --noise_levels $levels `
  --output_dir E:/ns_mc_gan_gi/outputs_phase5/noise_sweep/fixed_5pct `
  --device cuda

$exact = "E:/ns_mc_gan_gi/outputs_phase5/exact_binary_5pct/best_score.pt"
if (Test-Path $exact) {
  conda run -p $PY python -s -m src.eval_noise_sweep `
    --checkpoint $exact `
    --config configs/phase5_exact_binary_5pct.yaml `
    --noise_levels $levels `
    --output_dir E:/ns_mc_gan_gi/outputs_phase5/noise_sweep/exact_binary_5pct `
    --device cuda
} else {
  Write-Host "missing exact checkpoint $exact"
}
