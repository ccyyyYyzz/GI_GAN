$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$runs = @(
  @("configs/phase8_fixed_wide_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_5pct"),
  @("configs/phase8_fixed_wide_refiner_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_refiner_5pct"),
  @("configs/phase8_continuous_g_only_wide_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/continuous_g_only_wide_5pct"),
  @("configs/phase8_continuous_physical_wide_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/continuous_physical_wide_5pct"),
  @("configs/phase8_direct_y_fixed_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/direct_y_fixed_5pct")
)
foreach ($run in $runs) {
  $cfg = $run[0]
  $out = $run[1]
  $ckpt = Join-Path $out "best_score.pt"
  if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $out "best_ssim.pt" }
  if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $out "best_psnr.pt" }
  if (Test-Path $ckpt) {
    conda run -p $envPath python -m src.eval --checkpoint $ckpt --config $cfg --device cuda
  } else {
    Write-Host "Missing checkpoint for $out"
  }
}
