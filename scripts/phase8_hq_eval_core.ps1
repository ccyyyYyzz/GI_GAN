$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$runs = @(
  @("configs/phase8_hq/hadamard_hq_10pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8_hq/hadamard_hq_10pct"),
  @("configs/phase8_hq/hadamard_hq_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8_hq/hadamard_hq_5pct"),
  @("configs/phase8_hq/rademacher_hq_10pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8_hq/rademacher_hq_10pct"),
  @("configs/phase8_hq/continuous_physical_hq_10pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8_hq/continuous_physical_hq_10pct"),
  @("configs/phase8_hq/continuous_physical_hq_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8_hq/continuous_physical_hq_5pct")
)
foreach ($run in $runs) {
  $cfg = $run[0]
  $out = $run[1]
  $ckpt = Join-Path $out "best_hq.pt"
  if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $out "best_score.pt" }
  if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $out "best_ssim.pt" }
  if (Test-Path $ckpt) {
    conda run -p $envPath python -m src.eval --checkpoint $ckpt --config $cfg --device cuda
  } else {
    Write-Host "Missing checkpoint for $out"
  }
}
