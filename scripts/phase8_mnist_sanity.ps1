$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$runs = @(
  @("configs/phase8_mnist_fixed_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/mnist_fixed_5pct"),
  @("configs/phase8_mnist_continuous_5pct.yaml", "E:/ns_mc_gan_gi/outputs_phase8/mnist_continuous_5pct")
)
foreach ($run in $runs) {
  $cfg = $run[0]
  $out = $run[1]
  conda run -p $envPath python -m src.train --config $cfg --device cuda
  $ckpt = Join-Path $out "best_score.pt"
  if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $out "best_ssim.pt" }
  if (Test-Path $ckpt) {
    conda run -p $envPath python -m src.eval --checkpoint $ckpt --config $cfg --device cuda
  }
}
