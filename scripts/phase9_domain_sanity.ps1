$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  @("configs/phase9/mnist_hadamard5_hq.yaml", "E:/ns_mc_gan_gi/outputs_phase9/mnist_hadamard5_hq"),
  @("configs/phase9/fashion_hadamard5_hq.yaml", "E:/ns_mc_gan_gi/outputs_phase9/fashion_hadamard5_hq")
)
foreach ($Run in $Runs) {
  $Config = $Run[0]
  $Out = $Run[1]
  conda run -p $EnvPath python -m src.train --config $Config --device cuda
  $Checkpoint = $null
  foreach ($Name in @("best_hq.pt", "best_score.pt", "best_ssim.pt")) {
    $Candidate = Join-Path $Out $Name
    if (Test-Path $Candidate) {
      $Checkpoint = $Candidate
      break
    }
  }
  if ($Checkpoint -ne $null) {
    conda run -p $EnvPath python -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
    conda run -p $EnvPath python -m src.analyze_convergence --output_dir $Out
  }
}
