$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  @("configs/phase9/hadamard10_probe_nonoise.yaml", "E:/ns_mc_gan_gi/outputs_phase9/hadamard10_probe_nonoise"),
  @("configs/phase9/hadamard10_probe_noise001.yaml", "E:/ns_mc_gan_gi/outputs_phase9/hadamard10_probe_noise001"),
  @("configs/phase9/rademacher10_probe_noise001.yaml", "E:/ns_mc_gan_gi/outputs_phase9/rademacher10_probe_noise001"),
  @("configs/phase9/scrambled_hadamard10_probe_noise001.yaml", "E:/ns_mc_gan_gi/outputs_phase9/scrambled_hadamard10_probe_noise001"),
  @("configs/phase9/hadamard5_probe_noise001.yaml", "E:/ns_mc_gan_gi/outputs_phase9/hadamard5_probe_noise001")
)
foreach ($Run in $Runs) {
  $Config = $Run[0]
  $Out = $Run[1]
  $Checkpoint = $null
  foreach ($Name in @("best_hq.pt", "best_score.pt", "best_ssim.pt")) {
    $Candidate = Join-Path $Out $Name
    if (Test-Path $Candidate) {
      $Checkpoint = $Candidate
      break
    }
  }
  if ($Checkpoint -eq $null) {
    Write-Host "Skipping missing checkpoint in $Out"
    continue
  }
  conda run -p $EnvPath python -m src.eval --checkpoint $Checkpoint --config $Config --device cuda
  conda run -p $EnvPath python -m src.analyze_convergence --output_dir $Out
}
