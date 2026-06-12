$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Configs = @(
  "configs/phase4_best_2pct.yaml",
  "configs/phase4_best_5pct.yaml",
  "configs/phase4_best_10pct.yaml"
)
foreach ($Config in $Configs) {
  if (Test-Path -LiteralPath $Config) {
    conda run -p $EnvPath python -s -m src.train --config $Config --device cuda
  } else {
    Write-Host "missing $Config"
  }
}
