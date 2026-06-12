param(
  [string]$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

conda run -p $EnvPath python -m compileall src
conda run -p $EnvPath python -m src.phase15r_inventory
conda run -p $EnvPath python -m src.phase15r_inspect_exactA
conda run -p $EnvPath python -m src.phase15r_backprojection_test
conda run -p $EnvPath python -m src.phase15r_checkpoint_inspect
conda run -p $EnvPath python -m src.phase15r_eval_variants
conda run -p $EnvPath python -m src.phase15r_dataset_split_audit
conda run -p $EnvPath python -m src.phase15r_make_golden_bundle_script
conda run -p $EnvPath python -m src.phase15r_replay_golden_bundle
conda run -p $EnvPath python -m src.make_phase15r_repro_report
