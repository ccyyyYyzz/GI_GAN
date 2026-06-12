$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m compileall -q src
powershell -ExecutionPolicy Bypass -File scripts/phase26_pca_full.ps1
powershell -ExecutionPolicy Bypass -File scripts/phase26_arch_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts/phase26_gate_report.ps1
