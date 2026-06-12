$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m compileall src
powershell -ExecutionPolicy Bypass -File scripts/package_project_for_colab.ps1
Write-Host "Phase 14 Colab-first package is ready. See E:/ns_mc_gan_gi/outputs_phase14/COLAB_RUN_INSTRUCTIONS.md"
