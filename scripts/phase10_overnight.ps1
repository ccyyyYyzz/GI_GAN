$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.overnight_runner `
  --queue configs/phase10/overnight_queue.yaml `
  --status_path E:/ns_mc_gan_gi/outputs_phase10/overnight_status.json
