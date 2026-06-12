$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.eval_phase10_noise_sweep `
  --checkpoint E:/ns_mc_gan_gi/outputs_phase10/hadamard10_full_noise001/best_hq.pt `
  --config configs/phase10/hadamard10_full_noise001.yaml `
  --output_dir E:/ns_mc_gan_gi/outputs_phase10/noise_sweep_hadamard10 `
  --device cuda
