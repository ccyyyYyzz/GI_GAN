$ErrorActionPreference = "Continue"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Runs = @(
  "hadamard10_full_noise001",
  "hadamard10_full_nonoise",
  "hadamard5_medium_noise001",
  "hadamard5_full_noise001",
  "rademacher10_full_noise001",
  "scrambled_hadamard10_full_noise001",
  "mnist_hadamard5_full",
  "fashion_hadamard5_full",
  "cifar10_gray_hadamard10_medium"
)
foreach ($Run in $Runs) {
  conda run -p $EnvPath python -m src.eval_auto `
    --output_dir "E:/ns_mc_gan_gi/outputs_phase10/$Run" `
    --config "configs/phase10/$Run.yaml" `
    --device cuda
}
