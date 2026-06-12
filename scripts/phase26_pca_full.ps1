$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Root = if ($env:NS_MC_GAN_GI_ROOT) { $env:NS_MC_GAN_GI_ROOT } else { "E:/ns_mc_gan_gi" }
$TrainSamples = if ($env:TRAIN_SAMPLES) { $env:TRAIN_SAMPLES } else { "5000" }
$EvalSamples = if ($env:EVAL_SAMPLES) { $env:EVAL_SAMPLES } else { "500" }
conda run -p $EnvPath python -m src.phase26_pca_oracle_full --drive_root $Root --device cuda --train_samples $TrainSamples --eval_samples $EvalSamples
