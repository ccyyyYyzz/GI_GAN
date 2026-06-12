$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.train --config configs/phase10/hadamard10_full_noise001.yaml --device cuda
conda run -p $EnvPath python -m src.eval_auto --output_dir E:/ns_mc_gan_gi/outputs_phase10/hadamard10_full_noise001 --config configs/phase10/hadamard10_full_noise001.yaml --device cuda
conda run -p $EnvPath python -m src.analyze_convergence --output_dir E:/ns_mc_gan_gi/outputs_phase10/hadamard10_full_noise001
conda run -p $EnvPath python -m src.train --config configs/phase10/hadamard5_medium_noise001.yaml --device cuda
conda run -p $EnvPath python -m src.eval_auto --output_dir E:/ns_mc_gan_gi/outputs_phase10/hadamard5_medium_noise001 --config configs/phase10/hadamard5_medium_noise001.yaml --device cuda
conda run -p $EnvPath python -m src.analyze_convergence --output_dir E:/ns_mc_gan_gi/outputs_phase10/hadamard5_medium_noise001
conda run -p $EnvPath python -m src.aggregate_phase10
conda run -p $EnvPath python -m src.export_phase10_examples
conda run -p $EnvPath python -m src.make_phase10_report
