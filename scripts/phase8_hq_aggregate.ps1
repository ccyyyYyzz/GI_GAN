$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $envPath python -m src.aggregate_phase8_hq
conda run -p $envPath python -m src.export_hq_examples
conda run -p $envPath python -m src.make_phase8_hq_report
