$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $envPath python -m src.quality_audit
conda run -p $envPath python -m src.make_related_work_table
