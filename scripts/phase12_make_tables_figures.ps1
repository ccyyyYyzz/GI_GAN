$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.make_paper_tables
conda run -p $EnvPath python -m src.make_paper_figures
conda run -p $EnvPath python -m src.export_final_recon_examples
