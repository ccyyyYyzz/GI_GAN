$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m src.write_final_claims_draft
conda run -p $EnvPath python -m src.make_phase12_report
