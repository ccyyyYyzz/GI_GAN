$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m compileall src
conda run -p $EnvPath python -m src.phase12_monitor_fashion
conda run -p $EnvPath python -m src.build_final_result_registry
conda run -p $EnvPath python -m src.make_paper_tables
conda run -p $EnvPath python -m src.make_paper_figures
conda run -p $EnvPath python -m src.export_final_recon_examples
conda run -p $EnvPath python -m src.make_dc_row_control
conda run -p $EnvPath python -m src.run_minimal_traditional_baselines
conda run -p $EnvPath python -m src.write_final_claims_draft
conda run -p $EnvPath python -m src.make_phase12_report
