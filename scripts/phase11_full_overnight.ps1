$ErrorActionPreference = "Stop"
$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
conda run -p $EnvPath python -m compileall src
conda run -p $EnvPath python -m src.overnight_runner --queue configs/phase10/overnight_queue.yaml --status_path E:/ns_mc_gan_gi/outputs_phase10/overnight_status.json
conda run -p $EnvPath python -m src.aggregate_phase10
conda run -p $EnvPath python -m src.export_phase10_examples
conda run -p $EnvPath python -m src.make_phase10_report
conda run -p $EnvPath python -m src.phase11_adaptive_continue
conda run -p $EnvPath python -m src.phase11_run_adaptive_continue
conda run -p $EnvPath python -m src.phase11_attribution
conda run -p $EnvPath python -m src.aggregate_phase11
conda run -p $EnvPath python -m src.make_phase11_report
conda run -p $EnvPath python -m src.export_phase11_paper_assets
[console]::Beep(880, 250)
Start-Sleep -Milliseconds 80
[console]::Beep(1046, 350)
