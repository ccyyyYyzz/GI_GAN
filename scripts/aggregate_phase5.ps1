$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

conda run -p $PY python -s -m src.aggregate_phase5
conda run -p $PY python -s -m src.make_phase5_report
