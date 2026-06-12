$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"

conda run -p $PY python -s -m src.calibrate_operator_equivalence `
  --config configs/phase5_calibrate_exact_5pct.yaml `
  --device cuda `
  --fixed_pattern_type rademacher `
  --report_path E:/ns_mc_gan_gi/outputs_phase5/operator_calibration_5pct.json
