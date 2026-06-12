$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$PY = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$out = "E:/ns_mc_gan_gi/outputs_phase6"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$experiments = @"
[
  {
    "name": "Fixed Rademacher",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt",
    "config": "configs/clean_quick_5pct.yaml"
  },
  {
    "name": "Phase 5 Best Exact Slow",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase5/exact_binary_slow_5pct/best_score.pt",
    "config": "configs/phase5_exact_binary_slow_5pct.yaml"
  },
  {
    "name": "G-only Fine-tune",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/g_only_finetune_5pct/best_score.pt",
    "config": "configs/phase6_g_only_finetune_5pct.yaml"
  },
  {
    "name": "Pattern Trainable Alpha6",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha6_5pct/best_score.pt",
    "config": "configs/phase6_pattern_trainable_alpha6_5pct.yaml"
  },
  {
    "name": "Pattern Trainable Alpha2",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha2_5pct/best_score.pt",
    "config": "configs/phase6_pattern_trainable_alpha2_5pct.yaml"
  },
  {
    "name": "Pattern Trainable Alpha1",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha1_5pct/best_score.pt",
    "config": "configs/phase6_pattern_trainable_alpha1_5pct.yaml"
  },
  {
    "name": "Pattern Trainable Alpha0.5",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/pattern_trainable_alpha0p5_5pct/best_score.pt",
    "config": "configs/phase6_pattern_trainable_alpha0p5_5pct.yaml"
  },
  {
    "name": "Pattern-only Alpha1",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/pattern_only_alpha1_5pct/best_score.pt",
    "config": "configs/phase6_pattern_only_alpha1_5pct.yaml"
  },
  {
    "name": "Soft Signed Train",
    "checkpoint": "E:/ns_mc_gan_gi/outputs_phase6/soft_signed_train_5pct/best_score.pt",
    "config": "configs/phase6_soft_signed_train_5pct.yaml"
  }
]
"@
$experimentsPath = Join-Path $out "paired_5pct_experiments.json"
[System.IO.File]::WriteAllText($experimentsPath, $experiments, [System.Text.UTF8Encoding]::new($false))
conda run -p $PY python -s -m src.eval_paired_controls --experiments_json $experimentsPath --output_dir E:/ns_mc_gan_gi/outputs_phase6/paired_5pct --num_bootstrap 1000 --seed 42
