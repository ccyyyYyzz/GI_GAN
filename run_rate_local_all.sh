#!/usr/bin/env bash
# Resumable sequential local B-tier runner: 6 anchor-refiner retrains at new rates (4000 steps).
# Skips any run whose final reports/gate_report.json already exists, so a relaunch resumes.
PY='E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe'
cd /e/ns_mc_gan_gi_code_fcc_phase1 || exit 1
BASE=outputs/compatibility/measurement_conditioned_vqgan
LOG="$BASE/detail_fusion_paper/_bundle/rate_runs.log"
mkdir -p "$(dirname "$LOG")"
RUNS="rate10_seed0 rate10_seed1 rate10_seed2 rate02_seed0 rate02_seed1 rate02_seed2"
for tag in $RUNS; do
  gate="$BASE/anchor_${tag}_local/reports/gate_report.json"
  if [ -f "$gate" ]; then
    echo "=== SKIP $tag (already done) $(date +%H:%M:%S) ===" | tee -a "$LOG"
    continue
  fi
  echo "=== START $tag $(date +%H:%M:%S) ===" | tee -a "$LOG"
  "$PY" anchor_initialized_vqgan_inversion.py --config "configs/compatibility/anchor_vqgan_inversion_${tag}_local.yaml" >> "$LOG" 2>&1
  rc=$?
  echo "=== END $tag rc=$rc $(date +%H:%M:%S) ===" | tee -a "$LOG"
done
echo "ALL_RATE_RUNS_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
