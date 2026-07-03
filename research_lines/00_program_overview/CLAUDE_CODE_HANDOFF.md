# Claude Code Handoff: VQGAN Multi-Seed Pareto Confirmation

This branch is a handoff point for continuing the ghost-imaging GAN/VQGAN work with Claude Code.

## Repository And Branch

- GitHub repository: `https://github.com/ccyyyYyzz/GI_GAN`
- Handoff branch: `codex/vqgan-multiseed-handoff`
- Local Windows checkout used by Codex: `E:\ns_mc_gan_gi_code_fcc_phase1`
- Same checkout from WSL: `/mnt/e/ns_mc_gan_gi_code_fcc_phase1`

Suggested Claude Code entry:

```bash
git clone https://github.com/ccyyyYyzz/GI_GAN.git
cd GI_GAN
git checkout codex/vqgan-multiseed-handoff
```

## Current Scientific State

The latest completed goal is **VQGAN multi-seed Pareto confirmation** for anchor-initialized VQGAN inversion.

Mechanical classification:

```text
VQGAN_PRIOR_TRANSFER_CONFIRMED_MULTI_SEED
```

Interpretation:

- Quality mode is confirmed across 3 seeds.
- VQGAN improves LPIPS and RAPSD versus matched VQAE.
- Measurement consistency is preserved.
- This is a perception-distortion Pareto result, not a no-cost balanced win.
- Balanced mode is not confirmed because RAPSD worsens even though LPIPS improves.

Key multi-seed quality-mode numbers:

- LPIPS delta, VQGAN - VQAE: `-0.126208`, 95% cluster CI `[-0.130222, -0.122006]`
- RAPSD delta: `-0.00113292`, 95% cluster CI `[-0.00138029, -0.000970869]`
- PSNR delta: `-1.70134 dB`, within the preregistered 2.5 dB quality-mode tolerance
- Full RMSE delta: `+0.0163863`
- Centered RMSE delta: `+0.0163840`
- RelMeasErr mean: about `2.2e-7`
- All 3 seeds have same-direction LPIPS/RAPSD improvement.

## Frozen Result Package

Primary handoff artifact:

```text
outputs/compatibility/measurement_conditioned_vqgan/VQGAN_MULTI_SEED_PARETO_CONFIRMATION_PACKAGE.zip
```

SHA256:

```text
7D570BA627EAD19F71189AFBBD28749B96D2995768C293B27D51560EDFF9CAB4
```

Main reports inside the repository:

```text
outputs/compatibility/measurement_conditioned_vqgan/multiseed_pareto_confirmation/MULTISEED_PARETO_CONFIRMATION_REPORT.md
outputs/compatibility/measurement_conditioned_vqgan/multiseed_pareto_confirmation/multiseed_gate_report.json
outputs/compatibility/measurement_conditioned_vqgan/multiseed_pareto_confirmation/CLAIM_EVIDENCE_LEDGER.md
outputs/compatibility/measurement_conditioned_vqgan/multiseed_pareto_confirmation/quality_paired_per_image.csv
outputs/compatibility/measurement_conditioned_vqgan/multiseed_pareto_confirmation/balanced_paired_per_image.csv
```

Per-seed local-only result zips, useful if working on the same machine but intentionally not committed to GitHub because each is about 110 MB:

```text
outputs/compatibility/measurement_conditioned_vqgan/VQGAN_MULTI_SEED_LOCAL_SEED0_ARTIFACT.zip
outputs/compatibility/measurement_conditioned_vqgan/VQGAN_MULTI_SEED_LOCAL_SEED1_ARTIFACT.zip
outputs/compatibility/measurement_conditioned_vqgan/VQGAN_MULTI_SEED_LOCAL_SEED2_ARTIFACT.zip
```

The large `.pt` checkpoints are intentionally not included in GitHub.

## Core Code Added For This Line

Core scripts:

```text
measurement_conditioned_vqgan.py
mc_vqgan_prior_long_canary.py
anchor_initialized_vqgan_inversion.py
gan_high_quality_gi.py
scripts/run_vqgan_multiseed_local.py
scripts/aggregate_vqgan_multiseed_pareto.py
```

Colab wrappers:

```text
colab/vqgan_multiseed_colab_job_common.py
colab/vqgan_multiseed_colab_job_seed0.py
colab/vqgan_multiseed_colab_job_seed1.py
colab/vqgan_multiseed_colab_job_seed2.py
```

Tests:

```text
tests/test_measurement_conditioned_vqgan.py
tests/test_anchor_initialized_vqgan_inversion.py
```

Relevant configs:

```text
configs/compatibility/mc_vqgan_prior_multiseed_hashclean_seed0.yaml
configs/compatibility/mc_vqgan_prior_multiseed_hashclean_seed1.yaml
configs/compatibility/mc_vqgan_prior_multiseed_hashclean_seed2.yaml
configs/compatibility/anchor_vqgan_inversion_multiseed_hashclean_seed0.yaml
configs/compatibility/anchor_vqgan_inversion_multiseed_hashclean_seed1.yaml
configs/compatibility/anchor_vqgan_inversion_multiseed_hashclean_seed2.yaml
configs/compatibility/*_local.yaml
```

## Local Environment

Recommended Python:

```powershell
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe
```

This env was verified with:

- Python 3.11
- Torch `2.2.1+cu121`
- Torchvision `0.17.1+cu121`
- CUDA on local NVIDIA RTX 4060 Laptop GPU
- LPIPS and scikit-image installed

There is also a D-drive Anaconda install:

```text
D:\Anacondar\anaconda3\Scripts\conda.exe
```

but the `ns_mc_gan_gi_py311` project env above is the one that successfully completed the three-seed run.

Smoke tests:

```powershell
& 'E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe' -m pytest -q tests/test_anchor_initialized_vqgan_inversion.py tests/test_measurement_conditioned_vqgan.py
```

Aggregate existing per-seed outputs:

```powershell
& 'E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe' scripts\aggregate_vqgan_multiseed_pareto.py --seeds 0 1 2 --bootstrap-reps 2000
```

Re-run a local seed from scratch:

```powershell
& 'E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe' scripts\run_vqgan_multiseed_local.py --seed-id 0
```

Local configs use:

```text
dataset_root: E:/datasets
```

## Colab Accounts And CLI

Colab CLI path inside WSL:

```bash
/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
```

Account homes:

```bash
# user-described pro account, up to about 2 sessions
/var/tmp/codex-colab-accounts/pro1

# user-described pro+ account, up to about 3 sessions
/var/tmp/codex-colab-accounts/pro2
```

Always set `HOME` to isolate the account before calling the CLI:

```bash
COLAB=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab

HOME=/var/tmp/codex-colab-accounts/pro1 $COLAB status
HOME=/var/tmp/codex-colab-accounts/pro2 $COLAB status
HOME=/var/tmp/codex-colab-accounts/pro1 $COLAB sessions
HOME=/var/tmp/codex-colab-accounts/pro2 $COLAB sessions
```

At handoff time both accounts returned `No active sessions`.

Run one of the prepared Colab jobs:

```bash
COLAB=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
REPO=/mnt/e/ns_mc_gan_gi_code_fcc_phase1

HOME=/var/tmp/codex-colab-accounts/pro2 \
  $COLAB run --gpu L4 --timeout 43200 \
  $REPO/colab/vqgan_multiseed_colab_job_seed0.py
```

For manual sessions:

```bash
HOME=/var/tmp/codex-colab-accounts/pro2 \
  $COLAB new -s vqgan-seed0 --gpu L4

HOME=/var/tmp/codex-colab-accounts/pro2 \
  $COLAB exec -s vqgan-seed0 -f /mnt/e/ns_mc_gan_gi_code_fcc_phase1/colab/vqgan_multiseed_colab_job_seed0.py --timeout 43200
```

Operational note: previous long Colab sessions became stale despite heartbeat, so formal evidence in the latest result package is from the local 4060 run. Use Colab for future heavy exploration, but download artifacts early and often.

## Suggested Next Step

Do not over-claim balanced reconstruction. The honest next scientific move is to decide whether to:

1. take the confirmed quality-mode VQGAN Pareto result to a fresh locked test protocol, or
2. improve the balanced setting so RAPSD does not regress, likely by lowering null residual strength or adding a spectrum/RAPSD constraint during refiner training.

Use the existing gate report as the frozen development evidence; do not tune against any consumed locked/final-v4 set.
