# Publication Architecture Baseline Config Report

## Configs created

Windows configs:
- `configs/pub_baselines/unet_rad5_pub.yaml`
- `configs/pub_baselines/unet_scr5_pub.yaml`
- `configs/pub_baselines/unrolled_ista_rad5_pub.yaml`
- `configs/pub_baselines/unrolled_ista_scr5_pub.yaml`
- `configs/pub_baselines/resunet_rad5_pub.yaml`
- `configs/pub_baselines/resunet_scr5_pub.yaml`

Colab configs:
- `configs/pub_baselines/colab/unet_rad5_pub_colab.yaml`
- `configs/pub_baselines/colab/unet_scr5_pub_colab.yaml`
- `configs/pub_baselines/colab/unrolled_ista_rad5_pub_colab.yaml`
- `configs/pub_baselines/colab/unrolled_ista_scr5_pub_colab.yaml`
- `configs/pub_baselines/colab/resunet_rad5_pub_colab.yaml`
- `configs/pub_baselines/colab/resunet_scr5_pub_colab.yaml`

Support scripts:
- `scripts/render_pub_baseline_colab_configs.py`
- `scripts/validate_pub_baseline_configs.py`

## Exact path mappings

Dataset root:
- Windows: `E:\ns_mc_gan_gi\data`
- Colab: `/content/drive/MyDrive/ns_mc_gan_gi/data`

Output root:
- Windows: `E:\ns_mc_gan_gi\outputs_pub_baselines\<config_name>`
- Colab: `/content/drive/MyDrive/ns_mc_gan_gi/outputs_pub_baselines/<config_name>`

Rademacher exact-A path:
- Windows: `E:\ns_mc_gan_gi\outputs_phase15\imported_noleak\rademacher5_hq_noise001_colab\measurement_operator_exact.pt`
- Colab: `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase15/imported_noleak/rademacher5_hq_noise001_colab/measurement_operator_exact.pt`

The renderer rewrites both `phase25_measurement_lock.exact_A_path` and the top-level `measurement_operator_exact_path`.

## Shared settings

All publication baseline configs set:
- `use_adversarial: false`
- `lambda_adv: 0.0`
- `use_null_project: true`
- `use_dc_project: true`
- `use_final_dc_project: true`
- `output_range_mode: clamp_eval_only`
- `noise_std: 0.01`
- `lambda_solver: 0.001`
- `limit_train_samples: 50000`
- `limit_val_samples: 2000`
- `epochs: 80`

Rademacher configs set top-level `exact_A_required: true` and point to the imported no-leak exact-A tensor. Scrambled Hadamard configs set `exact_A_required: false` and leave `measurement_operator_exact_path` empty.

## Assumptions

- The phase25 architecture ablation configs are the template of record for model/loss/budget fields.
- STL10 has no native `val` split in the repo loader. To satisfy the no-test-split requirement, these configs use `train_split: unlabeled` and `val_split: train`.
- No g2r configs were modified, and no g2r posterior-sampler settings were added to these publication configs.
- The repo does not expose a dedicated config flag for saving an exact measurement operator. Exact-A reuse is supported through `exact_A_required` plus `measurement_operator_exact_path`, and the original phase25 measurement lock is preserved.

## Recommended Colab assignment

Minimum baseline set:
- Colab A: `unet_rad5_pub_colab.yaml`, then `unet_scr5_pub_colab.yaml`
- Colab B: `unrolled_ista_rad5_pub_colab.yaml`, then `unrolled_ista_scr5_pub_colab.yaml`

Optional ResUNet controls:
- Colab C: `resunet_rad5_pub_colab.yaml`
- Colab D: `resunet_scr5_pub_colab.yaml`

If only one Colab GPU is available, run the minimum set first in this order: `unet_rad5`, `unet_scr5`, `unrolled_ista_rad5`, `unrolled_ista_scr5`, then run the two ResUNet controls as resources allow.
