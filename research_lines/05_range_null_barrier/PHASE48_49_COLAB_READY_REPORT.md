# Phase 48/49 Colab Ready Report

## 1. Notebooks Created

- `colab/phase48_49/session_01_eval_probes.ipynb`
- `colab/phase48_49/session_02_rad5_no_gate.ipynb`
- `colab/phase48_49/session_03_rad5_no_final_audit.ipynb`
- `colab/phase48_49/session_04_scr5_no_gate.ipynb`
- `colab/phase48_49/session_05_scr5_no_final_audit.ipynb`

## 2. Required Input Bundles

Upload or place in Drive under `/content/drive/MyDrive/ns_mc_gan_gi/colab_upload`:

- `ns_mc_gan_gi_project_phase48_49.zip`
- `noleak_bundle_phase48_49.zip`

If browser upload is unstable for the large no-leak bundle, upload both generated input parts instead:

- `noleak_bundle_phase48_49.zip.part_000`
- `noleak_bundle_phase48_49.zip.part_001`

The notebooks automatically merge these parts back into `/content/noleak_bundle_phase48_49.zip`.

Create them locally with:

```powershell
.\scripts\phase48_49\phase48_49_prepare_upload_bundle.ps1
```

## 3. Expected Checkpoint Files In Bundle

- `rademacher5_hq_noise001_colab/last.pt`
- `scrambled_hadamard5_hq_noise001_colab/last.pt`
- `rademacher10_full_noise001_colab/last.pt`
- `scrambled_hadamard10_full_noise001_colab/last.pt`

These are expected from `E:/ns_mc_gan_gi/outputs_phase15/imported_noleak`.

## 4. Expected Exact-A Files

- `rademacher5_hq_noise001_colab/measurement_operator_exact.pt`
- `rademacher10_full_noise001_colab/measurement_operator_exact.pt`

Scrambled Hadamard sessions do not use exact-A files, but must preserve row/column randomization from the no-leak resolved config.

## 5. Colab Commands Per Session

Each notebook performs setup, upload/Drive copy, unzip, verification, run, zip, SHA256, and download. The core commands are:

- Session 01: `python -m src.phase48_49_mechanistic_probes`
- Session 02: `python -m src.phase48_49_train_ablation --task rad5 --variant no_gate`
- Session 03: `python -m src.phase48_49_train_ablation --task rad5 --variant no_final_audit`
- Session 04: `python -m src.phase48_49_train_ablation --task scr5 --variant no_gate`
- Session 05: `python -m src.phase48_49_train_ablation --task scr5 --variant no_final_audit`

## 6. Output Zip Names

- `session_01_eval_probes_outputs.zip`
- `session_02_rad5_no_gate_outputs.zip`
- `session_03_rad5_no_final_audit_outputs.zip`
- `session_04_scr5_no_gate_outputs.zip`
- `session_05_scr5_no_final_audit_outputs.zip`

Zips larger than 1.8GB are split into `.part_###` files.

## 7. Local Import Commands

```powershell
.\scripts\phase48_49\phase48_49_merge_colab_parts.ps1
.\scripts\phase48_49\phase48_49_import_colab_outputs.ps1
```

Default import target:

`E:/ns_mc_gan_gi/outputs_phase48_49_colab_import`

## 8. Train / No-Train Split

- Session 01: no training, eval-only mechanistic probes.
- Sessions 02-05: full train-time ablations in Colab.

## 9. Runtime Category

- Session 01: eval-only, short to medium depending on 1000-sample probes.
- Session 02: overnight-style full Rad-5 ablation.
- Session 03: overnight-style full Rad-5 ablation.
- Session 04: overnight-style full Scr-5 ablation.
- Session 05: overnight-style full Scr-5 ablation.

## 10. Known Risks

- Colab sessions on different Drive accounts must each receive the same project zip and no-leak bundle.
- Rademacher conclusions require `exact_A_loaded=true` and safe cache rebuild metadata.
- Session 01 plots are diagnostic, not train-time causal proof.
- Session 02-05 results are exploratory/ablation until the user approves main-table inclusion.
- Scrambled Hadamard configs use randomized Hadamard rows/columns and `hadamard_zero_filled`; do not relabel them as low-frequency primary HQ.

## Local Check

`python -m compileall src` completed successfully with the bundled Codex Python runtime. Existing SyntaxWarnings in older manuscript-generation scripts are unrelated to Phase 48/49.
