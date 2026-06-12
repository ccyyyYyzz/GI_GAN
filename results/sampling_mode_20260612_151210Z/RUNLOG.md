# Runlog

- UTC created: 2026-06-12T15:20:01+00:00
- Prompt copied from: `C:\Users\CYZ的computer\Downloads\prompt_codex.md`
- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/checks/gan_identity_gate.py --output results/sampling_mode_20260612_151210Z/checks/identity_gate_results.json`
- Identity gate pass: `True`.
- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m compileall results/sampling_mode_20260612_151210Z`
- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/build_sampling_dossier.py`
- Split-hash command not run because no main or pilot data split index file was locatable; this is recorded as a blocker in PROVENANCE_SAMPLING.json.
- Full G2 training launched: `false`.
- G2 smoke training launched: `false`; blocked by unsafe provenance.

## Follow-Up Gap Items

- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/test_infra_utilities.py`
- Infra unit-test status: `pass`; output `INFRA_UNIT_TEST_RESULTS.json`.
- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py`
- Certificate recheck status: mean checkpoint and G1 source checkpoint loaded with `generator_ema`; post-GAN pilot checkpoint not found.
- Command: `conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m compileall results/sampling_mode_20260612_151210Z`
- Training launched during follow-up: `false`.
