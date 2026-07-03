# Colab Runner Report

## Summary

Created a safe, reproducible Google Colab runner for publication baseline
workflows. The notebook is manual-use only and does not connect to an
authenticated Colab runtime from this local session.

Full training was run: no.

## Files Changed

- `colab/pub_baselines_colab_runner.ipynb`
- `colab/README_COLAB_RUNNER.md`
- `scripts/validate_colab_runner.py`
- `scripts/collect_colab_artifacts.py`
- `COLAB_RUNNER_REPORT.md`

The unrelated untracked file `configs/g2r/g2r_r2_sanity.yaml` was left
untouched and must remain uncommitted.

## Selected Python Interpreter

Initial discovery found:

- `C:\Python34\python.exe`: Python 3.4.1, rejected.
- `D:\bin\python.exe`: Python 3.12.9, modern but missing `PyYAML`, rejected for
  this validation session.
- `D:\Anacondar\anaconda3\python.exe`: Python 3.11.5 with `PyYAML 6.0.1`,
  selected.

Final local validation interpreter:

```powershell
$PY = "D:\Anacondar\anaconda3\python.exe"
```

Version:

```text
3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]
PyYAML 6.0.1
```

## Repository Preflight

```text
git rev-parse --show-toplevel
E:/ns_mc_gan_gi_code

git branch --show-current
pub-architecture-baselines

git status --short
?? configs/g2r/g2r_r2_sanity.yaml

git log --oneline --decorate -n 12
6bcf967 (HEAD -> pub-architecture-baselines) prepare publication architecture baseline configs
9c02267 (backup/g2r-modec, g2r-modec) Phase 3 three-arm report: stop rule FIRED (all arms fail G-MEAN), Round 2 authorized
7d897e8 Pilot arm 2/3 (adv3e-3) results: 3/6 gates, identical std plateau 0.4841, G-MEAN -15.9 dB
22b72ae Round 2 amendment pre-registered: closed-loop beta_SD controller (default OFF)
bcb0d09 Pilot arm 1/3 (adv1e-3) results: 3/6 gates, std runaway to 0.484 plateau, G-MEAN -15.6 dB
806e30e Phase 3 pilot infrastructure: gate-trajectory eval, collapse detection, eval_seed decoupling
15af60b Mode C posterior sampler (g2r series): z-injected refiner, P0 null-space head, conditional PatchGAN
057758a (backup/g2r-protocol, g2r-protocol) import eval toolkit (developed externally; tests green in project env)
05c6c1d Path hygiene after repo relocation: repo-relative fig1 fallback path
0c3e4c6 g2r protocol hardening: split guards, checkpoint discipline, exact-A cache assertions, P0 machinery, pre-registered gates
3bc11cf (backup/main, main) Baseline snapshot before g2r protocol hardening
```

Branch created:

```text
pub-colab-runner
```

## Local Validation Results

```powershell
& $PY -m compileall src scripts
```

Output:

```text
Listing 'src'...
Listing 'scripts'...
Listing 'scripts\\g2r'...
Listing 'scripts\\phase48_49'...
Listing 'scripts\\phase51A'...
Listing 'scripts\\phase53B'...
Listing 'scripts\\phase53C'...
Compiling 'scripts\\validate_colab_runner.py'...
```

```powershell
& $PY scripts\validate_pub_baseline_configs.py
```

Output:

```text
Validated 12 pub baseline configs.
Supported model types checked against src/models.py: direct_coeff_to_image, direct_y_to_image, hq_two_stage, hq_unet, nafnet_small, residual_unet_small, residual_unet_wide, residual_unet_wide_refiner, resunet, unet, unrolled_ista
```

```powershell
& $PY scripts\validate_colab_runner.py
```

Output:

```text
Colab runner validation summary
- notebook: colab\pub_baselines_colab_runner.ipynb
- cells: 21
- warnings: 0
- errors: 0
PASS: Colab runner notebook is structurally valid and uses conservative defaults.
```

```powershell
& $PY -c "import json; p='colab/pub_baselines_colab_runner.ipynb'; nb=json.load(open(p, encoding='utf-8')); print(nb.get('nbformat'), len(nb.get('cells', [])))"
```

Output:

```text
4 21
```

Also run:

```powershell
git status --short
git diff --stat
```

Output before staging:

```text
?? colab/README_COLAB_RUNNER.md
?? colab/pub_baselines_colab_runner.ipynb
?? configs/g2r/g2r_r2_sanity.yaml
?? scripts/collect_colab_artifacts.py
?? scripts/validate_colab_runner.py
```

`git diff --stat` produced no output before staging because the changed files
were new and untracked at that point.

## Notebook Sections

- `# Publication Baseline Colab Runner`
- `## Source Setup`
- `# User Configuration`
- `# Source Setup`
- `## Environment Check`
- `# Environment Check`
- `## Dependency Installation`
- `# Dependency Installation`
- `## Config Validation`
- `# Config Validation`
- `## Baseline Command Discovery`
- `# Baseline Command Discovery`
- `## Run Modes`
- `# Run Modes`
- `## Logging And Artifacts`
- `# Logging And Artifacts`
- `## End-of-run Summary`
- `# End-of-run Summary`
- `## Completion Notification`
- `# Completion Notification`

## How To Use In Colab

1. Create a repo zip from Windows:

   ```powershell
   cd E:\ns_mc_gan_gi_code
   git archive --format=zip --output "$env:USERPROFILE\Desktop\ns_mc_gan_gi_code_pub_colab.zip" pub-colab-runner
   ```

2. Open `colab/pub_baselines_colab_runner.ipynb` in Google Colab.
3. Keep defaults for the first run:

   ```python
   SOURCE_MODE = "upload_zip"
   RUN_MODE = "smoke"
   DRY_RUN = True
   ```

4. Upload the repo zip when Colab asks.
5. Run all cells through the final summary.
6. Copy the printed `COLAB_SMOKE_COPY_TO_CHATGPT` block back into ChatGPT.
7. Only after smoke succeeds, test `RUN_MODE = "single"` with `DRY_RUN = True`.
8. Set `DRY_RUN = False` only when ready to run one real baseline.
9. Use `RUN_MODE = "matrix"` only after smoke and single runs succeed.

## Remaining Risks And Manual Steps

- Colab package state can differ from local state. If `PyYAML` or other
  packages are missing, set `INSTALL_DEPENDENCIES = True` and install from the
  repo `requirements.txt`.
- The notebook cannot verify private Drive files from here; Drive paths are
  user-editable Colab-side settings.
- Real training can be expensive. The default smoke mode performs validation
  and manifest generation only.
- A committed file cannot contain its own final commit hash without changing
  that hash. The final commit hash is recorded in the handoff block after the
  commit.

## Commit

Commit hash: recorded in the final `COPY_TO_CHATGPT_C1` handoff after commit.
