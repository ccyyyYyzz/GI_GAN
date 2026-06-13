# Publication Baseline Colab Runner

This folder contains a conservative Google Colab runner for the publication
baseline configs under `configs/pub_baselines`.

The notebook is designed for manual Colab use. It does not connect to Colab
for you, does not bypass Google login, does not need private tokens, and
defaults to smoke mode with no training.

## Files

- `pub_baselines_colab_runner.ipynb`: manual Colab notebook.
- `../scripts/preflight_pub_baseline_run.py`: lightweight preflight guard used
  before any real single training command.
- `../scripts/validate_colab_runner.py`: local structural and safety checks for
  the notebook.
- `../scripts/collect_colab_artifacts.py`: JSON/CSV artifact manifest writer
  for Colab or local output folders.

## Create a Repo Zip on Windows

From PowerShell, use a Windows local example like this after committing the
runner branch:

```powershell
cd E:\ns_mc_gan_gi_code
git archive --format=zip --output "$env:USERPROFILE\Desktop\ns_mc_gan_gi_code_pub_colab.zip" HEAD
```

This creates a clean zip from tracked files only. It will not include
untracked local files such as scratch configs.

If you need to include the current branch name explicitly:

```powershell
cd E:\ns_mc_gan_gi_code
git archive --format=zip --output "$env:USERPROFILE\Desktop\ns_mc_gan_gi_code_pub_colab.zip" pub-colab-runner
```

## Open the Notebook in Colab

1. Upload `colab/pub_baselines_colab_runner.ipynb` to Google Colab.
2. Choose a runtime. GPU is useful for real training, but smoke mode should not
   require a GPU.
3. Run the user configuration cell first and keep the default:

```python
RUN_MODE = "smoke"
DRY_RUN = True
CONFIRM_REAL_TRAINING = False
```

## SOURCE_MODE Options

`SOURCE_MODE = "upload_zip"`

Use this first if you have the repo zip on your local Windows machine. The
notebook will ask you to upload the zip through `files.upload()`.

`SOURCE_MODE = "drive_zip"`

Upload the repo zip to Google Drive, set `DRIVE_ZIP_PATH` to the Drive path,
then run the source setup cell. The notebook mounts Drive and extracts the zip.
Use a user-editable Drive path such as:

```python
DRIVE_ZIP_PATH = "/content/drive/MyDrive/ns_mc_gan_gi_code_pub_colab.zip"
```

`SOURCE_MODE = "git_clone"`

Use this only for a public repository URL:

```python
GIT_URL = "https://github.com/OWNER/REPO.git"
GIT_BRANCH = "pub-colab-runner"
```

Private GitHub credentials are intentionally not part of this workflow.

## Recommended Run Order

1. Start with smoke mode:

```python
RUN_MODE = "smoke"
DRY_RUN = True
```

This validates the repo, renders Colab baseline configs when the script is
present, checks `configs/pub_baselines`, discovers the actual training
entrypoint, writes logs, and prints `COLAB_SMOKE_COPY_TO_CHATGPT`.

2. Try single mode as a dry run:

```python
RUN_MODE = "single"
DRY_RUN = True
CONFIRM_REAL_TRAINING = False
MAX_CONFIGS = 1
SINGLE_CONFIG = ""
```

With `SINGLE_CONFIG = ""`, the notebook selects the first valid config under
`configs/pub_baselines/colab`. The dry-run writes at least one planned
`python -m src.train --config ... --device ...` command to:

- `logs/planned_training_commands.json`
- `logs/command_log.json`
- `logs/return_code_summary.json`

It also prints `COLAB_SINGLE_DRYRUN_COPY_TO_CHATGPT`. Copy that block back into
ChatGPT after the manual Colab run.

Single dry-run is not an experiment result. It is only a command-manifest check:
`training_attempted` should remain `0`, and `dry_run_commands` should be at
least `1`.

3. Run single preflight after the dry-run command looks correct:

```python
RUN_MODE = "single"
DRY_RUN = False
CONFIRM_REAL_TRAINING = False
MAX_CONFIGS = 1
```

This runs `scripts/preflight_pub_baseline_run.py` and then stops safely before
training because `CONFIRM_REAL_TRAINING` is still `False`. The preflight checks
GPU availability, required imports, config readability, data and exact-A paths,
output path writability, disk space, command logging, obvious Windows path
mistakes in executable config fields, and obvious secrets/tokens. It writes:

- `logs/preflight_report.json`
- `logs/preflight_report.md`
- `logs/command_log.json`
- `logs/return_code_summary.json`

It also prints `COLAB_SINGLE_PREFLIGHT_COPY_TO_CHATGPT`. Single preflight is
not an experiment result; it is a go/no-go check for one real command.

4. Run one real baseline only after smoke, dry-run, and preflight look correct.
   Real single training requires a GPU Colab runtime.

```python
RUN_MODE = "single"
DRY_RUN = False
CONFIRM_REAL_TRAINING = True
MAX_CONFIGS = 1
```

Only set these three flags together after the dry-run command and preflight
report are correct:

```python
RUN_MODE = "single"
DRY_RUN = False
CONFIRM_REAL_TRAINING = True
```

The notebook runs the real training command only when all of these are true:

- `RUN_MODE == "single"`
- `DRY_RUN == False`
- `CONFIRM_REAL_TRAINING == True`
- preflight `ok == true`

If any condition fails, the command is recorded as blocked and the artifact zip
is still created. After a real run, download or save the artifact zip before
closing the Colab runtime. It contains the command logs, return-code summary,
preflight report, artifact manifest, and the copy block for ChatGPT.

Matrix mode is available for planning/dry-run command inspection only in this
guarded notebook. Do not use it for real training.

## Artifacts

The notebook writes a timestamped output directory under `OUTPUT_ROOT`, with:

- `logs/environment_log.json`
- `logs/config_list.json`
- `logs/command_discovery_manifest.json`
- `logs/planned_training_commands.json`
- `logs/preflight_report.json` and `logs/preflight_report.md` after single
  preflight or real single runs
- `logs/command_log.json`
- `logs/return_code_summary.json`
- `artifact_manifest.json`
- `artifact_manifest.csv`
- a zip bundle of the run directory when possible
- `logs/COLAB_SMOKE_COPY_TO_CHATGPT.txt`
- `logs/COLAB_SINGLE_DRYRUN_COPY_TO_CHATGPT.txt` after a single dry-run
- `logs/COLAB_SINGLE_PREFLIGHT_COPY_TO_CHATGPT.txt` after a blocked preflight
- `logs/COLAB_SINGLE_REAL_COPY_TO_CHATGPT.txt` after a real single command is
  attempted

To copy artifacts back to Windows, download the zip bundle from the Colab file
browser or copy the run directory to Drive and download it from Google Drive.

## What Not To Do

- Do not paste private tokens or passwords into the notebook.
- Do not hard-code secrets in notebook cells.
- Do not rely on `E:\` paths inside Colab. Those are Windows local examples
  only.
- Do not use keep-alive hacks.
- Do not run multiple notebooks or accounts to bypass Colab limits.
- Do not claim dry-run or smoke-mode output as full experiment results.

## Local Validation

Use a modern Python interpreter, not the system Python 3.4 install:

```powershell
$PY = "D:\Anacondar\anaconda3\python.exe"
& $PY scripts\validate_colab_runner.py
```
