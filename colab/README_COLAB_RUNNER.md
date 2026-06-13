# Publication Baseline Colab Runner

This folder contains a conservative Google Colab runner for the publication
baseline configs under `configs/pub_baselines`.

The notebook is designed for manual Colab use. It does not connect to Colab
for you, does not bypass Google login, does not need private tokens, and
defaults to smoke mode with no training.

## Files

- `pub_baselines_colab_runner.ipynb`: manual Colab notebook.
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
SINGLE_CONFIG = "configs/pub_baselines/colab/unet_scr5_pub_colab.yaml"
```

Inspect the planned command manifest before executing anything.

3. Run one real baseline only after smoke and single dry-run look correct:

```python
RUN_MODE = "single"
DRY_RUN = False
```

4. Use matrix mode only after smoke and single have succeeded:

```python
RUN_MODE = "matrix"
DRY_RUN = False
```

Matrix mode can be expensive and is intentionally not the default.

## Artifacts

The notebook writes a timestamped output directory under `OUTPUT_ROOT`, with:

- `logs/environment_log.json`
- `logs/config_list.json`
- `logs/command_discovery_manifest.json`
- `logs/planned_training_commands.json`
- `logs/command_log.json`
- `logs/return_code_summary.json`
- `artifact_manifest.json`
- `artifact_manifest.csv`
- a zip bundle of the run directory when possible
- `logs/COLAB_SMOKE_COPY_TO_CHATGPT.txt`

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
$PY = "D:\bin\python.exe"
& $PY scripts\validate_colab_runner.py
```
