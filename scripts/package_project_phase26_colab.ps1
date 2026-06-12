$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UploadDir = "E:/ns_mc_gan_gi/colab_upload"
$Stage = Join-Path $UploadDir "_staging_ns_mc_gan_gi_phase26_project"
$Zip = Join-Path $UploadDir "ns_mc_gan_gi_project_phase26_colab.zip"
$Sha = "$Zip.sha256.txt"
$InstructionsDir = "E:/ns_mc_gan_gi/outputs_phase26"
$Instructions = Join-Path $InstructionsDir "COLAB_PHASE26_RUNBOOK.md"

New-Item -ItemType Directory -Force -Path $UploadDir, $InstructionsDir | Out-Null
if (Test-Path $Stage) {
    $resolvedStage = (Resolve-Path $Stage).Path
    $resolvedUpload = (Resolve-Path $UploadDir).Path
    if (-not $resolvedStage.StartsWith($resolvedUpload)) {
        throw "Refusing to remove staging directory outside upload dir: $resolvedStage"
    }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$Include = @("src", "configs", "scripts", "colab", "README.md", "requirements.txt")
foreach ($item in $Include) {
    $src = Join-Path $Repo $item
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination $Stage -Recurse -Force
    }
}

if (Test-Path $Zip) { Remove-Item -LiteralPath $Zip -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -CompressionLevel Fastest
$hash = (Get-FileHash -Algorithm SHA256 -Path $Zip).Hash.ToLowerInvariant()
Set-Content -Path $Sha -Encoding UTF8 -Value $hash

$md = @"
# Phase 26 Colab Runbook

Upload this zip to Colab:

$Zip

SHA256:

$hash

## Colab setup

~~~python
from google.colab import drive
drive.mount('/content/drive')
~~~

~~~python
from google.colab import files
uploaded = files.upload()  # upload ns_mc_gan_gi_project_phase26_colab.zip
~~~

~~~bash
rm -rf /content/ns_mc_gan_gi_phase26
mkdir -p /content/ns_mc_gan_gi_phase26
unzip -q /content/ns_mc_gan_gi_project_phase26_colab.zip -d /content/ns_mc_gan_gi_phase26
cd /content/ns_mc_gan_gi_phase26
pip install -q -r requirements.txt
~~~

## Required cloud files

These must already exist in Drive:

- `/content/drive/MyDrive/ns_mc_gan_gi/data`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase15/imported_noleak/rademacher5_hq_noise001_colab/measurement_operator_exact.pt`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase15/imported_noleak/rademacher10_full_noise001_colab/measurement_operator_exact.pt`
- Preferably `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase16/supplementary_experiments/attribution/attribution_final.csv`

## Recommended split across Colab sessions

### Session A: PCA full

~~~bash
cd /content/ns_mc_gan_gi_phase26
bash scripts/colab_phase26_pca_full.sh
~~~

Outputs:

- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/pca_oracle_full/pca_oracle_full_results.csv`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/pca_oracle_full/pca_oracle_full_results.md`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/pca_oracle_full/pca_psnr_vs_k.png`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/pca_oracle_full/pca_ssim_vs_k.png`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/pca_oracle_full/pca_vs_current_model.png`

### Session B: architecture pilot

~~~bash
cd /content/ns_mc_gan_gi_phase26
bash scripts/colab_phase26_arch_pilot.sh
~~~

To run only the four faster pilots:

~~~bash
cd /content/ns_mc_gan_gi_phase26
CONFIGS=current_hq_rad5_pilot,nafnet_small_rad5_pilot,current_hq_scr5_pilot,nafnet_small_scr5_pilot bash scripts/colab_phase26_arch_pilot.sh
~~~

### Final aggregation after runs finish

~~~bash
cd /content/ns_mc_gan_gi_phase26
bash scripts/colab_phase26_gate_report.sh
~~~

Final outputs:

- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/arch_pilot_results.csv`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/PHASE26_GATE_DECISION.md`
- `/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase26/PHASE26_LIMIT_ARCHITECTURE_REPORT.md`

## Guardrails

- Do not mix medium pilot numbers into the strict 80-epoch main table.
- PCA oracle is a linear-prior baseline, not a deployable final method.
- Architecture pilot is planning evidence unless full no-leak training is explicitly approved.
- Rademacher runs require exact-A safe override; the scripts fail if the exact-A file is missing.
"@
Set-Content -Path $Instructions -Encoding UTF8 -Value $md

Write-Host "Project zip: $Zip"
Write-Host "SHA256 file: $Sha"
Write-Host "Runbook: $Instructions"
