$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UploadDir = "E:/ns_mc_gan_gi/colab_upload"
$Stage = Join-Path $UploadDir "_staging_ns_mc_gan_gi_project"
$Zip = Join-Path $UploadDir "ns_mc_gan_gi_project_phase14_colab.zip"
$Sha = "$Zip.sha256.txt"
$InstructionsDir = "E:/ns_mc_gan_gi/outputs_phase14"
$Instructions = Join-Path $InstructionsDir "COLAB_RUN_INSTRUCTIONS.md"

New-Item -ItemType Directory -Force -Path $UploadDir, $InstructionsDir | Out-Null
if (Test-Path $Stage) {
    $resolvedStage = (Resolve-Path $Stage).Path
    if (-not $resolvedStage.StartsWith((Resolve-Path $UploadDir).Path)) {
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
# Phase 14 Colab Run Instructions

## Upload

Upload this zip to Colab:

`$Zip`

SHA256:

`$hash`

Open notebook:

`colab/phase14_colab_5pct_hq.ipynb`

If Google Drive is full or cannot mount, use this notebook instead:

`colab/phase14_colab_5pct_hq_no_drive.ipynb`

## Run in Colab

### Drive workflow

1. Mount Google Drive.
2. Upload or copy the project zip into `/content`.
3. Run the notebook cells in order.
4. Train both configs:
   - `configs/phase14_colab/rademacher5_hq_noise001_colab.yaml`
   - `configs/phase14_colab/scrambled_hadamard5_hq_noise001_colab.yaml`
5. Run the summary and packaging cells.
6. Download `/content/phase14_colab_outputs.zip` and `/content/phase14_colab_outputs_manifest.json`.

### No-Drive workflow

1. Open `colab/phase14_colab_5pct_hq_no_drive.ipynb`.
2. Do not mount Drive.
3. Upload `ns_mc_gan_gi_project_phase14_colab.zip` from your PC when the first cell asks.
4. The notebook rewrites runtime configs to:
   - data: `/content/ns_mc_gan_gi_data`
   - outputs: `/content/outputs_phase14_colab`
5. Download:
   - `/content/phase14_colab_outputs_manifest.json`
   - `/content/phase14_download_parts/phase14_split_manifest.json`
   - all `/content/phase14_download_parts/phase14_colab_outputs.zip.part_*`
6. Keep all parts before stopping the runtime. `/content` disappears after the runtime is gone.

## Bring results back to this PC

For Drive workflow, put the downloaded zip in:

`E:/ns_mc_gan_gi/colab_downloads`

For No-Drive split workflow, put all downloaded parts and manifests in:

`E:/ns_mc_gan_gi/colab_downloads`

Then merge parts into `phase14_colab_outputs.zip` before import:

```powershell
Get-Content E:/ns_mc_gan_gi/colab_downloads/phase14_colab_outputs.zip.part_* -Encoding Byte -ReadCount 0 |
  Set-Content E:/ns_mc_gan_gi/colab_downloads/phase14_colab_outputs.zip -Encoding Byte
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase14_colab_import.ps1
```

## Important

No local Phase 14 5% large training was started by this package. Local scripts only compile, package, import, aggregate, and report.
"@
Set-Content -Path $Instructions -Encoding UTF8 -Value $md
Write-Host "Project zip: $Zip"
Write-Host "SHA256 file: $Sha"
Write-Host "Instructions: $Instructions"
