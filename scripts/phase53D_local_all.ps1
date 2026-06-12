$ErrorActionPreference = "Stop"

$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Out = "E:/ns_mc_gan_gi/outputs_phase53D_local_preflight"
$Bundle = "E:/ns_mc_gan_gi/outputs_phase15/imported_noleak"
$Data = "E:/ns_mc_gan_gi/data"
$Phase48 = "E:/ns_mc_gan_gi/outputs_phase48_49_colab_import"
$Phase51A = "E:/ns_mc_gan_gi/outputs_phase51A_colab_import"
$Device = "cuda"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "==== Phase53D: $Name ===="
    & conda run -p $EnvPath python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Phase53D step failed: $Name"
    }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

Run-Step "exact projector checks" @(
    "-m", "src.phase53D_exact_projector",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--device", $Device,
    "--limit_samples", "128"
)

Run-Step "E1-mini anchor/null pretest" @(
    "-m", "src.phase53D_anchor_info_pretest",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--device", $Device,
    "--limit_samples", "1024",
    "--batch_size", "32",
    "--num_workers", "2"
)

Run-Step "feasible hallucination figure" @(
    "-m", "src.phase53D_feasible_hallucination",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--device", $Device,
    "--limit_samples", "96",
    "--examples_per_task", "4"
)

Run-Step "residual shortcut audit" @(
    "-m", "src.phase53D_shortcut_audit",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--device", $Device,
    "--limit_samples", "512",
    "--batch_size", "32",
    "--num_workers", "2"
)

Run-Step "post-hoc certificate sweep" @(
    "-m", "src.phase53D_posthoc_certificate_sweep",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--phase48_root", $Phase48,
    "--phase51A_root", $Phase51A,
    "--device", $Device,
    "--limit_samples", "64",
    "--batch_size", "16",
    "--num_workers", "2"
)

Run-Step "aggregate reports" @(
    "-m", "src.phase53D_aggregate",
    "--bundle_root", $Bundle,
    "--output_dir", $Out,
    "--dataset_root", $Data,
    "--phase48_root", $Phase48,
    "--phase51A_root", $Phase51A,
    "--device", $Device
)

Write-Host ""
Write-Host "Phase53D local preflight complete:"
Write-Host $Out
