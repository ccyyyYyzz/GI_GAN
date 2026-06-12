$ErrorActionPreference = "Stop"

$EnvPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$Out = "E:/ns_mc_gan_gi/outputs_phase56_group_split_exact_null_critic"
$Bundle = "E:/ns_mc_gan_gi/outputs_phase15/imported_noleak"
$Data = "E:/ns_mc_gan_gi/data"
$Device = "cuda"
$Limit = "384"
$Batch = "32"
$Epochs = "4"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "==== Phase56: $Name ===="
    & conda run -p $EnvPath python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Phase56 step failed: $Name"
    }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

$Common = @(
    "--bundle_root", $Bundle,
    "--dataset_root", $Data,
    "--output_dir", $Out,
    "--device", $Device,
    "--limit_samples", $Limit,
    "--batch_size", $Batch,
    "--critic_epochs", $Epochs
)

Run-Step "input audit and image-ID group splits" (@("-m", "src.phase56_build_group_splits") + $Common)
Run-Step "exact P0 anchors and pair metadata" (@("-m", "src.phase56_build_exact_null_pairs") + $Common)
Run-Step "train/eval group-split exact-null critics" (@("-m", "src.phase56_train_group_split_critic") + $Common)
Run-Step "aggregate critic eval" (@("-m", "src.phase56_eval_group_split_critic") + $Common)
Run-Step "memorization and leakage diagnostics" (@("-m", "src.phase56_memorization_audit") + $Common)
Run-Step "Scr sanity audit" (@("-m", "src.phase56_scr_sanity_audit") + $Common)
Run-Step "feasible hallucination hardening" (@("-m", "src.phase56_harden_feasible_hallucination") + $Common)
Run-Step "final report" (@("-m", "src.phase56_make_report") + $Common)

Write-Host ""
Write-Host "Phase56 group-split exact-null critic repeat complete:"
Write-Host $Out
