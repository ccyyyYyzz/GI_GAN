$ErrorActionPreference = "Stop"

$Repo = "E:\ns_mc_gan_gi_code"
$Python = "D:\Anacondar\anaconda3\python.exe"
$Out = "E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion"
$LogDir = Join-Path $Out "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $Repo

foreach ($Seed in 4..10) {
    $stdout = Join-Path $LogDir ("phase83_seed{0:D2}_pair.stdout.log" -f $Seed)
    $stderr = Join-Path $LogDir ("phase83_seed{0:D2}_pair.stderr.log" -f $Seed)
    "phase83_start seed=$Seed $(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $LogDir "phase83_driver.log")
    & $Python -m src.phase81_96px_rad5_paper_completion pair --seed $Seed > $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        "phase83_fail seed=$Seed exit=$LASTEXITCODE $(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $LogDir "phase83_driver.log")
        throw "seed $Seed pair failed with exit code $LASTEXITCODE"
    }
    "phase83_done seed=$Seed $(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $LogDir "phase83_driver.log")
}

$aggOut = Join-Path $LogDir "phase83_aggregate.stdout.log"
$aggErr = Join-Path $LogDir "phase83_aggregate.stderr.log"
& $Python -m src.phase81_96px_rad5_paper_completion aggregate > $aggOut 2> $aggErr
if ($LASTEXITCODE -ne 0) {
    "phase83_fail aggregate exit=$LASTEXITCODE $(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $LogDir "phase83_driver.log")
    throw "aggregate failed with exit code $LASTEXITCODE"
}
"phase83_done aggregate $(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $LogDir "phase83_driver.log")
