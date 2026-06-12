$ErrorActionPreference = "Stop"

$Python = $env:NS_MCGAN_PYTHON
if (-not $Python) {
    $CondaPython = "D:/Anacondar/anaconda3/python.exe"
    if (Test-Path $CondaPython) { $Python = $CondaPython } else { $Python = "python" }
}

& $Python -m src.train --config configs/ablation_5pct_no_null.yaml --device cuda
& $Python -m src.train --config configs/ablation_5pct_no_dc.yaml --device cuda
& $Python -m src.train --config configs/ablation_5pct_no_adv.yaml --device cuda
