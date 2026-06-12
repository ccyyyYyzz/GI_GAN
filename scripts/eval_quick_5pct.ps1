$ErrorActionPreference = "Stop"

$Python = $env:NS_MCGAN_PYTHON
if (-not $Python) {
    $CondaPython = "D:/Anacondar/anaconda3/python.exe"
    if (Test-Path $CondaPython) {
        $Python = $CondaPython
    } else {
        $Python = "python"
    }
}

& $Python -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs/quick_5pct/best_ssim.pt --config configs/quick_train_5pct.yaml --device cuda
