$ErrorActionPreference = "Stop"

$Python = $env:NS_MCGAN_PYTHON
if (-not $Python) {
    $CondaPython = "D:/Anacondar/anaconda3/python.exe"
    if (Test-Path $CondaPython) { $Python = $CondaPython } else { $Python = "python" }
}

& $Python -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs/ablation_5pct_no_null/best_ssim.pt --config configs/ablation_5pct_no_null.yaml --device cuda
& $Python -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs/ablation_5pct_no_dc/best_ssim.pt --config configs/ablation_5pct_no_dc.yaml --device cuda
& $Python -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs/ablation_5pct_no_adv/best_ssim.pt --config configs/ablation_5pct_no_adv.yaml --device cuda
