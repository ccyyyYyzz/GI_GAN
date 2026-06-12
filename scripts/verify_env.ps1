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

& $Python -m src.verify_env --dataset_root E:/ns_mc_gan_gi/data --output_dir E:/ns_mc_gan_gi/outputs
