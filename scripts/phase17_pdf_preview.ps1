$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Python = "E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$env:PYTHONPATH = $Repo

& $Python -m src.phase17_make_pdf_preview

$PreviewDir = "E:\ns_mc_gan_gi\outputs_phase17\pdf_preview"
Push-Location $PreviewDir
& xelatex -interaction=nonstopmode -halt-on-error phase17_pdf_preview.tex
& xelatex -interaction=nonstopmode -halt-on-error phase17_pdf_preview.tex
Pop-Location

$PageDir = Join-Path $PreviewDir "page_previews"
New-Item -ItemType Directory -Force -Path $PageDir | Out-Null
Remove-Item -Path (Join-Path $PageDir "*.png") -ErrorAction SilentlyContinue
& pdftoppm -png -r 160 (Join-Path $PreviewDir "phase17_pdf_preview.pdf") (Join-Path $PageDir "page")

Write-Host "PDF: $(Join-Path $PreviewDir 'phase17_pdf_preview.pdf')"
Write-Host "Page previews: $PageDir"
