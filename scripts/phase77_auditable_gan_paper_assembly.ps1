$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "E:/ns_mc_gan_gi_code"
$Python = Join-Path $env:USERPROFILE ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $Python -m py_compile src/phase77_auditable_gan_paper_assembly.py
& $Python src/phase77_auditable_gan_paper_assembly.py
