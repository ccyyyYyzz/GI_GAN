$ErrorActionPreference = "Stop"
param(
    [string]$DownloadDir = "E:/ns_mc_gan_gi/colab_downloads",
    [string]$OutputZip = "E:/ns_mc_gan_gi/colab_downloads/phase14_colab_outputs.zip"
)

$dir = Resolve-Path $DownloadDir
$parts = Get-ChildItem -LiteralPath $dir -File |
    Where-Object { $_.Name -match '^phase14_colab_outputs\.zip(\s\(\d+\))?\.part_\d{3}$|^phase14_colab_outputs\.zip\.part_\d{3}(\s\(\d+\))?$' } |
    Sort-Object {
        if ($_.Name -match 'part_(\d{3})') { [int]$Matches[1] } else { 999999 }
    }

if (-not $parts -or $parts.Count -eq 0) {
    throw "No phase14_colab_outputs.zip.part_### files found in $DownloadDir"
}

if (Test-Path $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}

$out = [System.IO.File]::Open($OutputZip, [System.IO.FileMode]::CreateNew)
try {
    foreach ($part in $parts) {
        Write-Host "Appending $($part.Name) $([math]::Round($part.Length / 1MB, 2)) MB"
        $in = [System.IO.File]::OpenRead($part.FullName)
        try {
            $in.CopyTo($out)
        }
        finally {
            $in.Dispose()
        }
    }
}
finally {
    $out.Dispose()
}

$hash = (Get-FileHash -Algorithm SHA256 -Path $OutputZip).Hash.ToLowerInvariant()
Write-Host "Merged zip: $OutputZip"
Write-Host "SHA256: $hash"
