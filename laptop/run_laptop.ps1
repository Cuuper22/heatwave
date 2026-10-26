# Creates a virtual environment, installs Heatwave, and runs a 20-minute DRY-RUN session.
# Run from anywhere:  powershell -ExecutionPolicy RemoteSigned -File laptop\run_laptop.ps1
# Dry run only prints the nvidia-smi commands the controller would run; it changes nothing.
# For live control, start an administrator PowerShell and use the command in laptop\README.md.
param(
    [double]$Minutes = 20,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv-laptop"
if (-not (Test-Path $venv)) {
    # --system-site-packages keeps the CUDA build of torch that is already installed.
    & $Python -m venv $venv --system-site-packages
}
$py = Join-Path $venv "Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -e . nvidia-ml-py

& $py -c "import torch, sys; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "torch cannot see the GPU. Install a CUDA build of torch first (see laptop\README.md)."
}

& $py -m heatwave.laptop.run --minutes $Minutes --dry-run
Write-Host "Logs are in laptop\logs\. Scores are in results\laptop\."
