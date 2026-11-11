$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
}
$python = Join-Path $project '.venv\Scripts\python.exe'
& $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA torch installation failed' }
& $python -m pip install -r requirements-v2.txt tqdm
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency verification failed' }
& $python -m pip freeze | Set-Content -LiteralPath requirements-v2-lock.txt -Encoding utf8
