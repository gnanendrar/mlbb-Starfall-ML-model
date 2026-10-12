param(
    [ValidateSet('gpu', 'cpu')][string]$Mode = 'gpu',
    [ValidateSet('cu128', 'cu126', 'cu118')][string]$CudaBuild = 'cu128'
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $taskPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw 'Install 64-bit Python 3.11 from python.org, then run setup again.'
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $taskPython)) {
        throw 'Could not create the environment. Install 64-bit Python 3.11 and retry.'
    }
}

function Invoke-TaskPython {
    param([string[]]$Arguments)
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup step failed (exit $LASTEXITCODE)." }
}

Invoke-TaskPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
$taskBuild = if ($Mode -eq 'gpu') { $CudaBuild } else { 'cpu' }
Invoke-TaskPython -Arguments @('-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'torch', 'torchvision', '--index-url', "https://download.pytorch.org/whl/$taskBuild")
Invoke-TaskPython -Arguments @('-m', 'pip', 'install', '-e', '.[test,learn]')
Invoke-TaskPython -Arguments @('tools\bootstrap.py')
$taskCheckArgs = @('tools\check_training.py')
if ($Mode -eq 'gpu') { $taskCheckArgs += '--require-cuda' }
Invoke-TaskPython -Arguments $taskCheckArgs
Invoke-TaskPython -Arguments @('-m', 'starfall.cli', 'doctor')
Write-Host 'Setup finished. Read docs/FRIEND_SETUP.md for offline training and phone control.'
