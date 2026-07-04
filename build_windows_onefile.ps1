$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvDir = Join-Path $Root ".venv-build"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    if ($env:PYTHON) {
        & $env:PYTHON -m venv $VenvDir
    }
    else {
        & py -3 -m venv $VenvDir
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements-packaging.txt")

& $VenvPython -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --name GameJamCamera `
    --hidden-import cv2 `
    --hidden-import numpy `
    --hidden-import tkinter `
    --add-data "process_multi_live\live_multi_red_tracking.py;process_multi_live" `
    --add-data "process_live\live_camera_720_clahe.py;process_live" `
    --add-data "process_2487\stabilize_space_mask_preprocess.py;process_2487" `
    start_camera_pipeline.py

Write-Host ""
Write-Host "Built: $Root\dist\GameJamCamera.exe"
Write-Host "Give that single .exe file to other Windows users."
