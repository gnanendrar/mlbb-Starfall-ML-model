@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "tools\setup_training.ps1" -Mode gpu
if errorlevel 1 echo Setup failed. Read the error above before starting training.
pause
