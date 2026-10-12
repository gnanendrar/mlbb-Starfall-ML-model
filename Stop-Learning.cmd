@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True); Path('runs/stop-request').touch()"
echo Learning loop will stop and release controls.
