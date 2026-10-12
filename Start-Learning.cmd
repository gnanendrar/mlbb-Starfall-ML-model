@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m starfall.cli learning-loop
pause
