@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Training.cmd first.
  pause
  exit /b 1
)
echo Training in four synthetic arenas. These scores are not real game points.
".venv\Scripts\python.exe" -m starfall.cli train-policy --algorithm ppo --steps 200000 --n-envs 4 --vec-backend subproc --device cpu --threads 2
pause
