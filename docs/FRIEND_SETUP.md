# Run on the training laptop

The 6 GB **dedicated NVIDIA** GPU is used for neural detector training. Intel's 8 GB shared GPU memory is borrowed system RAM; it does not increase NVIDIA's dedicated memory to 14 GB. Setup reports the exact GPU name and tests a tensor operation. Windows Task Manager's GPU 1 is normally CUDA GPU 0 when there is one NVIDIA card, because CUDA does not enumerate the Intel adapter.

## Download and install

1. Download this repository using GitHub **Code → Download ZIP** and extract it, or clone it. Keep the whole folder together.
2. Install **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/windows/) and the current NVIDIA graphics driver.
3. Double-click **Setup-Training.cmd**. It creates `.venv`, installs CUDA PyTorch from its official CUDA 12.8 wheel index, installs this project, downloads the verified scrcpy bundle, and checks GPU computation. Internet is needed for setup and initial pretrained weight downloads. Phone connection is optional for setup.
4. Read the final GPU report. GPU setup must succeed before requesting `--device 0`; if CUDA is unavailable, setup fails explicitly. The laptop's exact GPU, driver, temperature and power settings determine speed. Keep it plugged in during training.

The current PC's `requirements-lock.txt` records a CPU environment. **Do not use that CPU snapshot to set up the NVIDIA laptop**, and do not copy `.venv` between PCs. For a CPU installation, run `powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup_training.ps1 -Mode cpu`. If the installed driver requires a different supported build, use `-Mode gpu -CudaBuild cu126` or `cu118` after checking the [official PyTorch installer](https://pytorch.org/get-started/locally/).

## Import recorded training data

Recordings and model weights are deliberately outside Git. On the collecting PC, run:

```powershell
.\.venv\Scripts\python.exe tools\export_training_data.py
```

Transfer the generated `exports/starfall-training-data-*.zip` privately and extract its contents **into the repository folder** on the training laptop. It contains captured frames, labels, reviewed player splits, movement logs, and current detector/response candidates. Dataset paths in the archive are portable. Its manifest lists each file and SHA256; the archive checksum sits next to the ZIP. This command does not publish or upload anything. Stop phone control while making a fresh export if it is actively recording.

The current reviewed player dataset has **31 selected positive frames: 7 train / 12 validation / 12 test**, grouped by actual attempt. It is an experimental subset. Raw game frames have many unfinished labels; they are not ready for full meteor/orb training. Importing existing models does not automatically enable them in the live controller.

## GPU detector training

Reproduce the small player candidate with automatic NVIDIA selection:

```powershell
.\.venv\Scripts\python.exe tools\train_player_candidate.py --device 0
```

For a larger manually reviewed dataset, use its split YAML:

```powershell
.\.venv\Scripts\python.exe -m starfall.cli train-detector data\game\YOUR_SPLIT\dataset.yaml --device 0 --imgsz 416 --batch 4 --epochs 40
.\.venv\Scripts\python.exe -m starfall.cli evaluate-detector models\detector-experiments\YOUR_RUN\weights\best.pt data\game\YOUR_SPLIT\dataset.yaml --device 0 --split test
```

`--device auto` prefers CUDA and otherwise uses CPU; `--device 0` requires a working CUDA GPU. Start with batch 4 for 6 GB VRAM. If memory runs out, lower the batch and then image size. Evaluation must use a separate held-out attempt. Do not deploy a candidate merely because training completed. The current player candidate has material false positives and remains disabled for live control.

## Multiple offline arenas

Double-click **Train-Offline.cmd** to train PPO in **four separate synthetic arenas** sharing one model. It uses CPU because this policy is a small network over numeric features. It needs no phone or tickets. Alternatively:

```powershell
.\.venv\Scripts\python.exe -m starfall.cli train-policy --algorithm ppo --steps 200000 --n-envs 4 --vec-backend subproc --device cpu
```

`--vec-backend dummy` runs several arenas in one process; `subproc` runs separate processes with Windows-safe spawn. For a cheap arena, process overhead can make `dummy` faster. Compare wall times recorded in each experiment before raising the arena count. `--threads 2` limits training CPU threads; the best arena count depends on the laptop's CPU. Checkpoints are saved under `models/` and requested/actual steps are recorded.

**This is an engineering sandbox, not an offline copy of MLBB.** Full shooting/damage, boss phases, orbs and ultimate mechanics are not yet represented faithfully. Synthetic rewards are not real game points. More training in this sandbox does not establish stronger real gameplay. Candidates still require paired held-out simulation checks and actual phone evaluation.

## Connect the phone and play

Connect the same calibrated Android phone by USB, enable USB debugging, authorize the laptop, and open Starfall's ready page. Double-click **Start-Learning.cmd**. You handle low health, Return to Lobby and reopening Starfall; heart scanning stays disabled. The controller starts once per confirmed ready screen, collects gameplay and fits movement response between attempts. Use **Stop-Learning.cmd** or `q` in the preview to stop.

Current coordinates/templates are for this phone at **960 × 430** capture resolution. Using a different phone or UI size requires recalibration. There is only one live controller per phone. A faster laptop improves computation, while phone attempts still proceed at normal game speed.

## Verify installation

```powershell
.\.venv\Scripts\python.exe tools\check_training.py --require-cuda
.\.venv\Scripts\python.exe -m starfall.cli doctor
.\.venv\Scripts\python.exe -m pytest -q
```

Tests include small regression image fixtures, not the full recordings. UI-only fixtures omit account/friend information and are marked as modified test examples. They must never enter a training dataset.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [PyTorch installation](https://pytorch.org/get-started/locally/), [PPO CPU/vector environment guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Windows vector environment precautions](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).
