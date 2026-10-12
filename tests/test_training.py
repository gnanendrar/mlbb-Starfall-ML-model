import json
from pathlib import Path
import shutil
import pytest
import yaml
from starfall.training import prepare_dataset


def test_relocated_dataset_repairs_an_old_machine_path(tmp_path):
    source = tmp_path / "moved" / "dataset.yaml"
    source.parent.mkdir()
    for subset in ("train", "val", "test"):
        (source.parent / "images" / subset).mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "missing-old-computer"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "player"},
            }
        )
    )
    original = source.read_bytes()
    normalized = prepare_dataset(source, tmp_path / "experiment" / "dataset.yaml")
    assert Path(yaml.safe_load(normalized.read_text())["path"]) == source.parent
    assert source.read_bytes() == original
    shutil.rmtree(source.parent / "images/test")
    with pytest.raises(FileNotFoundError, match="test"):
        prepare_dataset(source, normalized)


def test_requested_gpu_does_not_silently_fall_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from starfall.training import select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == select_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="NVIDIA"):
        select_device("0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert select_device("auto") == select_device("0") == "cuda:0"
    with pytest.raises(RuntimeError):
        select_device("1")


def test_multiple_arena_training_saves_a_loadable_candidate(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from starfall.experiments import train_policy
    from starfall.simulation import StarfallSim

    folder = train_policy(steps=128, n_envs=2, rollout_steps=32, output=tmp_path)
    report = json.loads((folder / "experiment.json").read_text())
    assert report["n_envs"] == 2 and report["device"] == "cpu"
    assert report["steps_actual"] >= 128 and not report["deployment_approved"]
    assert len(list((folder / "monitors").glob("*.monitor.csv"))) == 2
    policy = sb3.PPO.load(folder / "candidate.zip", device="cpu")
    env = StarfallSim()
    observation, _ = env.reset(seed=10000)
    action = int(policy.predict(observation, deterministic=True)[0])
    assert env.action_space.contains(action)
    env.step(action)
    env.close()
