"""Simulation experiments and real-frame detector timing, with explicit provenance."""

from datetime import datetime, timezone
from pathlib import Path
import json
import time
import cv2
import numpy as np
from .simulation import StarfallSim
from .planning import PredictivePlanner
from .types import DIRECTIONS
from .logging import percentiles


def evaluate(
    policy="mpc",
    episodes=30,
    seed=10000,
    model_path=None,
    algorithm="ppo",
    duration=30,
    randomize=True,
):
    env = StarfallSim(duration=duration, randomize=randomize)
    model = None
    if model_path:
        from stable_baselines3 import PPO, DQN

        model = (PPO if algorithm == "ppo" else DQN).load(model_path, device="cpu")
    results = []
    timings = []
    start = time.perf_counter()
    for episode in range(episodes):
        rng = np.random.default_rng(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        planner = PredictivePlanner(
            speed=env.speed,
            delay=0.0,
            uncertainty_scale=0.0,
            item_values={"item": 1.0, "shield": 1.5},
        )
        while True:
            t = time.perf_counter_ns()
            if model is not None:
                action = int(model.predict(obs, deterministic=True)[0])
                if policy == "shielded":
                    action, _ = planner.shield(env.world(), action)
            elif policy == "mpc":
                action = planner.choose(env.world()).action
            elif policy == "random":
                action = int(rng.integers(9))
            elif policy == "nearest":
                hazards = env.hazards
                if hazards:
                    h = min(
                        hazards,
                        key=lambda h: np.linalg.norm(h.position - env.player.position),
                    )
                    away = env.player.position - h.position
                    action = int(np.argmax(DIRECTIONS @ away))
                else:
                    action = 0
            elif policy == "stay":
                action = 0
            else:
                raise ValueError("Unknown policy")
            timings.append((time.perf_counter_ns() - t) / 1e6)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                results.append(dict(info, seed=seed + episode))
                break
    scores = [r["score"] for r in results]
    elapsed = time.perf_counter() - start
    return {
        "provenance": "synthetic_unvalidated_mechanics",
        "policy": policy,
        "algorithm": algorithm if model else None,
        "model": model_path,
        "episodes": episodes,
        "seed": seed,
        "duration_s": duration,
        "randomize": randomize,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "highest_score": max(scores),
        "mean_survival_s": float(np.mean([r["survival_s"] for r in results])),
        "mean_collections": float(np.mean([r["collections"] for r in results])),
        "collision_rate": float(np.mean([r["collisions"] for r in results])),
        "decision_latency": percentiles(timings),
        "wall_s": elapsed,
        "runs": results,
    }


def train_policy(
    algorithm="ppo",
    steps=20000,
    seed=42,
    output="models",
    device="cpu",
    n_envs=1,
    vec_backend="dummy",
    threads=2,
    rollout_steps=512,
):
    import torch
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .training import select_device

    if algorithm not in ("ppo", "dqn"):
        raise ValueError("Algorithm must be ppo or dqn")
    if n_envs < 1 or threads < 1 or steps < 1 or rollout_steps < 2:
        raise ValueError("Training sizes/threads must be positive; rollout steps >= 2")
    if vec_backend not in ("dummy", "subproc"):
        raise ValueError("Vector backend must be dummy or subproc")
    selected = select_device(device)
    torch.set_num_threads(threads)
    probe = StarfallSim()
    try:
        check_env(probe, warn=True)
    finally:
        probe.close()
    identity = f"{algorithm}-synthetic-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder = Path(output) / identity
    folder.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(
        StarfallSim,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(folder / "monitors"),
        monitor_kwargs=dict(info_keywords=("score", "collections", "collisions")),
        vec_env_cls=SubprocVecEnv if vec_backend == "subproc" else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if vec_backend == "subproc" else {},
    )
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=selected,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    print(
        f"Training {algorithm} in {n_envs} synthetic arenas ({vec_backend}, {selected}).",
        flush=True,
    )
    try:
        if algorithm == "ppo":
            model = PPO(
                **common,
                n_steps=rollout_steps,
                batch_size=64,
                n_epochs=5,
                learning_rate=3e-4,
            )
        else:
            model = DQN(
                **common,
                buffer_size=30000,
                learning_starts=512,
                batch_size=64,
                train_freq=4,
                gradient_steps=1,
                learning_rate=3e-4,
                target_update_interval=500,
            )
        start = time.perf_counter()
        model.learn(
            total_timesteps=steps,
            callback=CheckpointCallback(
                save_freq=max(1, max(1000, steps // 4) // n_envs),
                save_path=str(folder),
                name_prefix="checkpoint",
            ),
        )
        model.save(str(folder / "candidate"))
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    config = {
        "id": identity,
        "algorithm": algorithm,
        "seed": seed,
        "steps_requested": steps,
        "steps_actual": model.num_timesteps,
        "train_wall_s": elapsed,
        "device_requested": device,
        "device": str(model.device),
        "n_envs": n_envs,
        "vec_backend": vec_backend,
        "torch_threads": threads,
        "rollout_steps": rollout_steps if algorithm == "ppo" else None,
        "provenance": "synthetic_unvalidated_mechanics",
        "deployment_approved": False,
        "observations": "features-v1-132",
        "actions": "9-directions-v1",
    }
    config["hyperparameters"] = {
        "net_arch": [64, 64],
        "learning_rate": 3e-4,
        "batch_size": 64,
    }
    (folder / "experiment.json").write_text(json.dumps(config, indent=2))
    return folder


def compare(candidate, baseline, samples=2000):
    if [r["seed"] for r in candidate["runs"]] != [r["seed"] for r in baseline["runs"]]:
        raise ValueError("Compare paired, held-out seeds")
    if (
        candidate["duration_s"] != baseline["duration_s"]
        or candidate["randomize"] != baseline["randomize"]
    ):
        raise ValueError("Compare identical scenario configurations")
    delta = np.array(
        [c["score"] - b["score"] for c, b in zip(candidate["runs"], baseline["runs"])]
    )
    rng = np.random.default_rng(42)
    means = np.mean(rng.choice(delta, (samples, len(delta))), axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    passed = bool(
        lo > 0
        and candidate["collision_rate"] <= baseline["collision_rate"]
        and len(delta) >= 20
    )
    return {
        "paired_mean_score_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(lo), float(hi)],
        "simulation_gate_passed": passed,
        "real_deployment_approved": False,
        "reason": "Simulation-only evidence. Real detector, controls and paired game evaluation remain necessary.",
    }


def benchmark_detectors(
    images,
    output="reports/detector-benchmark.json",
    repeats=10,
    sizes=(320, 416, 640),
    export_openvino=False,
):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    frames = [cv2.imread(str(p)) for p in images]
    if not frames or any(f is None for f in frames):
        raise ValueError("Benchmark requires readable representative frames")
    results = []
    for name in ("yolo26n.pt", "yolo11n.pt"):
        model = YOLO(name)
        variants = ["pytorch"]
        exported = None
        if export_openvino:
            existing = Path(name).with_suffix("").name + "_openvino_model"
            exported = (
                existing
                if Path(existing).exists()
                else model.export(format="openvino", imgsz=416, dynamic=True)
            )
            variants.append("openvino_cpu")
        for backend in variants:
            for size in sizes:
                print(f"Timing {name} {backend} size={size}", flush=True)
                if backend == "pytorch":
                    kwargs = dict(imgsz=size, device="cpu", verbose=False, rect=False)

                    def infer(image):
                        return model.predict(image, **kwargs)
                else:
                    from .openvino_backend import OpenvinoDetector

                    runner = OpenvinoDetector(
                        exported, imgsz=size, device="CPU", confidence=0.25
                    )
                    infer = runner.boxes
                for _ in range(3):
                    infer(frames[0])
                    torch.set_num_threads(2)
                lat = []
                for j in range(repeats):
                    started = time.perf_counter_ns()
                    infer(frames[j % len(frames)])
                    lat.append((time.perf_counter_ns() - started) / 1e6)
                results.append(
                    {
                        "model": name,
                        "backend": backend,
                        "imgsz": size,
                        "warmup": 3,
                        "samples": repeats,
                        "latency": percentiles(lat),
                    }
                )
                Path(output).write_text(
                    json.dumps(
                        {"partial": True, "accuracy": None, "results": results},
                        indent=2,
                    )
                )
    report = {
        "accuracy": None,
        "note": "COCO models, real screen timing only; no validated game detections or accuracy evidence.",
        "images": list(map(str, images)),
        "torch_threads": 2,
        "results": results,
    }
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def train_detector(
    dataset,
    model="yolo26n.pt",
    epochs=40,
    imgsz=416,
    seed=42,
    device="auto",
    batch=4,
    workers=0,
    threads=2,
):
    import torch
    from ultralytics import YOLO
    from .training import select_device, prepare_dataset

    selected = select_device(device)
    torch.set_num_threads(threads)
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path("models/detector-experiments") / identity
    data = prepare_dataset(dataset, folder / "dataset.yaml")
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "source_dataset": str(dataset),
                "device_requested": device,
                "device": selected,
                "batch": batch,
                "imgsz": imgsz,
                "seed": seed,
                "epochs": epochs,
                "model": model,
                "deployment_approved": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model = YOLO(model)
    return model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=selected,
        seed=seed,
        deterministic=True,
        workers=workers,
        project=str(Path("models/detector-experiments").resolve()),
        name=identity,
        exist_ok=True,
    )
