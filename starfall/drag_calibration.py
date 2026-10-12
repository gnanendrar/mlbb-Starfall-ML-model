"""Measure drag-to-displacement from the heart row which follows Layla."""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import yaml
from .android import Joystick
from .safety import HealthSupervisor
from .runtime import load_config


def heart_position(reading):
    if reading.hearts is None or not reading.centers:
        raise RuntimeError("Heart row not reliably visible for position measurement")
    return np.mean(np.array(reading.centers), axis=0) / 960


def measure_drags(config_path, stream):
    cfg = load_config(config_path)
    traces = []
    joystick = None
    guard = None
    evidence = Path("reports/drag-calibration")
    evidence.mkdir(exist_ok=True)
    try:
        joystick = Joystick(stream, cfg["joystick"]["base"], cfg["joystick"]["radius"])
        guard = HealthSupervisor(stream, cfg["health_guard"], joystick).start()
        for i, direction in enumerate(
            ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (0.5, 0), (-0.5, 0))
        ):
            guard.check()
            frame = stream.get_frame()
            reading = guard.reader.read(frame.image)
            if reading.hearts is None:
                raise RuntimeError("Unreadable health before drag")
            if reading.hearts <= 1:
                guard.return_to_lobby(frame, reading)
                guard.check()
            anchor = heart_position(reading)
            initial_hearts = reading.hearts
            cv2.imwrite(str(evidence / f"{i}-before.png"), frame.image)
            seq = frame.sequence
            sent = time.perf_counter_ns()
            joystick.drag(direction, duration=0.08, steps=4)
            samples = []
            end = time.monotonic() + 0.6
            while time.monotonic() < end:
                frame = stream.get_frame(seq)
                seq = frame.sequence
                guard.check()
                reading = guard.reader.read(frame.image)
                if reading.hearts == initial_hearts:
                    samples.append(
                        {
                            "elapsed_s": (frame.received_ns - sent) / 1e9,
                            "position": heart_position(reading).tolist(),
                            "pts_us": frame.device_pts_us,
                        }
                    )
            if len(samples) < 3:
                traces.append(
                    {
                        "direction": list(direction),
                        "anchor": anchor.tolist(),
                        "samples": samples,
                        "valid": False,
                        "reason": "insufficient_stable_samples",
                    }
                )
                raise RuntimeError("Too few stable health-position observations")
            cv2.imwrite(str(evidence / f"{i}-after.png"), frame.image)
            u = np.array(direction, float)
            mag = np.linalg.norm(u)
            unit = u / mag
            positions = np.array([s["position"] for s in samples])
            ts = np.array([s["elapsed_s"] for s in samples])
            projected = (positions - anchor) @ unit
            displacement = float(np.median(projected[-3:]))
            onset = np.flatnonzero(projected > 0.004)
            traces.append(
                {
                    "direction": list(direction),
                    "anchor": anchor.tolist(),
                    "samples": samples,
                    "magnitude": mag,
                    "displacement": displacement,
                    "onset_s": max(0.0, float(ts[onset[0]])) if len(onset) else None,
                }
            )
        full = [t["displacement"] for t in traces[:4] if t["displacement"] > 0.01]
        delay = [t["onset_s"] for t in traces if t["onset_s"] is not None]
        if len(full) < 3 or not delay:
            raise RuntimeError("Drag displacement not measured consistently")
        period = 0.22
        cfg["motion"] = {
            "measured": True,
            "kind": "drag",
            "speed": float(np.median(full) / period),
            "delay": float(np.median(delay)),
            "action_period_s": period,
            "stroke_duration_s": 0.08,
            "stroke_steps": 4,
            "stroke_displacement": float(np.median(full)),
            "displacement_samples": full,
            "source": "real_phone_visual_drag_calibration",
            "model_limitations": "Average action speed; individual gesture response and hitboxes remain approximate.",
        }
        cfg["bounds"][1] = 0.10
        Path(config_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
        return cfg["motion"]
    finally:
        try:
            if guard:
                if guard.exiting.is_set():
                    guard.finished.wait(3.5)
                elif stream.latest:
                    reading = guard.reader.read(stream.latest.image)
                    if reading.gameplay:
                        guard.return_to_lobby(
                            stream.latest, reading, "drag_calibration_complete"
                        )
                guard.close()
        finally:
            if joystick:
                joystick.close()
            Path("reports/drag-calibration.json").write_text(
                json.dumps(traces, indent=2)
            )
