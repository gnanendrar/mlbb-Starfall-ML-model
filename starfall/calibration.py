"""Measure UI geometry and motion from visual detections and controlled pulses."""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import yaml
from .android import Adb, ScrcpyStream, Joystick
from .perception import create_detector
from .runtime import load_config
from .safety import HealthSupervisor


def select_geometry(config_path, serial=None):
    config = load_config(config_path)
    stream = ScrcpyStream(Adb(serial), **config["capture"]).start()
    try:
        image = stream.get_frame().image
        h, w = image.shape[:2]
        scale = max(h, w)
        print("Select the playable arena excluding HUD and joystick; Enter to confirm.")
        x, y, bw, bh = cv2.selectROI("Arena", image, False)
        if not bw or not bh:
            raise ValueError("No arena selected")
        config["bounds"] = [x / scale, y / scale, (x + bw) / scale, (y + bh) / scale]
        print(
            "Select a square around the joystick: center becomes touch base, half its smaller side becomes radius."
        )
        jx, jy, jw, jh = cv2.selectROI("Joystick base and radius", image, False)
        if not jw or not jh:
            raise ValueError("No joystick selected")
        config["joystick"].update(
            base=[jx + jw / 2, jy + jh / 2],
            radius=min(jw, jh) / 2,
            frame_size=[w, h],
            calibrated=True,
            source="manual_screen_measurement",
        )
        print(
            "Select player template, then any observed hazard/item templates. Empty ROI ends template collection."
        )
        folder = Path("data/templates")
        folder.mkdir(parents=True, exist_ok=True)
        specs = []
        while True:
            tx, ty, tw, th = cv2.selectROI("Observed sprite template", image, False)
            if not tw or not th:
                break
            kind = input("Role (player/hazard/item): ").strip()
            if kind not in ("player", "hazard", "item"):
                raise ValueError("Invalid role")
            label = input("Observed object name: ").strip()
            if not label or not label.replace("_", "").isalnum():
                raise ValueError("Use a simple alphanumeric name")
            path = folder / f"{label}-{time.time_ns()}.png"
            cv2.imwrite(str(path), image[ty : ty + th, tx : tx + tw])
            specs.append(
                dict(
                    path=str(path),
                    label=label,
                    kind=kind,
                    threshold=0.8,
                    max_objects=1 if kind == "player" else 30,
                )
            )
        config["detector"] = {"type": "templates", "templates": specs}
        config["classes"] = sorted({s["label"] for s in specs})
        config["detector_validated"] = False
        Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False))
    finally:
        stream.close()
        cv2.destroyAllWindows()
    return config


def measure_motion(config_path, serial=None, hold=0.4, stream=None):
    config = load_config(config_path)
    if not config["joystick"]["calibrated"]:
        raise ValueError("Measure joystick geometry first")
    detector = create_detector(config)
    owns_stream = stream is None
    stream = (
        stream or ScrcpyStream(Adb(serial), control=True, **config["capture"]).start()
    )
    joystick = None
    supervisor = None
    traces = []
    try:
        if [stream.width, stream.height] != config["joystick"]["frame_size"]:
            raise ValueError("Geometry resolution mismatch")
        joystick = Joystick(
            stream,
            config["joystick"]["base"],
            config["joystick"]["radius"],
            refresh_motion=True,
        )
        supervisor = HealthSupervisor(stream, config["health_guard"], joystick).start()
        initial = supervisor.reader.read(stream.get_frame().image)
        if initial.hearts is None:
            raise RuntimeError("Visible health is required before joystick calibration")
        if initial.hearts <= config["health_guard"]["exit_at_hearts"]:
            supervisor.return_to_lobby(stream.latest, initial)
            supervisor.check()
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.5, 0), (-0.5, 0)):
            first = stream.get_frame()
            supervisor.check()
            seq = first.sequence
            players = [d for d in detector.detect(first.image) if d.kind == "player"]
            if len(players) != 1:
                raise RuntimeError("Need one reliably detected player for calibration")
            anchor = players[0].position.copy()
            sent = time.perf_counter_ns()
            samples = []
            while (time.perf_counter_ns() - sent) / 1e9 < hold:
                supervisor.check()
                joystick.move(direction)
                frame = stream.get_frame(seq)
                supervisor.check()
                seq = frame.sequence
                observed = [
                    d for d in detector.detect(frame.image) if d.kind == "player"
                ]
                if len(observed) == 1:
                    samples.append(
                        {
                            "elapsed_s": (frame.received_ns - sent) / 1e9,
                            "position": observed[0].position.tolist(),
                            "pts_us": frame.device_pts_us,
                        }
                    )
            joystick.release()
            time.sleep(0.15)
            traces.append(
                dict(direction=list(direction), anchor=anchor.tolist(), samples=samples)
            )
        estimates = []
        delays = []
        for trace in traces:
            samples = trace["samples"]
            if len(samples) < 4:
                continue
            ts = np.array([s["elapsed_s"] for s in samples])
            positions = np.array([s["position"] for s in samples])
            u = np.array(trace["direction"], dtype=float)
            mag = np.linalg.norm(u)
            displacement = (positions - np.array(trace["anchor"])) @ (u / mag)
            onset = np.where(displacement > 0.005)[0]
            if len(onset):
                delays.append(max(0.0, float(ts[onset[0]])))
            tail = ts > hold * 0.4
            speed = (
                float(np.polyfit(ts[tail], displacement[tail], 1)[0])
                if tail.sum() >= 3
                else 0.0
            )
            if speed > 0.01 and mag > 0.9:
                estimates.append(speed)
        if len(estimates) < 3 or not delays:
            raise RuntimeError(
                "Insufficient measurable motion; boundary/animation/template may be interfering. Raw traces retained."
            )
        config["motion"] = {
            "measured": True,
            "speed": float(np.median(estimates)),
            "delay": float(np.median(delays)),
            "speed_samples": estimates,
            "delay_samples": delays,
            "source": "visual_pulse_calibration",
        }
        Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False))
    finally:
        try:
            if supervisor:
                if supervisor.exiting.is_set():
                    supervisor.finished.wait(3.5)
                elif stream.latest:
                    reading = supervisor.reader.read(stream.latest.image)
                    if reading.gameplay:
                        supervisor.return_to_lobby(
                            stream.latest, reading, "calibration_complete"
                        )
                supervisor.close()
        finally:
            if joystick:
                joystick.close()
            if owns_stream:
                stream.close()
            Path("reports/motion-calibration.json").write_text(
                json.dumps(traces, indent=2)
            )
    return config["motion"]
