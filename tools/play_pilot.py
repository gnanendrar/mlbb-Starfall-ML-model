"""One bounded, health-protected predictive-control pilot on the real phone."""

import time
import cv2
import json
from pathlib import Path
from starfall.android import ScrcpyStream, touch_packet
from starfall.runtime import load_config, run_live
from starfall.safety import HeartReader, HealthSupervisor


def main():
    cv2.setNumThreads(2)
    cfg = load_config("configs/phone.yaml")
    stream = ScrcpyStream(control=True, **cfg["capture"]).start()
    guard = None
    try:
        reader = HeartReader(cfg["health_guard"])
        frame = stream.get_frame()
        state = reader.read(frame.image)
        if state.ready:
            stream.control.sendall(
                touch_packet(0, 815, 348, stream.width, stream.height, pointer=1)
            )
            time.sleep(0.03)
            stream.control.sendall(
                touch_packet(1, 815, 348, stream.width, stream.height, pointer=1)
            )
        elif not state.gameplay:
            raise RuntimeError("Starfall ready screen is required")
        seq = frame.sequence
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            frame = stream.get_frame(seq)
            seq = frame.sequence
            state = reader.read(frame.image)
            if state.hearts is not None:
                break
        if state.hearts is None:
            raise RuntimeError("Health unreadable before pilot")
        if state.hearts <= 1:
            guard = HealthSupervisor(stream, cfg["health_guard"])
            guard.return_to_lobby(frame, state)
            return
        path, summary = run_live(
            "configs/phone.yaml",
            seconds=20,
            control=True,
            collect=True,
            preview=False,
            provided_stream=stream,
        )
        Path("reports/latest-pilot.json").write_text(
            json.dumps({"run_directory": str(path), "summary": summary}, indent=2)
        )
        print(
            json.dumps({"run_directory": str(path), "summary": summary}, indent=2),
            flush=True,
        )
    finally:
        if not stream.stop_event.is_set():
            state = HeartReader(cfg["health_guard"]).read(stream.latest.image)
            if state.gameplay:
                guard = HealthSupervisor(stream, cfg["health_guard"])
                guard.return_to_lobby(stream.latest, state, "pilot_cleanup")
            stream.close()


if __name__ == "__main__":
    main()
