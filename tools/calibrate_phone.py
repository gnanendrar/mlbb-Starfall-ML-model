"""One measured early-game pulse session, protected by the last-heart guard."""

import json
import time
import cv2
from starfall.android import ScrcpyStream, touch_packet
from starfall.runtime import load_config
from starfall.drag_calibration import measure_drags
from starfall.safety import HeartReader, HealthSupervisor


def main():
    cfg = load_config("configs/phone.yaml")
    cv2.setNumThreads(2)
    stream = ScrcpyStream(control=True, **cfg["capture"]).start()
    protector = None
    try:
        reader = HeartReader(cfg["health_guard"])
        frame = stream.get_frame()
        cv2.imwrite("reports/calibration-before.png", frame.image)
        reading = reader.read(frame.image)
        if reading.ready:
            # Inspected Challenge button. No repeat attempts or purchases.
            stream.control.sendall(
                touch_packet(0, 815, 348, stream.width, stream.height, pointer=1)
            )
            stream.control.sendall(
                touch_packet(1, 815, 348, stream.width, stream.height, pointer=1)
            )
        elif not reading.gameplay:
            raise RuntimeError("Phone must show Starfall ready screen or gameplay")
        deadline = time.monotonic() + 10
        seq = frame.sequence
        while time.monotonic() < deadline:
            frame = stream.get_frame(seq)
            seq = frame.sequence
            reading = reader.read(frame.image)
            if reading.hearts is not None:
                break
        if reading.hearts is None:
            raise RuntimeError("No readable hearts appeared")
        cv2.imwrite("reports/calibration-start.png", frame.image)
        print(
            json.dumps(measure_drags("configs/phone.yaml", stream), indent=2),
            flush=True,
        )
    finally:
        # If calibration could not run, preserve the game by exiting verified UI.
        if stream.latest:
            reader = HeartReader(cfg["health_guard"])
            reading = reader.read(stream.latest.image)
            if reading.gameplay:
                protector = HealthSupervisor(stream, cfg["health_guard"])
                report = protector.return_to_lobby(
                    stream.latest, reading, "calibration_cleanup"
                )
                print(json.dumps(report, indent=2), flush=True)
        cv2.imwrite("reports/calibration-after.png", stream.latest.image)
        stream.close()


if __name__ == "__main__":
    main()
