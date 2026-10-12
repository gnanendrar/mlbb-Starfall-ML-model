"""One passive real-game observation session; always protect the last heart."""

from pathlib import Path
import json
import time
import cv2
from starfall.android import ScrcpyStream, touch_packet
from starfall.dataset import Collector
from starfall.runtime import load_config
from starfall.safety import HealthSupervisor


def main():
    cfg = load_config("configs/phone.yaml")
    cv2.setNumThreads(2)
    stream = ScrcpyStream(control=True, **cfg["capture"]).start()
    guard = HealthSupervisor(stream, cfg["health_guard"]).start()
    collector = Collector(interval=0.35, session=f"real-guarded-{time.time_ns()}")
    rows = []
    reason = "duration_limit"
    started = time.monotonic()
    try:
        first = stream.get_frame()
        state = guard.reader.read(first.image)
        cv2.imwrite("reports/guarded-capture-before.png", first.image)
        if state.ready:
            stream.control.sendall(
                touch_packet(0, 815, 348, stream.width, stream.height, pointer=1)
            )
            stream.control.sendall(
                touch_packet(1, 815, 348, stream.width, stream.height, pointer=1)
            )
        elif not state.gameplay:
            raise RuntimeError("Reopen Starfall ready page; no game started")
        seq = first.sequence
        while time.monotonic() - started < 40:
            frame = stream.get_frame(seq)
            seq = frame.sequence
            if guard.exiting.is_set():
                reason = "health_protection"
                guard.finished.wait(3.5)
                break
            if guard.error:
                raise RuntimeError(str(guard.error))
            collector.offer(frame)
            state = guard.latest
            rows.append(
                {
                    "sequence": seq,
                    "pts_us": frame.device_pts_us,
                    "hearts": state.hearts if state else None,
                    "receive_to_monitor_ms": (
                        time.perf_counter_ns() - frame.received_ns
                    )
                    / 1e6,
                }
            )
    finally:
        if not guard.exiting.is_set() and stream.latest:
            state = guard.reader.read(stream.latest.image)
            if state.gameplay:
                guard.return_to_lobby(stream.latest, state, "observation_complete")
        guard.close()
        cv2.imwrite("reports/guarded-capture-after.png", stream.latest.image)
        stream.close()
        report = {
            "source": "real_phone",
            "policy": "passive_observation_no_movement",
            "reason": reason,
            "session": collector.session,
            "sampled_frames_in_session": sum(
                r["session"] == collector.session for r in collector.records
            ),
            "score": None,
            "heart_exit": guard.report,
            "rows": rows,
            "guard_error": str(guard.error) if guard.error else None,
        }
        Path("reports/guarded-capture.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
