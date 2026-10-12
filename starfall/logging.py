from datetime import datetime, timezone
from pathlib import Path
import json
import time
import uuid
from collections import deque
import numpy as np
import psutil


def percentiles(values):
    if not values:
        return None
    return dict(
        zip(
            ("p50_ms", "p95_ms", "p99_ms"),
            map(float, np.percentile(values, [50, 95, 99])),
        )
    )


class RunLog:
    def __init__(self, config, mode="live", root="runs"):
        self.id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.path = Path(root) / self.id
        self.path.mkdir(parents=True)
        self.meta = {
            "run_id": self.id,
            "mode": mode,
            "config": config,
            "agent_version": "0.1.0",
            "score": None,
            "health": None,
            "collision_events": None,
            "powerups_collected": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.start = time.perf_counter()
        self.file = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self.latencies = {}
        self.count = self.dropped = 0

    def event(self, event):
        self.file.write(json.dumps(event, allow_nan=False) + "\n")
        self.count += 1
        self.dropped += event.get("dropped", 0)
        for stage, value in event.get("latency_ms", {}).items():
            self.latencies.setdefault(stage, deque(maxlen=10000)).append(value)
        if self.count % 50 == 0:
            self.file.flush()

    def finish(self, reason, **extra):
        elapsed = time.perf_counter() - self.start
        self.file.close()
        self.meta.update(extra)
        self.meta.update(
            {
                "reason": reason,
                "session_duration_s": elapsed,
                "processed_frames": self.count,
                "dropped_frames": self.dropped,
                "processing_fps": self.count / max(elapsed, 0.001),
                "latency": {s: percentiles(v) for s, v in self.latencies.items()},
                "rss_mb": psutil.Process().memory_info().rss / 2**20,
                "game_survival_s": extra.get("game_survival_s"),
            }
        )
        (self.path / "summary.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )
        return self.meta
