"""Only one local process may send game controls at a time."""

from pathlib import Path
import json
import os
import psutil


class ControlLease:
    def __init__(self, path="runs/phone-control.lock"):
        self.path = Path(path)
        self.pid = os.getpid()

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                with self.path.open("x", encoding="utf-8") as file:
                    json.dump({"pid": self.pid}, file)
                return self
            except FileExistsError:
                try:
                    owner = json.loads(self.path.read_text())["pid"]
                except (json.JSONDecodeError, KeyError):
                    raise RuntimeError(
                        "Phone controller lock is being created; retry shortly"
                    )
                if psutil.pid_exists(owner):
                    raise RuntimeError(
                        f"Another phone controller is running (process {owner}); stop it first"
                    )
                self.path.unlink()
        raise RuntimeError("Could not acquire phone control")

    def close(self):
        if (
            self.path.exists()
            and json.loads(self.path.read_text()).get("pid") == self.pid
        ):
            self.path.unlink()
