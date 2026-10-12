"""Independent health watchdog. A detected last heart overrides every policy."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import threading
import time
import cv2
from .android import touch_packet


class HealthAbort(RuntimeError):
    pass


def load_template(path):
    image = cv2.imread(str(path))
    if image is None or image.std() < 2:
        raise ValueError(f"Missing or invalid health/UI template: {path}")
    return image


def find_template(image, template, roi, threshold, max_count=1):
    x0, y0, x1, y1 = roi
    crop = image[y0:y1, x0:x1]
    h, w = template.shape[:2]
    if crop.shape[0] < h or crop.shape[1] < w:
        return []
    scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    found = []
    for _ in range(max_count):
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < threshold:
            break
        found.append((x + x0 + w / 2, y + y0 + h / 2, float(score)))
        scores[
            max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1
        ] = -1
    return found


@dataclass
class HealthReading:
    hearts: int | None
    confidence: float
    gameplay: bool
    lobby: bool
    centers: list
    return_button: tuple | None
    ready: bool = False
    main_lobby: bool = False


class HeartReader:
    def __init__(self, config):
        self.config = config
        self.reference = tuple(config["frame_size"])
        self.heart = load_template(config["heart_template"])
        self.heart_upper = (
            load_template(config["upper_heart_template"])
            if config.get("upper_heart_template")
            else None
        )
        self.return_label = load_template(config["return_template"])
        self.lobby_label = load_template(config["lobby_template"])
        self.main_lobby_label = (
            load_template(config["main_lobby_template"])
            if config.get("main_lobby_template")
            else None
        )
        self.confirm = (
            load_template(config["confirm_template"])
            if config.get("confirm_template")
            else None
        )
        self.last_return_button = None
        self.last_game_context = None

    def read(self, image):
        h, w = image.shape[:2]
        if (w, h) != self.reference:
            raise ValueError(
                "Health calibration resolution changed; stop control and recalibrate"
            )
        ready = bool(find_template(image, self.lobby_label, (w // 2, 30, w, 120), 0.78))
        main_lobby = bool(
            self.main_lobby_label is not None
            and find_template(
                image, self.main_lobby_label, (w // 5, h * 2 // 3, w * 3 // 5, h), 0.85
            )
        )
        lobby = ready or main_lobby
        buttons = find_template(
            image, self.return_label, (w // 3, 0, w * 2 // 3, 50), 0.78
        )
        if lobby:
            self.last_game_context = None
            self.last_return_button = None
        elif buttons:
            self.last_return_button = (buttons[0][0], buttons[0][1])
            self.last_game_context = time.monotonic()
        cached = (
            self.last_game_context is not None
            and time.monotonic() - self.last_game_context < 0.8
        )
        gameplay = (bool(buttons) or cached) and not lobby
        return_button = (
            (buttons[0][0], buttons[0][1])
            if buttons
            else self.last_return_button
            if cached
            else None
        )
        if not gameplay:
            return HealthReading(
                None, 0.0, False, lobby, [], return_button, ready, main_lobby
            )
        candidates = find_template(
            image,
            self.heart,
            (8, 10, w - 8, h - 10),
            self.config.get("heart_threshold", 0.82),
            12,
        )

        def pink_fraction(cx, cy, template):
            th, tw = template.shape[:2]
            x, y = round(cx - tw / 2), round(cy - th / 2)
            patch = image[y : y + th, x : x + tw]
            if patch.shape[:2] != (th, tw):
                return 0.0
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            return float(
                (
                    (hsv[:, :, 0] > 125) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 140)
                ).mean()
            )

        candidates = [
            c for c in candidates if pink_fraction(c[0], c[1], self.heart) >= 0.20
        ]
        if self.heart_upper is not None:
            upper = find_template(
                image, self.heart_upper, (8, 10, w - 8, h - 10), 0.82, 12
            )
            offset = (self.heart.shape[0] - self.heart_upper.shape[0]) / 2
            for x, y, confidence in upper:
                if pink_fraction(x, y, self.heart_upper) < 0.22:
                    continue
                adjusted = (x, y + offset, confidence)
                if not any(
                    abs(c[0] - x) < 4 and abs(c[1] - adjusted[1]) < 4
                    for c in candidates
                ):
                    candidates.append(adjusted)
        # Actual hearts are a horizontal row above Layla. Keep a single row,
        # reject >3 matches and widely spaced unrelated UI icons.
        groups = []
        for anchor in candidates:
            group = [
                c
                for c in candidates
                if abs(c[1] - anchor[1]) <= 4 and abs(c[0] - anchor[0]) <= 60
            ]
            group.sort(key=lambda c: c[0])
            if 1 <= len(group) <= 3 and all(
                12 <= b[0] - a[0] <= 30 for a, b in zip(group, group[1:])
            ):
                groups.append(group)
        if not groups:
            return HealthReading(None, 0.0, True, False, [], return_button)
        group = max(groups, key=lambda g: (len(g), sum(c[2] for c in g)))
        return HealthReading(
            len(group),
            min(c[2] for c in group),
            True,
            False,
            [(c[0], c[1]) for c in group],
            return_button,
        )


class HealthSupervisor:
    """Runs beside expensive inference and shares no learned-policy decisions."""

    def __init__(
        self, stream, config, joystick=None, report_root="reports/health-events"
    ):
        if not config.get("enabled", False):
            raise ValueError("Optional health guard is disabled in this configuration")
        self.stream, self.config, self.joystick = stream, config, joystick
        self.reader = HeartReader(config)
        self.stop_event = threading.Event()
        self.exiting = threading.Event()
        self.finished = threading.Event()
        self.latest = None
        self.error = None
        self.report = None
        self.exit_lock = threading.Lock()
        self.report_root = Path(report_root)
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(
            target=self._watch, name="last-heart-guard", daemon=True
        )

    def start(self):
        self.worker.start()
        return self

    def _tap(self, point):
        x, y = map(int, point)
        # Joystick is disarmed before a distinct UI pointer is used.
        self.stream.control.sendall(
            touch_packet(0, x, y, self.stream.width, self.stream.height, pointer=1)
        )
        # Keep DOWN through at least one game UI tick; the first event is
        # dispatched immediately, while the joystick is already disarmed.
        time.sleep(0.025)
        self.stream.control.sendall(
            touch_packet(1, x, y, self.stream.width, self.stream.height, pointer=1)
        )

    def return_to_lobby(self, frame, reading, reason="one_heart"):
        with self.exit_lock:
            if self.finished.is_set():
                return self.report
            return self._return_to_lobby(frame, reading, reason)

    def _return_to_lobby(self, frame, reading, reason):
        self.exiting.set()
        if self.joystick:
            self.joystick.disarm()
        directory = self.report_root / f"{time.time_ns()}-{reason}"
        directory.mkdir()
        trigger_ns = time.perf_counter_ns()
        if not reading.return_button:
            raise RuntimeError("Return to Lobby button not visually verified")
        # Dispatch before writing screenshots or running another inference.
        self._tap(reading.return_button)
        dispatched_ns = time.perf_counter_ns()
        report = {
            "reason": reason,
            "heart_reading": asdict(reading),
            "trigger_sequence": frame.sequence,
            "received_ns": frame.received_ns,
            "trigger_ns": trigger_ns,
            "dispatch_ns": dispatched_ns,
            "trigger_to_dispatch_ms": (dispatched_ns - trigger_ns) / 1e6,
            "receive_to_exit_dispatch_ms": (dispatched_ns - frame.received_ns) / 1e6,
            "lobby_verified": False,
            "confirmation_needed": False,
            "ticket_preservation_verified": False,
        }
        cv2.imwrite(str(directory / "trigger.png"), frame.image)
        seq = frame.sequence
        deadline = time.monotonic() + 3
        last = frame
        attempts = 1
        last_tap = time.monotonic()
        while time.monotonic() < deadline and not self.stop_event.is_set():
            try:
                last = self.stream.get_frame(seq, timeout=0.25)
                seq = last.sequence
            except TimeoutError:
                continue
            state = self.reader.read(last.image)
            if state.lobby:
                report["lobby_verified"] = True
                break
            if (
                state.gameplay
                and state.return_button
                and attempts < 3
                and time.monotonic() - last_tap > 0.25
            ):
                self._tap(state.return_button)
                attempts += 1
                last_tap = time.monotonic()
            if self.reader.confirm is not None:
                found = find_template(
                    last.image,
                    self.reader.confirm,
                    (0, 50, self.stream.width, self.stream.height),
                    0.85,
                )
                if found:
                    report["confirmation_needed"] = True
                    self._tap(found[0][:2])
                    # Avoid sending repeated confirmations on a stale frame.
                    self.reader.confirm = None
        cv2.imwrite(str(directory / "after-exit.png"), last.image)
        report["evidence_directory"] = str(directory)
        report["exit_tap_attempts"] = attempts
        (directory / "event.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        self.report = report
        self.finished.set()
        return report

    def _watch(self):
        seq = 0
        last_valid = None
        try:
            while not self.stop_event.is_set() and not self.exiting.is_set():
                try:
                    frame = self.stream.get_frame(seq, timeout=0.15)
                except TimeoutError:
                    if self.joystick:
                        self.joystick.release()
                    continue
                seq = frame.sequence
                reading = self.reader.read(frame.image)
                self.latest = reading
                if reading.hearts is not None:
                    last_valid = time.monotonic()
                    if reading.hearts <= self.config.get("exit_at_hearts", 1):
                        self.return_to_lobby(frame, reading)
                        break
                elif (
                    reading.gameplay
                    and last_valid is not None
                    and time.monotonic() - last_valid
                    > self.config.get("unknown_timeout_s", 0.35)
                ):
                    self.return_to_lobby(frame, reading, "health_unreadable")
                    break
        except BaseException as exc:
            self.error = exc
            self.exiting.set()
            if self.joystick:
                try:
                    self.joystick.disarm()
                except OSError:
                    pass
            self.finished.set()

    def check(self):
        if self.error:
            raise HealthAbort(f"Health guard failed: {self.error}") from self.error
        if self.exiting.is_set():
            raise HealthAbort("Returned to lobby for health protection")

    def close(self):
        self.stop_event.set()
        self.worker.join(4)
