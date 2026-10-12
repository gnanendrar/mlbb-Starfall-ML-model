"""Version-pinned scrcpy 5.0.1 transport, in-memory decoding, latest-frame slot."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from collections import deque
import secrets
import socket
import struct
import subprocess
import threading
import time
import av

ROOT = Path(__file__).resolve().parents[1]
SCRCPY_VERSION = "5.0.1"
BUNDLE = ROOT / "tools" / f"scrcpy-win64-v{SCRCPY_VERSION}"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Frame:
    image: object
    sequence: int
    device_pts_us: int
    received_ns: int
    decoded_ns: int

    @property
    def time(self):
        return self.device_pts_us / 1_000_000


def recv_exact(sock, size):
    parts = bytearray()
    while len(parts) < size:
        chunk = sock.recv(size - len(parts))
        if not chunk:
            raise EOFError("scrcpy socket closed")
        parts.extend(chunk)
    return bytes(parts)


def touch_packet(action, x, y, width, height, pointer=0):
    if not (0 <= x < width and 0 <= y < height and width <= 65535 and height <= 65535):
        raise ValueError("Touch outside current video coordinates")
    # Verified against v5.0.1 app/tests/test_control_msg_serialize.c.
    return struct.pack(
        ">BBQiiHHHII",
        2,
        action,
        pointer,
        x,
        y,
        width,
        height,
        0 if action in (1, 3) else 65535,
        0,
        0,
    )


class Adb:
    def __init__(self, serial=None, executable=None):
        self.serial = serial
        self.executable = str(executable or BUNDLE / "adb.exe")

    def command(self, *args, timeout=15):
        cmd = (
            [self.executable]
            + (["-s", self.serial] if self.serial else [])
            + list(args)
        )
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=NO_WINDOW,
            check=True,
        ).stdout.strip()

    def devices(self):
        output = self.command("devices", "-l")
        return [
            dict(serial=line.split()[0], state=line.split()[1])
            for line in output.splitlines()[1:]
            if len(line.split()) >= 2
        ]

    def select(self):
        devices = self.devices()
        if self.serial:
            ready = [
                d
                for d in devices
                if d["serial"] == self.serial and d["state"] == "device"
            ]
        else:
            ready = [d for d in devices if d["state"] == "device"]
        if len(ready) != 1:
            raise RuntimeError(
                f"Need one authorized Android device or --serial. Found: {devices}"
            )
        self.serial = ready[0]["serial"]
        return self.serial


class ScrcpyStream:
    def __init__(
        self, adb=None, max_size=960, fps=60, bitrate=4_000_000, control=False
    ):
        self.adb = adb or Adb()
        self.max_size, self.fps, self.bitrate = max_size, fps, bitrate
        self.want_control = control
        self.width = self.height = 0
        self.error = None
        self.latest = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.video = self.control = self.process = self.worker = None
        self.port = None
        self.sequence = 0
        self.log = deque(maxlen=30)

    def start(self):
        self.adb.select()
        server = BUNDLE / "scrcpy-server"
        if not server.exists():
            raise RuntimeError("Run python tools/bootstrap.py first")
        # Name is unique per connection; dynamic local port avoids conflicts.
        scid = secrets.randbelow(2**31)
        remote = f"localabstract:scrcpy_{scid:08x}"
        jar = f"/data/local/tmp/starfall-scrcpy-{scid:08x}.jar"
        self.adb.command("push", str(server), jar)
        self.port = int(self.adb.command("forward", "tcp:0", remote))
        options = [
            f"scid={scid:08x}",
            "tunnel_forward=true",
            "audio=false",
            f"control={str(self.want_control).lower()}",
            "video_codec=h264",
            f"max_size={self.max_size}",
            f"max_fps={self.fps}",
            f"video_bit_rate={self.bitrate}",
            "send_device_meta=false",
            "send_dummy_byte=true",
            "send_stream_meta=true",
            "send_frame_meta=true",
            "clipboard_autosync=false",
            "cleanup=true",
            "log_level=warn",
        ]
        cmd = [
            self.adb.executable,
            "-s",
            self.adb.serial,
            "shell",
            f"CLASSPATH={jar}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            SCRCPY_VERSION,
            *options,
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=NO_WINDOW,
        )
        threading.Thread(target=self._drain_log, daemon=True).start()
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                candidate = None
                try:
                    candidate = socket.create_connection(
                        ("127.0.0.1", self.port), timeout=0.5
                    )
                    candidate.settimeout(2)
                    if recv_exact(candidate, 1) != b"\0":
                        raise RuntimeError("Invalid scrcpy forward handshake")
                    self.video = candidate
                    break
                except (OSError, EOFError):
                    if candidate:
                        candidate.close()
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "scrcpy server stopped: " + "".join(self.log)
                        )
                    time.sleep(0.1)
            if self.video is None:
                raise TimeoutError("scrcpy handshake timed out: " + "".join(self.log))
            if self.want_control:
                self.control = socket.create_connection(
                    ("127.0.0.1", self.port), timeout=2
                )
                self.control.settimeout(0.2)
                self.control.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            codec = recv_exact(self.video, 4)
            if codec != b"h264":
                raise RuntimeError(f"Unexpected scrcpy codec {codec!r}")
            self.video.settimeout(2.0)
            self.worker = threading.Thread(target=self._decode, daemon=True)
            self.worker.start()
            self.get_frame(timeout=10)
            return self
        except BaseException:
            self.close()
            raise

    def _drain_log(self):
        for line in self.process.stdout:
            self.log.append(line)

    def _decode(self):
        codec = av.CodecContext.create("h264", "r")
        codec.thread_count = 1
        config = b""
        try:
            while not self.stop_event.is_set():
                header = recv_exact(self.video, 12)
                if header[0] & 0x80:
                    self.width, self.height = struct.unpack(">II", header[4:])
                    if not (0 < self.width <= 16384 and 0 < self.height <= 16384):
                        raise RuntimeError("Invalid scrcpy session size")
                    codec = av.CodecContext.create("h264", "r")
                    codec.thread_count = 1
                    config = b""
                    continue
                flags, length = struct.unpack(">QI", header)
                if not 0 < length <= 16 * 1024 * 1024:
                    raise RuntimeError(f"Invalid scrcpy packet length: {length}")
                payload = recv_exact(self.video, length)
                received = time.perf_counter_ns()
                if flags & (1 << 62):
                    config = payload
                    continue
                pts = flags & ((1 << 61) - 1)
                packet = av.Packet(config + payload)
                config = b""
                packet.pts = packet.dts = pts
                packet.time_base = Fraction(1, 1_000_000)
                for decoded in codec.decode(packet):
                    image = decoded.to_ndarray(format="bgr24")
                    self.sequence += 1
                    frame = Frame(
                        image,
                        self.sequence,
                        int(decoded.pts or pts),
                        received,
                        time.perf_counter_ns(),
                    )
                    with self.condition:
                        self.latest = frame
                        self.condition.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                with self.condition:
                    self.condition.notify_all()

    def get_frame(self, after=0, timeout=1):
        with self.condition:
            ok = self.condition.wait_for(
                lambda: (
                    self.error
                    or self.stop_event.is_set()
                    or (self.latest is not None and self.latest.sequence > after)
                ),
                timeout,
            )
            if self.error:
                raise RuntimeError("Capture failed: " + str(self.error)) from self.error
            if not ok or self.stop_event.is_set():
                raise TimeoutError("No fresh frame")
            return self.latest

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        for sock in (self.control, self.video):
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(3)
        if self.port:
            try:
                self.adb.command("forward", "--remove", f"tcp:{self.port}")
            except (OSError, subprocess.SubprocessError):
                pass
        if self.worker:
            self.worker.join(3)


class Joystick:
    """One persistent finger, serialized events, watchdog release on stalled agent."""

    def __init__(self, stream, base, radius, watchdog=0.25, refresh_motion=False):
        if stream.control is None:
            raise RuntimeError("scrcpy control socket is disabled")
        self.stream, self.base, self.radius = stream, tuple(base), radius
        self.watchdog = watchdog
        self.refresh_motion = refresh_motion
        self.refresh_phase = False
        self.lock = threading.Lock()
        self.down = False
        self.last = time.monotonic()
        self.point = self.base
        self.closed = threading.Event()
        self.error = None
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.worker.start()

    def _send(self, action, point):
        w, h = self.stream.width, self.stream.height
        self.stream.control.sendall(
            touch_packet(action, int(point[0]), int(point[1]), w, h)
        )

    def _release(self):
        if self.down:
            try:
                self._send(1, self.point)
            finally:
                self.down = False

    def move(self, direction):
        with self.lock:
            if self.error:
                raise RuntimeError("Input watchdog failed") from self.error
            if self.closed.is_set():
                raise RuntimeError("Joystick closed")
            self.last = time.monotonic()
            dx, dy = map(float, direction)
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            if dx == 0 and dy == 0:
                self._release()
                return
            if not self.down:
                self._send(0, self.base)
                self.down = True
            self.refresh_phase = not self.refresh_phase
            factor = 0.96 if self.refresh_motion and self.refresh_phase else 1.0
            self.point = (
                self.base[0] + self.radius * factor * dx / length,
                self.base[1] + self.radius * factor * dy / length,
            )
            try:
                self._send(2, self.point)
            except Exception:
                self._release()
                raise

    def release(self):
        with self.lock:
            self._release()

    def drag(self, direction, duration=0.08, steps=4):
        """A bounded fresh stroke for controls that react to finger movement."""
        if duration <= 0 or steps < 1:
            raise ValueError("Invalid drag duration/steps")
        self.release()
        dx, dy = map(float, direction)
        if dx == 0 and dy == 0:
            return
        try:
            for k in range(1, steps + 1):
                self.move((dx * k / steps, dy * k / steps))
                if self.closed.wait(duration / steps):
                    raise RuntimeError("Joystick closed during drag")
        finally:
            self.release()

    def _watch(self):
        while not self.closed.wait(0.025):
            with self.lock:
                if self.down and time.monotonic() - self.last > self.watchdog:
                    try:
                        self._release()
                    except Exception as exc:
                        self.error = exc

    def close(self):
        self.closed.set()
        try:
            self.release()
        finally:
            self.worker.join(1)

    def disarm(self):
        """Permanently inhibit new joystick presses before a safety UI action."""
        self.closed.set()
        self.release()
