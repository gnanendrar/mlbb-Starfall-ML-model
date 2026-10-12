from pathlib import Path
import json
import struct
import time
import cv2
import numpy as np
from starfall.combat import CombatReader, CombatReading, UltimatePolicy
from starfall.planning import PredictivePlanner, segment_box_clearance
from starfall.types import World, Wall, make_track, Detection
from starfall.runtime import load_config


def test_recorded_orbs_core_and_real_charge_counter():
    config = load_config("configs/phone.yaml")
    reader = CombatReader(config["combat"])
    base = Path(__file__).resolve().parent / "fixtures/combat"
    examples = [
        ("20261012T015357Z-3bcef2c5-attempt2-00012095.jpg", "power_orb", (827, 262), 0),
        ("20261012T015357Z-3bcef2c5-attempt4-00024902.jpg", "power_orb", (714, 262), 0),
        (
            "20261012T015357Z-3bcef2c5-attempt5-00028706.jpg",
            "stellar_charge",
            (359, 263),
            0,
        ),
        ("20261012T015357Z-3bcef2c5-attempt1-00003001.jpg", "boss_core", (624, 213), 1),
        ("20261012T015357Z-3bcef2c5-attempt3-00017000.jpg", "boss_core", (506, 264), 0),
        ("20261012T015357Z-3bcef2c5-attempt6-00032651.jpg", "boss_core", (462, 255), 1),
    ]
    for filename, label, center, charges in examples:
        reading = reader.read(cv2.imread(str(base / filename)))
        matches = [d for d in reading.detections if d.label == label]
        assert any(np.linalg.norm(d.position * 960 - center) < 8 for d in matches)
        assert reading.charges == charges
        if label == "boss_core":
            assert reading.wall is not None
    negative = reader.read(
        cv2.imread(str(base / "20261012T015357Z-3bcef2c5-attempt1-00001893.jpg"))
    )
    assert not negative.detections and negative.charges == 0
    assert reader.read(np.zeros((960, 430, 3), np.uint8)).charges is None


def test_swept_wall_collision_including_parallel_and_tunneling():
    lower, upper = np.array([0.4, 0.4]), np.array([0.6, 0.6])
    starts = np.array([[0.1, 0.5], [0.1, 0.1], [0.5, 0.5]])
    ends = np.array([[0.9, 0.5], [0.9, 0.1], [0.5, 0.5]])
    distance = segment_box_clearance(starts, ends, lower, upper)
    assert distance[0] < 0 and distance[2] < 0
    assert abs(distance[1] - 0.3) < 1e-9


def test_safe_orb_is_pursued_but_dangerous_orb_cannot_buy_collision():
    player = make_track(0, "player", 0.5, 0.5, radius=0.02)
    orb = make_track(1, "item", 0.62, 0.5, radius=0.025, label="power_orb")
    planner = PredictivePlanner(
        delay=0,
        uncertainty_scale=0,
        item_values={"power_orb": 5},
        allowed_actions=[0, 2],
    )
    decision = planner.choose(World(player, [], [orb]))
    assert decision.action == 2
    meteor = make_track(2, "hazard", 0.59, 0.5, radius=0.04)
    decision = planner.choose(World(player, [meteor], [orb]))
    assert decision.action == 0 and decision.safe_actions[0]


def test_shooting_seeks_a_safe_lane_and_prioritizes_boss_core():
    player = make_track(0, "player", 0.2, 0.3, radius=0.02)
    meteor = make_track(1, "hazard", 0.8, 0.2, radius=0.03)
    planner = PredictivePlanner(
        delay=0,
        uncertainty_scale=0,
        shot_lane_weight=2,
        boss_lane_weight=8,
        muzzle_offset=0,
    )
    decision = planner.choose(World(player, [meteor], []))
    assert decision.path[-1, 1] < player.position[1]
    core = make_track(2, "target", 0.65, 0.4, radius=0.025, label="boss_core")
    world = World(
        player, [meteor], [], targets=[core], walls=[Wall((0.60, 0.0, 0.78, 1.0))]
    )
    decision = planner.choose(world)
    assert decision.path[-1, 1] > player.position[1]
    assert decision.objective == "boss_core" and decision.safe_actions[decision.action]
    assert decision.path[:, 0].max() < 0.60 - player.radius - planner.margin


def test_ultimate_requires_readable_positive_charge_and_useful_target():
    player = make_track(0, "player", 0.2, 0.2)
    core = make_track(1, "target", 0.65, 0.215, label="boss_core")
    world = World(player, [], [], targets=[core])
    policy = UltimatePolicy()
    for charges in (None, 0):
        reading = CombatReading(charges=charges)
        assert policy.choose(world, reading, 0) is None
        assert policy.choose(world, reading, 1) is None
    reading = CombatReading(charges=1)
    assert policy.choose(world, reading, 2) is None
    assert policy.choose(world, reading, 2.1) == "aligned_boss_core"
    policy.fired(1, 2.1)
    assert policy.choose(world, reading, 3.5) is None
    assert policy.choose(world, CombatReading(charges=0), 4) is None
    core.state[1] = 0.4
    assert policy.choose(world, reading, 6) is None
    assert policy.choose(world, reading, 6.1) is None


def test_runtime_taps_ultimate_once_then_suspends_for_orientation(
    monkeypatch, tmp_path
):
    from starfall import runtime
    from starfall.android import Frame
    from starfall.session import PageReading, EpisodeLoop
    from starfall.logging import RunLog
    from starfall.control_lease import ControlLease

    config = load_config("configs/phone.yaml")
    config["session"]["learn_response"] = False
    normal = np.zeros((430, 960, 3), np.uint8)
    portrait = np.zeros((960, 430, 3), np.uint8)
    packets = []

    class Sink:
        def sendall(self, packet):
            packets.append(packet)

    class Stream:
        width, height = 960, 430
        control = Sink()
        latest = None
        i = 0

        def get_frame(self, after=0, timeout=1):
            self.i += 1
            now = time.perf_counter_ns()
            # A rotated frame interrupts inputs without killing the loop.
            image = portrait if self.i == 5 else normal
            self.latest = Frame(image, self.i, self.i * 100000, now, now)
            return self.latest

        def close(self):
            pass

    class Detector:
        def detect(self, image):
            return [Detection("player", np.array([0.2, 0.225]), 0.018)]

    class Combat:
        def read(self, image):
            return CombatReading(
                [Detection("target", np.array([0.65, 0.24]), 0.025, 0.99, "boss_core")],
                charges=1,
            )

    class Pages:
        def __init__(self, _):
            self.i = 0

        def read(self, image):
            self.i += 1
            return PageReading("gameplay" if self.i <= 5 else "main_lobby")

    monkeypatch.setattr(runtime, "create_detector", lambda _: Detector())
    monkeypatch.setattr(runtime, "CombatReader", lambda _: Combat())
    monkeypatch.setattr(runtime, "PageReader", Pages)
    monkeypatch.setattr(runtime, "load_config", lambda _: config)
    monkeypatch.setattr(
        runtime, "ControlLease", lambda: ControlLease(tmp_path / "lock")
    )
    monkeypatch.setattr(
        runtime, "RunLog", lambda *a, **k: RunLog(*a, **k, root=tmp_path)
    )
    monkeypatch.setattr(
        runtime, "EpisodeLoop", lambda auto_start: EpisodeLoop(auto_start, stable_s=0)
    )
    path, report = runtime.run_live(
        "unused",
        control=True,
        repeat=True,
        preview=False,
        provided_stream=Stream(),
        max_episodes=1,
        seconds=5,
    )
    ultimate = [p for p in packets if struct.unpack(">Q", p[2:10])[0] == 2]
    assert [p[1] for p in ultimate] == [0, 1]
    assert all(struct.unpack(">ii", p[10:18]) == (863, 348) for p in ultimate)
    events = [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]
    rotated = next(e for e in events if e["page"] == "orientation_changed")
    assert not rotated["gesture_sent"] and not rotated["ultimate_sent"]
    assert report["reason"] == "episodes_limit" and report["health_exit"] is None
