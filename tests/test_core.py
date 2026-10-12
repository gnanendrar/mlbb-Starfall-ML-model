import socket
import struct
import threading
import time
import numpy as np
import pytest
from starfall.android import Joystick, recv_exact, touch_packet
from starfall.types import Detection, World, make_track
from starfall.tracking import Tracker
from starfall.planning import PredictivePlanner, time_to_collision
from starfall.simulation import StarfallSim
from starfall.features import encode, OBS_SIZE


def test_scrcpy_touch_wire_format():
    packet = touch_packet(0, 100, 200, 1080, 1920, pointer=0x1234567887654321)
    expected = bytes.fromhex(
        "0200123456788765432100000064000000c804380780ffff0000000000000000"
    )
    assert packet == expected
    assert len(packet) == 32
    assert struct.unpack(">H", touch_packet(1, 100, 200, 1080, 1920)[22:24])[0] == 0
    with pytest.raises(ValueError):
        touch_packet(0, 1080, 0, 1080, 1920)


def test_fragmented_reads_and_disconnect():
    a, b = socket.socketpair()

    def send():
        b.sendall(b"ab")
        time.sleep(0.01)
        b.sendall(b"cde")
        b.close()

    thread = threading.Thread(target=send)
    thread.start()
    assert recv_exact(a, 5) == b"abcde"
    with pytest.raises(EOFError):
        recv_exact(a, 1)
    a.close()
    thread.join()


def test_watchdog_releases_touch_once():
    class Sink:
        def __init__(self):
            self.packets = []

        def sendall(self, p):
            self.packets.append(p)

    class Stream:
        width = height = 500
        control = Sink()

    stream = Stream()
    j = Joystick(stream, (100, 100), 30, watchdog=0.03)
    j.move((1, 0))
    time.sleep(0.10)
    j.close()
    assert [p[1] for p in stream.control.packets] == [0, 2, 1]


def test_timestamp_tracking_velocity_and_expiry():
    tracker = Tracker(ttl=0.2, gate=0.15)
    identities = []
    for k in range(20):
        t = k * 0.05
        tracks = tracker.update(
            [Detection("hazard", np.array([0.1 + 0.2 * t, 0.2]), 0.02)], t
        )
        identities.append(tracks[0].id)
    assert len(set(identities)) == 1
    assert tracks[0].velocity[0] == pytest.approx(0.2, abs=0.025)
    assert np.linalg.eigvalsh(tracks[0].covariance).min() >= -1e-10
    assert tracker.update([], 1.3) == []
    with pytest.raises(ValueError):
        tracker.update([], 1.3)


@pytest.mark.parametrize(
    "r,v,radius,expected",
    [
        ([1, 0], [-1, 0], 0.1, 0.9),
        ([1, 0], [1, 0], 0.1, float("inf")),
        ([0, 0], [1, 0], 0.1, 0.0),
        ([1, 1], [-1, 0], 0.1, float("inf")),
    ],
)
def test_time_to_collision(r, v, radius, expected):
    assert time_to_collision(r, v, radius) == pytest.approx(expected)


def test_planner_avoids_crossing_hazards_and_dangerous_collectible():
    player = make_track(0, "player", 0.5, 0.5, radius=0.02)
    hazard = make_track(1, "hazard", 0.8, 0.5, -0.7, 0, radius=0.04)
    item = make_track(2, "item", 0.7, 0.5, radius=0.03, label="item")
    planner = PredictivePlanner(
        speed=0.35, delay=0, uncertainty_scale=0, item_values={"item": 5}
    )
    result = planner.choose(World(player, [hazard], [item]))
    assert result.action not in (0, 2)
    assert result.safe_actions[result.action]
    assert result.clearances[result.action] >= 0


def test_continuous_sweep_detects_fast_hazard_between_samples():
    p = make_track(0, "player", 0.5, 0.5, radius=0.02)
    h = make_track(1, "hazard", 0.8, 0.5, -20, 0, radius=0.02)
    result = PredictivePlanner(delay=0, uncertainty_scale=0).choose(World(p, [h], []))
    assert not result.safe_actions[0]


def test_uncertainty_can_remove_safe_action():
    p = make_track(0, "player", 0.5, 0.5)
    h = make_track(1, "hazard", 0.7, 0.5, radius=0.02)
    small = (
        PredictivePlanner(delay=0, uncertainty_scale=2)
        .choose(World(p, [h], []))
        .safe_actions[0]
    )
    h.covariance = np.eye(4) * 0.04
    large = (
        PredictivePlanner(delay=0, uncertainty_scale=2)
        .choose(World(p, [h], []))
        .safe_actions[0]
    )
    assert small and not large


def test_sim_seed_observations_and_termination():
    a, b = StarfallSim(duration=1), StarfallSim(duration=1)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)
    assert len(oa) == OBS_SIZE == 132
    for _ in range(20):
        ra, rb = a.step(0), b.step(0)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert a.observation_space.contains(ra[0])
        assert ra[1:] == rb[1:]
        if ra[2] or ra[3]:
            break
    assert ra[2] or ra[3]
    with pytest.raises(RuntimeError):
        a.step(0)


def test_unknown_health_and_shield_masks():
    obs = encode(World(make_track(0, "player", 0.5, 0.5), [], []))
    assert obs[5] == 0 and obs[7] == 0


def test_drag_stroke_and_cancellation():
    class Sink:
        def __init__(self):
            self.packets = []

        def sendall(self, p):
            self.packets.append(p)

    class Stream:
        width = height = 500
        control = Sink()

    stream = Stream()
    j = Joystick(stream, (100, 100), 40)
    j.drag((1, 0), duration=0.02, steps=4)
    assert [p[1] for p in stream.control.packets] == [0, 2, 2, 2, 2, 1]
    xs = [struct.unpack(">i", p[10:14])[0] for p in stream.control.packets if p[1] == 2]
    assert xs == [110, 120, 130, 140]
    timer = threading.Timer(0.03, j.disarm)
    timer.start()
    with pytest.raises(RuntimeError, match="closed"):
        j.drag((0, 1), duration=0.20, steps=10)
    timer.join()
    j.close()
    assert stream.control.packets[-1][1] == 1


def test_physical_drag_shortened_before_arena_boundary():
    from starfall.planning import limited_drag

    p = make_track(0, "player", 0.5, 0.15, radius=0.03)
    world = World(p, [], [], bounds=(0.02, 0.1, 0.98, 0.405))
    velocities = np.zeros((9, 2))
    velocities[3] = [0, -0.2]
    direction, fraction = limited_drag(world, 3, velocities, 0.22)
    assert 0 < fraction < 1
    final = p.position + velocities[3] * 0.22 * fraction
    assert final[1] >= 0.1 + p.radius + 0.006 - 1e-9
    assert direction[1] < 0
