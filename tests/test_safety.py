"""Synthetic unit fixtures made from observed UI sprites, not game training data."""

from pathlib import Path
import time
import cv2
import pytest
import yaml
from starfall.android import Frame, Joystick
from starfall.safety import HeartReader, HealthSupervisor, HealthAbort

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    # Legacy guard is tested explicitly; it is disabled in the human-reset loop.
    return dict(
        yaml.safe_load((ROOT / "configs/phone.yaml").read_text())["health_guard"],
        enabled=True,
    )


def synthetic_hearts(count):
    # Only remove hearts in an in-memory TEST fixture. Never append to real data.
    image = cv2.imread(str(ROOT / "tests/fixtures/gameplay.png"))
    for x in (199, 222, 245)[count:]:
        image[125:140, x : x + 17] = image[90:105, x : x + 17]
    return image


@pytest.mark.parametrize("hearts", [1, 2, 3])
def test_visible_filled_hearts_count(config, hearts):
    reading = HeartReader(config).read(synthetic_hearts(hearts))
    assert reading.gameplay
    assert reading.hearts == hearts
    assert reading.return_button == (479.5, 15.0)


def test_lobby_and_unknown_are_not_mistaken_for_one_heart(config):
    reader = HeartReader(config)
    lobby = reader.read(cv2.imread(str(ROOT / "tests/fixtures/ready.png")))
    assert lobby.lobby and lobby.hearts is None
    unknown = reader.read(synthetic_hearts(0))
    assert unknown.gameplay and unknown.hearts is None


def test_dimension_change_stops_health_processing(config):
    with pytest.raises(ValueError, match="resolution"):
        HeartReader(config).read(
            cv2.imread(str(ROOT / "tests/fixtures/ready.png"))[:300]
        )


def test_real_powerup_text_does_not_hide_two_healthy_hearts(config):
    image = cv2.imread(str(ROOT / "tests/fixtures/powerup-hearts.png"))
    assert HeartReader(config).read(image).hearts == 3


def test_gray_empty_hearts_do_not_count_as_filled(config):
    image = synthetic_hearts(3)
    for x in (222, 245):
        tile = image[125:140, x : x + 17]
        image[125:140, x : x + 17] = cv2.cvtColor(
            cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
    assert HeartReader(config).read(image).hearts == 1


def test_real_heart_row_near_top_is_readable(config):
    image = cv2.imread(str(ROOT / "tests/fixtures/upper-hearts.png"))
    assert HeartReader(config).read(image).hearts == 3


def test_one_heart_interrupts_and_disarms_joystick(config, tmp_path):
    packets = []

    class Sink:
        def sendall(self, p):
            packets.append(p)

    class Stream:
        width = 960
        height = 430
        control = Sink()

        def get_frame(self, after=0, timeout=1):
            image = (
                synthetic_hearts(1)
                if after == 0
                else cv2.imread(str(ROOT / "tests/fixtures/ready.png"))
            )
            return Frame(
                image,
                after + 1,
                after + 1,
                time.perf_counter_ns(),
                time.perf_counter_ns(),
            )

    stream = Stream()
    joystick = Joystick(stream, (100, 350), 43)
    joystick.move((1, 0))
    supervisor = HealthSupervisor(
        stream, config, joystick, report_root=tmp_path
    ).start()
    assert supervisor.finished.wait(2.0)
    assert supervisor.error is None
    assert supervisor.report["reason"] == "one_heart"
    assert supervisor.report["lobby_verified"]
    # Joystick UP precedes UI DOWN/UP, and no new movement can begin.
    assert [p[1] for p in packets] == [0, 2, 1, 0, 1]
    assert int.from_bytes(packets[-1][2:10], "big") == 1
    with pytest.raises(RuntimeError, match="closed"):
        joystick.move((1, 0))
    with pytest.raises(HealthAbort):
        supervisor.check()
    supervisor.close()
    joystick.close()


def test_two_hearts_never_trigger_exit(config, tmp_path):
    class Stream:
        width = 960
        height = 430
        control = None

        def get_frame(self, after=0, timeout=1):
            time.sleep(0.02)
            return Frame(
                synthetic_hearts(2),
                after + 1,
                after + 1,
                time.perf_counter_ns(),
                time.perf_counter_ns(),
            )

    supervisor = HealthSupervisor(Stream(), config, report_root=tmp_path).start()
    time.sleep(0.12)
    supervisor.close()
    assert not supervisor.exiting.is_set()
    assert supervisor.latest.hearts == 2
