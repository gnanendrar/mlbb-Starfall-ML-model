"""Build small regression fixtures from local captures; never training data."""

from pathlib import Path
import hashlib
import json
import shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []

    def add(source, destination, regions=None):
        source, destination = ROOT / source, ROOT / "tests/fixtures" / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if regions is None:
            shutil.copy2(source, destination)
        else:
            original = cv2.imread(str(source))
            fixture = np.zeros_like(original)
            for x1, y1, x2, y2 in regions:
                fixture[y1:y2, x1:x2] = original[y1:y2, x1:x2]
            assert cv2.imwrite(str(destination), fixture)
        records.append(
            {
                "fixture": destination.relative_to(ROOT).as_posix(),
                "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "regions": regions,
                "purpose": "regression_test_only",
            }
        )

    for attempt, sequence in [
        (2, 12095),
        (4, 24902),
        (5, 28706),
        (1, 3001),
        (3, 17000),
        (6, 32651),
        (1, 1893),
    ]:
        filename = f"20261012T015357Z-3bcef2c5-attempt{attempt}-{sequence:08d}.jpg"
        add("data/game/images/" + filename, "combat/" + filename)
    for name, source in [
        ("ready.png", "reports/starfall-frame.png"),
        ("ready-with-return.png", "reports/current-page.png"),
    ]:
        add(source, name, [(310, 0, 950, 120), (720, 280, 950, 430)])
    add("reports/challenge-after12s.png", "gameplay.png")
    add(
        "reports/health-events/1791766265065633800-one_heart/trigger.png",
        "powerup-hearts.png",
    )
    add(
        "reports/health-events/1791767134510931900-health_unreadable/trigger.png",
        "upper-hearts.png",
    )
    add(
        "reports/health-events/1791765595708144500-calibration_complete/after-exit.png",
        "lobby/main.png",
        [(270, 330, 445, 385)],
    )
    add("data/game/images/20261012T011902Z-12c37b4f-ep2-00004481.jpg", "transition.jpg")
    add("data/game/images/20261012T011902Z-12c37b4f-ep5-00008473.jpg", "paused.jpg")
    (ROOT / "tests/fixtures/provenance.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared {len(records)} test fixtures. Modified UI fixtures are never training examples."
    )


if __name__ == "__main__":
    main()
