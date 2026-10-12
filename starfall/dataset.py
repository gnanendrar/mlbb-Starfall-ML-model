"""Real-frame collection, manual annotation, validation and leakage-aware splits."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import shutil
import time
import cv2
import numpy as np
import yaml


def dhash(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


class Collector:
    def __init__(
        self, root="data/game", interval=0.5, distance=6, limit=2000, session=None
    ):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "labels").mkdir(exist_ok=True)
        self.interval, self.distance, self.limit = interval, distance, limit
        self.last = -float("inf")
        self.hashes = []
        self.session = session or f"session-{time.time_ns()}"
        self.records = []
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            self.records = [
                json.loads(line) for line in manifest.read_text().splitlines()
            ]
            self.hashes = [int(row["dhash"], 16) for row in self.records]

    def offer(self, frame):
        now = time.monotonic()
        if len(self.records) >= self.limit or now - self.last < self.interval:
            return None
        self.last = now
        hashed = dhash(frame.image)
        if any((hashed ^ other).bit_count() < self.distance for other in self.hashes):
            return None
        name = f"{self.session}-{frame.sequence:08d}.jpg"
        path = self.root / "images" / name
        if not cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise OSError(f"Could not write {path}")
        record = {
            "image": name,
            "session": self.session,
            "source": "real_phone",
            "sequence": frame.sequence,
            "pts_us": frame.device_pts_us,
            "received_ns": frame.received_ns,
            "dhash": f"{hashed:016x}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        self.hashes.append(hashed)
        return path


def validate(root, classes):
    root = Path(root)
    errors, counts, negatives = [], Counter(), 0
    manifest_path = root / "manifest.jsonl"
    records = (
        [json.loads(line) for line in manifest_path.read_text().splitlines()]
        if manifest_path.exists()
        else []
    )
    if not records:
        errors.append("No captured real frames")
    for row in records:
        name = row["image"]
        image_path = root / "images" / name
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"Unreadable image: {name}")
        elif hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append(f"Image changed since capture: {name}")
        label = root / "labels" / Path(name).with_suffix(".txt")
        if not label.exists():
            errors.append(f"Unreviewed image: {name}")
            continue
        lines = label.read_text().splitlines()
        negatives += not lines
        for line in lines:
            try:
                values = line.split()
                if len(values) != 5:
                    raise ValueError()
                cls = int(values[0])
                x, y, w, h = map(float, values[1:])
                if not (
                    0 <= cls < len(classes)
                    and np.isfinite([x, y, w, h]).all()
                    and w > 0
                    and h > 0
                    and 0 <= x - w / 2 <= x + w / 2 <= 1 + 1e-6
                    and 0 <= y - h / 2 <= y + h / 2 <= 1 + 1e-6
                ):
                    raise ValueError()
                counts[classes[cls]] += 1
            except ValueError:
                errors.append(f"Malformed YOLO label {label}: {line}")
    known = {Path(r["image"]).stem for r in records}
    for label in (root / "labels").glob("*.txt"):
        if label.stem not in known:
            errors.append(f"Orphan label: {label.name}")
    report = {
        "frames": len(records),
        "sessions": len({r["session"] for r in records}),
        "classes": classes,
        "boxes_per_class": dict(counts),
        "reviewed_negatives": negatives,
        "errors": errors,
        "valid": not errors,
    }
    (root / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def split(root, classes, seed=42, duplicate_distance=6):
    root = Path(root)
    report = validate(root, classes)
    if not report["valid"]:
        raise ValueError("Dataset validation failed; inspect validation.json")
    records = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    sessions = sorted({r["session"] for r in records})
    parent = {s: s for s in sessions}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    # Whole sessions stay together. Similar frames across sessions merge groups.
    for i, row in enumerate(records):
        for other in records[:i]:
            if (
                int(row["dhash"], 16) ^ int(other["dhash"], 16)
            ).bit_count() < duplicate_distance:
                parent[find(row["session"])] = find(other["session"])
    groups = sorted({find(s) for s in sessions})
    if len(groups) < 3:
        raise ValueError(
            "Need at least 3 independent session groups for train/val/test; duplicates may have merged sessions"
        )
    random.Random(seed).shuffle(groups)
    nv = max(1, round(len(groups) * 0.15))
    nt = max(1, round(len(groups) * 0.15))
    assignment = {
        g: "val" if i < nv else "test" if i < nv + nt else "train"
        for i, g in enumerate(groups)
    }
    if not any(v == "train" for v in assignment.values()):
        raise ValueError("Not enough train sessions")
    output = root / f"split-{time.time_ns()}"
    split_rows = []
    for row in records:
        subset = assignment[find(row["session"])]
        for kind in ("images", "labels"):
            folder = output / kind / subset
            folder.mkdir(parents=True, exist_ok=True)
            filename = (
                Path(row["image"])
                if kind == "images"
                else Path(row["image"]).with_suffix(".txt")
            )
            shutil.copy2(root / kind / filename, folder / filename)
        split_rows.append(dict(row, split=subset))
    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes)),
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "splits.json").write_text(
        json.dumps({"seed": seed, "rows": split_rows}, indent=2), encoding="utf-8"
    )
    return output


def annotate(root, classes):
    """Manual review only. Existing labels are displayed and may be replaced."""
    root = Path(root)
    if not classes:
        raise ValueError("Define observed class names first")
    images = sorted((root / "images").glob("*.jpg"))
    print(
        "Annotation: b=add box, c=clear, s=save reviewed labels, n=skip, q=quit. Existing boxes remain until cleared."
    )
    for path in images:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        target = root / "labels" / path.with_suffix(".txt").name
        boxes = []
        if target.exists():
            boxes = [
                list(map(float, line.split()))
                for line in target.read_text().splitlines()
            ]
        while True:
            canvas = image.copy()
            for cls, x, y, bw, bh in boxes:
                a = (int((x - bw / 2) * w), int((y - bh / 2) * h))
                b = (int((x + bw / 2) * w), int((y + bh / 2) * h))
                cv2.rectangle(canvas, a, b, (0, 255, 0), 1)
                cv2.putText(
                    canvas,
                    classes[int(cls)],
                    a,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("Manual game annotation", canvas)
            key = cv2.waitKey(0) & 255
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break
            if key == ord("c"):
                boxes = []
            if key == ord("b"):
                x, y, bw, bh = cv2.selectROI("Manual game annotation", canvas, False)
                if bw and bh:
                    print(dict(enumerate(classes)))
                    cls = int(input("Observed class number: "))
                    if not 0 <= cls < len(classes):
                        raise ValueError("Invalid class")
                    boxes.append(
                        [cls, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
                    )
            if key == ord("s"):
                target.write_text(
                    "".join(
                        f"{int(c)} {x:.7f} {y:.7f} {bw:.7f} {bh:.7f}\n"
                        for c, x, y, bw, bh in boxes
                    )
                )
                break
    cv2.destroyAllWindows()
