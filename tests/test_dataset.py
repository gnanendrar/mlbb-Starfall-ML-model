import hashlib
import json
import cv2
import numpy as np
from starfall.dataset import validate, split


def make_dataset(root):
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rows = []
    for i, hashed in enumerate(
        (0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA)
    ):
        name = f"s{i}.jpg"
        cv2.imwrite(
            str(root / "images" / name),
            np.random.default_rng(i).integers(0, 255, (32, 32, 3), dtype=np.uint8),
        )
        (root / "labels" / f"s{i}.txt").write_text("0 .5 .5 .2 .2\n")
        rows.append(
            {
                "image": name,
                "session": f"s{i}",
                "source": "synthetic_unit_test_only",
                "dhash": f"{hashed:016x}",
                "sha256": hashlib.sha256(
                    (root / "images" / name).read_bytes()
                ).hexdigest(),
            }
        )
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def test_group_splits_and_no_session_leakage(tmp_path):
    make_dataset(tmp_path)
    output = split(tmp_path, ["test_class"])
    rows = json.loads((output / "splits.json").read_text())["rows"]
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert len({(r["session"], r["split"]) for r in rows}) == 3


def test_invalid_boxes_and_unreviewed_images_rejected(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "labels" / "s0.txt").write_text("0 .99 .5 .4 .2\n")
    (tmp_path / "labels" / "s1.txt").unlink()
    report = validate(tmp_path, ["test_class"])
    assert not report["valid"]
    assert len(report["errors"]) == 2


def test_duplicate_sessions_are_merged_before_splitting(tmp_path):
    import pytest

    rows = make_dataset(tmp_path)
    rows[1]["dhash"] = rows[0]["dhash"]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="3 independent"):
        split(tmp_path, ["test_class"])
