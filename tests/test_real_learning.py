import numpy as np
from starfall.real_learning import extract_strokes, fit_response


def test_saved_response_is_bound_to_measured_phone_geometry():
    from starfall.real_learning import response_signature
    from starfall.runtime import load_config

    config = load_config("configs/phone.yaml")
    original = response_signature(config)
    config["joystick"]["radius"] += 1
    assert response_signature(config) != original


def test_midgame_controller_restart_cannot_create_new_independent_training_data():
    rows = [
        dict(event(1, 1, [0.3, 0.2], True), attempt_verified=False),
        dict(event(2, 1.19, [0.34, 0.2]), attempt_verified=False),
    ]
    assert not extract_strokes(
        rows, "new-process", dict(delay=0.1, stroke_duration_s=0.08), [0, 0, 1, 0.5]
    )


def event(seq, time, point, gesture=False, episode=1, action=2):
    return dict(
        sequence=seq,
        received_ns=int(time * 1e9),
        dispatch_ns=int((time + 0.08) * 1e9),
        gesture_started_ns=int(time * 1e9) if gesture else None,
        page="gameplay",
        player_detection=point,
        player_valid=True,
        player_track_id=7,
        episode=episode,
        gesture_sent=gesture,
        action=action,
        gesture_fraction=1,
        latency_ms={"receive_to_dispatch": 80},
    )


def test_response_targets_do_not_cross_a_manual_reset_or_second_stroke():
    motion = dict(delay=0.10, stroke_duration_s=0.08)
    rows = [
        event(1, 1, [0.3, 0.2], True),
        event(2, 1.19, [0.34, 0.2]),
        event(3, 2, [0.3, 0.2], True),
        event(4, 2.19, [0.34, 0.2], episode=2),
        event(5, 3, [0.3, 0.2], True),
        event(6, 3.10, [0.32, 0.2], True),
        event(7, 3.29, [0.36, 0.2]),
    ]
    samples = extract_strokes(rows, "run", motion, [0, 0, 1, 0.5])
    assert len(samples) == 2
    assert np.allclose(samples[0]["displacement"], [0.04, 0])
    assert samples[1]["id"].endswith(":6")


def test_response_targets_do_not_include_an_ultimate_cast():
    rows = [
        event(1, 1, [0.3, 0.2], True),
        dict(event(2, 1.1, [0.32, 0.2]), ultimate_sent=True),
        event(3, 1.19, [0.34, 0.2]),
    ]
    assert not extract_strokes(
        rows, "run", dict(delay=0.1, stroke_duration_s=0.08), [0, 0, 1, 0.5]
    )


def test_response_model_requires_independent_attempts_and_improvement():
    baseline = np.zeros((9, 2))
    baseline[2] = [0.04, 0]
    samples = [
        dict(
            group=group,
            action=2,
            fraction=1,
            displacement=[0.05 + (0.0002 if i % 2 else -0.0002), 0],
        )
        for group in ("a", "b")
        for i in range(20)
    ]
    good = fit_response(samples, baseline)
    assert good["promoted_actions"] == [2]
    assert np.isclose(good["validated_displacements"][2][0], 0.05)
    assert not fit_response(samples[:20], baseline)["promoted_actions"]
    bad = [dict(s, displacement=[0.13, 0]) for s in samples]
    assert not fit_response(bad, baseline)["promoted_actions"]


def test_template_search_preserves_players_whose_sprite_overlaps_bounds():
    import cv2
    from starfall.runtime import load_config
    from starfall.perception import create_detector

    cfg = load_config("configs/phone.yaml")
    image = cv2.imread("reports/challenge-after12s.png")
    cfg["bounds"][1] = 0.19  # Head above boundary, center still inside.
    assert any(d.kind == "player" for d in create_detector(cfg).detect(image))
