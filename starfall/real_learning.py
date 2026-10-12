"""Learn a compact action-response model from actual phone trajectories.

Supervised system identification for MPC; no health scans or guessed score rewards.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import numpy as np
from .types import DIRECTIONS


def response_signature(config):
    geometry = {key: config[key] for key in ("joystick", "motion", "bounds")}
    geometry["action_velocities"] = config["planner"]["action_velocities"]
    geometry["allowed_actions"] = config["planner"]["allowed_actions"]
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def read_events(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A running process can have an incomplete final line.


def extract_strokes(events, run_id, motion, bounds):
    pending = None
    samples = []
    minimum_age = max(
        0.14, motion.get("delay", 0.10) + motion.get("stroke_duration_s", 0.08)
    )
    for event in events:
        point = event.get("player_detection")
        if (
            event.get("page") != "gameplay"
            or not event.get("attempt_verified", True)
            or point is None
            or not event.get("player_valid")
            or event.get("ultimate_sent")
        ):
            pending = None
            continue
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            pending = None
            continue
        near_edge = (
            min(
                point[0] - bounds[0],
                bounds[2] - point[0],
                point[1] - bounds[1],
                bounds[3] - point[1],
            )
            < 0.035
        )
        latency = event.get("latency_ms", {}).get("receive_to_dispatch", 1000)
        if pending:
            age = (event["received_ns"] - pending["sent_ns"]) / 1e9
            same = (
                event.get("episode") == pending["episode"]
                and event.get("player_track_id") == pending["track_id"]
            )
            if not same or age > 0.4:
                pending = None
            elif age >= minimum_age:
                delta = point - pending["origin"]
                length = np.linalg.norm(delta)
                # Reject edge-clipped strokes, wrong-way jumps and side motion.
                if (
                    not near_edge
                    and 0.004 < length < 0.09
                    and delta @ pending["direction"] > 0.70 * length
                ):
                    samples.append(
                        {
                            "id": f"{run_id}:{pending['episode']}:{pending['sequence']}",
                            "group": pending["attempt_id"]
                            or f"{run_id}:ep{pending['episode']}",
                            "action": pending["action"],
                            "fraction": pending["fraction"],
                            "displacement": delta.tolist(),
                            "observed_after_s": age,
                            "source": "real_visual_gameplay",
                        }
                    )
                pending = None
        if event.get("gesture_sent"):
            pending = None  # Never fold a second stroke into the first target.
            action = event["action"]
            fraction = event.get("gesture_fraction", 1)
            if action and fraction >= 0.5 and not near_edge and latency < 180:
                sent = event.get("gesture_started_ns")
                if sent is None:
                    sent = event["dispatch_ns"] - int(
                        motion.get("stroke_duration_s", 0.08) * 1e9
                    )
                if 0 <= (sent - event["received_ns"]) / 1e9 < 0.10:
                    pending = {
                        "sent_ns": sent,
                        "origin": point,
                        "action": action,
                        "fraction": fraction,
                        "episode": event.get("episode"),
                        "sequence": event["sequence"],
                        "direction": DIRECTIONS[action],
                        "track_id": event.get("player_track_id"),
                        "attempt_id": event.get("attempt_id"),
                    }
    return samples


def fit_response(samples, baseline_displacements, minimum_train=10, minimum_test=5):
    groups = list(dict.fromkeys(s["group"] for s in samples))
    report = {
        "samples": len(samples),
        "independent_attempts": len(groups),
        "model_type": "action_conditioned_linear_drag_response",
        "target": "normalized visible player displacement per stroke",
        "score_optimization": False,
        "reward_available": False,
        "actions": {},
        "promoted_actions": [],
        "holdout_group": groups[-1] if groups else None,
    }
    model = np.asarray(baseline_displacements, float).copy()
    for action in range(1, 9):
        train = [
            s
            for s in samples
            if s["action"] == action and s["group"] != report["holdout_group"]
        ]
        test = [
            s
            for s in samples
            if s["action"] == action and s["group"] == report["holdout_group"]
        ]
        result = {
            "train_samples": len(train),
            "test_samples": len(test),
            "promoted": False,
        }
        report["actions"][str(action)] = result
        if len(groups) < 2 or len(train) < minimum_train or len(test) < minimum_test:
            result["reason"] = "insufficient_independent_attempt_coverage"
            continue
        x = np.array([s["fraction"] for s in train])
        y = np.array([s["displacement"] for s in train])
        center = np.median(y / x[:, None], axis=0)
        distance = np.linalg.norm(y / x[:, None] - center, axis=1)
        mad = np.median(abs(distance - np.median(distance)))
        keep = distance <= max(0.012, np.median(distance) + 3 * mad)
        x, y = x[keep], y[keep]
        result["retained_train_samples"] = len(x)
        if len(x) < minimum_train:
            result["reason"] = "too_many_uncertain_samples"
            continue
        candidate = (x[:, None] * y).sum(axis=0) / (x @ x)
        tx = np.array([s["fraction"] for s in test])
        ty = np.array([s["displacement"] for s in test])
        base_error = np.linalg.norm(ty - tx[:, None] * model[action], axis=1)
        new_error = np.linalg.norm(ty - tx[:, None] * candidate, axis=1)
        difference = new_error - base_error
        rng = np.random.default_rng(42)
        boot = difference[rng.integers(len(test), size=(2000, len(test)))].mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975])
        result.update(
            candidate_displacement=candidate.tolist(),
            baseline_mean_error=float(base_error.mean()),
            candidate_mean_error=float(new_error.mean()),
            error_difference_95pct_bootstrap=interval.tolist(),
        )
        base_norm, new_norm = np.linalg.norm(model[action]), np.linalg.norm(candidate)
        plausible = base_norm > 0 and 0.5 < new_norm / base_norm < 1.5
        if plausible and interval[1] < 0 and new_error.mean() < 0.8 * base_error.mean():
            model[action] = candidate
            result["promoted"] = True
            report["promoted_actions"].append(action)
        else:
            result["reason"] = (
                "heldout_improvement_or_calibration_consistency_gate_failed"
            )
    report["validated_displacements"] = model.tolist()
    return report


def train_real_response(config, runs_root="runs", output_root="models/real-response"):
    samples = []
    for path in sorted(Path(runs_root).glob("*/events.jsonl")):
        samples.extend(
            extract_strokes(
                read_events(path), path.parent.name, config["motion"], config["bounds"]
            )
        )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    period = config["motion"]["action_period_s"]
    report = fit_response(
        samples, np.asarray(config["planner"]["action_velocities"]) * period
    )
    report["trained_utc"] = datetime.now(timezone.utc).isoformat()
    report["input_signature"] = response_signature(config)
    report["dataset"] = str(output / "strokes.jsonl")
    (output / "strokes.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8"
    )
    (output / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output / f"{version}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
