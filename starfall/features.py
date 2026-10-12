import numpy as np
from .planning import time_to_collision

HAZARDS = 12
ITEMS = 4
OBS_SIZE = 12 + HAZARDS * 8 + ITEMS * 6


def encode(world):
    p = world.player
    x0, y0, x1, y1 = world.bounds
    # Unknown visual health/score/effects have explicit validity masks.
    output = [
        p.position[0],
        p.position[1],
        *p.velocity,
        world.health or 0.0,
        float(world.health is not None),
        float(world.shield_until is not None and world.shield_until > world.time),
        float(world.shield_until is not None),
        p.position[0] - x0,
        x1 - p.position[0],
        p.position[1] - y0,
        y1 - p.position[1],
    ]
    hazards = sorted(
        world.hazards,
        key=lambda h: (
            time_to_collision(
                h.position - p.position, h.velocity - p.velocity, h.radius + p.radius
            ),
            np.linalg.norm(h.position - p.position),
        ),
    )[:HAZARDS]
    for h in hazards:
        ttc = time_to_collision(
            h.position - p.position, h.velocity - p.velocity, h.radius + p.radius
        )
        output += [
            *(h.position - p.position),
            *h.velocity,
            h.radius,
            min(ttc, 2.0) / 2,
            min(np.sqrt(np.trace(h.covariance[:2, :2])), 1),
            1.0,
        ]
    output += [0.0] * ((HAZARDS - len(hazards)) * 8)
    items = sorted(world.items, key=lambda i: np.linalg.norm(i.position - p.position))[
        :ITEMS
    ]
    for i in items:
        output += [
            *(i.position - p.position),
            *i.velocity,
            float(i.label == "shield"),
            1.0,
        ]
    output += [0.0] * ((ITEMS - len(items)) * 6)
    return np.clip(np.array(output, dtype=np.float32), -2, 2)
