"""Receding-horizon path evaluation with swept-circle collision tests."""

from dataclasses import dataclass
import numpy as np
from .types import DIRECTIONS


def time_to_collision(relative_position, relative_velocity, radius, horizon=2.0):
    r, v = np.asarray(relative_position), np.asarray(relative_velocity)
    c = float(r @ r - radius * radius)
    if c <= 0:
        return 0.0
    a, b = float(v @ v), 2 * float(r @ v)
    disc = b * b - 4 * a * c
    if a < 1e-12 or b >= 0 or disc < 0:
        return float("inf")
    t = (-b - np.sqrt(disc)) / (2 * a)
    return float(t) if 0 <= t <= horizon else float("inf")


def limited_drag(world, action, action_velocities, period, margin=0.006):
    """Shorten the physical stroke, so model clipping cannot hide overshoot."""
    delta = np.asarray(action_velocities[action]) * period
    lo = np.array(world.bounds[:2]) + world.player.radius + margin
    hi = np.array(world.bounds[2:]) - world.player.radius - margin
    fraction = 1.0
    for axis in range(2):
        if delta[axis] > 1e-8:
            fraction = min(
                fraction, (hi[axis] - world.player.position[axis]) / delta[axis]
            )
        elif delta[axis] < -1e-8:
            fraction = min(
                fraction, (lo[axis] - world.player.position[axis]) / delta[axis]
            )
    fraction = float(np.clip(fraction, 0, 1))
    return DIRECTIONS[action] * fraction, fraction


def segment_box_clearance(start, end, lower, upper):
    """Distance from swept segments to a rectangle, negative for intersections."""
    delta = end - start
    moving = np.abs(delta) > 1e-12
    t0, t1 = np.full_like(delta, -np.inf), np.full_like(delta, np.inf)
    np.divide(lower - start, delta, out=t0, where=moving)
    np.divide(upper - start, delta, out=t1, where=moving)
    enter = np.maximum(np.minimum(t0, t1).max(axis=-1), 0)
    leave = np.minimum(np.maximum(t0, t1).min(axis=-1), 1)
    outside = (~moving & ((start < lower) | (start > upper))).any(axis=-1)
    intersects = (enter <= leave) & ~outside
    distance = np.minimum(
        np.linalg.norm(start - np.clip(start, lower, upper), axis=-1),
        np.linalg.norm(end - np.clip(end, lower, upper), axis=-1),
    )
    denom = np.maximum(np.sum(delta * delta, axis=-1), 1e-12)
    for corner in (
        (lower[0], lower[1]),
        (lower[0], upper[1]),
        (upper[0], lower[1]),
        (upper[0], upper[1]),
    ):
        relative = np.array(corner) - start
        u = np.clip(np.sum(relative * delta, axis=-1) / denom, 0, 1)
        distance = np.minimum(
            distance, np.linalg.norm(relative - u[..., None] * delta, axis=-1)
        )
    return np.where(intersects, -0.001, distance)


@dataclass
class Decision:
    action: int
    costs: np.ndarray
    clearances: np.ndarray
    safe_actions: np.ndarray
    path: np.ndarray
    objective: str = "survival"


class PredictivePlanner:
    def __init__(
        self,
        speed=0.35,
        horizon=0.6,
        step=0.05,
        delay=0.05,
        margin=0.012,
        uncertainty_scale=1.5,
        item_values=None,
        shot_lane_weight=0.0,
        action_velocities=None,
        allowed_actions=None,
        clearance_weight=0.25,
        boss_lane_weight=0.0,
        muzzle_offset=0.015,
        projectile_speed=1.2,
    ):
        self.speed, self.horizon, self.step = speed, horizon, step
        self.delay, self.margin, self.uncertainty_scale = (
            delay,
            margin,
            uncertainty_scale,
        )
        self.item_values = item_values or {"item": 1.0}
        self.shot_lane_weight = shot_lane_weight
        self.clearance_weight, self.boss_lane_weight = (
            clearance_weight,
            boss_lane_weight,
        )
        self.muzzle_offset, self.projectile_speed = muzzle_offset, projectile_speed
        if projectile_speed <= 0:
            raise ValueError("Projectile speed must be positive")
        self.action_velocities = (
            np.array(action_velocities, dtype=float)
            if action_velocities is not None
            else DIRECTIONS * self.speed
        )
        if (
            self.action_velocities.shape != (9, 2)
            or not np.isfinite(self.action_velocities).all()
        ):
            raise ValueError("Expected nine finite action velocity vectors")
        self.allowed_actions = set(allowed_actions or range(9))
        if 0 not in self.allowed_actions or not self.allowed_actions.issubset(range(9)):
            raise ValueError("Allowed actions must include stop and use indices 0..8")
        self.last_action = 0

    def choose(self, world, observation_age=0.0):
        n = max(2, int(np.ceil(self.horizon / self.step)))
        dt = self.horizon / n
        times = np.arange(n + 1) * dt
        # Evaluate 81 two-stage trajectories to avoid choosing a dead end.
        first = np.repeat(np.arange(9), 9)
        second = np.tile(np.arange(9), 9)
        path = np.empty((81, n + 1, 2))
        path[:, 0] = world.player.position
        delay = min(self.horizon, self.delay + max(0.0, observation_age))
        lo = np.array(world.bounds[:2]) + world.player.radius
        hi = np.array(world.bounds[2:]) - world.player.radius
        if np.any(hi <= lo):
            raise ValueError("Arena bounds smaller than player")
        for k in range(1, n + 1):
            if times[k] <= delay:
                velocity = np.broadcast_to(world.player.velocity, (81, 2))
            else:
                actions = first if times[k] <= max(0.2, delay + dt) else second
                velocity = self.action_velocities[actions]
            path[:, k] = np.clip(path[:, k - 1] + velocity * dt, lo, hi)
        clearance = np.full(81, np.inf)
        for hazard in world.hazards:
            hp = hazard.position + (times[:, None] + observation_age) * hazard.velocity
            relative = path - hp[None]
            # Minimum distance along each relative motion segment (no tunneling).
            start, delta = relative[:, :-1], np.diff(relative, axis=1)
            denom = np.sum(delta * delta, axis=2)
            u = np.clip(-np.sum(start * delta, axis=2) / np.maximum(denom, 1e-12), 0, 1)
            distance = np.linalg.norm(start + u[..., None] * delta, axis=2)
            age = max(0.0, world.time - hazard.last_seen)
            pvar = np.trace(hazard.covariance[:2, :2])
            vvar = np.trace(hazard.covariance[2:, 2:])
            uncertainty = self.uncertainty_scale * np.sqrt(
                pvar + (times[1:] + age) ** 2 * vvar
            )
            radius = world.player.radius + hazard.radius + self.margin + uncertainty
            clearance = np.minimum(clearance, np.min(distance - radius, axis=1))
        for wall in world.walls:
            wp = (times[:, None] + observation_age) * wall.velocity
            radius = world.player.radius + self.margin
            lower = np.array(wall.bounds[:2]) - radius
            upper = np.array(wall.bounds[2:]) + radius
            wall_clearance = segment_box_clearance(
                path[:, :-1] - wp[None, :-1], path[:, 1:] - wp[None, 1:], lower, upper
            )
            clearance = np.minimum(clearance, wall_clearance.min(axis=1))
        finite = np.where(np.isfinite(clearance), clearance, 1.0)
        # Within safe paths, proximity is a bounded preference. It must not
        # overwhelm shooting and pickups simply because a meteor is nearby.
        costs = self.clearance_weight * np.exp(-np.maximum(finite, 0) / 0.035)
        edge = np.minimum(path - lo, hi - path).min(axis=(1, 2))
        costs += 0.003 / np.maximum(edge, 0.004)
        costs += 0.03 * (first != self.last_action)
        costs += 0.02 * (first != second)
        disabled = ~np.isin(first, list(self.allowed_actions)) | ~np.isin(
            second, list(self.allowed_actions)
        )
        # Automatic shots travel right. Align throughout the candidate path,
        # with a provisional projectile lead; actual hitboxes/speed need data.
        targets = [t for t in world.targets if t.label == "boss_core"] or world.hazards
        attack_reward = np.zeros(81)
        for target in targets:
            future = (
                target.position + (times[:, None] + observation_age) * target.velocity
            )
            ahead = future[None, :, 0] > path[:, :, 0] + world.player.radius
            flight = (
                np.maximum(future[None, :, 0] - path[:, :, 0], 0)
                / self.projectile_speed
            )
            shot_y = path[:, :, 1] + self.muzzle_offset
            difference = future[None, :, 1] + flight * target.velocity[1] - shot_y
            width = 0.015 + min(target.radius * 0.25, 0.018)
            alignment = np.exp(-0.5 * (difference / width) ** 2) * ahead
            weight = (
                self.boss_lane_weight
                if target.label == "boss_core"
                else self.shot_lane_weight
            )
            attack_reward = np.maximum(attack_reward, weight * alignment.mean(axis=1))
        costs -= attack_reward
        for item in world.items:
            value = self.item_values.get(item.label, self.item_values.get("item", 0.0))
            if (
                item.label == "shield"
                and world.shield_until is not None
                and world.shield_until > world.time
            ):
                value *= 0.15
            ip = item.position + (times[:, None] + observation_age) * item.velocity
            distances = np.linalg.norm(path - ip[None], axis=2)
            # Reward progress throughout the path and earlier collection. A
            # minimum-distance-only reward lets MPC postpone the pickup forever.
            value = min(max(value, 0), 5.0)
            costs -= value * np.exp(-distances / 0.15).mean(axis=1)
            contact = distances < world.player.radius + item.radius
            costs -= value * (contact * np.exp(-times / 0.35)).max(axis=1)
        # Collision feasibility is decided after goals, so even a boss or many
        # valuable orbs cannot buy permission to cross a predicted obstacle.
        costs = np.clip(costs, -100, 100)
        costs[finite < 0] = 10000 - 100000 * finite[finite < 0]
        costs[disabled] += 1e8
        best_second = np.argmin(costs.reshape(9, 9), axis=1)
        rows = np.arange(9) * 9 + best_second
        action_costs = costs[rows]
        selected = int(np.argmin(action_costs))
        self.last_action = selected
        return Decision(
            selected,
            action_costs,
            clearance[rows],
            (clearance[rows] >= 0) & np.isin(np.arange(9), list(self.allowed_actions)),
            path[rows[selected]],
            "boss_core"
            if world.targets
            else "shoot_and_collect"
            if world.hazards or world.items
            else "survival",
        )

    def shield(self, world, proposed, observation_age=0.0):
        result = self.choose(world, observation_age)
        if result.safe_actions[int(proposed)]:
            return int(proposed), False
        return result.action, True
