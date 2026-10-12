"""Synthetic engineering sandbox. Defaults are NOT measured MLBB mechanics."""

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .types import DIRECTIONS, World, make_track
from .features import encode, OBS_SIZE
from .planning import time_to_collision


class StarfallSim(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, duration=30, randomize=True, render_mode=None):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_SIZE,), dtype=np.float32)
        self.duration, self.randomize, self.render_mode = (
            duration,
            randomize,
            render_mode,
        )
        self.dt = 0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = self.score = self.collections = 0
        self.speed = self.np_random.uniform(0.28, 0.42) if self.randomize else 0.35
        self.response = self.np_random.uniform(0.3, 0.8) if self.randomize else 0.7
        self.rate = self.np_random.uniform(2.0, 4.0) if self.randomize else 3.0
        self.player = make_track(0, "player", 0.5, 0.65, radius=0.018)
        self.hazards, self.items = [], []
        self.identity = 1
        self.shield = 0.0
        self.collisions = 0
        self.done = False
        return encode(self.world()), {"synthetic": True}

    def world(self):
        return World(
            self.player,
            self.hazards,
            self.items,
            (0.0, 0.0, 1.0, 1.0),
            self.time,
            health=1.0,
            score=self.score,
            shield_until=self.shield,
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset after termination")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        rng, dt = self.np_random, self.dt
        self.time += dt
        old_player = self.player.position.copy()
        self.player.state[2:] += self.response * (
            DIRECTIONS[action] * self.speed - self.player.velocity
        )
        self.player.state[:2] = np.clip(
            old_player + self.player.velocity * dt,
            self.player.radius,
            1 - self.player.radius,
        )
        if rng.random() < self.rate * dt:
            # Real inspection established right-to-left hazards; speed, angle,
            # spawn distribution and hitboxes below remain synthetic guesses.
            x, y = 1.05, rng.uniform(0, 1)
            velocity = np.array([-rng.uniform(0.15, 0.5), rng.uniform(-0.10, 0.10)])
            self.hazards.append(
                make_track(
                    self.identity,
                    "hazard",
                    x,
                    y,
                    *velocity,
                    radius=rng.uniform(0.012, 0.035),
                    time=self.time,
                )
            )
            self.identity += 1
        if rng.random() < 0.22 * dt:
            self.items.append(
                make_track(
                    self.identity,
                    "item",
                    rng.uniform(0.08, 0.92),
                    rng.uniform(0.15, 0.9),
                    radius=0.016,
                    label="shield" if rng.random() < 0.3 else "item",
                    time=self.time,
                )
            )
            self.identity += 1
        hit = False
        removed = set()
        pv = (self.player.position - old_player) / dt
        for h in self.hazards:
            if (
                time_to_collision(
                    h.position - old_player,
                    h.velocity - pv,
                    h.radius + self.player.radius,
                    dt,
                )
                <= dt
            ):
                if self.shield > self.time:
                    removed.add(h.id)
                else:
                    hit = True
            h.state[:2] += h.velocity * dt
            h.time = h.last_seen = self.time
        self.hazards = [
            h
            for h in self.hazards
            if -0.15 < h.position[0] < 1.15
            and -0.15 < h.position[1] < 1.15
            and h.id not in removed
        ]
        collected = []
        for item in self.items:
            if (
                time_to_collision(
                    item.position - old_player,
                    -pv,
                    item.radius + self.player.radius,
                    dt,
                )
                <= dt
            ):
                collected.append(item.id)
                if item.label == "shield":
                    self.shield = self.time + 2.5
        self.items = [i for i in self.items if i.id not in collected]
        self.collections += len(collected)
        self.collisions += int(hit)
        gained = 10 * dt + 25 * len(collected)
        self.score += gained
        components = {
            "score": gained / 100,
            "survival": 0.01 * dt,
            "death": -5.0 * int(hit),
        }
        self.done = bool(hit or self.time >= self.duration)
        info = {
            "synthetic": True,
            "score": self.score,
            "survival_s": self.time,
            "collections": self.collections,
            "collisions": self.collisions,
            "reward_components": components,
        }
        return (
            encode(self.world()),
            sum(components.values()),
            bool(hit),
            bool(self.time >= self.duration),
            info,
        )

    def render(self):
        image = np.full((640, 640, 3), 20, dtype=np.uint8)
        for track, color in (
            [(self.player, (100, 240, 120))]
            + [(h, (80, 80, 240)) for h in self.hazards]
            + [(i, (240, 200, 50)) for i in self.items]
        ):
            center = tuple(np.rint(track.position * 640).astype(int))
            cv2.circle(image, center, max(2, int(track.radius * 640)), color, -1)
        cv2.putText(
            image,
            f"SYNTHETIC SANDBOX  score {self.score:.0f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
        )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
