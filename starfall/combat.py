"""Game-frame combat cues and charge-aware ultimate decisions; no health scans."""

from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from .types import Detection, Wall
from .planning import time_to_collision


@dataclass
class CombatReading:
    detections: list = field(default_factory=list)
    wall: Wall | None = None
    charges: int | None = None
    charge_confidence: float = 0.0
    button: tuple = (863, 348)


class CombatReader:
    def __init__(self, config):
        self.config = config
        self.reference = tuple(config.get("frame_size", [960, 430]))
        self.mask = np.sum((np.indices((32, 32)) - 15.5) ** 2, axis=0) < 14.5**2
        self.orb_refs = []
        self.orb_names = []
        for path in sorted(Path(config["orb_templates"]).glob("*.png")):
            self.orb_names.append(path.stem)
            self.orb_refs.append(self._vector(cv2.imread(str(path), 0)))
        if not self.orb_refs:
            raise ValueError("Combat requires reviewed orb icon references")
        self.orb_refs = np.asarray(self.orb_refs)
        self.core = cv2.imread(config["core_template"], 0)
        self.cores = [self.core] + [
            cv2.imread(path, 0) for path in config.get("core_variants", [])
        ]
        self.glyph = cv2.imread(config["ultimate_template"], 0)
        self.digits = [cv2.imread(path, 0) for path in config["charge_templates"]]
        if any(x is None for x in [*self.cores, self.glyph, *self.digits]):
            raise ValueError("Missing combat cue templates")

    def _vector(self, image):
        vector = cv2.resize(image, (32, 32)).astype(float)[self.mask]
        vector -= vector.mean()
        return vector / max(np.linalg.norm(vector), 1e-9)

    def _orbs(self, gray, hsv):
        # Separate gold/blue components avoid merging a gold orb with a blue
        # background glow. Shape proposals still require reviewed icon matches.
        h, s, v = cv2.split(hsv)
        gold = ((h <= 38) & (s >= 70) & (v >= 180)).astype(np.uint8) * 255
        blue = (
            ((h >= 95) & (h <= 145) & (s >= 90) & (v >= 165)) | ((v > 220) & (s < 90))
        ).astype(np.uint8) * 255
        candidates = []
        for mask in (gold, blue):
            mask[:55] = 0
            mask[345:] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for xx, yy, w, height, area in stats[1:]:
                if (
                    20 < w < 65
                    and 20 < height < 65
                    and 0.7 < w / height < 1.35
                    and area / (w * height) > 0.30
                ):
                    candidates.append((xx + w / 2, yy + height / 2, (w + height) / 4))
        result = []
        for x, y, radius in candidates[:30]:
            if (x < 158 and y > 292) or (x > 790 and y > 286) or (x > 700 and y < 82):
                continue
            best, name, center, size = 0.0, None, (x, y), radius
            for dx, dy in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
                for factor in (0.85, 0.95, 1.05):
                    r = radius * factor
                    a, b, c, d = map(
                        int, (x + dx - r, y + dy - r, x + dx + r + 1, y + dy + r + 1)
                    )
                    if a < 0 or b < 0 or c > gray.shape[1] or d > gray.shape[0]:
                        continue
                    scores = self.orb_refs @ self._vector(gray[b:d, a:c])
                    index = int(np.argmax(scores))
                    if scores[index] > best:
                        best, name, center, size = (
                            float(scores[index]),
                            self.orb_names[index],
                            (x + dx, y + dy),
                            r,
                        )
            if best >= self.config.get("orb_threshold", 0.74):
                label = "stellar_charge" if name == "actual-charge" else "power_orb"
                if not any(
                    np.linalg.norm(d.position * 960 - center) < 20 for d in result
                ):
                    result.append(
                        Detection(
                            "item", np.array(center) / 960, size / 960, best, label
                        )
                    )
        return result

    def _boss(self, image, gray):
        hsv = cv2.cvtColor(cv2.resize(image, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2HSV)
        warm = cv2.inRange(hsv, (0, 100, 100), (35, 255, 255))
        warm[:28] = 0
        warm[205:] = 0
        warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((15, 3), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(warm)
        candidates = [
            s for s in stats[1:] if s[3] > 105 and s[3] > s[2] * 2 and s[2] < 90
        ]
        if not candidates:
            return None, None
        x, y, w, h = (int(a) * 2 for a in max(candidates, key=lambda s: s[3])[:4])
        # Orange spine alone understates the dark rock spikes on its right.
        bounds = np.array([max(0, x - 10), 40, min(960, x + w + 80), 410]) / 960
        wall = Wall(tuple(bounds))
        left, right = max(0, x - 50), min(960, x + w + 50)
        crop = cv2.resize(gray[80:355, left:right], None, fx=0.5, fy=0.5)
        best, center = 0.0, None
        for core in self.cores:
            for factor in (0.85, 1.0, 1.15):
                template = cv2.resize(core, None, fx=factor * 0.5, fy=factor * 0.5)
                th, tw = template.shape
                if crop.shape[0] < th or crop.shape[1] < tw:
                    continue
                _, score, _, point = cv2.minMaxLoc(
                    cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
                )
                if score > best:
                    best = score
                    center = (
                        np.array([point[0] * 2 + left + tw, point[1] * 2 + 80 + th])
                        / 960
                    )
        core = (
            Detection("target", center, 0.025, best, "boss_core")
            if best >= self.config.get("core_threshold", 0.64)
            else None
        )
        return wall, core

    def _charges(self, gray):
        glyph_score = cv2.minMaxLoc(
            cv2.matchTemplate(gray[318:363, 834:894], self.glyph, cv2.TM_CCOEFF_NORMED)
        )[1]
        if glyph_score < 0.85:
            return None, 0.0
        scores = [
            cv2.minMaxLoc(
                cv2.matchTemplate(gray[366:383, 855:870], digit, cv2.TM_CCOEFF_NORMED)
            )[1]
            for digit in self.digits
        ]
        order = np.argsort(scores)
        best = int(order[-1])
        if scores[best] < 0.80 or scores[best] - scores[order[-2]] < 0.08:
            return None, float(scores[best])
        return best, float(scores[best])

    def read(self, image):
        if (image.shape[1], image.shape[0]) != self.reference:
            return CombatReading()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wall, core = self._boss(image, gray)
        detections = self._orbs(gray, cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        if core:
            detections.append(core)
        if wall:
            detections.append(
                Detection(
                    "wall",
                    np.array(
                        [
                            (wall.bounds[0] + wall.bounds[2]) / 2,
                            (wall.bounds[1] + wall.bounds[3]) / 2,
                        ]
                    ),
                    0.02,
                    0.9,
                    "magma_wall",
                )
            )
        charges, confidence = self._charges(gray)
        return CombatReading(
            detections,
            wall,
            charges,
            confidence,
            tuple(self.config.get("button", [863, 348])),
        )


class UltimatePolicy:
    def __init__(self, muzzle_offset=0.015):
        self.muzzle_offset = muzzle_offset
        self.reset()

    def reset(self):
        self.previous = None
        self.stable = 0
        self.last_fired = -float("inf")
        self.pending = None

    def choose(self, world, reading, now):
        charges = reading.charges
        self.stable = self.stable + 1 if charges == self.previous else 1
        self.previous = charges
        if self.pending is not None and charges is not None and charges < self.pending:
            self.pending = None
        if (
            charges is None
            or charges < 1
            or self.stable < 2
            or now - self.last_fired < 1.0
        ):
            return None
        if self.pending is not None and now - self.last_fired < 3.0:
            return None
        shot_y = world.player.position[1] + self.muzzle_offset
        cores = [
            t
            for t in world.targets
            if t.label == "boss_core" and world.time - t.last_seen < 0.15
        ]
        if any(
            t.position[0] > world.player.position[0]
            and abs(t.position[1] - shot_y) < 0.04
            for t in cores
        ):
            return "aligned_boss_core"
        threats = [
            h
            for h in world.hazards
            if h.position[0] > world.player.position[0]
            and h.position[0] - world.player.position[0] < 0.55
        ]
        aligned = [h for h in threats if abs(h.position[1] - shot_y) < h.radius + 0.04]
        if len(threats) >= 3 and aligned:
            return "meteor_cluster"
        imminent = any(
            time_to_collision(
                h.position - world.player.position,
                h.velocity - world.player.velocity,
                h.radius + world.player.radius,
                0.45,
            )
            <= 0.45
            for h in threats
        )
        if imminent:
            return "imminent_meteor"
        if charges == 2 and aligned:
            return "full_charges_with_target"
        return None

    def fired(self, charges, now):
        self.last_fired, self.pending = now, charges
