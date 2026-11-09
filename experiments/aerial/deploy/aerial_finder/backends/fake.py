"""Kinematic stand-in world for tests and service smoke runs without a simulator.

Ground plane at z = 0, optional box obstacles, and point targets. ``FakeDetector``
reports a target's projected box when it is in view and within range, so the
whole mission loop (detect -> locate -> track -> confirm -> report) runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geo import GeoRef
from ..perception.locate import world_to_pixel
from ..types import Frame, Intrinsics, VehicleState
from .base import body_delta_to_world_velocity


@dataclass
class FakeTarget:
    pos: np.ndarray
    label: str = "stool"
    color: str = "red"
    size_m: float = 0.5


@dataclass
class FakeWorld:
    targets: List[FakeTarget] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float, float]] = field(default_factory=list)  # x0,y0,x1,y1,height


class FakeBackend:
    def __init__(self, world: FakeWorld, start=(0.0, 0.0, 5.0), yaw: float = 0.0, step_hz: float = 5.0,
                 width: int = 640, height: int = 360, hfov_deg: float = 90.0,
                 geo_ref: GeoRef = GeoRef(47.641468, -122.140165, 122.0)) -> None:
        self.world = world
        self.pos = np.asarray(start, np.float64).copy()
        self.yaw = float(yaw)
        self.step_hz = float(step_hz)
        self.intr = Intrinsics.from_hfov(width, height, hfov_deg)
        self.geo_ref = geo_ref
        self.t = 0.0
        self.collided = False
        self.n_steps = 0

    def _occupied(self, p: np.ndarray) -> bool:
        return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] <= h for x0, y0, x1, y1, h in self.world.boxes)

    def state(self) -> VehicleState:
        return VehicleState(pos=self.pos.copy(), yaw=self.yaw, collided=self.collided, t=self.t)

    def observe(self) -> Tuple[Frame, VehicleState]:
        rgb = np.full((self.intr.height, self.intr.width, 3), 128, np.uint8)
        depth = self._depth()
        return Frame(rgb=rgb, rgb_intr=self.intr, depth=depth, depth_intr=self.intr, t=self.t), self.state()

    def _depth(self) -> np.ndarray:
        """Planar depth of the ground plane (sky = inf); obstacles are not rendered."""
        i = self.intr
        v = np.arange(i.height, dtype=np.float64)[:, None]
        yn = (v - i.cy) / i.fy
        with np.errstate(divide="ignore"):
            z = np.where(yn > 1e-6, self.pos[2] / yn, np.inf)
        return np.repeat(z, i.width, axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> None:
        dt = 1.0 / self.step_hz
        vn, ve, vu, yr = body_delta_to_world_velocity(action, self.yaw, dt)
        nxt = self.pos + dt * np.array([vn, ve, vu])
        if self._occupied(nxt):
            self.collided = True
        else:
            self.pos = nxt
        self.yaw = float((self.yaw + dt * yr + np.pi) % (2 * np.pi) - np.pi)
        self.t += dt
        self.n_steps += 1

    def hover(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeYolo:
    """Stands in for ``YoloWorld``: boxes for in-view targets within ``max_range_m``."""

    def __init__(self, backend: FakeBackend, max_range_m: float = 35.0, conf: float = 0.4,
                 pixel_noise: float = 1.0, seed: int = 0) -> None:
        self.backend = backend
        self.max_range_m = float(max_range_m)
        self.conf = float(conf)
        self.noise = float(pixel_noise)
        self.rng = np.random.default_rng(seed)
        self.prompts: List[str] = []
        self.last_rgb: Optional[np.ndarray] = None

    def set_prompts(self, prompts: Sequence[str]) -> None:
        self.prompts = list(prompts)

    def __call__(self, rgb: np.ndarray):
        b = self.backend
        st = b.state()
        out = []
        for tg in b.world.targets:
            if not any(tg.label in p for p in self.prompts):
                continue
            uvz = world_to_pixel(tg.pos, st, b.intr)
            if uvz is None:
                continue
            u, v, z = uvz
            r = float(np.linalg.norm(tg.pos - st.pos))
            if r > self.max_range_m or not (0 <= u < b.intr.width and 0 <= v < b.intr.height):
                continue
            half = 0.5 * tg.size_m * b.intr.fx / z
            u += self.rng.normal(0, self.noise)
            v += self.rng.normal(0, self.noise)
            bbox = (u - half, v - half, u + half, v + half)
            x0, y0, x1, y1 = (int(max(0, round(c))) for c in bbox)
            if tg.color == "red":
                rgb[y0:y1 + 1, x0:x1 + 1] = (200, 20, 20)
            prompt = next(p for p in self.prompts if tg.label in p)
            out.append((bbox, self.conf, prompt))
        self.last_rgb = rgb
        return out


class GoalSeekNavigator:
    """Turn toward the goal, then fly forward (fake-world tests; no learned model)."""

    uses_env = False

    def reset(self) -> None:
        pass

    def set_band(self, z_now: float, goal_z: float) -> None:
        pass

    def observe(self, depth, env, executed) -> None:
        pass

    def act(self, vs: VehicleState, goal: np.ndarray) -> np.ndarray:
        d = np.asarray(goal[:2], np.float64) - vs.pos[:2]
        b = float((np.arctan2(d[1], d[0]) - vs.yaw + np.pi) % (2 * np.pi) - np.pi)
        turn = float(np.clip(b, -0.314, 0.314))
        fwd = 0.8 if abs(b) < 0.5 else 0.1
        dz = float(np.clip(float(goal[2]) - vs.pos[2], -0.3, 0.3))
        return np.array([min(fwd, float(np.linalg.norm(d))), 0.0, dz, turn])
