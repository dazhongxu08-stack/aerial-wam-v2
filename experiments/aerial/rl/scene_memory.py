"""Seen-scene memory and a prior schematic, joined only by a subgoal.

The consequence model is unchanged. A subgoal replaces the task goal inside
``goal_features`` for one decision; survival is still scored on the live depth.
Nothing here vetoes a candidate.

Pieces taken from public systems, and the part deliberately left out:

* RTAB-Map rehearsal (Labbé & Michaud, IROS 2011 / TRO 2013): a new view that
  is similar to a stored place is merged and raises that place's weight. Their
  threshold is on bag-of-words similarity; here the score is cosine distance
  on the L2-normalized DA3 scene vector the progress head already consumes.
* PlaceNav (Suomela et al., 2024) and ViKiNG (Shah & Levine, RSS 2022): a map
  proposes the next place, and the point handed to the local model lies inside
  that model's horizon.
* Hydra (Hughes et al., RSS 2022): the active window stays the live view. Stored
  depth is for later visits, never a model input.
* NanoMap (Florence, Tedrake et al., ICRA 2018): each depth image stays with the
  pose that saw it. A query reads the newest image whose camera still contains
  the sample. Images are not fused into a voxel grid, and this memory does not
  veto a candidate.

A blocked heading is a direction that already lost survival or hit something.
It removes that schematic edge. Another free edge inside the horizon may replace
it. If none does, the subgoal is dropped instead of aiming along the task-goal
chord. Recorded path points are not a subgoal, and a block does not forbid
candidates or invent a passage that was never flown.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.aerial.rl.far_continuation import FlightContinuation

# New keyframe when the nose has turned this far, even if the vector still matches.
YAW_NEW_RAD = np.deg2rad(30.0)
# Same-looking corridor: drop another keyframe after this much travel.
GAP_M = 12.0
# Cosine on a unit scene vector. Merge is stricter than recall.
MERGE_COSINE = 0.85
MATCH_COSINE = 0.75
METRIC_MATCH_M = 15.0
ROUGH_MATCH_M = 40.0
BLOCK_YAW_RAD = 0.50
PASS_MIN_M = 1.0
# Camera that produced the stored depth. Matches ``depth_points`` in the eval.
CAM_DZ = -1.0
HFOV_DEG = 90.0
VFOV_DEG = 58.7
# A sample this far past the measured surface is through it. NanoMap's vehicle
# radius is 0.4 m. This is only for placing the subgoal, not for deleting candidates.
SURFACE_PASS_M = 0.4
_AIM_OFFSETS_DEG = (10, 20, 30, 45, 60, 75)


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-8:
        return np.zeros_like(x)
    return x / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_delta(a: float, b: float) -> float:
    return abs(_wrap(a - b))


def _heading(here: np.ndarray, there: np.ndarray) -> float:
    return float(np.arctan2(there[1] - here[1], there[0] - here[0]))


def _blocked(heading: float, yaws: Sequence[float], tol: float = BLOCK_YAW_RAD) -> bool:
    return any(_yaw_delta(heading, float(y)) <= tol for y in yaws)


def _clamp_xy(here: np.ndarray, target: np.ndarray, horizon_m: float) -> Optional[np.ndarray]:
    """Point on the segment here→target, at most ``horizon_m`` away. None if closer than 0.5 m."""
    delta = np.asarray(target, np.float64)[:2] - np.asarray(here, np.float64)[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 0.5:
        return None
    step = min(dist, float(horizon_m))
    xy = np.asarray(here, np.float64)[:2] + delta / dist * step
    z = float(np.asarray(here, np.float64)[2]) if len(np.asarray(here).reshape(-1)) > 2 else 0.0
    return np.array([xy[0], xy[1], z], np.float64)


def _intrinsics(shape: Sequence[int]) -> Tuple[float, float]:
    h, w = int(shape[0]), int(shape[1])
    fx = (w / 2) / np.tan(np.deg2rad(HFOV_DEG) / 2)
    fy = (h / 2) / np.tan(np.deg2rad(VFOV_DEG) / 2)
    return float(fx), float(fy)


def point_at_pixel(v: int, u: int, depth_m: float, pose: np.ndarray, yaw: float,
                   shape: Sequence[int]) -> np.ndarray:
    """World point on the ray through pixel (v, u), ``depth_m`` along the optical axis."""
    h, w = int(shape[0]), int(shape[1])
    fx, fy = _intrinsics((h, w))
    fwd = float(depth_m)
    right = (u + 0.5 - w / 2) / fx * fwd
    down = (v + 0.5 - h / 2) / fy * fwd
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    origin = np.asarray(pose, np.float64)[:3]
    return np.array([
        origin[0] + fwd * c - right * s,
        origin[1] + fwd * s + right * c,
        origin[2] + CAM_DZ - down,
    ], np.float64)


def project_pixel(point: np.ndarray, pose: np.ndarray, yaw: float,
                  shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Pixel (v, u) if ``point`` lies in this camera's image. None otherwise."""
    h, w = int(shape[0]), int(shape[1])
    fx, fy = _intrinsics((h, w))
    origin = np.asarray(pose, np.float64)[:3]
    d = np.asarray(point, np.float64)[:3] - np.array([origin[0], origin[1], origin[2] + CAM_DZ])
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    fwd = c * d[0] + s * d[1]
    if fwd <= 0.05:
        return None
    right = -s * d[0] + c * d[1]
    down = -d[2]
    u = right / fwd * fx + w / 2 - 0.5
    v = down / fwd * fy + h / 2 - 0.5
    ui, vi = int(np.rint(u)), int(np.rint(v))
    if 0 <= ui < w and 0 <= vi < h:
        return vi, ui
    return None


@dataclass
class DepthView:
    """One depth image and the body pose that captured it. Not fused with other views."""

    pose: np.ndarray
    yaw: float
    depth: np.ndarray

    def to_json(self) -> Dict[str, Any]:
        return {
            "pose": np.asarray(self.pose, np.float64).round(3).tolist(),
            "yaw": round(float(self.yaw), 4),
            "depth": np.asarray(self.depth, np.float32).round(2).tolist(),
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "DepthView":
        return cls(
            pose=np.asarray(raw["pose"], np.float64),
            yaw=float(raw["yaw"]),
            depth=np.asarray(raw["depth"], np.float32),
        )


@dataclass
class Keyframe:
    id: int
    vector: np.ndarray
    pose: np.ndarray
    yaw: float
    weight: int = 1
    passed: List[np.ndarray] = field(default_factory=list)
    blocked_yaw: List[float] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vector": np.asarray(self.vector, np.float32).round(5).tolist(),
            "pose": np.asarray(self.pose, np.float64).round(3).tolist(),
            "yaw": round(float(self.yaw), 4),
            "weight": int(self.weight),
            "passed": [np.asarray(p, np.float64).round(3).tolist() for p in self.passed],
            "blocked_yaw": [round(float(y), 4) for y in self.blocked_yaw],
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "Keyframe":
        return cls(
            id=int(raw["id"]),
            vector=_unit(np.asarray(raw["vector"], np.float64)),
            pose=np.asarray(raw["pose"], np.float64),
            yaw=float(raw["yaw"]),
            weight=int(raw.get("weight", 1)),
            passed=[np.asarray(p, np.float64) for p in raw.get("passed") or []],
            blocked_yaw=[float(y) for y in raw.get("blocked_yaw") or []],
        )


@dataclass
class Proposal:
    """    ``aim`` is the goal point the consequence model should score. ``source`` is
    ``schematic``, ``far`` (this flight's continuation, clamped to the horizon),
    ``goal`` (task goal, unchanged), or ``dropped`` (the edge was
    blocked and no other edge remains; do not fly the task-goal chord)."""

    aim: np.ndarray
    source: str
    kf_id: Optional[int]


class SceneMemory:
    def __init__(self, merge_cosine: float = MERGE_COSINE, match_cosine: float = MATCH_COSINE,
                 yaw_new_rad: float = YAW_NEW_RAD, gap_m: float = GAP_M) -> None:
        self.merge_cosine = float(merge_cosine)
        self.match_cosine = float(match_cosine)
        self.yaw_new_rad = float(yaw_new_rad)
        self.gap_m = float(gap_m)
        self.frames: List[Keyframe] = []
        self.views: List[DepthView] = []
        self._next = 0

    def match(self, pose: np.ndarray, vector: np.ndarray, radius_m: float) -> Optional[Keyframe]:
        if vector is None or not self.frames:
            return None
        here = np.asarray(pose, np.float64)[:3]
        best, score = None, self.match_cosine
        for kf in self.frames:
            if float(np.linalg.norm(kf.pose[:2] - here[:2])) > radius_m:
                continue
            s = cosine(vector, kf.vector)
            if s >= score:
                best, score = kf, s
        return best

    def observe(self, pose: np.ndarray, yaw: float, vector: np.ndarray,
                depth: Optional[np.ndarray], trust: str) -> Keyframe:
        """Store this view. Similar places still merge into one keyframe.

        The depth image is kept with this pose and is not fused into other images.
        A rough pose that matches an old place is written at that place, not at
        the drifted odometry. The depth image stays with the pose that captured it.
        """
        here = np.asarray(pose, np.float64)[:3].copy()
        yaw = float(yaw)
        vec = _unit(vector)
        self._remember_depth(here, yaw, depth)
        radius = ROUGH_MATCH_M if trust == "rough" else METRIC_MATCH_M
        known = self.match(here, vec, radius)
        last = self.frames[-1] if self.frames else None
        if last is not None and self._same_view(last, here, yaw, vec):
            self._merge(last, vec)
            return last
        # Revisit of an older place. Extending past gap_m from the latest frame
        # stays a new keyframe so one corridor does not collapse onto its first pose.
        if (known is not None and known is not last
                and _yaw_delta(known.yaw, yaw) <= self.yaw_new_rad
                and cosine(vec, known.vector) >= self.merge_cosine):
            self._merge(known, vec)
            return known
        kf = Keyframe(id=self._next, vector=vec, pose=here, yaw=yaw)
        self._next += 1
        self.frames.append(kf)
        return kf

    def _remember_depth(self, pose: np.ndarray, yaw: float, depth: Optional[np.ndarray]) -> None:
        if depth is None:
            return
        image = np.asarray(depth, np.float32)
        if image.ndim != 2 or image.size == 0:
            return
        self.views.append(DepthView(pose=np.asarray(pose, np.float64)[:3].copy(), yaw=float(yaw),
                                    depth=image.copy()))

    def query_depth(self, point: np.ndarray) -> Optional[float]:
        """Depth from the newest stored image whose camera still contains ``point``.

        Older images are not consulted once a newer one sees that direction.
        No stored image yields None. This does not score or reject a candidate.
        """
        found = self._containing(point)
        if found is None:
            return None
        _view, _pix, reading = found
        if not np.isfinite(reading) or reading <= 0.05:
            return None
        return reading

    def _containing(self, point: np.ndarray) -> Optional[Tuple["DepthView", Tuple[int, int], float]]:
        sample = np.asarray(point, np.float64)[:3]
        for view in reversed(self.views):
            pix = project_pixel(sample, view.pose, view.yaw, view.depth.shape)
            if pix is None:
                continue
            return view, pix, float(view.depth[pix])
        return None

    def ray_known(self, here: np.ndarray, aim: np.ndarray,
                  margin: float = SURFACE_PASS_M) -> Optional[bool]:
        """Whether any stored view sees the segment here→aim.

        True when the segment passes a surface. False when a view contains it and
        the surface is farther than the segment. None when no view contains it.
        A direction outside every camera is not treated as clear.
        """
        a = np.asarray(here, np.float64)[:3]
        b = np.asarray(aim, np.float64)[:3]
        delta = b - a
        dist = float(np.linalg.norm(delta[:2]))
        if dist < 0.5:
            return False
        contained = False
        n = max(2, int(np.ceil(dist / 0.5)))
        for t in np.linspace(0.5 / dist, 1.0, n):
            sample = a + float(t) * delta
            found = self._containing(sample)
            if found is None:
                continue
            contained = True
            view, _pix, reading = found
            if not np.isfinite(reading) or reading <= 0.05:
                continue
            cam = np.array([view.pose[0], view.pose[1], float(view.pose[2]) + CAM_DZ], np.float64)
            if float(np.linalg.norm(sample - cam)) > reading + margin:
                return True
        return False if contained else None

    def ray_hits_surface(self, here: np.ndarray, aim: np.ndarray, margin: float = SURFACE_PASS_M) -> bool:
        """True when the segment here→aim passes the surface in the newest view that sees it."""
        return self.ray_known(here, aim, margin) is True

    def ray_status(self, here: np.ndarray, aim: np.ndarray, margin: float = SURFACE_PASS_M
                   ) -> Optional[bool]:
        """False when here→aim stays in stored views and in front of the surface.

        True when a stored view sees the segment pass a surface. None when no
        view contains it, or the segment leaves the seen region before the end.
        Samples before the first seen one are the camera's near gap, not open space.
        """
        a = np.asarray(here, np.float64)[:3]
        b = np.asarray(aim, np.float64)[:3]
        delta = b - a
        dist = float(np.linalg.norm(delta[:2]))
        if dist < 0.5:
            return False
        seen = False
        n = max(2, int(np.ceil(dist / 0.5)))
        for t in np.linspace(0.5 / dist, 1.0, n):
            sample = a + float(t) * delta
            found = self._containing(sample)
            if found is None:
                if seen:
                    return None
                continue
            seen = True
            view, _pix, reading = found
            if not np.isfinite(reading) or reading <= 0.05:
                return None
            cam = np.array([view.pose[0], view.pose[1], float(view.pose[2]) + CAM_DZ], np.float64)
            if float(np.linalg.norm(sample - cam)) > float(reading) + margin:
                return True
        return False if seen else None

    def _same_view(self, kf: Keyframe, pose: np.ndarray, yaw: float, vector: np.ndarray) -> bool:
        return (float(np.linalg.norm(kf.pose[:2] - pose[:2])) <= self.gap_m
                and _yaw_delta(kf.yaw, yaw) <= self.yaw_new_rad
                and cosine(vector, kf.vector) >= self.merge_cosine)

    def _merge(self, kf: Keyframe, vector: np.ndarray) -> None:
        kf.weight += 1
        kf.vector = _unit(kf.vector * (kf.weight - 1) + _unit(vector))

    def note(self, kf_id: Optional[int], yaw: float, surv: float, d_before: float, d_after: float,
             collided: bool, surv_min: float, pose: np.ndarray) -> None:
        kf = self._by_id(kf_id)
        if kf is None:
            return
        heading = float(yaw)
        if collided or float(surv) < float(surv_min):
            if not _blocked(heading, kf.blocked_yaw, 0.2):
                kf.blocked_yaw.append(heading)
            return
        if float(d_before) - float(d_after) < 0.3:
            return
        # This heading got through, so a miss recorded on the same view no longer stands.
        kf.blocked_yaw = [y for y in kf.blocked_yaw if not _blocked(heading, (y,))]
        point = np.asarray(pose, np.float64)[:3]
        if any(float(np.linalg.norm(point[:2] - p[:2])) < PASS_MIN_M for p in kf.passed):
            return
        kf.passed.append(point.copy())

    def _by_id(self, kf_id: Optional[int]) -> Optional[Keyframe]:
        if kf_id is None:
            return None
        for kf in self.frames:
            if kf.id == kf_id:
                return kf
        return None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "keyframes": [kf.to_json() for kf in self.frames],
            "views": [view.to_json() for view in self.views],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SceneMemory":
        mem = cls()
        path = Path(path)
        if not path.is_file():
            return mem
        raw = json.loads(path.read_text(encoding="utf-8"))
        mem.frames = [Keyframe.from_json(k) for k in raw.get("keyframes") or []]
        mem.views = [DepthView.from_json(v) for v in raw.get("views") or []]
        mem._next = 1 + max((kf.id for kf in mem.frames), default=-1)
        return mem


class _Panel:
    """One free-space raster. ``dist`` is geodesic metres to the bound goal."""

    def __init__(self, free: np.ndarray, x0: float, y0: float, res: float) -> None:
        self.free = np.asarray(free, bool)
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.dist: Optional[np.ndarray] = None

    def contains(self, xy: np.ndarray) -> bool:
        i, j = self.cell(xy)
        return i is not None

    def cell(self, xy: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        i = int((float(xy[0]) - self.x0) / self.res)
        j = int((float(xy[1]) - self.y0) / self.res)
        if 0 <= i < self.free.shape[0] and 0 <= j < self.free.shape[1]:
            return i, j
        return None, None

    def center(self, i: int, j: int) -> np.ndarray:
        return np.array([self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res])


class SchematicMap:
    """Occupancy from the sim scan. The next subgoal is read off this grid.

    No polyline is stored. When a goal is bound, distances are filled from that
    goal through free cells. A step then walks at most one horizon downhill.
    """

    def __init__(self, panels: Sequence[_Panel]) -> None:
        self.panels = list(panels)
        self._active: Optional[_Panel] = None

    @classmethod
    def load(cls, path: Path) -> "SchematicMap":
        z = np.load(path)
        panels = []
        for name in ("east", "west"):
            key = f"{name}_free"
            if key not in z:
                continue
            panels.append(_Panel(z[key], float(z[f"{name}_x0"]), float(z[f"{name}_y0"]), float(z[f"{name}_res"])))
        if not panels:
            raise ValueError(f"no east_free/west_free in {path}")
        return cls(panels)

    def prepare(self, goal: np.ndarray) -> None:
        goal = np.asarray(goal, np.float64)
        self._active = None
        for panel in self.panels:
            if panel.contains(goal):
                self._active = panel
                break
        panel = self._active
        if panel is None:
            return
        seed = _nearest_free(panel, panel.cell(goal), 8.0)
        if seed is None:
            panel.dist = None
            return
        panel.dist = _geodesic(panel, seed)
        reachable = int(np.isfinite(panel.dist).sum())
        print(f"schematic grid goal ({goal[0]:.0f},{goal[1]:.0f}) reachable {reachable}", flush=True)

    def carrot(self, here: np.ndarray, horizon_m: float, blocked_yaws: Sequence[float]) -> Optional[np.ndarray]:
        panel = self._active
        if panel is None or panel.dist is None:
            return None
        here = np.asarray(here, np.float64)
        ij = panel.cell(here)
        if ij[0] is None or not np.isfinite(panel.dist[ij]):
            ij = _nearest_free(panel, ij if ij[0] is not None else _closest_cell(panel, here), float(horizon_m))
            if ij is None:
                return None
        end = _walk_down(panel, ij, float(horizon_m))
        if end is None:
            return None
        pt = panel.center(*end)
        if _blocked(_heading(here, pt), blocked_yaws):
            alt = _alternate_down(panel, ij, here, float(horizon_m), blocked_yaws)
            if alt is None:
                return None
            end = alt
            pt = panel.center(*end)
        return _clamp_xy(here, np.array([pt[0], pt[1], here[2] if here.size > 2 else 0.0]), horizon_m)


def _closest_cell(panel: _Panel, xy: np.ndarray) -> Tuple[int, int]:
    i = int(np.clip(round((float(xy[0]) - panel.x0) / panel.res - 0.5), 0, panel.free.shape[0] - 1))
    j = int(np.clip(round((float(xy[1]) - panel.y0) / panel.res - 0.5), 0, panel.free.shape[1] - 1))
    return i, j


def _nearest_free(panel: _Panel, ij: Tuple[Optional[int], Optional[int]], rad_m: float) -> Optional[Tuple[int, int]]:
    if ij[0] is None:
        return None
    i, j = ij
    if panel.free[i, j]:
        return i, j
    r = int(math_ceil(rad_m / panel.res))
    best = None
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            a, b = i + di, j + dj
            if 0 <= a < panel.free.shape[0] and 0 <= b < panel.free.shape[1] and panel.free[a, b]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, a, b)
    return None if best is None else (best[1], best[2])


def math_ceil(v: float) -> int:
    return int(np.ceil(v))


_NBR = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356), (-1, -1, 1.41421356))


def _geodesic(panel: _Panel, seed: Tuple[int, int]) -> np.ndarray:
    free, res = panel.free, panel.res
    dist = np.full(free.shape, np.inf, np.float64)
    dist[seed] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, seed[0], seed[1])]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > float(dist[i, j]) + 1e-6:
            continue
        for di, dj, w in _NBR:
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b]:
                nd = d + w * res
                if nd < float(dist[a, b]):
                    dist[a, b] = nd
                    heapq.heappush(heap, (float(dist[a, b]), a, b))
    return dist


def _walk_down(panel: _Panel, start: Tuple[int, int], horizon_m: float) -> Optional[Tuple[int, int]]:
    dist = panel.dist
    i, j = start
    if not np.isfinite(dist[i, j]):
        return None
    traveled = 0.0
    while traveled < horizon_m - 1e-6:
        best = None
        for di, dj, w in _NBR:
            a, b = i + di, j + dj
            if not (0 <= a < dist.shape[0] and 0 <= b < dist.shape[1]):
                continue
            if float(dist[a, b]) < float(dist[i, j]) - 1e-3:
                step = w * panel.res
                if best is None or float(dist[a, b]) < best[0]:
                    best = (float(dist[a, b]), step, a, b)
        if best is None or traveled + best[1] > horizon_m + 1e-6:
            break
        traveled += best[1]
        i, j = best[2], best[3]
    if traveled < 0.5:
        return None
    return i, j


def _alternate_down(panel: _Panel, start: Tuple[int, int], here: np.ndarray, horizon_m: float,
                    blocked_yaws: Sequence[float]) -> Optional[Tuple[int, int]]:
    """Another downhill step whose end is not on a blocked heading."""
    dist = panel.dist
    i, j = start
    best = None
    for di, dj, w in _NBR:
        a, b = i + di, j + dj
        if not (0 <= a < dist.shape[0] and 0 <= b < dist.shape[1]):
            continue
        if not np.isfinite(dist[a, b]) or float(dist[a, b]) >= float(dist[i, j]) - 1e-3:
            continue
        step = w * panel.res
        if step >= horizon_m:
            continue
        end = _walk_down(panel, (a, b), horizon_m - step)
        if end is None:
            if step < 0.5:
                continue
            end = (a, b)
        pt = panel.center(*end)
        if _blocked(_heading(here, pt), blocked_yaws):
            continue
        d = float(dist[end])
        if best is None or d < best[0]:
            best = (d, end)
    return None if best is None else best[1]


def load_schematic(path: Path) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], Optional[SchematicMap]]:
    """A polyline, per-route polylines, or an occupancy grid.

    Accepted JSON: ``{"polyline": [...]}``, ``{"points": [...]}``,
    ``{"routes": {"route_id": [...]}}``, ``{"grid": "map.npz"}``, or a bare list.
    A grid is the schematic itself. Route polylines in the same file are ignored
    once a grid is present.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    grid = None
    if isinstance(raw, dict) and raw.get("grid"):
        gpath = Path(raw["grid"])
        if not gpath.is_absolute():
            gpath = path.parent / gpath
        grid = SchematicMap.load(gpath)
    if isinstance(raw, list):
        return _poly(raw), {}, grid
    routes = {} if grid is not None else {k: _poly(v) for k, v in (raw.get("routes") or {}).items()}
    default = None if grid is not None else raw.get("polyline", raw.get("points"))
    return (None if default is None else _poly(default)), routes, grid


def _poly(points: Sequence) -> np.ndarray:
    arr = np.asarray(points, np.float64).reshape(-1, 3)
    if len(arr) < 2:
        raise ValueError("a schematic polyline needs at least two points")
    return arr


def _project_arc(poly: np.ndarray, here: np.ndarray) -> Tuple[float, np.ndarray]:
    """Arc length of the closest point on the polyline, and that point."""
    xy = np.asarray(here, np.float64)[:2]
    best_d, best_s, best_p = 1e18, 0.0, poly[0]
    acc = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b[:2] - a[:2]
        lab = float(np.dot(ab, ab))
        t = 0.0 if lab < 1e-9 else float(np.clip(np.dot(xy - a[:2], ab) / lab, 0.0, 1.0))
        p = a[:2] + t * ab
        d = float(np.linalg.norm(xy - p))
        if d < best_d:
            best_d, best_s, best_p = d, acc + t * float(np.sqrt(lab)), np.array([p[0], p[1], a[2]])
        acc += float(np.linalg.norm(ab))
    return best_s, best_p


def schematic_point(poly: Optional[np.ndarray], here: np.ndarray, task_goal: np.ndarray,
                    horizon_m: float, blocked_yaws: Sequence[float]) -> Optional[np.ndarray]:
    """Farthest point along the schematic still within the body horizon, skipping a blocked heading."""
    if poly is None or len(poly) < 2:
        return None
    pts = np.asarray(poly, np.float64)
    goal = np.asarray(task_goal, np.float64)
    if float(np.linalg.norm(pts[0, :2] - goal[:2])) + 1e-6 < float(np.linalg.norm(pts[-1, :2] - goal[:2])):
        pts = pts[::-1].copy()
    s0, _ = _project_arc(pts, here)
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    total = float(seg.sum())
    reach = min(total, s0 + float(horizon_m))
    if reach <= s0 + 0.5:
        return None
    samples = np.linspace(s0 + 0.5, reach, num=max(2, int(np.ceil(reach - s0))))
    chosen = None
    acc = np.concatenate([[0.0], np.cumsum(seg)])
    for s in samples:
        i = int(np.searchsorted(acc, s, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        span = float(seg[i]) if seg[i] > 1e-9 else 1.0
        t = float(np.clip((s - acc[i]) / span, 0.0, 1.0))
        p = pts[i] + t * (pts[i + 1] - pts[i])
        if _blocked(_heading(here, p), blocked_yaws):
            continue
        chosen = p
    if chosen is None:
        return None
    return _clamp_xy(here, chosen, horizon_m)


def _clear_aim(memory: "SceneMemory", here: np.ndarray, heading: float,
               horizon_m: float) -> Optional[np.ndarray]:
    """Farthest point along ``heading`` that does not pass a seen surface."""
    last = None
    dist = 0.5
    while dist <= float(horizon_m) + 1e-9:
        aim = np.array([
            here[0] + dist * float(np.cos(heading)),
            here[1] + dist * float(np.sin(heading)),
            here[2],
        ], np.float64)
        if memory.ray_hits_surface(here, aim):
            break
        last = aim
        dist += 0.5
    return last


# A surface this close beside the next step moves the subgoal to the open side.
# 4 m reaches the corner that sits about 3 m off the forward line, and leaves an
# 8 m street wall alone.
_BESIDE_M = 4.0


def _side_known(memory: "SceneMemory", here: np.ndarray, heading: float, sign: float) -> Optional[bool]:
    """Surface within ``_BESIDE_M`` on this side of ``heading``.

    True when a probe passes a surface. False when a stored view sees the side and
    it is clear. None when every probe leaves the stored cameras. Turning away from
    a corner must not count as the corner having gone.
    """
    seen = False
    for deg in (15.0, 25.0, 40.0):
        ang = heading + sign * float(np.deg2rad(deg))
        probe = np.array([
            here[0] + _BESIDE_M * float(np.cos(ang)),
            here[1] + _BESIDE_M * float(np.sin(ang)),
            here[2],
        ], np.float64)
        known = memory.ray_known(here, probe)
        if known is True:
            return True
        if known is False:
            seen = True
    return False if seen else None


def _goal_ray_clear(memory: "SceneMemory", here: np.ndarray, heading: float, horizon_m: float) -> bool:
    aim = _clear_aim(memory, here, heading, horizon_m)
    if aim is None:
        return False
    return float(np.linalg.norm(aim[:2] - here[:2])) >= float(horizon_m) - 0.6


def _best_aim(memory: "SceneMemory", here: np.ndarray, goal: np.ndarray, headings: Sequence[float],
              horizon_m: float, goal_d: float) -> np.ndarray:
    closer: Optional[Tuple[float, np.ndarray]] = None
    any_clear: Optional[Tuple[float, np.ndarray]] = None
    for heading in headings:
        aim = _clear_aim(memory, here, float(heading), horizon_m)
        if aim is None:
            continue
        dist = float(np.linalg.norm(aim[:2] - goal[:2]))
        if dist < goal_d - 0.5 and (closer is None or dist < closer[0]):
            closer = (dist, aim)
        if any_clear is None or dist < any_clear[0]:
            any_clear = (dist, aim)
    if closer is not None:
        return closer[1]
    if any_clear is not None:
        return any_clear[1]
    heading = float(headings[0])
    step = min(0.5, goal_d)
    return np.array([
        here[0] + step * float(np.cos(heading)),
        here[1] + step * float(np.sin(heading)),
        here[2],
    ], np.float64)


def view_subgoal(memory: "SceneMemory", here: np.ndarray, task_goal: np.ndarray,
                 horizon_m: float, prefer_sign: float = 0.0,
                 prefer_heading: Optional[float] = None
                 ) -> Tuple[str, Optional[np.ndarray], float, Optional[float]]:
    """Place one subgoal from stored depth views.

    The point stays inside ``horizon_m`` and short of a seen surface. A surface
    on one side adds headings toward the open side, and the point closest to the
    task goal is used, so the path returns to the goal line. A latched side is
    kept only while that side is still blocked. It does not hold a right-angle
    heading. The third value is the side (+1 left of the goal heading, -1 right,
    0 none). The fourth is unused and stays None.
    """
    if not memory.views:
        return "keep", None, 0.0, None
    here = np.asarray(here, np.float64)[:3]
    goal = np.asarray(task_goal, np.float64)[:3]
    goal_d = float(np.linalg.norm(goal[:2] - here[:2]))
    if goal_d < 0.5:
        return "keep", None, 0.0, None
    base = float(np.arctan2(goal[1] - here[1], goal[0] - here[0]))
    left = _side_known(memory, here, base, 1.0)
    right = _side_known(memory, here, base, -1.0)
    if left is True and right is not True:
        sign = -1.0
    elif right is True and left is not True:
        sign = 1.0
    elif prefer_sign != 0.0:
        blocked_sign = -float(np.sign(prefer_sign))
        blocked = _side_known(memory, here, base, blocked_sign)
        # Looking along the detour, the corner is gone when that side is seen and clear.
        past_corner = (prefer_heading is not None
                       and _side_known(memory, here, float(prefer_heading), blocked_sign) is False)
        if past_corner or (blocked is False and _goal_ray_clear(memory, here, base, horizon_m)):
            sign = 0.0
        else:
            sign = float(np.sign(prefer_sign))
    else:
        sign = 0.0
    headings = [base]
    if sign == 0.0:
        for deg in _AIM_OFFSETS_DEG:
            rad = float(np.deg2rad(deg))
            headings.append(base + rad)
            headings.append(base - rad)
    else:
        for deg in _AIM_OFFSETS_DEG:
            headings.append(base + sign * float(np.deg2rad(deg)))
    return "seen", _best_aim(memory, here, goal, headings, horizon_m, goal_d), sign, None


def seen_point(kf: Optional[Keyframe], here: np.ndarray, task_goal: np.ndarray,
               horizon_m: float) -> Optional[np.ndarray]:
    """Nearest passed point that got closer to the task goal, clamped to the horizon.

    Not used as the flight subgoal. A recorded path does not replace the schematic.
    """
    if kf is None or not kf.passed:
        return None
    here = np.asarray(here, np.float64)[:3]
    goal_d = float(np.linalg.norm(here[:2] - np.asarray(task_goal, np.float64)[:2]))
    ahead = []
    for p in kf.passed:
        if float(np.linalg.norm(p[:2] - np.asarray(task_goal, np.float64)[:2])) >= goal_d - 0.5:
            continue
        if _blocked(_heading(here, p), kf.blocked_yaw):
            continue
        ahead.append(p)
    if not ahead:
        return None
    ahead.sort(key=lambda p: float(np.linalg.norm(p[:2] - here[:2])))
    return _clamp_xy(here, ahead[0], horizon_m)


class SceneGuide:
    """One object for a flight. ``record`` writes views. ``use_seen`` may steer.

    With both flags clear and no schematic, ``step_view`` returns the task goal.
    """

    def __init__(self, memory: SceneMemory, horizon_m: float, trust: str = "metric",
                 record: bool = False, use_seen: bool = False,
                 default_poly: Optional[np.ndarray] = None,
                 routes: Optional[Dict[str, np.ndarray]] = None,
                 grid: Optional[SchematicMap] = None, use_far: bool = False) -> None:
        if trust not in ("metric", "rough"):
            raise ValueError(trust)
        self.memory = memory
        self.horizon_m = float(horizon_m)
        self.trust = trust
        self.record = bool(record)
        self.use_seen = bool(use_seen)
        self.default_poly = default_poly
        self.routes = dict(routes or {})
        self.grid = grid
        self.use_far = bool(use_far)
        self.continuation = FlightContinuation(self.horizon_m)
        self.poly: Optional[np.ndarray] = None if grid is not None else default_poly
        self.path: Optional[Path] = None
        self._active_blocked: Tuple[float, ...] = ()
        self._hold_blocked: Tuple[float, ...] = ()
        self._hold_xy: Optional[np.ndarray] = None
        self._around_sign = 0.0
        self._around_heading: Optional[float] = None

    @property
    def wants_vector(self) -> bool:
        return self.record or self.use_seen

    def bind(self, route_id: str, goal: Optional[np.ndarray] = None) -> None:
        self.poly = None if self.grid is not None else self.routes.get(route_id, self.default_poly)
        self._active_blocked = ()
        self._hold_blocked = ()
        self._hold_xy = None
        self._around_sign = 0.0
        self._around_heading = None
        self.continuation.reset()
        if self.grid is not None and goal is not None:
            self.grid.prepare(np.asarray(goal, np.float64))

    def _far_blocked(self, here: np.ndarray, aim: np.ndarray) -> Optional[bool]:
        return self.memory.ray_status(here, aim)

    def blocks_heading(self, here: np.ndarray, there: np.ndarray) -> bool:
        """True when the current match still marks the direction here→there as blocked."""
        return bool(self._active_blocked) and _blocked(_heading(here, there), self._active_blocked)

    def _take_blocked(self, pose: np.ndarray, matched: Optional[Keyframe]) -> Tuple[float, ...]:
        if not self.use_seen:
            self._active_blocked = ()
            return ()
        if matched is not None:
            self._hold_blocked = tuple(float(y) for y in matched.blocked_yaw)
            self._hold_xy = np.asarray(matched.pose[:2], np.float64).copy()
            self._active_blocked = self._hold_blocked
            return self._active_blocked
        if self._hold_xy is None:
            self._active_blocked = ()
            return ()
        radius = ROUGH_MATCH_M if self.trust == "rough" else METRIC_MATCH_M
        if float(np.linalg.norm(np.asarray(pose, np.float64)[:2] - self._hold_xy)) <= radius:
            self._active_blocked = self._hold_blocked
            return self._active_blocked
        self._hold_blocked = ()
        self._hold_xy = None
        self._active_blocked = ()
        return ()

    def _schematic_aim(self, loc: np.ndarray, goal: np.ndarray, blocked: Sequence[float]) -> Optional[np.ndarray]:
        if self.grid is not None:
            return self.grid.carrot(loc, self.horizon_m, blocked)
        return schematic_point(self.poly, loc, goal, self.horizon_m, blocked)

    def step_view(self, state: np.ndarray, task_goal: np.ndarray, vector: Optional[np.ndarray],
                  depth: Optional[np.ndarray]) -> Proposal:
        state = np.asarray(state, np.float64)
        goal = np.asarray(task_goal, np.float64)
        pose, yaw = state[:3], float(state[6])
        kf_id = None
        if self.record and vector is not None:
            kf_id = self.memory.observe(pose, yaw, vector, depth, self.trust).id
        matched = None
        loc = pose
        if self.use_seen and vector is not None:
            radius = ROUGH_MATCH_M if self.trust == "rough" else METRIC_MATCH_M
            matched = self.memory.match(pose, vector, radius)
            if matched is not None and self.trust == "rough":
                loc = matched.pose
        blocked = self._take_blocked(pose, matched)
        carrot = self._schematic_aim(loc, goal, blocked)
        kf_out = kf_id if kf_id is not None else (matched.id if matched is not None else None)
        if carrot is not None and self.use_seen and self.memory.views and self.memory.ray_hits_surface(pose, carrot):
            carrot = None
        if carrot is not None:
            aim, source = carrot, "schematic"
        elif self.use_far and self.use_seen and self.memory.views:
            point = self.continuation.update(pose, goal, self._far_blocked)
            return Proposal(_at_altitude(point, pose), "far", kf_out)
        elif self.use_seen and self.memory.views:
            kind, point, sign, heading = view_subgoal(
                self.memory, pose, goal, self.horizon_m, self._around_sign, self._around_heading)
            self._around_sign = sign
            self._around_heading = heading
            if kind == "dropped":
                return Proposal(goal.copy(), "dropped", kf_out)
            if kind == "seen" and point is not None:
                return Proposal(_at_altitude(point, pose), "seen", kf_out)
            return Proposal(goal.copy(), "goal", kf_out)
        elif blocked and (
            self._schematic_aim(loc, goal, ()) is not None or _blocked(_heading(loc, goal), blocked)
        ):
            self._active_blocked = blocked
            return Proposal(goal.copy(), "dropped", kf_out)
        else:
            return Proposal(goal.copy(), "goal", kf_out)
        if self.trust == "rough" and matched is not None:
            # Offset from the remembered place, applied at the live (drifted) pose.
            delta = np.asarray(aim, np.float64)[:2] - np.asarray(loc, np.float64)[:2]
            aim = np.array([pose[0] + delta[0], pose[1] + delta[1], pose[2]], np.float64)
        else:
            aim = _at_altitude(aim, pose)
        return Proposal(aim, source, kf_out)

    def note(self, kf_id: Optional[int], yaw: float, surv: float, d_before: float, d_after: float,
             collided: bool, surv_min: float, pose: np.ndarray) -> None:
        if self.record:
            self.memory.note(kf_id, yaw, surv, d_before, d_after, collided, surv_min, pose)

    def save(self) -> None:
        if self.record and self.path is not None:
            self.memory.save(self.path)


def _at_altitude(point: np.ndarray, pose: np.ndarray) -> np.ndarray:
    out = np.asarray(point, np.float64).copy()
    out[2] = float(np.asarray(pose, np.float64)[2])
    return out


def open_guide(schematic: str, scene_memory: str, horizon_m: float, trust: str, mode: str,
               use_far: bool = False) -> Optional[SceneGuide]:
    """Build a guide, or None when both paths are empty (the trim path)."""
    if not schematic and not scene_memory:
        return None
    if mode not in ("record", "use", "both"):
        raise ValueError(mode)
    default, routes, grid = (None, {}, None) if not schematic else load_schematic(schematic)
    memory = SceneMemory.load(scene_memory) if scene_memory else SceneMemory()
    # A schematic alone never writes or reads a scene library. Recording starts
    # only when a library path is set, so a prior map cannot turn into a hard memory.
    guide = SceneGuide(memory, horizon_m, trust,
                       record=bool(scene_memory) and mode in ("record", "both"),
                       use_seen=bool(scene_memory) and mode in ("use", "both"),
                       default_poly=default, routes=routes, grid=grid,
                       use_far=bool(use_far) and bool(scene_memory) and mode in ("use", "both"))
    guide.path = Path(scene_memory) if scene_memory else None
    return guide
