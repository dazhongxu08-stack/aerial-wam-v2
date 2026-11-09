"""Fuse per-frame located detections into world-frame object tracks."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..types import Detection


@dataclass
class Track:
    id: int
    points: List[np.ndarray] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    ranges: List[float] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    viewpoints: List[np.ndarray] = field(default_factory=list)
    n_verified: int = 0
    n_unverified: int = 0
    n_hint: int = 0
    verified_ranges: List[float] = field(default_factory=list)
    verified_confs: List[float] = field(default_factory=list)
    verified_focus: List[bool] = field(default_factory=list)
    last_t: float = 0.0
    best_crop: Optional[np.ndarray] = None
    best_score: float = -1.0
    rejected: bool = False
    looked: bool = False                 # rejected after a close look (not just unreached)
    unreached: int = 0

    @property
    def position(self) -> np.ndarray:
        w = np.asarray(self.weights)
        return (np.asarray(self.points) * w[:, None]).sum(0) / w.sum()

    @property
    def spread_m(self) -> float:
        if len(self.points) < 2:
            return float("inf")
        p = np.asarray(self.points)
        w = np.asarray(self.weights)
        c = self.position
        return float(np.sqrt((w * ((p[:, :2] - c[:2]) ** 2).sum(1)).sum() / w.sum()))

    @property
    def min_range(self) -> float:
        return float(min(self.ranges)) if self.ranges else float("inf")

    @property
    def baseline_m(self) -> float:
        """Largest distance between two viewpoints (multi-view support)."""
        if len(self.viewpoints) < 2:
            return 0.0
        v = np.asarray(self.viewpoints)[:, :2]
        return float(np.linalg.norm(v.max(0) - v.min(0)))

    @property
    def score(self) -> float:
        return self.n_verified * (1.0 + float(np.mean(self.confs or [0.0]))) + 0.1 * self.n_hint

    def uncertainty_m(self) -> float:
        return float(max(self.spread_m if np.isfinite(self.spread_m) else 0.0, 0.3 + 0.04 * self.min_range))

    def summary(self) -> dict:
        p = self.position
        return {"id": self.id, "pos": [round(float(v), 2) for v in p], "n_verified": self.n_verified,
                "n_unverified": self.n_unverified, "n_hint": self.n_hint,
                "n_focus_verified": int(sum(self.verified_focus)),
                "spread_m": None if not np.isfinite(self.spread_m) else round(self.spread_m, 2),
                "min_range_m": round(self.min_range, 1), "baseline_m": round(self.baseline_m, 1),
                "mean_conf": round(float(np.mean(self.confs)), 3) if self.confs else 0.0,
                "rejected": self.rejected}


class Tracker:
    def __init__(self, gate_m: float = 2.0, gate_frac: float = 0.08, max_tracks: int = 64,
                 veto_m: float = 3.0) -> None:
        self.veto_m = float(veto_m)
        self.vetoed: List[np.ndarray] = []   # places rejected after a close look; outlive track eviction
        self.gate_m = float(gate_m)
        self.gate_frac = float(gate_frac)
        self.max_tracks = int(max_tracks)
        self.tracks: List[Track] = []
        self._ids = itertools.count(1)

    def add(self, det: Detection, viewpoint: np.ndarray, t: float, crop: Optional[np.ndarray] = None,
            prefer_id: Optional[int] = None, prefer_gate_m: float = 0.0) -> Track:
        """``det.world`` / ``det.range_m`` must be set. ``prefer_id``: track the detection was aimed at
        (a verification look), which takes it within the wider ``prefer_gate_m``."""
        p = np.asarray(det.world, np.float64)
        r = float(det.range_m)
        gate = max(self.gate_m, self.gate_frac * r)
        best, best_d = None, gate
        pref = next((tr for tr in self.tracks if tr.id == prefer_id), None) if prefer_id is not None else None
        if pref is not None and float(np.linalg.norm(pref.position[:2] - p[:2])) <= max(gate, prefer_gate_m):
            best = pref
        for tr in ([] if best is not None else self.tracks):
            d = float(np.linalg.norm(tr.position[:2] - p[:2]))
            if d <= best_d:
                best, best_d = tr, d
        if best is None:
            if len(self.tracks) >= self.max_tracks:
                self.tracks.remove(min(self.tracks, key=lambda x: (not x.rejected, x.score, x.last_t)))
            best = Track(next(self._ids))
            self.tracks.append(best)
        w = (1.0 if det.source == "yolo" else 0.3) / max(1.0, r) ** 2
        best.points.append(p)
        best.weights.append(w)
        best.ranges.append(r)
        best.viewpoints.append(np.asarray(viewpoint, np.float64))
        best.last_t = t
        if det.source == "yolo":
            best.confs.append(det.conf)
            if det.verified:
                best.n_verified += 1
                best.verified_ranges.append(r)
                best.verified_confs.append(float(det.conf))
                best.verified_focus.append(det.extra.get("mode") == "focus")
                # A colour blob that failed a look must not keep a later real detection.
                if best.rejected and not best.looked:
                    best.rejected = False
            else:
                best.n_unverified += 1
        else:
            best.n_hint += 1
        s = det.conf * (1.0 if det.verified else 0.2) / max(1.0, r)
        if crop is not None and s > best.best_score:
            best.best_crop, best.best_score = crop, s
        return best

    def confirmed(self, *, min_verified: int = 3, max_spread_m: float = 1.5, max_range_m: float = 30.0,
                  min_verified_frac: float = 0.6, min_conf: float = 0.0, center: Optional[np.ndarray] = None,
                  radius: Optional[float] = None, only_id: Optional[int] = None,
                  focus_only: bool = False) -> Optional[Track]:
        """Best track verified enough times from within ``max_range_m``, consistently (optionally inside an area)."""
        ok = []
        for tr in self.tracks:
            if only_id is not None and tr.id != only_id:
                continue
            if center is not None and radius is not None and np.linalg.norm(tr.position[:2] - center[:2]) > radius:
                continue
            n = tr.n_verified + tr.n_unverified
            if (not tr.rejected and tr.n_verified >= min_verified and n and tr.n_verified / n >= min_verified_frac
                    and tr.spread_m <= max_spread_m
                    and _close_ok(tr, max_range_m, min_verified, min_conf, focus_only)):
                ok.append(tr)
        return max(ok, key=lambda x: x.score) if ok else None

    def candidates(self, center: Optional[np.ndarray] = None, radius: Optional[float] = None) -> List[Track]:
        """Unrejected tracks worth a closer look, best first (optionally inside the search area)."""
        out = []
        vetoed = self.vetoed + [t.position[:2] for t in self.tracks if t.looked]
        for tr in self.tracks:
            if tr.rejected or (tr.n_verified == 0 and tr.n_hint < 2 and tr.n_unverified < 2):
                continue
            if any(float(np.linalg.norm(tr.position[:2] - v)) <= self.veto_m for v in vetoed):
                continue
            if center is not None and radius is not None and np.linalg.norm(tr.position[:2] - center[:2]) > radius:
                continue
            out.append(tr)
        return sorted(out, key=lambda x: -x.score)


def _close_ok(tr: Track, max_range_m: float, min_verified: int, min_conf: float, focus_only: bool = False) -> bool:
    c = [conf for r, conf, f in zip(tr.verified_ranges, tr.verified_confs, tr.verified_focus)
         if r <= max_range_m and (f or not focus_only)]
    return len(c) >= min_verified and float(np.mean(c)) >= min_conf


def veto(tracker: "Tracker", tr: Track) -> None:
    """Reject ``tr`` after a close look and remember the place."""
    tr.rejected = tr.looked = True
    tracker.vetoed.append(tr.position[:2].copy())


def reject_look(tracker: "Tracker", tr: Track) -> None:
    """Drop a track after a look. Only a verified detection blacklists the neighbourhood.

    A colour blob that never matched the detector must not block a real target a couple of metres away.
    """
    if tr.n_verified > 0:
        veto(tracker, tr)
    else:
        tr.rejected = True
