"""Find-and-report mission: transit -> search -> approach / verify -> report.

The mission layer only chooses goals (and in-place yaw scans / dwells); flying
to a goal is the navigator's job (the consequence MPC, unchanged). Altitude is
held at the start altitude for the whole mission.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..comm.dog import target_message
from ..geo import wrap
from ..perception.locate import depth_scale_from_ground, estimate_ground_z, locate, world_to_pixel
from ..perception.tracker import Track, Tracker, reject_look
from ..types import Frame, VehicleState
from .parser import MissionSpec

log = logging.getLogger("aerial_finder.mission")
YAW_STEP = 0.3141592653589793
# Largest plausible extent (m) of a target class; larger boxes of that class are look-alikes (kiosks, booths).
MAX_SIZE_M = {"stool": 1.2, "chair": 1.5, "bench": 2.5, "backpack": 1.0, "bicycle": 2.2, "box": 1.5,
              "barrel": 1.5, "trash can": 1.6, "traffic cone": 1.2, "suitcase": 1.2, "umbrella": 2.5}


@dataclass
class MissionConfig:
    detect_every: int = 2
    search_radius_min_m: float = 15.0
    search_radius_max_m: float = 60.0
    subgoal_m: float = 40.0
    reach_m: float = 4.0
    stall_s: float = 25.0
    stall_gain_m: float = 1.0
    scan_rate: float = 0.6 * YAW_STEP
    standoff_m: float = 8.0              # minimum; raised so a ground object stays inside the vertical view
    view_margin_deg: float = 6.0
    verify_agl_m: Optional[float] = 7.0  # approach altitude above the candidate (None: stay at cruise);
                                         # kept above street-lamp height
    focus_verify: bool = True            # magnified crop pass on the candidate while verifying
    max_unreached: int = 2               # approach timeouts before a candidate is dropped
    focus_gate_m: float = 4.0            # crop-pass detections join the verified track within this distance
    verify_alt_tol_m: float = 1.0
    vertical_step_m: float = 0.25        # per-step climb/descent over the observation point (no lateral motion)
    reach_obs_m: float = 2.0             # observation point reached (horizontal)
    dwell_steps: int = 15
    candidate_timeout_s: float = 60.0
    max_candidate_visits: int = 10
    max_hint_visits: int = 3             # approaches to colour-only tracks (YOLO never agreed)
    confirm_min_verified: int = 3
    confirm_max_spread_m: float = 1.5
    confirm_min_conf: float = 0.25        # mean detector confidence of those close verified views
    confirm_max_range_m: float = 18.0     # verified views must come from this close (after descending)
    max_locate_m: float = 80.0
    max_above_ground_m: float = 3.0      # targets stand on the ground; higher boxes are lamp arms, signs, ...
    async_detect: bool = True            # detect on a worker thread so control keeps step_hz
    log_min_conf: float = 0.15
    locate_depth: str = "model"          # "model" (DA3, deployable) | "sensor" (backend depth, sim only)
    ground_z: Optional[float] = None     # fixed ground height; None = home_agl_m or estimate from depth
    home_agl_m: Optional[float] = None   # height above ground at mission start (takeoff / rangefinder)
    ground_objects: bool = True          # targets stand on the ground: position from the ground ray, depth only
                                         # rejects elevated look-alikes
    hint_max_range_m: float = 45.0
    transit_min_verified: int = 5        # leave the transit leg only for strong candidates
    cam_pitch_deg: float = 0.0
    cam_offset_frd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_time_s: float = 1200.0
    geofence_margin_m: float = 60.0
    save_frames: bool = True


@dataclass
class Task:
    kind: str                       # goto | scan | approach
    point: Optional[np.ndarray] = None
    reason: str = ""
    track_id: Optional[int] = None
    t_start: float = -1.0
    best_d: float = float("inf")
    best_t: float = 0.0
    turned: float = 0.0
    last_yaw: Optional[float] = None
    phase: str = ""
    dwell_left: int = 0


class _DetectWorker:
    """Runs the detector on the newest submitted frame; one frame in flight, results polled."""

    def __init__(self, detector) -> None:
        self.detector = detector
        self._job = None
        self._done = None
        self._cv = threading.Condition()
        self._stop = False
        self.busy = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._job = self._job, None
            try:
                focus = job[4]
                dets = self.detector(job[0].rgb) if focus is None else self.detector.focus(job[0].rgb, focus)
            except Exception:  # noqa: BLE001
                log.exception("detector failed")
                dets = []
            with self._cv:
                self._done = (dets, *job[:4])
                self.busy = False

    def submit(self, frame: Frame, st: VehicleState, depth: np.ndarray, step: int, focus=None) -> bool:
        with self._cv:
            if self.busy:
                return False
            self.busy = True
            self._job = (frame, st, depth, step, focus)
            self._cv.notify()
            return True

    def poll(self):
        with self._cv:
            done, self._done = self._done, None
            return done

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()


class MissionRunner:
    def __init__(self, backend, depth_model, navigator, detector, reporters: List[Any], cfg: MissionConfig,
                 out_root: Path, drone_id: str = "drone_1") -> None:
        self.backend = backend
        self.depth_model = depth_model
        self.nav = navigator
        self.detector = detector
        self.reporters = list(reporters)
        self.cfg = cfg
        self.out_root = Path(out_root)
        self.drone_id = drone_id
        self._abort = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {"state": "IDLE"}

    # -- public API ---------------------------------------------------------
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, spec: MissionSpec) -> str:
        if self.busy():
            raise RuntimeError("a mission is already running")
        mid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self._abort.clear()
        self._set(state="STARTING", mission_id=mid, spec=spec.to_dict())
        self._thread = threading.Thread(target=self.run, args=(spec, mid), daemon=True)
        self._thread.start()
        return mid

    def abort(self) -> None:
        self._abort.set()

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status, default=_jsonable))

    # -- internals ----------------------------------------------------------
    def _set(self, **kw: Any) -> None:
        with self._lock:
            self._status.update(kw)

    def _event(self, event: str, **kw: Any) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **kw}
        log.info("%s %s", event, json.dumps(kw, default=_jsonable)[:300])
        with open(self.out / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    def _search_tasks(self, home: np.ndarray, center: np.ndarray, radius: float, z: float) -> Deque[Task]:
        tasks: Deque[Task] = deque()
        d = float(np.linalg.norm(center[:2] - home[:2]))
        n_sub = int(math.floor(max(0.0, d - 0.5 * radius) / self.cfg.subgoal_m))
        for i in range(1, n_sub + 1):
            p = home[:2] + (center[:2] - home[:2]) * (i * self.cfg.subgoal_m / d)
            tasks.append(Task("goto", np.array([p[0], p[1], z]), f"transit {i}/{n_sub}"))
        tasks.append(Task("goto", np.array([center[0], center[1], z]), "search centre"))
        tasks.append(Task("scan", reason="scan centre"))
        heading = math.atan2(center[1] - home[1], center[0] - home[0]) if d > 1.0 else 0.0
        for frac, n in ((0.5, 6), (1.0, 8)):
            r = frac * radius
            if r < self.cfg.reach_m * 2:
                continue
            for k in range(n):
                a = heading + math.pi + 2.0 * math.pi * k / n
                p = center[:2] + r * np.array([math.cos(a), math.sin(a)])
                tasks.append(Task("goto", np.array([p[0], p[1], z]), f"ring {frac:.1f}r #{k}"))
                tasks.append(Task("scan", reason=f"scan ring {frac:.1f}r #{k}"))
        return tasks

    def _detect(self, frame: Frame, st: VehicleState, depth: np.ndarray, tracker: Tracker, step: int,
                focus=None) -> int:
        dets = self.detector(frame.rgb) if focus is None else self.detector.focus(frame.rgb, focus)
        return self._ingest(dets, frame, st, depth, tracker, step)

    def _focus(self, cur: Optional[Task], tracker: Tracker, frame: Frame, st: VehicleState):
        """Pixel of the candidate being verified (face/dwell of an approach), for a magnified crop pass."""
        if not self.cfg.focus_verify or cur is None or cur.kind != "approach" or cur.phase not in ("face", "dwell"):
            return None
        tr = next((t for t in tracker.tracks if t.id == cur.track_id), None)
        if tr is None or not hasattr(self.detector, "focus"):
            return None
        self._looking_id = tr.id
        px = world_to_pixel(tr.position, st, frame.rgb_intr, math.radians(self.cfg.cam_pitch_deg),
                            self.cfg.cam_offset_frd)
        if px is None or not (0 <= px[0] < frame.rgb_intr.width and 0 <= px[1] < frame.rgb_intr.height):
            return None
        return float(px[0]), float(px[1])

    def _ingest(self, dets, frame: Frame, st: VehicleState, depth: np.ndarray, tracker: Tracker, step: int) -> int:
        """Locate detections made on ``frame`` (with the pose and depth of that frame) and track them."""
        cfg = self.cfg
        if not dets:
            return 0
        cam_pitch = math.radians(cfg.cam_pitch_deg)
        gz = cfg.ground_z if cfg.ground_z is not None else self._gz_known
        if cfg.locate_depth == "sensor" and frame.depth is not None:
            lf = frame
        else:
            d_intr = frame.rgb_intr.scaled(depth.shape[1], depth.shape[0])
            if gz is not None:
                k = depth_scale_from_ground(depth, d_intr, st, gz, cam_pitch, cfg.cam_offset_frd)
                if k is not None:
                    self._scale_hist.append(k)
                scale = float(np.median(self._scale_hist)) if self._scale_hist else None
                lf = Frame(rgb=frame.rgb, rgb_intr=frame.rgb_intr,
                           depth=None if scale is None else depth * scale, depth_intr=d_intr)
            else:
                lf = Frame(rgb=frame.rgb, rgb_intr=frame.rgb_intr, depth=depth, depth_intr=d_intr)
        if gz is None and lf.depth is not None:
            g = estimate_ground_z(lf.depth, lf.depth_intr, st, cam_pitch, cam_offset_frd=cfg.cam_offset_frd)
            if g is not None:
                self._gz_hist.append(float(g))
            gz = float(np.percentile(self._gz_hist, 20)) if self._gz_hist else None
        n = 0
        for d in dets:
            loc = locate(d.bbox, lf, st, cam_pitch=cam_pitch, ground_z=gz, max_depth=cfg.max_locate_m,
                         cam_offset_frd=cfg.cam_offset_frd, prefer_ground=d.source != "yolo")
            if loc is None:
                continue
            if gz is not None and float(loc[0][2]) > gz + cfg.max_above_ground_m:
                continue
            if loc[2] == "depth" and gz is not None and cfg.ground_objects:
                g = locate(d.bbox, lf, st, cam_pitch=cam_pitch, ground_z=gz, max_depth=cfg.max_locate_m,
                           cam_offset_frd=cfg.cam_offset_frd, prefer_ground=True)
                if g is not None:
                    loc = g
            if d.source != "yolo" and loc[1] > cfg.hint_max_range_m:
                continue
            d.world, d.range_m, d.extra["method"] = loc[0], loc[1], loc[2]
            if d.source == "yolo" and self._max_size is not None:
                size = max(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) * loc[1] / frame.rgb_intr.fx
                if size > self._max_size:
                    d.verified, d.extra["oversize_m"] = False, round(float(size), 2)
            aimed = d.extra.get("mode") == "focus"
            tr = tracker.add(d, st.pos, st.t, _crop(frame.rgb, d.bbox),
                             prefer_id=self._looking_id if aimed else None,
                             prefer_gate_m=max(cfg.focus_gate_m, 0.25 * float(d.range_m)))
            n += 1
            if d.source == "yolo" and (d.verified or d.conf >= cfg.log_min_conf):
                self._event("detection", step=step, track=tr.id, prompt=d.prompt, conf=round(d.conf, 3),
                            verified=d.verified, color_frac=round(d.color_frac, 2), range_m=round(d.range_m, 1),
                            method=loc[2], world=np.round(d.world, 2))
                if cfg.save_frames and d.verified:
                    img = np.ascontiguousarray(frame.rgb[..., ::-1]).copy()
                    x0, y0, x1, y1 = (int(round(v)) for v in d.bbox)
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.imwrite(str(self.out / f"det_{step:05d}_t{tr.id}.jpg"), img)
        return n

    def run(self, spec: MissionSpec, mission_id: Optional[str] = None) -> Dict[str, Any]:
        mid = mission_id or time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.out = self.out_root / mid
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "mission.json").write_text(json.dumps({"spec": spec.to_dict(), "config": asdict(self.cfg)},
                                                          indent=1, default=_jsonable), encoding="utf-8")
        try:
            res = self._run(spec, mid)
        except Exception as exc:  # noqa: BLE001
            log.exception("mission crashed")
            self.backend.hover()
            res = {"state": "FAILED", "reason": f"error: {exc}"}
        self._set(**res)
        self._event("end", **res)
        (self.out / "result.json").write_text(json.dumps(self.status(), indent=1, default=_jsonable),
                                              encoding="utf-8")
        return self.status()

    def _run(self, spec: MissionSpec, mid: str) -> Dict[str, Any]:
        cfg, be = self.cfg, self.backend
        self.detector.configure(spec.label, spec.prompts, spec.color)
        self.depth_model.reset()
        self.nav.reset()
        self._gz_hist: Deque[float] = deque(maxlen=50)
        self._scale_hist: Deque[float] = deque(maxlen=100)
        self._max_size = MAX_SIZE_M.get(spec.label)
        self._looking_id: Optional[int] = None
        tracker = Tracker()
        prepare = getattr(be, "prepare", None)
        if prepare is not None:
            prepare()
        frame, st = be.observe()
        home, yaw0 = st.pos.copy(), float(st.yaw)
        z = float(home[2])
        self._gz_known = None if cfg.home_agl_m is None else z - float(cfg.home_agl_m)
        if spec.distance_m:
            h = spec.world_heading if spec.world_heading is not None else yaw0 + float(spec.body_bearing or 0.0)
            center = home + np.array([spec.distance_m * math.cos(h), spec.distance_m * math.sin(h), 0.0])
            radius = float(np.clip(spec.tolerance_m or cfg.search_radius_min_m, cfg.search_radius_min_m,
                                   cfg.search_radius_max_m))
        else:
            center, radius = home.copy(), cfg.search_radius_max_m
        fence = float(np.linalg.norm(center[:2] - home[:2])) + radius + cfg.geofence_margin_m
        tasks = self._search_tasks(home, center, radius, z)
        self.nav.set_band(z, z)
        visited: Dict[int, int] = {}
        self._set(state="TRANSIT", mission_id=mid, home=home, center=center, radius_m=radius,
                  home_gps=be.geo_ref.to_gps(*home), center_gps=be.geo_ref.to_gps(*center), result=None)
        self._event("start", text=spec.text, home=home, center=center, radius=radius, n_tasks=len(tasks))
        t0, step, last_a = st.t, 0, None
        cur: Optional[Task] = None
        worker = _DetectWorker(self.detector) if cfg.async_detect else None
        try:
            return self._loop(spec, mid, tracker, worker, home, center, radius, fence, tasks, z, frame, st)
        finally:
            if worker is not None:
                worker.stop()

    def _loop(self, spec, mid, tracker, worker, home, center, radius, fence, tasks, z, frame, st):
        cfg, be = self.cfg, self.backend
        visited = self._visited = {}
        self._hint_visits = 0
        t0, step, last_a = st.t, 0, None
        cur: Optional[Task] = None
        while True:
            if step:
                frame, st = be.observe()
            self._vfov_half = math.atan2(frame.rgb_intr.cy, frame.rgb_intr.fy)
            elapsed = st.t - t0
            if self._abort.is_set():
                be.hover()
                return {"state": "ABORTED", "reason": "abort requested"}
            if st.collided:
                be.hover()
                obj = getattr(be, "collision_object", lambda: "")()
                return {"state": "FAILED", "reason": f"collision {obj}".strip()}
            if elapsed > cfg.max_time_s:
                be.hover()
                return {"state": "FAILED", "reason": "timeout"}
            if np.linalg.norm(st.pos[:2] - home[:2]) > fence:
                be.hover()
                return {"state": "FAILED", "reason": "geofence"}
            raw = frame.depth if getattr(self.depth_model, "needs_sensor", False) else None
            if raw is not None:
                depth, env = self.depth_model(frame.rgb, raw)
            elif getattr(self.depth_model, "uses_valid", False):
                depth, env = self.depth_model(frame.rgb, frame.valid)
            else:
                depth, env = self.depth_model(frame.rgb)
            done_a = be.executed(last_a) if hasattr(be, "executed") else last_a
            self.nav.observe(depth, env, None if last_a is None else done_a)
            scanning = cur is not None and (cur.kind == "scan" or cur.phase in ("face", "dwell"))
            if worker is not None:
                done = worker.poll()
                if done is not None:
                    self._ingest(done[0], done[1], done[2], done[3], tracker, done[4])
                worker.submit(frame, st, depth, step, focus=self._focus(cur, tracker, frame, st))
            elif step % cfg.detect_every == 0 or scanning:
                self._detect(frame, st, depth, tracker, step, focus=self._focus(cur, tracker, frame, st))
            looking = cur is not None and cur.kind == "approach" and cur.phase in ("face", "dwell")
            tr = None if not looking else tracker.confirmed(
                min_verified=cfg.confirm_min_verified, max_spread_m=cfg.confirm_max_spread_m,
                max_range_m=cfg.confirm_max_range_m, min_conf=cfg.confirm_min_conf, center=center,
                radius=radius * 1.3, only_id=cur.track_id, focus_only=cfg.focus_verify)
            if tr is not None:
                be.hover()
                return self._report(spec, mid, tr, home)
            if cur is None or cur.kind != "approach":
                nxt = cur if cur is not None else (tasks[0] if tasks else None)
                transit = nxt is not None and nxt.kind == "goto" and nxt.reason.startswith("transit")
                hint_only = lambda c: c.n_verified == 0
                cands = [c for c in tracker.candidates(center, radius) if c.id not in visited
                         and (not spec.color or c.n_verified > 0 or c.n_hint >= 2)
                         and (not transit or c.n_verified >= cfg.transit_min_verified)
                         and (not hint_only(c) or self._hint_visits < cfg.max_hint_visits)]
                cands.sort(key=lambda c: (hint_only(c),
                                          -c.score / (1.0 + float(np.linalg.norm(c.position[:2] - center[:2])) / radius)))
                if cands and len(visited) < cfg.max_candidate_visits:
                    c = cands[0]
                    visited[c.id] = step
                    self._hint_visits += int(hint_only(c))
                    if cur is not None:
                        cur.t_start = -1.0
                        tasks.appendleft(cur)
                    cur = Task("approach", track_id=c.id, reason=f"candidate {c.id}", phase="go")
                    self._event("approach", track=c.summary(), depth_scale=self.status().get("depth_scale"))
            if cur is None:
                if not tasks:
                    be.hover()
                    return {"state": "FAILED", "reason": "not_found", "tracks": [t.summary() for t in tracker.tracks]}
                cur = tasks.popleft()
            if cur.t_start < 0:
                cur.t_start, cur.best_t = st.t, st.t
                self._event("task", kind=cur.kind, reason=cur.reason, point=cur.point)
            a, done = self._execute(cur, st, tracker, z)
            if done:
                if cur.kind == "approach":
                    self.nav.set_band(float(st.pos[2]), z)
                cur = None
            self._set(state=_phase_name(cur, tasks), step=step, elapsed_s=round(elapsed, 1), pos=st.pos,
                      yaw=st.yaw, gps=be.geo_ref.to_gps(*st.pos),
                      task=None if cur is None else {"kind": cur.kind, "reason": cur.reason, "phase": cur.phase},
                      tasks_left=len(tasks), nav=getattr(self.nav, "last", {}),
                      depth_scale=round(float(np.median(self._scale_hist)), 3) if self._scale_hist else None,
                      tracks=[t.summary() for t in tracker.candidates()[:5]])
            be.step(a)
            last_a = a
            step += 1

    def _execute(self, task: Task, st: VehicleState, tracker: Tracker, z: float) -> Tuple[np.ndarray, bool]:
        cfg = self.cfg
        hold = np.zeros(4)
        if task.kind == "scan":
            # Measured yaw, not the commanded rate: a slow vehicle turns less than it is asked to.
            if task.last_yaw is not None:
                task.turned += abs(wrap(float(st.yaw) - task.last_yaw))
            task.last_yaw = float(st.yaw)
            return np.array([0.0, 0.0, 0.0, cfg.scan_rate]), task.turned >= 2.0 * math.pi
        if task.kind == "goto":
            d = float(np.linalg.norm(task.point[:2] - st.pos[:2]))
            if d <= cfg.reach_m:
                return hold, True
            if d < task.best_d - cfg.stall_gain_m:
                task.best_d, task.best_t = d, st.t
            elif st.t - task.best_t > cfg.stall_s:
                self._event("skip", reason=task.reason, d_m=round(d, 1))
                return hold, True
            return self.nav.act(st, task.point), False
        z_now = float(st.pos[2])
        if task.phase == "climb":
            if z - z_now <= cfg.verify_alt_tol_m:
                return hold, True
            return np.array([0.0, 0.0, min(cfg.vertical_step_m, z - z_now), 0.0]), False
        tr = next((t for t in tracker.tracks if t.id == task.track_id), None)
        if tr is None or tr.rejected:
            return hold, True
        if task.phase == "go" and st.t - task.t_start > cfg.candidate_timeout_s:
            tr.unreached += 1
            if tr.unreached >= cfg.max_unreached:
                tr.rejected = True
            else:
                self._visited.pop(tr.id, None)
            self._event("reject", track=tr.id, reason="timeout", unreached=tr.unreached, final=tr.rejected)
            return hold, True
        p = tr.position
        off = p[:2] - st.pos[:2]
        dist = float(np.linalg.norm(off))
        gz = z if cfg.verify_agl_m is None else min(z, float(p[2]) + cfg.verify_agl_m)
        standoff = self._standoff(gz, p)
        bearing = wrap(math.atan2(off[1], off[0]) - st.yaw)
        yaw_to = float(np.clip(bearing, -YAW_STEP, YAW_STEP))
        if task.phase == "go":
            # Horizontal leg at cruise altitude to the observation point; all lateral motion is model-checked.
            g = p[:2] - off / max(dist, 1e-6) * min(standoff, dist)
            if float(np.linalg.norm(g - st.pos[:2])) > cfg.reach_obs_m and dist > standoff + 2.0:
                return self.nav.act(st, np.array([g[0], g[1], z])), False
            task.phase = "descend"
        if task.phase == "descend":
            # Straight down over the observation point, turning toward the target; no horizontal motion.
            if z_now - gz > cfg.verify_alt_tol_m:
                return np.array([0.0, 0.0, -min(cfg.vertical_step_m, z_now - gz), yaw_to]), False
            task.phase = "face"
        if task.phase == "face":
            if abs(bearing) > math.radians(5.0):
                return np.array([0.0, 0.0, 0.0, yaw_to]), False
            task.phase, task.dwell_left = "dwell", cfg.dwell_steps
        task.dwell_left -= 1
        if task.dwell_left <= 0:
            reject_look(tracker, tr)
            self._event("reject", track=tr.id, reason="not confirmed at standoff", track_state=tr.summary())
            task.phase = "climb"
        return hold, False

    def _standoff(self, vehicle_z: float, target: np.ndarray) -> float:
        """Horizontal distance at which ``target`` sits ``view_margin_deg`` above the image's bottom edge."""
        cfg = self.cfg
        cam_z = float(vehicle_z) - float(cfg.cam_offset_frd[2])
        below = math.atan2(1.0, 1.0) if not hasattr(self, "_vfov_half") else self._vfov_half
        ang = below - math.radians(cfg.cam_pitch_deg) - math.radians(cfg.view_margin_deg)
        need = max(0.0, cam_z - float(target[2])) / math.tan(max(ang, math.radians(5.0)))
        return max(cfg.standoff_m, need)

    def _report(self, spec: MissionSpec, mid: str, tr: Track, home: np.ndarray) -> Dict[str, Any]:
        be = self.backend
        msg = target_message(mission_id=mid, drone_id=self.drone_id,
                             query={"text": spec.text, "label": spec.label, "color": spec.color},
                             pos_local=tr.position, geo=be.geo_ref, uncertainty_m=tr.uncertainty_m(),
                             confidence=float(np.mean(tr.confs)) if tr.confs else 0.0,
                             observations=tr.n_verified, crop_rgb=tr.best_crop)
        sent = [r.send(msg) for r in self.reporters] if "dog" in spec.report_to else []
        if tr.best_crop is not None:
            cv2.imwrite(str(self.out / "target_crop.jpg"), np.ascontiguousarray(tr.best_crop[..., ::-1]))
        light = {k: v for k, v in msg.items() if k != "image_jpeg_b64"}
        (self.out / "report.json").write_text(json.dumps({"message": light, "sent": sent}, indent=1),
                                              encoding="utf-8")
        self._event("report", track=tr.summary(), position=light["position"], sent=sent)
        ok = all(s.get("ok") for s in sent) if sent else True
        return {"state": "DONE" if ok else "FOUND_REPORT_FAILED", "reason": "found",
                "result": {"message": light, "sent": sent, "track": tr.summary()}}


def _crop(rgb: np.ndarray, bbox, pad: float = 0.3) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    px, py = pad * (x1 - x0) + 4, pad * (y1 - y0) + 4
    return rgb[max(0, int(y0 - py)):min(h, int(y1 + py)), max(0, int(x0 - px)):min(w, int(x1 + px))].copy()


def _phase_name(cur: Optional[Task], tasks: Deque[Task]) -> str:
    if cur is None:
        return "SEARCH"
    if cur.kind == "approach":
        return "APPROACH"
    if cur.kind == "goto" and cur.reason.startswith("transit"):
        return "TRANSIT"
    return "SEARCH"


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [round(float(v), 4) for v in o.ravel()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)
