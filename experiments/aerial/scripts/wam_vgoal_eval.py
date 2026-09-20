#!/usr/bin/env python3
"""Phase-2 monocular visual goal eval (M1+M2).

Product stack (no GT world-goal control by default):
  YOLO / open-vocab detector
    → monocular D̂ back-projection (``bbox_to_goal_rel``)
    → TargetTracker (TRACKING / OCCLUDED / SEARCHING)
    → goal_rel → LatentActorDeployPolicy + ImaginationPlanner
    → ThreeZoneSpeedShield → env.step

SEARCHING (M3): ``--search-pattern scan`` = slow fwd+yaw; ``lawnmower``/``spiral`` =
``AreaSearchPlanner`` waypoint goals via Phase-2 π (perception still runs each step).
TRACKING/APPROACH: visual ``G`` → ``TowardGoalIntent`` clip → π/planner (Phase-2
toward_g shell). ``--fallback-toward-g`` is opt-in ablation only.

Ckpt defaults match Phase-2 close (E2 toward_g · tti=2.5 · long routes).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from experiments.aerial.scripts.vgoal_area_search import make_area_search_planner
from experiments.aerial.scripts.vgoal_dynamic_follow import (
    dynamic_tracker_step,
    make_dynamic_tracker,
)
from experiments.aerial.scripts.wam_phase2_long_eval import (
    PASS_THRESHOLDS,
    _goal_closure,
    _goal_dist,
    _segment_min_dist,
    aggregate_metrics,
)


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wam_vgoal_eval")


def _select_route_indices(n_available: int, episodes: int, routes_arg: Optional[str]) -> List[int]:
    if not routes_arg:
        return list(range(min(int(episodes), int(n_available))))
    idxs = [int(t) for t in str(routes_arg).split(",") if t.strip()]
    bad = [i for i in idxs if not 0 <= i < n_available]
    if bad:
        raise SystemExit(f"--routes {bad} out of range (annotation has {n_available})")
    if len(set(idxs)) != len(idxs):
        raise SystemExit(f"--routes has duplicates: {idxs}")
    return idxs


def _resolve_search_fwd_step(
    search_fwd_speed: Optional[float],
    *,
    search_at_cruise: bool,
    vx_max_step: float,
    slow_default: float = 0.2,
) -> float:
    if search_fwd_speed is not None:
        return float(search_fwd_speed)
    if search_at_cruise:
        return float(vx_max_step)
    return float(slow_default)


def _episode_search_z_hold(
    start_z: float,
    *,
    mode: str,
    hold_m: Optional[float],
    z_min: float,
    z_max: float,
) -> Optional[float]:
    if str(mode).lower() == "off":
        return None
    if hold_m is not None:
        return float(hold_m)
    # auto: hold spawn altitude, clipped to outdoor search band
    return float(np.clip(float(start_z), float(z_min), float(z_max)))


def _body_to_world(pos: np.ndarray, yaw: float, g_rel: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    g = np.asarray(g_rel, dtype=np.float64).reshape(-1)
    return pos + np.array([
        c * g[0] - s * g[1],
        s * g[0] + c * g[1],
        g[2] if g.size > 2 else 0.0,
    ], dtype=np.float64)


@dataclass
class VisionStepResult:
    goal_rel: Optional[np.ndarray]
    target_world: Optional[np.ndarray]
    tracker_state: str
    det_hit: bool
    using_vision: bool
    using_fallback: bool
    search_action: Optional[np.ndarray]
    perception: Optional[Dict[str, Any]] = None
    using_area_search: bool = False
    using_det_steer: bool = False
    dynamic_mode: Optional[str] = None


class _FfmpegVideoWriter:
    """Stream RGB frames to an H.264 mp4 via ffmpeg."""

    def __init__(self, out_mp4: Path, *, fps: float) -> None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        self.path = out_mp4
        self._fps = float(fps)
        self._proc: Optional[subprocess.Popen] = None
        self._size: Optional[Tuple[int, int]] = None
        self.n_frames = 0

    def write(self, rgb: np.ndarray) -> None:
        fr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if fr.ndim != 3 or fr.shape[2] < 3:
            return
        h, w = fr.shape[:2]
        ww, hh = w - (w % 2), h - (h % 2)
        if self._proc is None:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{ww}x{hh}", "-r", str(self._fps),
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(self.path),
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self._size = (ww, hh)
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(fr[:hh, :ww, :3], dtype=np.uint8).tobytes())
        self.n_frames += 1

    def close(self) -> None:
        if self._proc is None:
            return
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed rc={rc}\n{err[-2000:]}")
        logger.info("Wrote %d frames → %s", self.n_frames, self.path)
        self._proc = None


def _draw_vgoal_demo_frame(
    rgb: np.ndarray,
    *,
    step: int,
    max_steps: int,
    target_class: str,
    vstep: VisionStepResult,
    d_vis: Optional[float],
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    perc = vstep.perception or {}
    bbox = perc.get("bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        color = (0, 220, 80) if vstep.det_hit else (80, 80, 220)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{target_class} {perc.get('conf', 0):.2f}"
        if perc.get("d_fused_m") is not None:
            label += f" d={perc['d_fused_m']:.1f}m"
        cv2.putText(bgr, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.72, bgr, 0.28, 0, bgr)
    mode = vstep.tracker_state
    if vstep.using_det_steer:
        mode += " +det-steer"
    dist_s = f"{d_vis:.1f}m" if d_vis is not None and np.isfinite(d_vis) else "—"
    cv2.putText(
        bgr,
        f"VGoal demo | {target_class} | step {step + 1}/{max_steps} | {mode}",
        (12, 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"det={'Y' if vstep.det_hit else 'N'}  vision={'Y' if vstep.using_vision else 'N'}  d_vis={dist_s}",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bbox_det_steer_yaw_rate(
    det: Any,
    image_width: int,
    *,
    gain: float = 1.0,
    max_yaw_rate: float = 0.314,
) -> float:
    """Yaw rate (rad/s step) to center a detection bbox in the image."""
    bb = np.asarray(det.bbox, dtype=np.float64).reshape(-1)
    if bb.size < 4:
        return 0.0
    cu = 0.5 * (float(bb[0]) + float(bb[2]))
    half_w = max(float(image_width) * 0.5, 1.0)
    err_norm = (cu - half_w) / half_w
    return float(
        np.clip(-float(gain) * err_norm * float(max_yaw_rate), -float(max_yaw_rate), float(max_yaw_rate))
    )


def _nearest_scene_object_goal(
    env: Any,
    pos: np.ndarray,
    *,
    yaw: float = 0.0,
    pattern: str = "Cart.*",
    max_dist_m: float = 600.0,
    fov_deg: float = 160.0,
    min_fwd_m: float = 3.0,
) -> Optional[np.ndarray]:
    """Pick scene object for GT smoke: nearest in FOV cone, else boresight fallback."""
    connect = getattr(env, "_connect", None)
    if not callable(connect):
        return None
    names: List[str] = []
    for attempt in range(3):
        try:
            client = connect()
            names = list(client.simListSceneObjects(str(pattern)) or [])
            if names:
                break
        except Exception as exc:
            logger.warning("gt scene list attempt %d failed: %s", attempt + 1, exc)
        if attempt < 2:
            time.sleep(0.25)
    if not names:
        return None

    half_fov = math.radians(float(max(10.0, fov_deg)) * 0.5)
    c_yaw, s_yaw = math.cos(float(yaw)), math.sin(float(yaw))
    pos_xy = np.asarray(pos[:2], dtype=np.float64)
    client = connect()
    candidates: List[Tuple[float, float, float, str, np.ndarray]] = []
    for name in names:
        try:
            pose = client.simGetObjectPose(name)
        except Exception:
            continue
        goal = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=np.float64)
        d_world = goal - np.asarray(pos, dtype=np.float64).reshape(3)
        d_fwd = float(c_yaw * d_world[0] + s_yaw * d_world[1])
        d_left = float(-s_yaw * d_world[0] + c_yaw * d_world[1])
        if d_fwd < float(min_fwd_m):
            continue
        horiz = float(np.linalg.norm(goal[:2] - pos_xy))
        if horiz > float(max_dist_m):
            continue
        bearing = math.atan2(d_left, d_fwd)
        candidates.append((horiz, abs(bearing), bearing, str(name), goal))

    if not candidates:
        return None

    in_fov = [c for c in candidates if abs(c[2]) <= half_fov]
    pick_pool = in_fov if in_fov else candidates
    pick_mode = "fov" if in_fov else "boresight"
    if in_fov:
        horiz, _abs_bear, bearing, best_name, best_goal = min(pick_pool, key=lambda c: c[0])
    else:
        horiz, _abs_bear, bearing, best_name, best_goal = min(pick_pool, key=lambda c: (c[1], c[0]))

    logger.info(
        "GT scene pick (%s) %s bearing=%.1f° fwd=%.1fm horiz=%.1fm",
        pick_mode,
        best_name,
        math.degrees(bearing),
        float(c_yaw * (best_goal[0] - pos[0]) + s_yaw * (best_goal[1] - pos[1])),
        horiz,
    )
    return best_goal


def _raw_perception_record(
    det: Any,
    depth_map: Optional[np.ndarray],
    intrinsics: Any,
    src_shape: Tuple[int, int],
    measured_gr: Optional[np.ndarray],
    *,
    object_width_m: float,
    det_conf: float,
    near_bbox_px: float = 20.0,
    near_prior_dist_m: float = 25.0,
    bbox_prior_near: bool = True,
) -> Dict[str, Any]:
    from vgoal.geometry import bbox_forward_depth_prior, extract_target_depth, fuse_target_depth

    rec: Dict[str, Any] = {
        "det_raw": det is not None,
        "conf": round(float(det_conf), 4) if det_conf else None,
        "measured_dist_m": round(float(measured_gr[3]), 3) if measured_gr is not None else None,
        "measured_fwd_m": round(float(measured_gr[0]), 3) if measured_gr is not None else None,
        "d_patch_m": None,
        "d_bbox_prior_m": None,
        "d_fused_m": None,
        "bbox_w_px": None,
        "gt_direct_depth_m": None,
    }
    if det is None:
        return rec
    bb = [float(x) for x in det.bbox]
    rec["bbox"] = bb
    rec["bbox_w_px"] = round(bb[2] - bb[0], 1)
    d_direct = float(getattr(det, "direct_depth", 0.0) or 0.0)
    if d_direct > 0.0:
        rec["gt_direct_depth_m"] = round(d_direct, 3)
    if depth_map is not None:
        dp = extract_target_depth(depth_map, bb, src_shape=src_shape)
        db = bbox_forward_depth_prior(bb, intrinsics, src_shape=src_shape, object_width_m=object_width_m)
        df = fuse_target_depth(
            dp,
            db,
            bb[2] - bb[0],
            near_bbox_px=near_bbox_px,
            near_prior_dist_m=near_prior_dist_m,
            bbox_prior_near=bbox_prior_near,
        )
        if np.isfinite(dp):
            rec["d_patch_m"] = round(float(dp), 3)
        if np.isfinite(db):
            rec["d_bbox_prior_m"] = round(float(db), 3)
        if np.isfinite(df):
            rec["d_fused_m"] = round(float(df), 3)
    return rec


def _build_detector(args: argparse.Namespace, vgoal_repo: Path) -> Any:
    from vgoal.detector import MockDetector, OpenVocabPromptDetector, YOLOTargetDetector

    kind = str(args.detector).lower()
    if kind == "mock":
        return MockDetector(confidence=0.0, class_name=str(args.target_class or "target"))
    if kind == "gt":
        logger.warning("--detector gt is DEBUG ONLY — not valid for product eval")
        return _GroundTruthDetector(
            fov_deg=float(args.camera_fov_deg),
            img_w=int(args.capture_w),
            img_h=int(args.capture_h),
        )
    if kind == "humen_bridge":
        from experiments.aerial.scripts.humen_bridge_detector import HumenBridgeDetector

        prompt = str(args.visual_prompt or args.target_class or "bridge")
        return HumenBridgeDetector(
            model_path=str(args.yolo_model),
            conf_threshold=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            device=str(args.yolo_device),
            visual_prompt=prompt,
        )
    if kind == "humen_corridor":
        from experiments.aerial.scripts.humen_corridor_detector import HumenCorridorDetector

        prompt = str(args.visual_prompt or args.target_class or "bridge")
        return HumenCorridorDetector(
            model_path=str(args.yolo_model),
            conf_threshold=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            device=str(args.yolo_device),
            visual_prompt=prompt,
        )
    if kind in ("open_vocab", "semantic"):
        prompt = str(args.visual_prompt or args.target_class or "car")
        return OpenVocabPromptDetector(
            visual_prompt=prompt,
            model_path=str(args.yolo_model),
            conf_threshold=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            device=str(args.yolo_device),
        )
    classes = [str(args.target_class)] if args.target_class else None
    return YOLOTargetDetector(
        model_path=str(args.yolo_model),
        target_classes=classes,
        conf_threshold=float(args.yolo_conf),
        imgsz=int(args.yolo_imgsz),
        device=str(args.yolo_device),
    )


class _GroundTruthDetector:
    """DEBUG: project annotation goal into image (not product path)."""

    def __init__(self, fov_deg: float = 80.0, img_w: int = 224, img_h: int = 224) -> None:
        from vgoal.geometry import CameraIntrinsics
        from vgoal.detector import DetectionResult

        self.intrinsics = CameraIntrinsics.from_fov(fov_deg, width=img_w, height=img_h)
        self._DetectionResult = DetectionResult
        self._goal_world: Optional[np.ndarray] = None
        self._pos: Optional[np.ndarray] = None
        self._yaw: float = 0.0

    def set_goal(self, goal_world: np.ndarray) -> None:
        self._goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)

    def set_pose(self, pos: np.ndarray, yaw: float) -> None:
        self._pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self._yaw = float(yaw)

    def detect(self, rgb: Optional[np.ndarray] = None):
        if self._goal_world is None or self._pos is None:
            return None
        from vgoal.geometry import project_3d_to_pixel

        d_world = self._goal_world - self._pos
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        d_fwd = float(c * d_world[0] + s * d_world[1])
        d_left = float(-s * d_world[0] + c * d_world[1])
        d_up = float(d_world[2])
        if d_fwd <= 0.5:
            return None
        u, v, _z = project_3d_to_pixel([d_fwd, d_left, d_up], self.intrinsics)
        if math.isnan(u) or math.isnan(v):
            return None
        w, h = self.intrinsics.width, self.intrinsics.height
        if not (0 <= u < w and 0 <= v < h):
            return None
        half_box = max(6.0, 20.0 * (10.0 / max(1.0, d_fwd)))
        bbox = np.array([
            max(0.0, u - half_box), max(0.0, v - half_box),
            min(float(w - 1), u + half_box), min(float(h - 1), v + half_box),
        ], dtype=np.float32)
        res = self._DetectionResult(bbox=bbox, confidence=0.95, class_id=0, class_name="goal")
        setattr(res, "direct_depth", float(d_fwd))
        setattr(res, "_d_left", d_left)
        setattr(res, "_d_up", d_up)
        return res


def _det_to_goal_rel(
    det: Any,
    intrinsics: Any,
    depth_map: Optional[np.ndarray],
    src_shape: Tuple[int, int],
    *,
    object_width_m: float = 2.0,
    fuse_bbox_depth: bool = True,
    near_bbox_px: float = 20.0,
    near_prior_dist_m: float = 25.0,
    bbox_prior_near: bool = True,
) -> Optional[np.ndarray]:
    from vgoal.geometry import bbox_to_goal_rel

    d_fwd = float(getattr(det, "direct_depth", 0.0) or 0.0)
    if d_fwd > 0.0:
        d_left_stored = getattr(det, "_d_left", None)
        d_up_stored = getattr(det, "_d_up", None)
        if d_left_stored is not None and d_up_stored is not None:
            d_left = float(d_left_stored)
            d_up = float(d_up_stored)
            dist = float(np.sqrt(d_fwd**2 + d_left**2 + d_up**2))
            return np.array([d_fwd, d_left, d_up, dist], dtype=np.float64)
    if depth_map is None:
        return None
    gr = bbox_to_goal_rel(
        det.bbox,
        depth_map,
        intrinsics,
        src_shape=src_shape,
        object_width_m=object_width_m,
        fuse_bbox_depth=fuse_bbox_depth,
        near_bbox_px=near_bbox_px,
        near_prior_dist_m=near_prior_dist_m,
        bbox_prior_near=bbox_prior_near,
    )
    if gr is None:
        return None
    return np.asarray(gr, dtype=np.float64)


def _vision_step(
    *,
    obs: Any,
    detector: Any,
    tracker: Any,
    dynamic_tracker: Any,
    dynamic_min_meas_conf: float,
    depth_pred: Any,
    intrinsics: Any,
    pos: np.ndarray,
    yaw: float,
    prev_pos: np.ndarray,
    prev_yaw: float,
    dt: float,
    search_fwd_step: float,
    search_yaw_rate: float,
    area_search_planner: Any,
    fallback_intent: Any,
    annot_goal: np.ndarray,
    allow_fallback: bool,
    prefer_nearest: bool,
    camera_fov_deg: float,
    object_width_m: float = 2.0,
    fuse_bbox_depth: bool = True,
    near_bbox_px: float = 20.0,
    near_prior_dist_m: float = 25.0,
    bbox_prior_near: bool = True,
    reject_far_lock_m: float = 0.0,
    spawn_acquire_yaw_rate: float = 0.0,
    search_det_steer: bool = False,
    search_det_steer_gain: float = 1.0,
    search_det_steer_fwd: float = 0.0,
    search_det_steer_max_yaw: float = 0.314,
    search_area_priority: bool = False,
) -> VisionStepResult:
    from vgoal.geometry import CameraIntrinsics
    from vgoal.tracker import TargetState

    rgb = getattr(obs, "rgb", None)
    rgb_det = getattr(obs, "rgb_yolo", None)
    rgb_det_arr = np.asarray(rgb_det if rgb_det is not None else rgb, dtype=np.uint8)
    det_h, det_w = rgb_det_arr.shape[:2]
    if det_w != intrinsics.width or det_h != intrinsics.height:
        intrinsics = CameraIntrinsics.from_fov(float(camera_fov_deg), width=det_w, height=det_h)

    depth_map = depth_pred.predict_depth(obs) if depth_pred is not None else None

    if hasattr(detector, "set_pose"):
        detector.set_pose(pos, yaw)
    det = None
    measured_gr: Optional[np.ndarray] = None
    det_conf = 0.0

    detect_all = getattr(detector, "detect_all", None)
    if prefer_nearest and callable(detect_all) and depth_map is not None:
        best_gr = None
        best_conf = 0.0
        best_det = None
        for cand in detect_all(rgb_det_arr) or []:
            gr = _det_to_goal_rel(
                cand,
                intrinsics,
                depth_map,
                (det_w, det_h),
                object_width_m=object_width_m,
                fuse_bbox_depth=fuse_bbox_depth,
                near_bbox_px=near_bbox_px,
                near_prior_dist_m=near_prior_dist_m,
                bbox_prior_near=bbox_prior_near,
            )
            if gr is None:
                continue
            if best_gr is None or float(gr[3]) < float(best_gr[3]):
                best_gr = gr
                best_conf = float(cand.confidence)
                best_det = cand
        if best_gr is not None:
            det = best_det
            measured_gr = best_gr
            det_conf = best_conf
    else:
        det = detector.detect(rgb_det_arr)
        if det is not None:
            measured_gr = _det_to_goal_rel(
                det,
                intrinsics,
                depth_map,
                (det_w, det_h),
                object_width_m=object_width_m,
                fuse_bbox_depth=fuse_bbox_depth,
                near_bbox_px=near_bbox_px,
                near_prior_dist_m=near_prior_dist_m,
                bbox_prior_near=bbox_prior_near,
            )
            if measured_gr is not None:
                det_conf = float(det.confidence)

    perception_rec = _raw_perception_record(
        det,
        depth_map,
        intrinsics,
        (det_w, det_h),
        measured_gr,
        object_width_m=object_width_m,
        det_conf=det_conf,
        near_bbox_px=near_bbox_px,
        near_prior_dist_m=near_prior_dist_m,
        bbox_prior_near=bbox_prior_near,
    )

    d_world = pos - prev_pos
    dyaw = float(yaw - prev_yaw)
    dyaw = float((dyaw + math.pi) % (2.0 * math.pi) - math.pi)
    c_yaw, s_yaw = math.cos(prev_yaw), math.sin(prev_yaw)
    ego_delta = np.array([
        c_yaw * d_world[0] + s_yaw * d_world[1],
        -s_yaw * d_world[0] + c_yaw * d_world[1],
        d_world[2],
    ], dtype=np.float64)

    det_hit = det is not None and det_conf >= 1e-6
    if (
        measured_gr is not None
        and float(reject_far_lock_m) > 0.0
        and float(measured_gr[3]) > float(reject_far_lock_m)
    ):
        perception_rec["far_lock_rejected"] = True
        perception_rec["rejected_goal_d_m"] = round(float(measured_gr[3]), 3)
        measured_gr = None

    dynamic_mode: Optional[str] = None
    tracker_state_str = str(TargetState.SEARCHING.value)
    vision_would_lock = False
    vision_g_rel: Optional[np.ndarray] = None
    vision_target: Optional[np.ndarray] = None

    if dynamic_tracker is not None:
        from vgoal.dynamic_tracker import TrackingMode

        mode, cur_goal_rel = dynamic_tracker_step(
            dynamic_tracker,
            measured_gr,
            det_conf,
            pos,
            yaw,
            dt,
            min_meas_conf=float(dynamic_min_meas_conf),
        )
        tracker_state_str = str(mode.value)
        dynamic_mode = tracker_state_str
        if cur_goal_rel is not None and mode not in (TrackingMode.SEARCHING, TrackingMode.LOST):
            vision_would_lock = True
            vision_g_rel = np.asarray(cur_goal_rel, dtype=np.float64)
            vision_target = _body_to_world(pos, yaw, vision_g_rel)
            if not search_area_priority:
                return VisionStepResult(
                    goal_rel=vision_g_rel,
                    target_world=vision_target,
                    tracker_state=tracker_state_str,
                    det_hit=det_hit,
                    using_vision=True,
                    using_fallback=False,
                    using_area_search=False,
                    search_action=None,
                    perception=perception_rec,
                    dynamic_mode=dynamic_mode,
                )
    else:
        tracker_state = tracker.update(
            measured_gr,
            dt=dt,
            ego_delta_body=ego_delta,
            ego_delta_yaw=dyaw,
            confidence=det_conf,
        )
        tracker_state_str = str(tracker_state.value)
        cur_goal_rel = tracker.goal_rel
        if cur_goal_rel is not None and tracker_state != TargetState.SEARCHING:
            vision_would_lock = True
            vision_g_rel = np.asarray(cur_goal_rel, dtype=np.float64)
            vision_target = _body_to_world(pos, yaw, vision_g_rel)
            if not search_area_priority:
                return VisionStepResult(
                    goal_rel=vision_g_rel,
                    target_world=vision_target,
                    tracker_state=tracker_state_str,
                    det_hit=det_hit,
                    using_vision=True,
                    using_fallback=False,
                    using_area_search=False,
                    search_action=None,
                    perception=perception_rec,
                    dynamic_mode=None,
                )

    if measured_gr is not None and det_hit:
        perception_rec["bridge_goal_rel"] = [round(float(x), 4) for x in measured_gr]

    if allow_fallback and fallback_intent is not None:
        d_fwd_hat = obs.info.get("depth_min_pred")
        g_rel_body, s_info = fallback_intent.compute(
            curr_pos=pos, curr_yaw=yaw, goal=annot_goal, d_fwd_hat=d_fwd_hat
        )
        target_world = np.array(s_info["target_world"], dtype=np.float64)
        return VisionStepResult(
            goal_rel=np.asarray(g_rel_body, dtype=np.float64),
            target_world=target_world,
            tracker_state=str(TargetState.SEARCHING.value),
            det_hit=det_hit,
            using_vision=False,
            using_fallback=True,
            using_area_search=False,
            search_action=None,
            perception=perception_rec,
        )

    if float(spawn_acquire_yaw_rate) != 0.0:
        search_action = np.array([0.0, 0.0, 0.0, float(spawn_acquire_yaw_rate)], dtype=np.float64)
        return VisionStepResult(
            goal_rel=None,
            target_world=None,
            tracker_state=str(TargetState.SEARCHING.value),
            det_hit=det_hit,
            using_vision=False,
            using_fallback=False,
            using_area_search=False,
            search_action=search_action,
            perception=perception_rec,
        )

    if search_det_steer and det is not None:
        yaw_rate = bbox_det_steer_yaw_rate(
            det,
            det_w,
            gain=float(search_det_steer_gain),
            max_yaw_rate=float(search_det_steer_max_yaw),
        )
        fwd = float(search_det_steer_fwd) if float(search_det_steer_fwd) > 0.0 else float(search_fwd_step) * 0.25
        search_action = np.array([fwd, 0.0, 0.0, yaw_rate], dtype=np.float64)
        return VisionStepResult(
            goal_rel=None,
            target_world=None,
            tracker_state=str(TargetState.SEARCHING.value),
            det_hit=det_hit,
            using_vision=False,
            using_fallback=False,
            using_area_search=False,
            using_det_steer=True,
            search_action=search_action,
            perception=perception_rec,
        )

    if area_search_planner is not None:
        g_rel = np.asarray(area_search_planner.update(pos, yaw), dtype=np.float64)
        target_world = _body_to_world(pos, yaw, g_rel)
        return VisionStepResult(
            goal_rel=g_rel,
            target_world=target_world,
            tracker_state=tracker_state_str if vision_would_lock else str(TargetState.SEARCHING.value),
            det_hit=det_hit,
            using_vision=False,
            using_fallback=False,
            using_area_search=True,
            search_action=None,
            perception=perception_rec,
            dynamic_mode=dynamic_mode,
        )

    search_action = np.array([search_fwd_step, 0.0, 0.0, search_yaw_rate], dtype=np.float64)
    return VisionStepResult(
        goal_rel=None,
        target_world=None,
        tracker_state=str(TargetState.SEARCHING.value),
        det_hit=det_hit,
        using_vision=False,
        using_fallback=False,
        using_area_search=False,
        search_action=search_action,
        perception=perception_rec,
    )


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Phase-2 monocular visual goal eval")
    parser.add_argument("--config", default="configs/aerial_rl.yaml")
    parser.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt",
    )
    parser.add_argument(
        "--actor-ckpt",
        default=(
            "experiments/aerial/rl/artifacts/"
            "v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt"
        ),
    )
    parser.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
    )
    parser.add_argument(
        "--tau-ckpt",
        default="experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt",
    )
    parser.add_argument("--annotation", default="artifacts/seen_airsim16_long_routes.json")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--routes", type=str, default=None)
    parser.add_argument("--step-hz", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--cruise-speed", type=float, default=10.0)
    parser.add_argument("--tti-coeff", type=float, default=2.5)
    parser.add_argument("--success-dist", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--planner-horizon", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", default="artifacts/wam_vgoal_eval_result.json")
    parser.add_argument("--spawn-tol-m", type=float, default=12.0)
    parser.add_argument("--traj-out", default=None)
    parser.add_argument(
        "--video-out",
        default=None,
        help="Ego FPV mp4 path or directory (uses rgb_yolo when fan-out enabled)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Video fps (default: --step-hz)",
    )
    parser.add_argument(
        "--perception-log",
        default=None,
        help="Per-step raw perception JSONL dir/prefix (route{idx}_perception.jsonl)",
    )
    parser.add_argument(
        "--gt-nearest-scene-object",
        action="store_true",
        help="DEBUG/GT: use nearest AirSim scene object as goal (--detector gt)",
    )
    parser.add_argument("--gt-scene-pattern", default="Cart.*")
    parser.add_argument("--gt-scene-max-dist-m", type=float, default=600.0)
    parser.add_argument(
        "--gt-scene-fov-deg",
        type=float,
        default=160.0,
        help="Forward cone for GT object pick (degrees)",
    )
    parser.add_argument("--gt-scene-min-fwd-m", type=float, default=3.0)
    parser.add_argument("--vgoal-repo", default=os.path.expanduser("~/Projects/aerial-vgoal-wam"))
    parser.add_argument("--camera-fov-deg", type=float, default=80.0)
    parser.add_argument(
        "--capture-w",
        type=int,
        default=int(os.environ.get("AERIAL_CAPTURE_W", os.environ.get("INDOOR_CAPTURE_W", "640"))),
        help="AirSim CaptureSettings width (native grab before fan-out)",
    )
    parser.add_argument(
        "--capture-h",
        type=int,
        default=int(os.environ.get("AERIAL_CAPTURE_H", os.environ.get("INDOOR_CAPTURE_H", "480"))),
        help="AirSim CaptureSettings height",
    )
    parser.add_argument("--wam-encode-size", type=int, default=224)
    parser.add_argument(
        "--fanout-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Single grab → rgb_vio(native) + rgb_yolo(native) + rgb(224 WAM)",
    )
    parser.add_argument("--tracker-max-occlusion-s", type=float, default=2.0)
    parser.add_argument("--tracker-ema-alpha", type=float, default=0.7)
    parser.add_argument("--tracker-near-dist-m", type=float, default=35.0)
    parser.add_argument("--tracker-near-ema-alpha", type=float, default=0.92)
    parser.add_argument("--tracker-inflate-reject-m", type=float, default=4.0)
    parser.add_argument("--tracker-inflate-alpha", type=float, default=0.15)
    parser.add_argument(
        "--car-width-m",
        type=float,
        default=2.0,
        help="Assumed car width for bbox→depth prior (pinhole Z ≈ fx·W/w_px)",
    )
    parser.add_argument(
        "--bbox-depth-fuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse D̂ patch depth with bbox-width prior (default ON)",
    )
    parser.add_argument(
        "--bbox-prior-near",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Near range: min(d_bbox,d_patch) when bbox>=near-px or prior<=near-dist (default ON)",
    )
    parser.add_argument("--bbox-near-px", type=float, default=20.0)
    parser.add_argument("--bbox-near-dist-m", type=float, default=25.0)
    parser.add_argument(
        "--tracker-freeze-dist-on-occlude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OCCLUDED dead-reckoning cannot inflate range (static target, default ON)",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo", "open_vocab", "semantic", "humen_bridge", "humen_corridor", "mock", "gt"),
        default="yolo",
        help="Perception frontend (default: yolo pure vision)",
    )
    parser.add_argument("--target-class", default="car", help="YOLO COCO class filter")
    parser.add_argument("--visual-prompt", default=None, help="Open-vocab prompt (open_vocab detector)")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", default="cuda")
    parser.add_argument("--prefer-nearest-target", action="store_true", default=True)
    parser.add_argument(
        "--search-fwd-speed",
        type=float,
        default=None,
        help="SEARCHING forward m/step; default 0.2 (slow scan) unless --search-at-cruise",
    )
    parser.add_argument(
        "--search-at-cruise",
        action="store_true",
        help="SEARCHING uses cruise_speed/step_hz forward (default: slow 0.2 m/step)",
    )
    parser.add_argument("--search-yaw-rate", type=float, default=0.314)
    parser.add_argument(
        "--reject-far-lock-m",
        type=float,
        default=40.0,
        help="Ignore vision measurements with goal_rel dist above this (0=off). Rejects billboard far-locks.",
    )
    parser.add_argument(
        "--det-early-steps",
        type=int,
        default=50,
        help="Window for det_recall_early metric (any det_hit in first N steps).",
    )
    parser.add_argument(
        "--spawn-yaw-acquire-steps",
        type=int,
        default=0,
        help="Hover yaw micro-sweep at spawn before M3 search (0=off).",
    )
    parser.add_argument(
        "--spawn-yaw-acquire-deg",
        type=float,
        default=60.0,
        help="Total yaw span for spawn acquire ping-pong sweep.",
    )
    parser.add_argument(
        "--spawn-yaw-acquire-step-deg",
        type=float,
        default=10.0,
        help="Yaw delta per step during spawn acquire (degrees).",
    )
    parser.add_argument(
        "--search-det-steer",
        action="store_true",
        help="When SEARCHING with det but no lock, yaw toward bbox center (before M3 area search).",
    )
    parser.add_argument("--search-det-steer-gain", type=float, default=1.0)
    parser.add_argument(
        "--search-det-steer-fwd",
        type=float,
        default=0.0,
        help="Forward m/step during det-steer (0 = 25%% of search_fwd_speed).",
    )
    parser.add_argument(
        "--search-det-steer-max-yaw",
        type=float,
        default=None,
        help="Max yaw rate during det-steer (default: --search-yaw-rate).",
    )
    parser.add_argument(
        "--search-yaw-hold-deg",
        type=float,
        default=0.0,
        help="During M3 area search, pull yaw toward spawn heading within this band (0=off).",
    )
    parser.add_argument("--search-yaw-hold-gain", type=float, default=2.0)
    parser.add_argument(
        "--search-delay-area-until-acquire",
        action="store_true",
        help="Skip M3 area search until spawn-yaw-acquire window ends (or vision lock).",
    )
    parser.add_argument(
        "--search-pattern",
        choices=("scan", "lawnmower", "spiral", "corridor"),
        default="lawnmower",
        help="SEARCHING: scan=fwd+yaw; lawnmower/spiral/corridor=AreaSearchPlanner (M3)",
    )
    parser.add_argument(
        "--corridor-yaw-sweep-interval",
        type=int,
        default=0,
        help="Corridor search: repeat hover yaw sweep every N steps (0=off).",
    )
    parser.add_argument("--corridor-yaw-sweep-steps", type=int, default=12)
    parser.add_argument("--corridor-yaw-sweep-deg", type=float, default=60.0)
    parser.add_argument("--corridor-yaw-sweep-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--search-area-priority",
        action="store_true",
        help="M3 SEARCH: keep flying area waypoints even when tracker locks (log det/bridge_goal_rel).",
    )
    parser.add_argument(
        "--no-shield",
        action="store_true",
        help=(
            "Disable the three_zone safety shield entirely (e.g. Humen open-water depth "
            "hallucination causes near-100% intervention_rate and blocks forward progress "
            "during SEARCH/APPROACH, same issue --no-shield already works around in "
            "wam_phase2_long_eval for SURVEY)."
        ),
    )
    parser.add_argument(
        "--search-direct-area",
        action="store_true",
        help="M3 SEARCH: drive toward area waypoints with cruise body deltas (skip π during area search).",
    )
    parser.add_argument(
        "--search-area-half-m",
        type=float,
        default=40.0,
        help="Half-width of search box centered on spawn (world m)",
    )
    parser.add_argument("--search-sweep-spacing-m", type=float, default=15.0)
    parser.add_argument("--search-waypoint-radius-m", type=float, default=3.5)
    parser.add_argument("--search-spiral-radius-m", type=float, default=25.0)
    parser.add_argument(
        "--follow-mode",
        choices=("static", "standoff"),
        default="static",
        help="static=TargetTracker approach (M2); standoff=DynamicTargetTracker intercept/follow (M4)",
    )
    parser.add_argument("--standoff-dist-m", type=float, default=6.0)
    parser.add_argument("--standoff-height-m", type=float, default=3.0)
    parser.add_argument("--intercept-dist-m", type=float, default=12.0)
    parser.add_argument(
        "--follow-success-dist-m",
        type=float,
        default=4.0,
        help="Success when FOLLOWING and goal_rel dist <= this (standoff mode)",
    )
    parser.add_argument(
        "--dynamic-meas-conf",
        type=float,
        default=None,
        help="Min conf passed to DynamicTargetTracker measurement update (default: yolo_conf)",
    )
    parser.add_argument(
        "--search-z-hold-mode",
        choices=("auto", "off"),
        default="auto",
        help="SEARCHING altitude hold: auto clips spawn z to [--search-z-min, --search-z-max]",
    )
    parser.add_argument("--search-z-hold-m", type=float, default=None, help="Fixed SEARCHING z (m); overrides auto")
    parser.add_argument("--search-z-min", type=float, default=20.0)
    parser.add_argument("--search-z-max", type=float, default=40.0)
    parser.add_argument(
        "--search-z-gain",
        type=float,
        default=1.0,
        help="SEARCHING z-hold P gain: dz_step ~= gain * (z_hold - z) clipped per step",
    )
    parser.add_argument(
        "--toward-g-r-m",
        type=float,
        default=25.0,
        help="Visual G TowardGoalIntent clip radius (m)",
    )
    parser.add_argument(
        "--visual-toward-g",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TRACKING: clip visual target through TowardGoalIntent before π (default ON)",
    )
    parser.add_argument(
        "--tracker-min-confidence",
        type=float,
        default=None,
        help="TargetTracker lock threshold; default aligns with --yolo-conf (cap 0.5)",
    )
    parser.add_argument(
        "--fallback-toward-g",
        action="store_true",
        help="Ablation: geometric toward_g when SEARCHING (default OFF)",
    )
    args = parser.parse_args()
    if args.dynamic_meas_conf is None:
        args.dynamic_meas_conf = float(args.yolo_conf)

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    vgoal_repo = Path(args.vgoal_repo).expanduser().resolve()
    if not vgoal_repo.is_dir():
        raise SystemExit(f"--vgoal-repo not found: {vgoal_repo}")
    if str(vgoal_repo) not in sys.path:
        sys.path.insert(0, str(vgoal_repo))

    from vgoal.geometry import CameraIntrinsics
    from vgoal.tracker import TargetTracker, TrackerConfig

    import torch
    from experiments.aerial.rl.actor_critic import LatentActorCritic, LatentActorDeployPolicy
    from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
    from experiments.aerial.rl.goal_features import body_vel_from_obs
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.tau_predictor import make_tau_predictor
    from experiments.aerial.rl.scene_intent import TowardGoalIntent
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics

    cfg_file = (root / args.config).resolve()
    cfg = yaml.safe_load(cfg_file.read_text()) if cfg_file.is_file() else {}

    device_str = "cpu" if (args.mock or not torch.cuda.is_available()) else args.device
    device = torch.device(device_str)
    logger.info("device=%s mock=%s detector=%s fallback=%s", device, args.mock, args.detector, args.fallback_toward_g)

    anno_path = (root / args.annotation).resolve() if not Path(args.annotation).is_absolute() else Path(args.annotation)
    with open(anno_path, "r", encoding="utf-8") as f:
        anno_data = json.load(f)
    routes = anno_data.get("routes", anno_data) if isinstance(anno_data, dict) else anno_data
    route_idxs = _select_route_indices(len(routes), args.episodes, args.routes)
    n_routes = len(route_idxs)

    env_cfg = dict(cfg.get("env") or {})
    env_cfg["backend"] = "mock" if args.mock else "airsim"
    env_cfg["step_hz"] = float(args.step_hz)
    env_cfg["grab_depth"] = True
    use_fanout = bool(args.fanout_rgb) and str(args.detector).lower() != "mock"
    env_cfg["fanout_rgb"] = use_fanout
    env_cfg["width"] = int(args.capture_w)
    env_cfg["height"] = int(args.capture_h)
    env_cfg["wam_encode_size"] = int(args.wam_encode_size)
    # Humen / multi-scene: aerial_inspect sets AIRSIM_* — override yaml default :41451.
    if os.environ.get("AIRSIM_HOST"):
        env_cfg["host"] = os.environ["AIRSIM_HOST"]
    if os.environ.get("AIRSIM_PORT"):
        env_cfg["port"] = int(os.environ["AIRSIM_PORT"])
    if os.environ.get("AIRSIM_VEHICLE"):
        env_cfg["vehicle"] = os.environ["AIRSIM_VEHICLE"]
    if os.environ.get("AIRSIM_CAMERA"):
        env_cfg["camera"] = os.environ["AIRSIM_CAMERA"]
    env = _build_env(env_cfg)

    wm_cfg = cfg.get("world_model") or {}
    wm_path = (root / args.wm_ckpt).resolve() if not Path(args.wm_ckpt).is_absolute() else Path(args.wm_ckpt)
    dynamics, _ = load_torch_dynamics(wm_cfg, str(wm_path), device=device_str, success_dist_m=float(args.success_dist))

    actor_path = (root / args.actor_ckpt).resolve() if not Path(args.actor_ckpt).is_absolute() else Path(args.actor_ckpt)
    if not args.mock and actor_path.exists():
        actor_ac = LatentActorCritic.load_from_checkpoint(actor_path, device=device_str)
        actor_ac.config.goal_feat_mode = "meter"
        logger.info("Loaded actor-critic from %s", actor_path)
    else:
        actor_ac = LatentActorCritic.from_config({"latent_dim": dynamics.latent_dim, "device": device_str})

    depth_path = (root / args.depth_ckpt).resolve() if not Path(args.depth_ckpt).is_absolute() else Path(args.depth_ckpt)
    depth_pred = (
        DepthMinPredictor.from_checkpoint(depth_path, device=device_str)
        if (not args.mock and depth_path.is_file() and str(args.detector).lower() not in ("gt", "mock"))
        else None
    )
    if depth_pred is None and not args.mock and args.detector not in ("gt", "mock"):
        raise SystemExit(f"depth checkpoint required for YOLO back-projection: {depth_path}")

    tau_path = (root / args.tau_ckpt).resolve() if not Path(args.tau_ckpt).is_absolute() else Path(args.tau_ckpt)
    tau_pred = make_tau_predictor(
        kind="foe_calibrated",
        ckpt=tau_path if (not args.mock and tau_path.is_file()) else None,
        device=device_str,
    )

    dt_step = 1.0 / float(args.step_hz)
    phys_limits = body_delta_limits(dt_step)
    vx_max_step = float(min(float(args.cruise_speed) / float(args.step_hz), float(phys_limits[0])))
    action_limits = np.array(
        [vx_max_step, float(phys_limits[1]), float(phys_limits[2]), float(phys_limits[3])],
        dtype=np.float64,
    )
    search_fwd_step = _resolve_search_fwd_step(
        args.search_fwd_speed,
        search_at_cruise=bool(args.search_at_cruise),
        vx_max_step=vx_max_step,
    )
    search_fwd_label = (
        "override"
        if args.search_fwd_speed is not None
        else ("cruise" if args.search_at_cruise else "slow")
    )
    det_steer_max_yaw = (
        float(args.search_det_steer_max_yaw)
        if args.search_det_steer_max_yaw is not None
        else float(args.search_yaw_rate)
    )
    tracker_min_conf = (
        float(args.tracker_min_confidence)
        if args.tracker_min_confidence is not None
        else float(max(0.15, min(0.5, float(args.yolo_conf))))
    )

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)

    planner = None
    if args.planner:
        planner = ImaginationPlanner(
            dynamics=dynamics,
            horizon=int(args.planner_horizon),
            reward_cfg=reward_cfg,
            action_limits=action_limits,
        )

    policy = LatentActorDeployPolicy(dynamics, actor_ac, deterministic=True, stream_latent=True)

    shield = None
    if bool(args.no_shield):
        logger.info("shield disabled (--no-shield)")
    else:
        safety_cfg = dict(cfg.get("safety") or {})
        if str(safety_cfg.get("kind", "null")) in ("null", "none", "None"):
            safety_cfg["kind"] = "three_zone"
        safety_cfg["v_cruise_m_s"] = float(args.cruise_speed)
        safety_cfg["tti_coeff"] = float(args.tti_coeff)
        safety_cfg.pop("schedule_margin_l1_m", None)
        safety_cfg.pop("schedule_margin_l2_m", None)
        safety_cfg.pop("disc_lag_steps", None)
        shield = _build_safety(safety_cfg)
        if hasattr(shield, "zone"):
            logger.info(
                "three_zone v_cruise=%.1f engage_outer=%.1fm tti_coeff=%.1f",
                float(shield.zone.v_cruise_m_s),
                float(shield.zone.engage_outer_m),
                float(shield.tti_coeff),
            )

    fallback_intent = (
        TowardGoalIntent(r_m=100.0, mode="toward_g", cruise_speed=float(args.cruise_speed))
        if args.fallback_toward_g
        else None
    )
    visual_intent = (
        TowardGoalIntent(
            r_m=float(args.toward_g_r_m),
            mode="toward_g",
            cruise_speed=float(args.cruise_speed),
        )
        if args.visual_toward_g
        else None
    )

    det_w = int(args.capture_w)
    det_h = int(args.capture_h)
    intrinsics = CameraIntrinsics.from_fov(args.camera_fov_deg, width=det_w, height=det_h)
    tracker_cfg = TrackerConfig(
        success_dist_m=float(args.success_dist),
        max_occlusion_s=float(args.tracker_max_occlusion_s),
        ema_alpha=float(args.tracker_ema_alpha),
        min_confidence=tracker_min_conf,
        near_dist_m=float(args.tracker_near_dist_m),
        near_ema_alpha=float(args.tracker_near_ema_alpha),
        inflate_reject_m=float(args.tracker_inflate_reject_m),
        inflate_alpha=float(args.tracker_inflate_alpha),
        freeze_dist_on_occlude=bool(args.tracker_freeze_dist_on_occlude),
    )
    detector = _build_detector(args, vgoal_repo)

    visual_prompt = str(args.visual_prompt or args.target_class or "car")
    logger.info(
        "phase2_vgoal: %d routes | cs=%.1f tti=%.1f | det=%s prompt=%s "
        "fanout=%s capture=%dx%d wam=%d search=%s area_half=%.0fm "
        "search_fwd=%.3f(%s) yaw=%.2f z_hold=%s visual_toward_g=%s "
        "tracker_conf=%.2f yolo_conf=%.2f bbox_fuse=%s prior_near=%s "
        "car_w=%.1fm freeze_occ=%s follow=%s standoff=%.0f/%.0fm",
        n_routes, args.cruise_speed, args.tti_coeff, args.detector, visual_prompt,
        use_fanout, int(args.capture_w), int(args.capture_h), int(args.wam_encode_size),
        args.search_pattern, float(args.search_area_half_m),
        search_fwd_step, search_fwd_label, args.search_yaw_rate,
        args.search_z_hold_mode, bool(args.visual_toward_g), tracker_min_conf, args.yolo_conf,
        bool(args.bbox_depth_fuse), bool(args.bbox_prior_near), float(args.car_width_m),
        bool(args.tracker_freeze_dist_on_occlude),
        args.follow_mode, float(args.standoff_dist_m), float(args.standoff_height_m),
    )

    results: List[Dict[str, Any]] = []

    for slot, ep_idx in enumerate(route_idxs):
        r_info = routes[ep_idx]
        pts = np.array(r_info.get("pos", r_info.get("positions")), dtype=np.float64)
        goal_world = pts[-1].copy()
        if r_info.get("goal_pos"):
            goal_world = np.asarray(r_info["goal_pos"], dtype=np.float64).reshape(3)
        annot_goal = goal_world.copy()
        start_pos = pts[0].copy()
        yaws = np.array(r_info.get("yaw", [0.0] * len(pts)), dtype=np.float64)
        start_yaw = float(yaws[0]) if len(yaws) else 0.0
        ref_len = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

        policy.reset()
        if shield is not None:
            shield.reset()
        tau_pred.reset()
        if depth_pred is not None:
            depth_pred.reset()
        if planner is not None:
            planner.reset()
        if fallback_intent is not None:
            fallback_intent.reset()
        if visual_intent is not None:
            visual_intent.reset()
        ep_z_hold = _episode_search_z_hold(
            float(start_pos[2]),
            mode=str(args.search_z_hold_mode),
            hold_m=args.search_z_hold_m,
            z_min=float(args.search_z_min),
            z_max=float(args.search_z_max),
        )
        dynamic_tracker = None
        if str(args.follow_mode) == "standoff":
            dynamic_tracker = make_dynamic_tracker(
                standoff_dist_m=float(args.standoff_dist_m),
                standoff_height_m=float(args.standoff_height_m),
                intercept_dist_m=float(args.intercept_dist_m),
                max_occlusion_s=float(args.tracker_max_occlusion_s),
            )
            tracker = None
        else:
            tracker = TargetTracker(tracker_cfg)
        area_search_planner = make_area_search_planner(
            start_pos,
            pattern=str(args.search_pattern),
            altitude_z=float(ep_z_hold if ep_z_hold is not None else start_pos[2]),
            half_m=float(args.search_area_half_m),
            sweep_spacing_m=float(args.search_sweep_spacing_m),
            waypoint_reach_radius_m=float(args.search_waypoint_radius_m),
            spiral_max_radius_m=float(args.search_spiral_radius_m),
            route_info=r_info,
        )

        ep_dict = {
            "pos": pts.tolist(),
            "yaw": yaws.tolist() if len(yaws) == len(pts) else [start_yaw] * len(pts),
            "gpt_instruction": r_info.get("gpt_instruction", visual_prompt),
        }
        obs = env.reset(ep_dict)

        p_curr = np.array(obs.position, dtype=np.float64)
        curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else 0.0
        if bool(args.gt_nearest_scene_object) and str(args.detector).lower() == "gt":
            gt_fov = float(args.gt_scene_fov_deg or args.camera_fov_deg)
            resolved = _nearest_scene_object_goal(
                env,
                p_curr,
                yaw=curr_yaw,
                pattern=str(args.gt_scene_pattern),
                max_dist_m=float(args.gt_scene_max_dist_m),
                fov_deg=gt_fov,
                min_fwd_m=float(args.gt_scene_min_fwd_m),
            )
            if resolved is not None:
                annot_goal = np.asarray(resolved, dtype=np.float64)
                goal_world = annot_goal.copy()
                logger.info(
                    "Route %02d GT goal from scene %s dist=%.1fm pos=%s",
                    ep_idx + 1,
                    args.gt_scene_pattern,
                    float(np.linalg.norm(annot_goal[:2] - p_curr[:2])),
                    [round(float(x), 2) for x in annot_goal],
                )
            else:
                logger.warning("Route %02d GT scene goal lookup failed — using annotation goal", ep_idx + 1)
        if hasattr(detector, "set_goal"):
            detector.set_goal(annot_goal)
        spawn_err = float(np.linalg.norm(p_curr - start_pos))

        if spawn_err > float(args.spawn_tol_m) and not args.mock:
            bump = start_pos.copy()
            bump[2] = float(bump[2]) + 2.0
            logger.warning("Route %02d spawn_err=%.1fm — retry z+=2", ep_idx + 1, spawn_err)
            ep_retry = dict(ep_dict)
            pts_retry = np.asarray(ep_retry["pos"], dtype=np.float64).copy()
            pts_retry[0] = bump
            ep_retry["pos"] = pts_retry.tolist()
            obs = env.reset(ep_retry)
            p_curr = np.array(obs.position, dtype=np.float64)
            curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else curr_yaw
            spawn_err = float(np.linalg.norm(p_curr - bump))

        if spawn_err > float(args.spawn_tol_m):
            logger.error("Route %02d spawn_fail err=%.1fm — skip", ep_idx + 1, spawn_err)
            results.append({
                "route_idx": ep_idx,
                "base_route_idx": r_info.get("base_route_idx", ep_idx),
                "L_ref": ref_len, "L_act": 0.0, "d0": float("nan"), "min_d": float("nan"),
                "arrived": False, "collided": False, "severe_collision": False,
                "progress_ratio": 0.0, "spl": 0.0, "intervention_rate": 0.0,
                "spawn_fail": True, "spawn_err_m": spawn_err, "fail_tag": "F1",
            })
            continue

        d0_annot = _goal_dist(p_curr, annot_goal)
        min_d_annot = d0_annot
        min_d_vision = float("inf")
        min_d_measured = float("inf")
        d_final_vision = float("inf")
        had_vision_lock = False
        traj = [p_curr.copy()]
        traj_writer = None
        perception_writer = None
        if args.traj_out:
            _tp = Path(args.traj_out).with_suffix("") / f"route{ep_idx:02d}.jsonl"
            _tp.parent.mkdir(parents=True, exist_ok=True)
            traj_writer = _tp.open("w")
        if args.perception_log:
            _pp = Path(args.perception_log).with_suffix("") / f"route{ep_idx:02d}_perception.jsonl"
            _pp.parent.mkdir(parents=True, exist_ok=True)
            perception_writer = _pp.open("w")
        video_writer: Optional[_FfmpegVideoWriter] = None
        if args.video_out:
            _vp = Path(args.video_out)
            if _vp.suffix.lower() == ".mp4":
                video_path = _vp
            else:
                _vp.mkdir(parents=True, exist_ok=True)
                video_path = _vp / f"route{ep_idx:02d}_ego.mp4"
            video_writer = _FfmpegVideoWriter(
                video_path,
                fps=float(args.video_fps if args.video_fps is not None else args.step_hz),
            )

        arrived = False
        collided = False
        severe_coll = False
        fail_tag: Optional[str] = None
        interventions = 0
        intervened_steps: set = set()
        s_prog = 0.0
        p_prev_tracker = p_curr.copy()
        prev_yaw_tracker = curr_yaw
        detections_hit = 0
        far_lock_rejects = 0
        det_early = False
        det_early_steps = max(1, int(args.det_early_steps))
        steps_searching = 0
        steps_area_search = 0
        steps_intercepting = 0
        steps_following = 0
        steps_vision = 0
        steps_fallback = 0
        steps_spawn_acquire = 0
        steps_det_steer = 0
        arrived_follow = False
        vision_target_last: Optional[np.ndarray] = None
        acquire_steps = max(0, int(args.spawn_yaw_acquire_steps))
        acquire_half = max(
            1,
            int(math.ceil(float(args.spawn_yaw_acquire_deg) / max(float(args.spawn_yaw_acquire_step_deg), 1.0))),
        )
        acquire_yaw_step = math.radians(float(args.spawn_yaw_acquire_step_deg))

        from vgoal.tracker import TargetState as TS

        for step in range(args.max_steps):
            obs.info.pop("depth_min_pred", None)
            obs.info.pop("depth_cones_pred", None)
            obs.info.pop("tau_pred", None)
            d_fwd = None
            if depth_pred is not None and obs.rgb is not None:
                pred_both = getattr(depth_pred, "predict_min_and_cones", None)
                if callable(pred_both):
                    d_min, cones = pred_both(obs)
                    if d_min is not None:
                        obs.info["depth_min_pred"] = float(d_min)
                        d_fwd = float(d_min)
                    if isinstance(cones, dict):
                        obs.info["depth_cones_pred"] = {k: (float(v) if v is not None else None) for k, v in cones.items()}
                        cf = cones.get("forward")
                        if cf is not None and np.isfinite(float(cf)):
                            d_fwd = float(cf)
                else:
                    d_fwd = depth_pred.predict_min(obs)
                    if d_fwd is not None:
                        obs.info["depth_min_pred"] = float(d_fwd)
            tau_v = tau_pred.predict_tau(obs)
            if tau_v is not None:
                obs.info["tau_pred"] = float(tau_v)

            spawn_acquire_yaw = 0.0
            corridor_interval = max(0, int(args.corridor_yaw_sweep_interval))
            if (
                corridor_interval > 0
                and str(args.search_pattern) == "corridor"
                and not had_vision_lock
            ):
                corridor_steps = max(1, int(args.corridor_yaw_sweep_steps))
                corridor_half = max(
                    1,
                    int(
                        math.ceil(
                            float(args.corridor_yaw_sweep_deg)
                            / max(float(args.corridor_yaw_sweep_step_deg), 1.0)
                        )
                    ),
                )
                corridor_yaw_step = math.radians(float(args.corridor_yaw_sweep_step_deg))
                cycle_pos = step % corridor_interval
                if cycle_pos < corridor_steps:
                    phase = cycle_pos % (2 * corridor_half)
                    sign = 1.0 if phase < corridor_half else -1.0
                    spawn_acquire_yaw = sign * corridor_yaw_step
            elif acquire_steps > 0 and step < acquire_steps and not had_vision_lock:
                phase = step % (2 * acquire_half)
                sign = 1.0 if phase < acquire_half else -1.0
                spawn_acquire_yaw = sign * acquire_yaw_step

            area_planner_step = area_search_planner
            if (
                bool(args.search_delay_area_until_acquire)
                and acquire_steps > 0
                and step < acquire_steps
                and not had_vision_lock
            ):
                area_planner_step = None

            vstep = _vision_step(
                obs=obs,
                detector=detector,
                tracker=tracker,
                dynamic_tracker=dynamic_tracker,
                dynamic_min_meas_conf=float(args.dynamic_meas_conf),
                depth_pred=depth_pred,
                intrinsics=intrinsics,
                pos=p_curr,
                yaw=curr_yaw,
                prev_pos=p_prev_tracker,
                prev_yaw=prev_yaw_tracker,
                dt=dt_step,
                search_fwd_step=search_fwd_step,
                search_yaw_rate=float(args.search_yaw_rate),
                area_search_planner=area_planner_step,
                fallback_intent=fallback_intent,
                annot_goal=annot_goal,
                allow_fallback=bool(args.fallback_toward_g),
                prefer_nearest=bool(args.prefer_nearest_target),
                camera_fov_deg=float(args.camera_fov_deg),
                object_width_m=float(args.car_width_m),
                fuse_bbox_depth=bool(args.bbox_depth_fuse),
                near_bbox_px=float(args.bbox_near_px),
                near_prior_dist_m=float(args.bbox_near_dist_m),
                bbox_prior_near=bool(args.bbox_prior_near),
                reject_far_lock_m=float(args.reject_far_lock_m),
                spawn_acquire_yaw_rate=spawn_acquire_yaw,
                search_det_steer=bool(args.search_det_steer),
                search_det_steer_gain=float(args.search_det_steer_gain),
                search_det_steer_fwd=float(args.search_det_steer_fwd),
                search_det_steer_max_yaw=det_steer_max_yaw,
                search_area_priority=bool(args.search_area_priority),
            )
            p_prev_tracker = p_curr.copy()
            prev_yaw_tracker = curr_yaw

            if vstep.perception and vstep.perception.get("measured_dist_m") is not None:
                min_d_measured = min(min_d_measured, float(vstep.perception["measured_dist_m"]))
            if vstep.perception and vstep.perception.get("far_lock_rejected"):
                far_lock_rejects += 1

            if video_writer is not None:
                rgb_vid = getattr(obs, "rgb_yolo", None)
                if rgb_vid is None:
                    rgb_vid = obs.rgb
                if rgb_vid is not None:
                    d_vis_hud = None
                    if vstep.target_world is not None:
                        d_vis_hud = float(_goal_dist(p_curr, vstep.target_world))
                    frame = _draw_vgoal_demo_frame(
                        np.asarray(rgb_vid, dtype=np.uint8),
                        step=step,
                        max_steps=int(args.max_steps),
                        target_class=str(args.target_class or "target"),
                        vstep=vstep,
                        d_vis=d_vis_hud,
                    )
                    video_writer.write(frame)

            if vstep.det_hit:
                detections_hit += 1
                if step < det_early_steps:
                    det_early = True
            if vstep.using_fallback:
                steps_fallback += 1
            elif vstep.using_area_search:
                steps_area_search += 1
                steps_searching += 1
            elif vstep.using_det_steer:
                steps_det_steer += 1
                steps_searching += 1
            elif vstep.search_action is not None:
                steps_searching += 1
                if spawn_acquire_yaw != 0.0:
                    steps_spawn_acquire += 1
            elif vstep.using_vision:
                steps_vision += 1
            if vstep.dynamic_mode == "intercepting":
                steps_intercepting += 1
            elif vstep.dynamic_mode == "following":
                steps_following += 1

            if vstep.target_world is not None:
                vision_target_last = vstep.target_world.copy()
                if vstep.using_vision:
                    had_vision_lock = True
                d_vis = _goal_dist(p_curr, vstep.target_world)
                min_d_vision = min(min_d_vision, d_vis)
                d_final_vision = d_vis
                if str(args.follow_mode) == "standoff":
                    if (
                        vstep.dynamic_mode == "following"
                        and vstep.goal_rel is not None
                        and float(vstep.goal_rel[3]) <= float(args.follow_success_dist_m)
                    ):
                        arrived = True
                        arrived_follow = True
                elif (
                    not vstep.using_area_search
                    and (
                        d_vis <= float(args.success_dist)
                        or vstep.tracker_state == TS.ARRIVED.value
                    )
                ):
                    arrived = True

            d_annot = _goal_dist(p_curr, annot_goal)
            min_d_annot = min(min_d_annot, d_annot)
            s_prog = float(max(0.0, d0_annot - d_annot))

            if arrived:
                break

            phys = body_delta_limits(dt_step)
            if vstep.search_action is not None:
                search_action = np.asarray(vstep.search_action, dtype=np.float64).copy()
                if ep_z_hold is not None:
                    z_err = float(ep_z_hold) - float(p_curr[2])
                    search_action[2] = float(
                        np.clip(z_err * float(args.search_z_gain), -float(phys[2]), float(phys[2]))
                    )
                vx_step_limit = float(min(float(args.cruise_speed) / float(args.step_hz), float(phys[0])))
                search_limits = np.array(
                    [vx_step_limit, float(phys[1]), float(phys[2]), float(phys[3])],
                    dtype=np.float64,
                )
                action = clip_body_delta(search_action, search_limits)
                g_rel_body = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
                target_world = p_curr + np.array([1.0, 0.0, 0.0])
                wm_out = None
            else:
                assert vstep.goal_rel is not None and vstep.target_world is not None
                g_vis = np.asarray(vstep.target_world, dtype=np.float64)
                if vstep.using_vision and visual_intent is not None:
                    g_rel_body, s_info = visual_intent.compute(
                        curr_pos=p_curr,
                        curr_yaw=curr_yaw,
                        goal=g_vis,
                        d_fwd_hat=obs.info.get("depth_min_pred"),
                    )
                    target_world = np.array(s_info["target_world"], dtype=np.float64)
                    safe_v = float(s_info.get("safe_speed_limit", args.cruise_speed))
                elif vstep.using_area_search:
                    g_rel_body = np.asarray(vstep.goal_rel, dtype=np.float64)
                    target_world = g_vis
                    safe_v = float(args.cruise_speed)
                else:
                    g_rel_body = np.asarray(vstep.goal_rel, dtype=np.float64)
                    target_world = g_vis
                    safe_v = float(args.cruise_speed)
                vx_step_limit = float(min(safe_v / float(args.step_hz), float(phys[0])))
                cur_limits = np.array([vx_step_limit, float(phys[1]), float(phys[2]), float(phys[3])], dtype=np.float64)
                if planner is not None:
                    planner.action_limits = cur_limits
                obs.info["goal"] = target_world.tolist()
                obs.info["goal_rel"] = g_rel_body.tolist()
                if planner is not None:
                    planner.set_goal(target_world)
                if (
                    bool(args.search_direct_area)
                    and vstep.using_area_search
                    and vstep.goal_rel is not None
                ):
                    gr = np.asarray(vstep.goal_rel, dtype=np.float64)
                    vx_step_limit = float(min(float(args.cruise_speed) / float(args.step_hz), float(phys[0])))
                    cur_limits = np.array(
                        [vx_step_limit, float(phys[1]), float(phys[2]), float(phys[3])],
                        dtype=np.float64,
                    )
                    fwd = float(np.clip(gr[0], 0.0, cur_limits[0]))
                    if fwd < 0.05 * cur_limits[0] and float(gr[3]) > 1.0:
                        fwd = 0.25 * cur_limits[0]
                    left = float(np.clip(gr[1], -cur_limits[1], cur_limits[1]))
                    z_cmd = float(np.clip(gr[2], -cur_limits[2], cur_limits[2]))
                    if ep_z_hold is not None:
                        z_err = float(ep_z_hold) - float(p_curr[2])
                        z_cmd = float(
                            np.clip(z_err * float(args.search_z_gain), -cur_limits[2], cur_limits[2])
                        )
                    yaw_cmd = float(
                        np.clip(
                            math.atan2(float(gr[1]), max(float(gr[0]), 0.5)) * 0.5,
                            -cur_limits[3],
                            cur_limits[3],
                        )
                    )
                    action = clip_body_delta(np.array([fwd, left, z_cmd, yaw_cmd], dtype=np.float64), cur_limits)
                    wm_out = None
                else:
                    action = policy.act(obs)
                    if planner is not None:
                        action = planner.plan(obs, action, latent=policy._latent)
                if vstep.using_area_search and float(args.search_yaw_hold_deg) > 0.0:
                    hold_rad = math.radians(float(args.search_yaw_hold_deg))
                    yaw_err = float(curr_yaw - start_yaw)
                    yaw_err = float((yaw_err + math.pi) % (2.0 * math.pi) - math.pi)
                    if abs(yaw_err) > hold_rad:
                        excess = abs(yaw_err) - hold_rad
                        corr = -math.copysign(excess * float(args.search_yaw_hold_gain), yaw_err)
                        action = np.asarray(action, dtype=np.float64).copy()
                        action[3] = float(np.clip(action[3] + corr, -float(phys[3]), float(phys[3])))
                action = clip_body_delta(action, cur_limits)
                wm_out = None
                if policy._latent is not None and hasattr(dynamics, "step"):
                    try:
                        wm_out = dynamics.step(
                            policy._latent, action, goal_rel=g_rel_body,
                            body_vel=body_vel_from_obs(obs),
                        )
                    except Exception:
                        wm_out = None

            if shield is not None:
                act_safe, overridden = shield.apply_action(action, obs, wm_out=wm_out, limits=action_limits)
                if overridden:
                    interventions += 1
                    intervened_steps.add(step)
                action = act_safe

            step_out = env.step(action)
            if len(step_out) == 4:
                obs, _rew, done, step_info = step_out
            else:
                obs, step_info = step_out
                done = bool(getattr(obs, "collided", False))

            p_prev = p_curr.copy()
            p_curr = np.array(obs.position, dtype=np.float64)
            curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else curr_yaw

            if float(np.linalg.norm(p_curr - p_prev)) > 20.0:
                logger.error("Route %02d F-tele step=%d — invalidated", ep_idx + 1, step)
                fail_tag = "F-tele"
                break

            traj.append(p_curr.copy())

            if traj_writer is not None:
                bridge_gr = None
                if vstep.perception and vstep.perception.get("bridge_goal_rel"):
                    bridge_gr = vstep.perception["bridge_goal_rel"]
                traj_writer.write(json.dumps({
                    "step": step,
                    "pos": p_curr.tolist(),
                    "yaw": round(float(curr_yaw), 4),
                    "tracker_state": vstep.tracker_state,
                    "det_hit": vstep.det_hit,
                    "using_vision": vstep.using_vision,
                    "using_area_search": vstep.using_area_search,
                    "dynamic_mode": vstep.dynamic_mode,
                    "using_fallback": vstep.using_fallback,
                    "goal_rel": None if vstep.goal_rel is None else [round(float(x), 3) for x in vstep.goal_rel],
                    "bridge_goal_rel": bridge_gr,
                }) + "\n")

            if perception_writer is not None and vstep.perception is not None:
                perc_row = dict(vstep.perception)
                perc_row.update({
                    "step": step,
                    "tracker_state": vstep.tracker_state,
                    "det_hit": vstep.det_hit,
                    "goal_rel_dist": (
                        round(float(vstep.goal_rel[3]), 3) if vstep.goal_rel is not None else None
                    ),
                })
                perception_writer.write(json.dumps(perc_row) + "\n")

            if vision_target_last is not None:
                seg_d = _segment_min_dist(p_prev, p_curr, vision_target_last)
                if seg_d <= float(args.success_dist):
                    arrived = True
                    min_d_vision = min(min_d_vision, seg_d)
                    d_final_vision = float(seg_d)
                    break

            if done:
                collided = bool(getattr(obs, "collided", False) or step_info.get("collided", False))
                if step_info.get("severe_collision", False) or collided:
                    severe_coll = True
                break

        if traj_writer is not None:
            traj_writer.close()
        if perception_writer is not None:
            perception_writer.close()
        if video_writer is not None:
            video_writer.close()
            ep_result_video = str(video_writer.path)
        else:
            ep_result_video = None

        actual_len = float(np.sum(np.linalg.norm(np.diff(np.array(traj), axis=0), axis=1))) if len(traj) > 1 else 0.0
        prog_ratio = float(np.clip(s_prog / max(1e-3, ref_len), 0.0, 1.0))
        ep_spl = (ref_len / max(ref_len, actual_len, 1e-6)) if arrived else 0.0
        goal_closure = _goal_closure(d0_annot, min_d_annot)
        n_steps = max(1, len(traj))

        ep_result = {
            "route_idx": ep_idx,
            "base_route_idx": r_info.get("base_route_idx"),
            "nominal_length_m": round(ref_len, 2),
            "actual_length_m": round(actual_len, 2),
            "steps": len(traj),
            "d_start_m": round(d0_annot, 2),
            "d_min_m": round(min_d_annot, 2),
            "d_final_m": round(float(min_d_vision if had_vision_lock else d_annot), 2),
            "d_min_vision_m": round(float(min_d_vision), 2) if had_vision_lock else None,
            "d_min_measured_m": round(float(min_d_measured), 2) if np.isfinite(min_d_measured) else None,
            "d_final_vision_m": round(float(d_final_vision), 2) if had_vision_lock else None,
            "gt_goal_world": [round(float(x), 2) for x in annot_goal] if args.gt_nearest_scene_object else None,
            "goal_closure": round(goal_closure, 4),
            "arrived": arrived,
            "arrived_vision": bool(arrived and had_vision_lock and steps_fallback == 0 and not arrived_follow),
            "arrived_follow": bool(arrived_follow),
            "collided": collided,
            "severe_collision": severe_coll,
            "progress_ratio": round(prog_ratio, 4),
            "spl": round(ep_spl, 4),
            "intervention_rate": round(interventions / n_steps, 4),
            "subgoal_source": "visual",
            "goal_from": "vision" if had_vision_lock and steps_fallback == 0 else ("mixed" if steps_fallback else "search_only"),
            "detections_hit": detections_hit,
            "steps_searching": steps_searching,
            "steps_area_search": steps_area_search,
            "steps_intercepting": steps_intercepting,
            "steps_following": steps_following,
            "steps_vision": steps_vision,
            "steps_fallback": steps_fallback,
            "detection_frac": round(detections_hit / n_steps, 4),
            "det_recall_early": bool(det_early),
            "far_lock_rejects": far_lock_rejects,
            "video": ep_result_video,
            "steps_spawn_acquire": steps_spawn_acquire,
            "steps_det_steer": steps_det_steer,
            "vision_frac": round(steps_vision / n_steps, 4),
            "area_search_frac": round(steps_area_search / n_steps, 4),
            "fail_tag": fail_tag,
        }
        results.append(ep_result)

        scored_so_far = [r for r in results if not r.get("spawn_fail")]
        sr_now = float(np.mean([r["arrived"] for r in scored_so_far])) if scored_so_far else 0.0
        logger.info(
            "Route %02d/%02d | arrived=%s vision=%s det=%.0f%% vis=%.0f%% fb=%d IR=%.0f%% | SR=%.1f%%",
            slot + 1, n_routes, arrived, ep_result["goal_from"],
            ep_result["detection_frac"] * 100, ep_result["vision_frac"] * 100,
            steps_fallback, ep_result["intervention_rate"] * 100, sr_now * 100,
        )

    scored, spawn_fails, metrics, verdict = aggregate_metrics(results)
    metrics["mean_detection_frac"] = round(
        float(np.mean([r["detection_frac"] for r in scored])), 4
    ) if scored else 0.0
    metrics["mean_vision_frac"] = round(
        float(np.mean([r["vision_frac"] for r in scored])), 4
    ) if scored else 0.0

    summary = {
        "verdict": verdict,
        "n_scored": len(scored),
        "n_spawn_fail": len(spawn_fails),
        "metrics": metrics,
        "protocol_version": "phase2_vgoal_m4",
        "method": "monocular_visual",
        "goal_from": "vision",
        "config": {
            "actor_ckpt": str(args.actor_ckpt),
            "wm_ckpt": str(args.wm_ckpt),
            "cruise_speed": args.cruise_speed,
            "tti_coeff": args.tti_coeff,
            "detector": args.detector,
            "target_class": args.target_class,
            "visual_prompt": visual_prompt,
            "yolo_model": args.yolo_model,
            "fallback_toward_g": bool(args.fallback_toward_g),
            "search_fwd_speed": search_fwd_step,
            "search_fwd_mode": search_fwd_label,
            "search_yaw_rate": args.search_yaw_rate,
            "search_det_steer": bool(args.search_det_steer),
            "search_det_steer_gain": float(args.search_det_steer_gain),
            "search_det_steer_fwd": float(args.search_det_steer_fwd),
            "search_det_steer_max_yaw": det_steer_max_yaw,
            "search_yaw_hold_deg": float(args.search_yaw_hold_deg),
            "search_yaw_hold_gain": float(args.search_yaw_hold_gain),
            "search_delay_area_until_acquire": bool(args.search_delay_area_until_acquire),
            "reject_far_lock_m": float(args.reject_far_lock_m),
            "spawn_yaw_acquire_steps": int(args.spawn_yaw_acquire_steps),
            "search_pattern": str(args.search_pattern),
            "search_area_half_m": float(args.search_area_half_m),
            "search_sweep_spacing_m": float(args.search_sweep_spacing_m),
            "search_waypoint_radius_m": float(args.search_waypoint_radius_m),
            "search_spiral_radius_m": float(args.search_spiral_radius_m),
            "follow_mode": str(args.follow_mode),
            "standoff_dist_m": float(args.standoff_dist_m),
            "standoff_height_m": float(args.standoff_height_m),
            "intercept_dist_m": float(args.intercept_dist_m),
            "follow_success_dist_m": float(args.follow_success_dist_m),
            "dynamic_meas_conf": float(args.dynamic_meas_conf),
            "search_z_hold_mode": args.search_z_hold_mode,
            "search_z_min": float(args.search_z_min),
            "search_z_max": float(args.search_z_max),
            "search_z_gain": float(args.search_z_gain),
            "visual_toward_g": bool(args.visual_toward_g),
            "toward_g_r_m": float(args.toward_g_r_m),
            "tracker_min_confidence": tracker_min_conf,
            "tracker_near_dist_m": float(args.tracker_near_dist_m),
            "tracker_near_ema_alpha": float(args.tracker_near_ema_alpha),
            "tracker_inflate_reject_m": float(args.tracker_inflate_reject_m),
            "tracker_inflate_alpha": float(args.tracker_inflate_alpha),
            "car_width_m": float(args.car_width_m),
            "bbox_depth_fuse": bool(args.bbox_depth_fuse),
            "bbox_prior_near": bool(args.bbox_prior_near),
            "bbox_near_px": float(args.bbox_near_px),
            "bbox_near_dist_m": float(args.bbox_near_dist_m),
            "tracker_freeze_dist_on_occlude": bool(args.tracker_freeze_dist_on_occlude),
            "perception_log": bool(args.perception_log),
            "gt_nearest_scene_object": bool(args.gt_nearest_scene_object),
            "gt_scene_pattern": str(args.gt_scene_pattern),
            "gt_scene_fov_deg": float(args.gt_scene_fov_deg or 160.0),
            "gt_scene_min_fwd_m": float(args.gt_scene_min_fwd_m),
            "yolo_conf": float(args.yolo_conf),
            "fanout_rgb": use_fanout,
            "capture_w": int(args.capture_w),
            "capture_h": int(args.capture_h),
            "wam_encode_size": int(args.wam_encode_size),
        },
        "episodes": results,
    }

    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Written: %s", out_path)
    logger.info(
        "FINAL | Verdict=%s SR=%.1f%% SCR=%.1f%% det=%.0f%% vision=%.0f%%",
        verdict,
        metrics["arrival_rate"] * 100,
        metrics["severe_collision_rate"] * 100,
        metrics.get("mean_detection_frac", 0) * 100,
        metrics.get("mean_vision_frac", 0) * 100,
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
