#!/usr/bin/env python3
"""Shield-off urban-interior collection for the occ_topo occupancy prior.

The network vector is 10-D and is not the simulator world pose:

  (dx, dy, dz, dyaw, ω前, ω左, ω上, a前, a左, a上)

The first four numbers are the body delta commanded on the step that produced
this frame (zeros on the spawn frame). They are not a difference of simulator
pose. The last six are ``getImuData`` angular velocity and linear acceleration,
converted from AirSim body NED (forward, right, down) into the occupancy
frame (forward, left, up). Gravity stays in the acceleration.

Simulator pose and depth are stored beside that vector for geography review
and occupancy labels. They are separate arrays and are not copied into
``proprio``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("collect_occ_topo")

DX = 0.6
DYAW = 0.314
YAW_EVERY = 8
MAX_STEPS = 240
MIN_KEEP_STEPS = 40
KEEP_EPISODES = 200
FACADE_STEPS = 2000
FACADE_M = 15.0
DEPTH_RANGE_M = 80.0
HOLDOUT_M = 50.0
MIN_SPAWN_INLAND_M = 100.0
MIN_PATH_INLAND_M = 80.0
MIN_FREE_GB = 8.0
STEP_HZ = 5.0


def imu6_from_airsim(ang_vel: Sequence[float], lin_acc: Sequence[float]) -> np.ndarray:
    """``getImuData`` body NED → (ω前, ω左, ω上, a前, a左, a上)."""
    wx, wy, wz = (float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]))
    ax, ay, az = (float(lin_acc[0]), float(lin_acc[1]), float(lin_acc[2]))
    return np.array([wx, -wy, -wz, ax, -ay, -az], dtype=np.float32)


def imu6_from_obs(imu: Optional[Dict[str, Any]]) -> np.ndarray:
    if not imu or "ang_vel" not in imu or "lin_acc" not in imu:
        raise RuntimeError("getImuData returned no ang_vel/lin_acc")
    ang = imu["ang_vel"]
    lin = imu["lin_acc"]
    if len(ang) < 3 or len(lin) < 3:
        raise RuntimeError("getImuData vectors shorter than 3")
    return imu6_from_airsim(ang, lin)


def pack_proprio(body_delta: Sequence[float], imu6: Sequence[float]) -> np.ndarray:
    odom = np.asarray(body_delta, dtype=np.float32).reshape(4)
    imu = np.asarray(imu6, dtype=np.float32).reshape(6)
    return np.concatenate([odom, imu])


def scripted_action(step_index: int, yaw_sign: float = 1.0) -> np.ndarray:
    """Step ``step_index`` of the fixed collector. Yaw alternates every 8 steps.

    ``yaw_sign`` flips the whole pattern. Even laps pass ``-1`` so a repeated
    route does not copy the previous trajectory.
    """
    action = np.array([DX, 0.0, 0.0, 0.0], dtype=np.float64)
    if int(step_index) % YAW_EVERY == 0:
        sign = float(yaw_sign) if ((int(step_index) // YAW_EVERY) % 2 == 0) else -float(yaw_sign)
        action[3] = sign * DYAW
    return action


def forward_hit_m(depth: np.ndarray) -> float:
    """Median finite depth in the center third, or inf when nothing is in range."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        return float("inf")
    h, w = d.shape
    patch = d[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    valid = patch[np.isfinite(patch) & (patch > 0.5) & (patch <= DEPTH_RANGE_M)]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid))


def facade_ahead(depth: np.ndarray, limit_m: float = FACADE_M) -> bool:
    return forward_hit_m(depth) <= float(limit_m)


def holdout_index(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    paths = set()
    starts = []
    for ep in episodes:
        image = str(ep.get("image_path") or "")
        if image:
            paths.add(image)
        pos = ep.get("pos") or []
        if pos:
            starts.append((float(pos[0][0]), float(pos[0][1])))
    return {"image_paths": paths, "starts": starts}


def overlaps_holdout(route: Dict[str, Any], holdout: Dict[str, Any], radius_m: float = HOLDOUT_M) -> bool:
    image = str(route.get("image_path") or "")
    if image and image in holdout["image_paths"]:
        return True
    pos = route.get("pos") or []
    if not pos or not holdout["starts"]:
        return False
    x, y = float(pos[0][0]), float(pos[0][1])
    radius = float(radius_m)
    for hx, hy in holdout["starts"]:
        if (x - hx) ** 2 + (y - hy) ** 2 < radius * radius:
            return True
    return False


def geography_reasons(positions: np.ndarray, regions: Dict[str, Any]) -> List[str]:
    from experiments.aerial.phase3_unified.region_geometry import (
        classify_spawn_xy,
        path_inland_metrics,
    )

    reasons: List[str] = []
    if len(positions) == 0:
        return ["empty"]
    spawn = classify_spawn_xy(float(positions[0, 0]), float(positions[0, 1]), regions)
    if not (spawn["in_urban"] and spawn["in_interior"]) or spawn["in_water"]:
        reasons.append("spawn_geography")
    if float(spawn["spawn_inland_m"]) < MIN_SPAWN_INLAND_M:
        reasons.append("spawn_inland<100")
    inland = path_inland_metrics(positions.tolist(), regions)
    if float(inland["path_min_inland_m"]) < MIN_PATH_INLAND_M:
        reasons.append("path_inland<80")
    for x, y, _z in positions:
        here = classify_spawn_xy(float(x), float(y), regions)
        if (not here["in_interior"]) or here["in_water"] or (not here["in_urban"]):
            reasons.append("left_interior")
            break
    return reasons


def review_episode(
    positions: np.ndarray,
    collided: np.ndarray,
    regions: Dict[str, Any],
    *,
    holdout_overlap: bool,
) -> Tuple[bool, List[str]]:
    """``collided[k]`` is the flag on frame k. Frame 0 is the spawn observation."""
    reasons: List[str] = []
    steps = max(0, int(len(positions)) - 1)
    if steps < MIN_KEEP_STEPS:
        reasons.append("too_short")
    if len(collided) >= 2 and bool(np.any(collided[1:3])):
        reasons.append("spawn_collision")
    if holdout_overlap:
        reasons.append("holdout_overlap")
    reasons.extend(geography_reasons(positions, regions))
    return (len(reasons) == 0), reasons


def load_episodes(path: Path, key: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key:
        if key not in payload:
            raise KeyError("annotation %s has no %s" % (path, key))
        return list(payload[key])
    if "episodes" in payload:
        return list(payload["episodes"])
    if "train" in payload:
        return list(payload["train"])
    raise KeyError("annotation has neither episodes nor train: %s" % path)


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / (1024.0 ** 3)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg",
        np.asarray(rgb)[:, :, ::-1],
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class EpisodeBuffer:
    def __init__(self) -> None:
        self.jpeg: List[bytes] = []
        self.depth: List[np.ndarray] = []
        self.proprio: List[np.ndarray] = []
        self.action: List[np.ndarray] = []
        self.label_state: List[np.ndarray] = []
        self.collided: List[bool] = []
        self.facade: List[bool] = []

    def add(
        self,
        obs: Any,
        body_delta: Sequence[float],
        next_action: Sequence[float],
    ) -> None:
        imu6 = imu6_from_obs(getattr(obs, "imu", None))
        depth = np.asarray(obs.depth, dtype=np.float32)
        self.jpeg.append(encode_jpeg(obs.rgb))
        self.depth.append(depth.astype(np.float16))
        self.proprio.append(pack_proprio(body_delta, imu6))
        self.action.append(np.asarray(next_action, dtype=np.float32).reshape(4))
        self.label_state.append(np.asarray(obs.state, dtype=np.float32).reshape(7))
        self.collided.append(bool(obs.collided))
        self.facade.append(bool(facade_ahead(depth)))

    def __len__(self) -> int:
        return len(self.proprio)

    def save(self, path: Path, meta: Dict[str, Any]) -> None:
        offsets = np.cumsum([0] + [len(b) for b in self.jpeg]).astype(np.int64)
        np.savez(
            path,
            jpeg=np.frombuffer(b"".join(self.jpeg), dtype=np.uint8),
            jpeg_offsets=offsets,
            depth=np.stack(self.depth),
            proprio=np.stack(self.proprio),
            action=np.stack(self.action),
            label_state=np.stack(self.label_state),
            collided=np.asarray(self.collided, dtype=bool),
            facade=np.asarray(self.facade, dtype=bool),
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )


def run_episode(
    env: Any,
    route: Dict[str, Any],
    args: argparse.Namespace,
    regions: Dict[str, Any],
    yaw_sign: float = 1.0,
) -> Tuple[Optional[EpisodeBuffer], Dict[str, Any]]:
    from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

    pos = [list(map(float, p)) for p in route["pos"]]
    yaw0 = float(route["yaw"][0])
    ep = {"pos": pos, "yaw": [yaw0] * len(route["yaw"]), "gpt_instruction": route.get("gpt_instruction", "")}
    used, obs, _err, failed = reset_with_spawn_retries(
        env,
        ep,
        min_spawn_z=float(args.min_spawn_z),
        spawn_z_retry_m=5.0,
        spawn_z_max_retries=4,
        spawn_tol_m=12.0,
        mock=False,
        min_spawn_clear_m=float(args.min_spawn_clear_m),
    )
    used_pos = np.asarray(used["pos"], dtype=np.float64)
    meta: Dict[str, Any] = {
        "route_id": route.get("route_id"),
        "route_idx": route.get("route_idx"),
        "image_path": route.get("image_path") or "",
        "odom_source": "commanded_body_delta",
        "imu_source": "getImuData",
        "yaw_sign": float(yaw_sign),
    }
    if failed:
        meta["end_reason"] = "spawn_fail"
        return None, meta
    if float(np.linalg.norm(used_pos[0, :2] - np.asarray(pos[0][:2]))) > 1.0:
        meta["end_reason"] = "start_xy_nudged"
        return None, meta

    buf = EpisodeBuffer()
    buf.add(obs, np.zeros(4), scripted_action(0, yaw_sign))
    end = "max_steps"
    for step_index in range(int(args.max_steps)):
        action = np.asarray(buf.action[-1], dtype=np.float64)
        obs, info = env.step(action)
        cmd = np.asarray(info.get("cmd", action), dtype=np.float64).reshape(4)
        nxt = scripted_action(step_index + 1, yaw_sign) if step_index + 1 < int(args.max_steps) else np.zeros(4)
        here = np.asarray(obs.state, dtype=np.float64).reshape(7)[:3]
        positions = np.vstack([np.stack(buf.label_state)[:, :3], here.reshape(1, 3)])
        outside = geography_reasons(positions, regions)
        if outside:
            end = "geography"
            meta["geography_stop"] = outside
            break
        buf.add(obs, cmd, nxt)
        if bool(obs.collided):
            end = "collision"
            break
    meta["steps"] = len(buf) - 1
    meta["end_reason"] = end
    meta["facade_steps"] = int(sum(buf.facade))
    return buf, meta


def write_review(path: Path, rows: List[Dict[str, Any]], kept_episodes: int, facade_steps: int) -> None:
    payload = {
        "kept_episodes": int(kept_episodes),
        "facade_steps": int(facade_steps),
        "target_episodes": KEEP_EPISODES,
        "target_facade_steps": FACADE_STEPS,
        "pass": kept_episodes >= KEEP_EPISODES and facade_steps >= FACADE_STEPS,
        "episodes": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_candidates.json",
    )
    parser.add_argument(
        "--holdout",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/aerial/rl/artifacts/occ_topo_collect",
    )
    parser.add_argument("--min-spawn-z", type=float, default=24.0)
    parser.add_argument("--min-spawn-clear-m", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-episodes", type=int, default=100000)
    parser.add_argument(
        "--route-key",
        default="holdout",
        help="List inside the annotation. Default holdout is interior_east, away from the G1 routes.",
    )
    args = parser.parse_args()

    from experiments.aerial.phase3_unified.region_geometry import load_regions
    from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    routes = load_episodes(Path(args.annotation), args.route_key)
    holdout_eps = load_episodes(Path(args.holdout))
    holdout = holdout_index(holdout_eps)
    regions = load_regions()
    logger.info("routes=%s key=%s holdout_starts=%s", len(routes), args.route_key, len(holdout["starts"]))
    rows: List[Dict[str, Any]] = []
    kept = 0
    facade_total = 0
    cfg = AirSimEnvConfig(step_hz=STEP_HZ, grab_depth=True, health_check=True)
    with AirSimDroneEnv(cfg) as env:
        flown = 0
        lap = 0
        stop = False
        while kept < KEEP_EPISODES or facade_total < FACADE_STEPS:
            if stop or flown >= int(args.max_episodes):
                break
            lap += 1
            yaw_sign = 1.0 if lap % 2 == 1 else -1.0
            logger.info("lap=%s yaw_sign=%s kept=%s facade=%s", lap, yaw_sign, kept, facade_total)
            flew_this_lap = 0
            for route in routes:
                if kept >= KEEP_EPISODES and facade_total >= FACADE_STEPS:
                    stop = True
                    break
                if flown >= int(args.max_episodes):
                    stop = True
                    break
                if free_gb(out) < MIN_FREE_GB:
                    logger.error("free space %.1f GB < %.1f GB, stop", free_gb(out), MIN_FREE_GB)
                    stop = True
                    break
                if overlaps_holdout(route, holdout):
                    if lap == 1:
                        rows.append({"route_id": route.get("route_id"), "keep": False, "reasons": ["holdout_overlap"]})
                        logger.info("skip %s holdout_overlap", route.get("route_id"))
                    continue
                pos0 = np.asarray(route["pos"][0], dtype=np.float64)
                pre = geography_reasons(pos0.reshape(1, 3), regions)
                if pre:
                    if lap == 1:
                        rows.append({"route_id": route.get("route_id"), "keep": False, "reasons": pre})
                        logger.info("skip %s %s", route.get("route_id"), ",".join(pre))
                    continue
                t0 = time.time()
                buf, meta = run_episode(env, route, args, regions, yaw_sign)
                flown += 1
                flew_this_lap += 1
                meta["seconds"] = round(time.time() - t0, 1)
                meta["lap"] = lap
                if buf is None:
                    meta["keep"] = False
                    meta["reasons"] = [str(meta.get("end_reason"))]
                    rows.append(meta)
                    write_review(out / "AUTO_REVIEW.json", rows, kept, facade_total)
                    continue
                positions = np.stack(buf.label_state)[:, :3]
                ok, reasons = review_episode(
                    positions,
                    np.asarray(buf.collided),
                    regions,
                    holdout_overlap=False,
                )
                meta["keep"] = bool(ok)
                meta["reasons"] = reasons
                if ok:
                    kept += 1
                    facade_total += int(meta["facade_steps"])
                    buf.save(out / f"ep_{kept:04d}.npz", meta)
                rows.append(meta)
                write_review(out / "AUTO_REVIEW.json", rows, kept, facade_total)
                logger.info(
                    "kept=%s facade=%s last=%s route=%s",
                    kept,
                    facade_total,
                    meta.get("end_reason"),
                    meta.get("route_id"),
                )
            if flew_this_lap == 0:
                logger.error("lap %s flew no episodes, stop", lap)
                break
    write_review(out / "AUTO_REVIEW.json", rows, kept, facade_total)
    passed = kept >= KEEP_EPISODES and facade_total >= FACADE_STEPS
    logger.info("done kept=%s facade=%s pass=%s", kept, facade_total, passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
