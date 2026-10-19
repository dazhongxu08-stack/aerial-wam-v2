#!/usr/bin/env python3
"""Shield-off rollouts of random action sequences for consequence labels.

No teacher and no imitation target: the behaviour is a mix of randomised
macro sequences (straight, yaw then forward, turn toward goal then forward,
yaw only, strafe, climb, descend, smooth noise), and every label is what the
simulator returned. Collision ends the episode (one contact per episode).

Train routes only (``outdoor_complex_od_train.json``). Each episode starts
with the nose rotated a random amount off the goal bearing so facade-ahead
states are common.

Per step it stores the RGB frame (JPEG), GT depth block-min pooled to 56x56
(float16), GT five-cone clearances and the center-band column profile at full
width, the kinematic state, the executed action, the collision flag, and the
macro id. ``AUTO_REVIEW.json`` keeps a geography + outcome review per episode;
the gate keeps collecting until it passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("collect_consequence")

LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793], dtype=np.float64)
MACROS = (
    "fwd",
    "yaw_fwd",
    "goal_turn_fwd",
    "yaw_only",
    "strafe",
    "climb",
    "descend",
    "smooth_random",
)
MACRO_P = np.array([0.18, 0.24, 0.16, 0.10, 0.10, 0.08, 0.04, 0.10])
# Only sampled by name (branching / MPC extra families), never by the random collector above.
EXTRA_MACROS = ("hover", "spin", "back")
BACK_LIMIT = 0.6
CONES = ("forward", "left", "right", "up", "down")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def goal_bearing_body(pos: np.ndarray, yaw: float, goal: np.ndarray) -> float:
    d = np.asarray(goal[:2], dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64)
    return wrap(math.atan2(d[1], d[0]) - yaw)


def sample_macro(
    rng: np.random.Generator, k: int, bearing: float, name: Optional[str] = None
) -> Tuple[str, np.ndarray]:
    if name is None:
        name = str(rng.choice(MACROS, p=MACRO_P))
    a = np.zeros((k, 4), dtype=np.float64)
    if name == "fwd":
        a[:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_fwd":
        m = int(rng.integers(1, 5))
        a[:m, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 1.0) * LIMITS[3]
        a[:m, 0] = rng.uniform(0.0, 0.25)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "goal_turn_fwd":
        left = bearing
        for t in range(k):
            step = float(np.clip(left, -LIMITS[3], LIMITS[3]))
            if abs(left) > 0.05:
                a[t, 3] = step
                a[t, 0] = rng.uniform(0.0, 0.25)
                left -= step
            else:
                a[t, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_only":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 1.0) * LIMITS[3]
        a[:, 0] = rng.uniform(0.0, 0.2)
    elif name == "strafe":
        a[:, 1] = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0) * LIMITS[1]
        a[:, 0] = rng.uniform(0.0, 0.4)
    elif name == "climb":
        a[:, 2] = rng.uniform(0.25, 1.0) * LIMITS[2]
        a[:, 0] = rng.uniform(0.0, 0.5)
    elif name == "descend":
        a[:, 2] = -rng.uniform(0.25, 0.75) * LIMITS[2]
        a[:, 0] = rng.uniform(0.0, 0.5)
    elif name == "hover":
        pass
    elif name == "spin":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * LIMITS[3]
    elif name == "back":
        m = int(rng.integers(3, k + 1))
        a[:m, 0] = -rng.uniform(0.2, 0.6) * BACK_LIMIT
    else:
        u = np.zeros(4)
        for t in range(k):
            u = 0.7 * u + 0.3 * rng.normal(0.0, 1.0, size=4)
            a[t] = np.tanh(u) * LIMITS
    a[:, 0] = np.clip(a[:, 0], -BACK_LIMIT if name == "back" else 0.0, LIMITS[0])
    return name, np.clip(a, -LIMITS, LIMITS)


def pool_min(depth: np.ndarray, out: int = 56) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    fh, fw = h // out, w // out
    d = d[: fh * out, : fw * out].reshape(out, fh, out, fw)
    return d.min(axis=(1, 3)).astype(np.float16)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.asarray(rgb)[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class EpisodeBuffer:
    def __init__(self) -> None:
        self.jpeg: List[bytes] = []
        self.depth56: List[np.ndarray] = []
        self.cones: List[List[float]] = []
        self.profile: List[np.ndarray] = []
        self.state: List[np.ndarray] = []
        self.action: List[np.ndarray] = []
        self.collided: List[bool] = []
        self.macro: List[int] = []
        self.segment: List[int] = []

    def add(self, obs: Any, action: np.ndarray, macro: int, segment: int) -> None:
        from experiments.aerial.rl.cone_transition import column_profile
        from experiments.aerial.rl.depth_geometry import cone_clearances

        depth = np.asarray(obs.depth, dtype=np.float32)
        cc = cone_clearances(depth)
        self.jpeg.append(encode_jpeg(obs.rgb))
        self.depth56.append(pool_min(depth))
        self.cones.append([float(cc[k]) for k in CONES])
        self.profile.append(np.asarray(column_profile(depth), dtype=np.float16))
        self.state.append(np.asarray(obs.state, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.collided.append(bool(obs.collided))
        self.macro.append(int(macro))
        self.segment.append(int(segment))

    def __len__(self) -> int:
        return len(self.state)

    def save(self, path: Path, meta: Dict[str, Any]) -> None:
        offsets = np.cumsum([0] + [len(b) for b in self.jpeg]).astype(np.int64)
        np.savez(
            path,
            jpeg=np.frombuffer(b"".join(self.jpeg), dtype=np.uint8),
            jpeg_offsets=offsets,
            depth56=np.stack(self.depth56),
            cones=np.asarray(self.cones, dtype=np.float32),
            profile=np.stack(self.profile),
            state=np.stack(self.state),
            action=np.stack(self.action),
            collided=np.asarray(self.collided, dtype=bool),
            macro=np.asarray(self.macro, dtype=np.int16),
            segment=np.asarray(self.segment, dtype=np.int32),
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )


def run_episode(
    env: Any,
    route: Dict[str, Any],
    rng: np.random.Generator,
    args: argparse.Namespace,
    mask: Any,
) -> Tuple[Optional[EpisodeBuffer], Dict[str, Any]]:
    from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

    pos = [list(map(float, p)) for p in route["pos"]]
    heading = float(route["yaw"][0])
    yaw0 = wrap(heading + float(rng.uniform(-math.pi, math.pi)))
    ep = {"pos": pos, "yaw": [yaw0] * len(pos), "gpt_instruction": ""}
    used, obs, err, failed = reset_with_spawn_retries(
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
        "route_id": route["route_id"],
        "yaw0_off_deg": round(math.degrees(wrap(yaw0 - heading)), 1),
        "spawn_z": float(used_pos[0, 2]),
        "start_xy_shift_m": round(float(np.linalg.norm(used_pos[0, :2] - np.asarray(pos[0][:2]))), 1),
    }
    if failed or meta["start_xy_shift_m"] > 1.0:
        meta["end_reason"] = "spawn_fail" if failed else "start_xy_nudged"
        return None, meta
    goal = used_pos[-1]
    z0 = float(used_pos[0, 2])
    buf = EpisodeBuffer()
    seg = 0
    end = "max_steps"
    left_city_steps = 0
    zero = np.zeros(4)
    buf.add(obs, zero, -1, -1)
    while len(buf) < int(args.max_steps):
        k = int(rng.choice([6, 8, 10]))
        p = obs.position
        name, seq = sample_macro(rng, k, goal_bearing_body(p, float(obs.state[6]), goal))
        mid = MACROS.index(name)
        for a in seq:
            buf.action[-1] = np.asarray(a, dtype=np.float32)
            buf.macro[-1] = mid
            buf.segment[-1] = seg
            obs, _ = env.step(a)
            buf.add(obs, zero, -1, -1)
            p = obs.position
            if bool(obs.collided):
                end = "collision"
                break
            if float(np.linalg.norm(p - goal)) <= 3.0:
                end = "arrived"
                break
            if p[2] > z0 + float(args.max_climb_m) or p[2] < float(args.min_z_m):
                end = "altitude_out"
                break
            if mask is not None and not mask.is_urban(float(p[0]), float(p[1])):
                left_city_steps += 1
                if left_city_steps >= 3:
                    end = "left_city"
                    break
            else:
                left_city_steps = 0
            if len(buf) >= int(args.max_steps):
                break
        seg += 1
        if end != "max_steps":
            break
    meta.update({
        "steps": len(buf) - 1,
        "segments": seg,
        "end_reason": end,
        "collided": end == "collision",
        "goal": [float(x) for x in goal],
    })
    return buf, meta


def review_episode(buf: EpisodeBuffer, meta: Dict[str, Any], regions: Any, mask: Any) -> Tuple[bool, List[str]]:
    from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, path_inland_metrics

    reasons: List[str] = []
    xyz = np.stack(buf.state)[:, :3].astype(np.float64)
    if int(meta.get("steps", 0)) < 3:
        reasons.append("too_short")
    if meta.get("end_reason") == "left_city":
        reasons.append("left_city")
    if path_inland_metrics(xyz.tolist(), regions)["path_min_inland_m"] < 80.0:
        reasons.append("path_inland<80")
    s = classify_spawn_xy(float(xyz[0, 0]), float(xyz[0, 1]), regions)
    if not (s["in_urban"] and s["in_interior"]) or s["in_water"] or s["spawn_inland_m"] < 100.0:
        reasons.append("spawn_geography")
    if mask is not None:
        frac = float(np.mean([mask.is_urban(float(x), float(y)) for x, y, _ in xyz]))
        meta["urban_mask_frac"] = round(frac, 3)
        if frac < 0.9:
            reasons.append("off_city_mask")
    return not reasons, reasons


def gate_status(review: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    kept = [e for e in review["episodes"] if e["keep"]]
    steps = int(sum(e["steps"] for e in kept))
    coll = int(sum(1 for e in kept if e["end_reason"] == "collision"))
    facade = int(sum(e.get("facade_ahead_steps", 0) for e in kept))
    return {
        "kept_episodes": len(kept),
        "kept_steps": steps,
        "collisions": coll,
        "facade_ahead_steps": facade,
        "pass": steps >= int(args.target_steps)
        and coll >= int(args.min_collisions)
        and facade >= int(args.min_facade_steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json")
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--out-dir", default="experiments/aerial/rl/artifacts/consequence_data_20260930")
    ap.add_argument("--urban-mask", default="experiments/aerial/rl/artifacts/interior_od_routes_20260930/urban_mask.json")
    ap.add_argument("--min-spawn-z", type=float, default=24.0)
    ap.add_argument("--min-spawn-clear-m", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--max-climb-m", type=float, default=30.0)
    ap.add_argument("--min-z-m", type=float, default=6.0)
    ap.add_argument("--target-steps", type=int, default=60000)
    ap.add_argument("--min-collisions", type=int, default=300)
    ap.add_argument("--min-facade-steps", type=int, default=6000)
    ap.add_argument("--facade-m", type=float, default=15.0)
    ap.add_argument("--max-episodes", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import yaml

    from experiments.aerial.phase3_unified.region_geometry import load_regions
    from experiments.aerial.rl.train_rl import _build_env
    from experiments.aerial.scripts.generate_interior_od_routes import UrbanMask

    ann = json.loads((root / args.annotation).read_text(encoding="utf-8"))
    routes = ann["episodes"]
    if any(r.get("complexity_meta", {}).get("split") != "train" for r in routes):
        raise SystemExit("refusing: annotation contains non-train routes")
    regions = load_regions()
    mask = UrbanMask(root / args.urban_mask, 0.0) if args.urban_mask else None
    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)

    out = root / args.out_dir
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    review_path = out / "AUTO_REVIEW.json"
    review: Dict[str, Any] = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.is_file()
        else {"episodes": [], "macros": list(MACROS), "args": vars(args)}
    )
    ep_idx = len(review["episodes"])
    rng = np.random.default_rng(int(args.seed) + ep_idx)
    t0 = time.time()
    stop_file = out / "STOP"
    while ep_idx < int(args.max_episodes):
        status = gate_status(review, args)
        if status["pass"]:
            logger.info("GATE PASS %s", status)
            break
        if stop_file.exists():
            logger.info("STOP file found; pausing at episode %d %s", ep_idx, status)
            env.close()
            return 3
        route = routes[int(rng.integers(0, len(routes)))]
        buf, meta = run_episode(env, route, rng, args, mask)
        meta["episode"] = ep_idx
        if buf is None:
            meta.update({"keep": False, "drop_reasons": [meta["end_reason"]], "steps": 0})
        else:
            keep, reasons = review_episode(buf, meta, regions, mask)
            cones = np.asarray(buf.cones, dtype=np.float32)
            meta["facade_ahead_steps"] = int(np.sum(cones[:, 0] <= float(args.facade_m)))
            meta.update({"keep": keep, "drop_reasons": reasons})
            if keep:
                fname = f"episode_{ep_idx:05d}.npz"
                buf.save(out / "episodes" / fname, meta)
                meta["file"] = fname
        review["episodes"].append(meta)
        review["gate"] = gate_status(review, args)
        review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
        logger.info(
            "ep %d %s steps=%s end=%s keep=%s %s | gate %s | %.0fs",
            ep_idx, meta["route_id"], meta.get("steps"), meta.get("end_reason"),
            meta["keep"], meta.get("drop_reasons"), review["gate"], time.time() - t0,
        )
        ep_idx += 1
    logger.info("DONE gate=%s", gate_status(review, args))
    return 0 if gate_status(review, args)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
