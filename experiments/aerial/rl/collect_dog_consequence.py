#!/usr/bin/env python3
"""Consequence data for the LUD dog: street-level random macros under dog dynamics.

Same file format and labels as ``collect_consequence_dataset`` (so
``train_consequence`` reads it unchanged), with the dog in place of the drone:
the camera rides ``cam_height_m`` above the street (``AirSimDogEnv``), actions
use only ``dx`` / ``dyaw`` (model units, mapped to dog commands), and every
episode draws its own dog response parameters (``DogDynamicsRanges``) so the
labels cover the unidentified gait controller. Extra per-step arrays: ``cmd``
(commanded vx, vyaw), ``dog_vel`` (modelled vx, yaw rate) and ``ground_z``.

Spawns are random points beside train routes (the routes fly above low roofs),
probed for ground and kept only where it is flat, within ``--street-z-min``..
``--street-z-max`` (street level in the AirVLN city is about -0.8..-0.6 m) and
clear ahead. Goals sit ``--goal-min-m``..``--goal-max-m`` further along the
route, shifted by the same lateral offset. ``AUTO_REVIEW.json``
keeps the geography + outcome review; collection runs until the gate passes.
Needs a dog-settings renderer (``scripts/make_dog_airsim_settings.py``).
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

from experiments.aerial.rl.collect_consequence_dataset import CONES, EpisodeBuffer, goal_bearing_body, wrap
from experiments.aerial.rl.env.dog_dynamics import MODEL_LIMITS

logger = logging.getLogger("collect_dog_consequence")

DOG_MACROS = ("fwd", "yaw_fwd", "goal_turn_fwd", "yaw_only", "smooth_random", "hover")
DOG_MACRO_P = np.array([0.22, 0.28, 0.18, 0.14, 0.12, 0.06])


def sample_dog_macro(
    rng: np.random.Generator, k: int, bearing: float, *, rad_per_step: float, name: Optional[str] = None
) -> Tuple[str, np.ndarray]:
    """``k`` model actions; ``rad_per_step`` is the nominal heading change of a full yaw command."""
    L = MODEL_LIMITS
    if name is None:
        name = str(rng.choice(DOG_MACROS, p=DOG_MACRO_P))
    a = np.zeros((k, 4), dtype=np.float64)
    if name == "fwd":
        a[:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_fwd":
        m = int(rng.integers(2, k))
        a[:m, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * L[3]
        a[:m, 0] = rng.uniform(0.0, 0.3)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "goal_turn_fwd":
        m = min(k - 1, int(round(abs(bearing) / max(rad_per_step, 1e-3))))
        a[:m, 3] = math.copysign(L[3], bearing)
        a[:m, 0] = rng.uniform(0.0, 0.3)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_only":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * L[3]
    elif name == "smooth_random":
        u = np.zeros(2)
        for t in range(k):
            u = 0.7 * u + 0.3 * rng.normal(0.0, 1.0, size=2)
            a[t, 0] = 0.5 * (1.0 + math.tanh(u[0]))
            a[t, 3] = math.tanh(u[1]) * L[3]
    elif name != "hover":
        raise ValueError(f"unknown dog macro {name!r}")
    a[:, 0] = np.clip(a[:, 0], 0.0, L[0])
    a[:, 3] = np.clip(a[:, 3], -L[3], L[3])
    return name, a


def is_stuck(buf: Any, n: int, dt: float, *, min_move_m: float = 0.03, min_yaw_rad: float = 0.02) -> bool:
    """The last ``n`` steps modelled real motion (``dog_vel``) but the vehicle neither moved nor turned.

    Seen in AirSim as a frozen body with no collision event; such episodes carry no valid labels.
    """
    if n <= 0 or len(buf.state) <= n:
        return False
    vel = np.asarray(buf.dog_vel[-n:], dtype=np.float64)
    st = np.asarray(buf.state[-n - 1:], dtype=np.float64)
    expect_move = float(np.sum(vel[:, 0])) * dt > 0.1
    expect_turn = float(np.sum(np.abs(vel[:, 1]))) * dt > 0.1
    moved = float(np.linalg.norm(st[-1, :2] - st[0, :2])) > min_move_m
    turned = abs(wrap(float(st[-1, 6] - st[0, 6]))) > min_yaw_rad
    return (expect_move and not moved) or (expect_turn and not turned and not moved)


def route_point_ahead(xy: np.ndarray, i: int, dist: float) -> Tuple[np.ndarray, float]:
    """Point ``dist`` metres along the polyline ``xy`` from vertex ``i`` and the segment heading there."""
    p = xy[i].copy()
    left = float(dist)
    for j in range(i, len(xy) - 1):
        seg = xy[j + 1] - xy[j]
        n = float(np.linalg.norm(seg))
        if n < 1e-6:
            continue
        if n >= left:
            return xy[j] + seg * (left / n), math.atan2(seg[1], seg[0])
        left -= n
        p = xy[j + 1].copy()
    seg = xy[-1] - xy[max(0, len(xy) - 2)]
    return p, math.atan2(seg[1], seg[0]) if np.linalg.norm(seg) > 1e-6 else 0.0


def run_episode(env: Any, route: Dict[str, Any], rng: np.random.Generator, args: argparse.Namespace,
                mask: Any, ranges: Any) -> Tuple[Optional[Any], Dict[str, Any]]:
    from experiments.aerial.rl.depth_geometry import directional_clearance_m
    from experiments.aerial.rl.env.airsim_dog_env import mask_ground

    pos = np.asarray(route["pos"], dtype=np.float64)
    xy = pos[:, :2]
    params = ranges.sample(rng)
    env.set_dynamics(params)
    meta: Dict[str, Any] = {"route_id": route["route_id"], "dog_dynamics": params.to_dict()}
    probes: List[Tuple[float, float]] = []
    found = None
    for _ in range(int(args.spawn_tries)):
        i = int(rng.integers(0, max(1, len(xy) - 1)))
        _, heading = route_point_ahead(xy, i, 1.0)
        off = float(rng.uniform(-args.spawn_lateral_m, args.spawn_lateral_m))
        sx = xy[i, 0] - off * math.sin(heading)
        sy = xy[i, 1] + off * math.cos(heading)
        gz, spread = env.probe_ground(sx, sy, float(pos[i, 2]))
        probes.append((round(gz, 2), round(spread, 2)))
        on_street = args.street_z_min <= gz <= args.street_z_max
        if np.isfinite(gz) and spread <= args.probe_flat_m and on_street:
            found = (i, sx, sy, heading, gz, spread)
            break
    meta["spawn_probes"] = probes
    if found is None:
        meta["end_reason"] = "spawn_no_street"
        return None, meta
    i, sx, sy, heading, gz, spread = found
    goal_xy, _ = route_point_ahead(xy, i, float(rng.uniform(args.goal_min_m, args.goal_max_m)))
    goal_xy = goal_xy + np.array([sx, sy]) - xy[i]
    yaw0 = wrap(heading + float(rng.uniform(-math.pi, math.pi)))
    top = float(pos[i, 2])
    meta.update({"route_vertex": i, "spawn_xy": [round(sx, 2), round(sy, 2)],
                 "yaw0_off_deg": round(math.degrees(wrap(yaw0 - heading)), 1),
                 "ground_z": round(gz, 2), "ground_spread": round(spread, 2)})
    try:
        obs = env.reset({"pos": [[sx, sy, top], [goal_xy[0], goal_xy[1], top]], "yaw": [yaw0, yaw0]},
                        ground=(gz, spread))
    except RuntimeError as e:
        meta["end_reason"] = f"spawn_fail:{e}"
        return None, meta
    hfov = float(env.config.hfov_deg)

    def above_ground(o: Any) -> float:
        return env.camera_z(float(o.state[2])) - env.ground_z

    clear = mask_ground(np.asarray(obs.depth, np.float64), hfov, above_ground(obs))
    d_fwd = directional_clearance_m(clear, np.array([1.0, 0.0, 0.0]))
    if np.isfinite(float(d_fwd)) and float(d_fwd) < args.min_spawn_clear_m:
        meta["end_reason"] = "spawn_blocked"
        return None, meta
    goal = np.array([goal_xy[0], goal_xy[1], float(obs.state[2])], dtype=np.float64)
    rad_per_step = params.vyaw_max * params.yaw_gain / float(env.config.step_hz)
    buf = DogEpisodeBuffer(hfov)
    zero = np.zeros(4)
    buf.add(obs, zero, -1, -1, None, env.ground_z, above_ground(obs))
    seg, end, left_city, lost = 0, "max_steps", 0, 0
    while len(buf) < int(args.max_steps):
        k = int(rng.choice([6, 8, 10]))
        name, seq = sample_dog_macro(rng, k, goal_bearing_body(obs.position, float(obs.state[6]), goal),
                                     rad_per_step=rad_per_step)
        mid = DOG_MACROS.index(name)
        for a in seq:
            buf.action[-1] = np.asarray(a, dtype=np.float32)
            buf.macro[-1] = mid
            buf.segment[-1] = seg
            obs, info = env.step(a)
            buf.add(obs, zero, -1, -1, info, env.ground_z, above_ground(obs))
            p = obs.position
            if bool(obs.collided):
                end = "collision"
                meta["hit_object"] = info.get("hit_object", "")
                break
            if float(np.linalg.norm(p[:2] - goal[:2])) <= args.arrive_m:
                end = "arrived"
                break
            if abs(env.camera_z(float(p[2])) - (env.ground_z + env.config.cam_height_m)) > args.height_tol_m:
                lost += 1
                if lost >= 3:
                    end = "ground_lost"
                    break
            else:
                lost = 0
            if mask is not None and not mask.is_urban(float(p[0]), float(p[1])):
                left_city += 1
                if left_city >= 3:
                    end = "left_city"
                    break
            else:
                left_city = 0
            if is_stuck(buf, int(args.stuck_steps), 1.0 / float(env.config.step_hz)):
                end = "stuck"
                break
            if len(buf) >= int(args.max_steps):
                break
        seg += 1
        if end != "max_steps":
            break
    meta.update({"steps": len(buf) - 1, "segments": seg, "end_reason": end, "collided": end == "collision",
                 "goal": [float(x) for x in goal]})
    return buf, meta


class DogEpisodeBuffer(EpisodeBuffer):
    """``depth56`` keeps the raw frame (the model input); cones / profile see it with the ground masked."""

    def __init__(self, hfov_deg: float = 90.0) -> None:
        super().__init__()
        self.hfov_deg = float(hfov_deg)
        self.cmd: List[List[float]] = []
        self.dog_vel: List[List[float]] = []
        self.ground_z: List[float] = []

    def add(self, obs: Any, action: np.ndarray, macro: int, segment: int,  # type: ignore[override]
            info: Optional[Dict[str, Any]] = None, ground_z: float = float("nan"),
            cam_above_ground_m: float = 0.6) -> None:
        from experiments.aerial.rl.collect_consequence_dataset import encode_jpeg, pool_min
        from experiments.aerial.rl.cone_transition import column_profile
        from experiments.aerial.rl.depth_geometry import cone_clearances
        from experiments.aerial.rl.env.airsim_dog_env import mask_ground

        depth = np.asarray(obs.depth, dtype=np.float32)
        clear = mask_ground(depth, self.hfov_deg, cam_above_ground_m)
        cc = cone_clearances(clear)
        self.jpeg.append(encode_jpeg(obs.rgb))
        self.depth56.append(pool_min(depth))
        self.cones.append([float(cc[k]) for k in CONES])
        self.profile.append(np.asarray(column_profile(clear), dtype=np.float16))
        self.state.append(np.asarray(obs.state, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.collided.append(bool(obs.collided))
        self.macro.append(int(macro))
        self.segment.append(int(segment))
        info = info or {}
        self.cmd.append(list(info.get("cmd", [0.0, 0.0])))
        self.dog_vel.append(list(info.get("dog_vel", [0.0, 0.0])))
        self.ground_z.append(float(ground_z))

    def save(self, path: Path, meta: Dict[str, Any]) -> None:
        super().save(path, meta)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays.update(cmd=np.asarray(self.cmd, np.float32), dog_vel=np.asarray(self.dog_vel, np.float32),
                      ground_z=np.asarray(self.ground_z, np.float32))
        np.savez(path, **arrays)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json")
    ap.add_argument("--out-dir", default="experiments/aerial/rl/artifacts/dog_consequence_data_20261019")
    ap.add_argument("--urban-mask", default="experiments/aerial/rl/artifacts/interior_od_routes_20260930/urban_mask.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=41481)
    ap.add_argument("--step-hz", type=float, default=2.0)
    ap.add_argument("--cam-height-m", type=float, default=0.6)
    ap.add_argument("--cam-offset-down-m", type=float, default=-0.15)
    ap.add_argument("--goal-min-m", type=float, default=10.0)
    ap.add_argument("--goal-max-m", type=float, default=40.0)
    ap.add_argument("--arrive-m", type=float, default=1.0)
    ap.add_argument("--probe-flat-m", type=float, default=0.3)
    ap.add_argument("--street-z-max", type=float, default=0.5,
                    help="highest ground (+up) that counts as street; above it the probe hit a roof")
    ap.add_argument("--street-z-min", type=float, default=-2.0)
    ap.add_argument("--spawn-tries", type=int, default=12)
    ap.add_argument("--spawn-lateral-m", type=float, default=12.0)
    ap.add_argument("--min-spawn-clear-m", type=float, default=1.0)
    ap.add_argument("--height-tol-m", type=float, default=0.5)
    ap.add_argument("--stuck-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--target-steps", type=int, default=40000)
    ap.add_argument("--min-collisions", type=int, default=300)
    ap.add_argument("--min-facade-steps", type=int, default=4000)
    ap.add_argument("--facade-m", type=float, default=3.0)
    ap.add_argument("--max-episodes", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.aerial.phase3_unified.region_geometry import load_regions
    from experiments.aerial.rl.collect_consequence_dataset import gate_status, review_episode
    from experiments.aerial.rl.env.airsim_dog_env import AirSimDogEnv, AirSimDogEnvConfig
    from experiments.aerial.rl.env.dog_dynamics import DogDynamicsRanges
    from experiments.aerial.scripts.generate_interior_od_routes import UrbanMask

    ann = json.loads((root / args.annotation).read_text(encoding="utf-8"))
    routes = ann["episodes"]
    if any(r.get("complexity_meta", {}).get("split") != "train" for r in routes):
        raise SystemExit("refusing: annotation contains non-train routes")
    regions = load_regions()
    mask = UrbanMask(root / args.urban_mask, 0.0) if args.urban_mask else None
    env = AirSimDogEnv(AirSimDogEnvConfig(host=args.host, port=args.port, step_hz=args.step_hz,
                                          cam_height_m=args.cam_height_m, cam_offset_down_m=args.cam_offset_down_m))
    ranges = DogDynamicsRanges()

    out = root / args.out_dir
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    review_path = out / "AUTO_REVIEW.json"
    review: Dict[str, Any] = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.is_file()
        else {"episodes": [], "macros": list(DOG_MACROS), "args": vars(args), "vehicle": "lud_dog"}
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
        buf, meta = run_episode(env, route, rng, args, mask, ranges)
        meta["episode"] = ep_idx
        if buf is None:
            meta.update({"keep": False, "drop_reasons": [meta["end_reason"]], "steps": 0})
        else:
            keep, reasons = review_episode(buf, meta, regions, mask)
            if meta["end_reason"] in ("ground_lost", "stuck"):
                keep, reasons = False, reasons + [meta["end_reason"]]
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
        logger.info("ep %d %s steps=%s end=%s keep=%s %s | gate %s | %.0fs", ep_idx, meta["route_id"],
                    meta.get("steps"), meta.get("end_reason"), meta["keep"], meta.get("drop_reasons"),
                    review["gate"], time.time() - t0)
        ep_idx += 1
    logger.info("DONE gate=%s", gate_status(review, args))
    return 0 if gate_status(review, args)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
