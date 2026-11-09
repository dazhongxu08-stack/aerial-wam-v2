#!/usr/bin/env python3
"""Closed-loop dog sim: compare consequence ensembles on the collection val routes.

Replays kept episodes (spawn, goal, recorded ground and dynamics, start yaw) in
the dog AirSim. Both ensembles see the same plant. Candidates are the finder
dog families in full model units; only the first action of the chosen plan is
executed, through ``model_action_to_cmd``. No 0.06 m action scale, no
``realize_motion``, no guard, no near-goal / brake / spin rules.

Set ``DOG_CONSEQUENCE_MODEL`` to the training repo's ``consequence_model.py``
when that file differs from this tree. Depth still comes from this tree, which
is the code that wrote the DA3 sidecars.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DOG_FAMILIES = ("fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "goal_turn_fwd", "yaw_only", "smooth_random", "hover")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_consequence_module():
    override = os.environ.get("DOG_CONSEQUENCE_MODEL", "").strip()
    if not override:
        from experiments.aerial.rl import consequence_model as cm
        return cm
    path = Path(override)
    spec = importlib.util.spec_from_file_location("dog_ft_consequence_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def goal_bearing_body(pos: np.ndarray, yaw: float, goal: np.ndarray) -> float:
    d = np.asarray(goal[:2], dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64)
    return wrap(math.atan2(float(d[1]), float(d[0])) - float(yaw))


def episode_seed(base: int, file_name: str) -> int:
    return (int(base) + zlib.crc32(file_name.encode())) & 0xFFFFFFFF


def select_episodes(review: Dict[str, Any], routes: Sequence[str], per_route: int) -> List[Dict[str, Any]]:
    want = set(routes)
    groups: Dict[str, List[Dict[str, Any]]] = {r: [] for r in routes}
    for ep in review["episodes"]:
        if not ep.get("keep") or ep.get("route_id") not in want:
            continue
        groups[ep["route_id"]].append(ep)
    out: List[Dict[str, Any]] = []
    for rid in routes:
        eps = sorted(groups[rid], key=lambda e: str(e["file"]))
        if per_route > 0:
            eps = eps[:per_route]
        out.extend(eps)
    return out


def parse_models(specs: Sequence[str]) -> List[Tuple[str, List[str]]]:
    out = []
    for spec in specs:
        name, _, paths = spec.partition("=")
        ckpts = [p for p in paths.split(",") if p]
        if not name or not ckpts:
            raise SystemExit(f"--model expects name=ckpt[,ckpt...], got {spec!r}")
        out.append((name, ckpts))
    return out


def load_models(cm: Any, paths: Sequence[str], device: str) -> List[Any]:
    models = [cm.load_checkpoint(Path(p), device) for p in paths]
    for m in models:
        m.eval()
    horizons = {int(m.horizon) for m in models}
    if len(horizons) != 1:
        raise SystemExit(f"ensemble horizons differ: {horizons}")
    return models


def score_candidates(cm: Any, models: Sequence[Any], depth56: np.ndarray, state: np.ndarray,
                     goal: np.ndarray, prev: np.ndarray, cands: np.ndarray, device: str
                     ) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    m0 = len(cands)
    g = torch.as_tensor(np.repeat(cm.goal_features(state, goal)[None], m0, 0), device=device)
    pa = torch.as_tensor(np.repeat(np.asarray(prev, np.float32)[None], m0, 0), device=device)
    acts = torch.as_tensor(np.asarray(cands, np.float32), device=device)
    s_all, p_all = [], []
    with torch.inference_mode():
        for m in models:
            d_k, a_k = cm.history_inputs([depth56], [], m.frame_offsets, m.hist_actions)
            d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
            d = torch.as_tensor(np.repeat(d_k[None], m0, 0), device=device)
            ha = torch.as_tensor(np.repeat(a_k[None], m0, 0), device=device) if m.hist_actions else None
            out = m(d, g, pa, acts, ha)
            s_all.append(cm.survival(out["hazard_logit"])[..., -1].float().cpu().numpy())
            p_all.append((-out["dgoal"][..., -1]).float().cpu().numpy())
    return np.mean(s_all, 0), np.mean(p_all, 0)


def choose_action(cm: Any, models: Sequence[Any], rng: np.random.Generator, depth56: np.ndarray,
                  state: np.ndarray, goal: np.ndarray, prev: np.ndarray, *, horizon: int,
                  surv_min: float, fov_deg: float, fwd_dx: float, rad_per_step: float, device: str
                  ) -> Tuple[np.ndarray, str, float, float, int]:
    from experiments.aerial.rl.collect_dog_consequence import sample_dog_macro

    bearing = goal_bearing_body(state[:3], float(state[6]), goal)
    cands, names = [], []
    for fam in DOG_FAMILIES:
        name, seq = sample_dog_macro(rng, horizon, bearing, rad_per_step=rad_per_step, name=fam)
        if fam == "fwd":
            seq[:, 0] = float(fwd_dx)
        cands.append(seq)
        names.append(name)
    stacked = np.stack(cands)
    allowed = np.ones(len(stacked), dtype=bool)
    if fov_deg > 0:
        seen = allowed & cm.visible_motion_allowed(stacked, 1, fov_deg)
        if seen.any():
            allowed = seen
    surv, prog = score_candidates(cm, models, depth56, state, goal, prev, stacked, device)
    idx = cm.select_candidate(surv, prog, allowed, surv_min)
    n_pass = int((allowed & (surv >= float(surv_min))).sum())
    return np.asarray(stacked[idx, 0], np.float64), names[idx], float(surv[idx]), float(prog[idx]), n_pass


class _Motion:
    def __init__(self) -> None:
        self.state: List[np.ndarray] = []
        self.dog_vel: List[List[float]] = []


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(row["model"], []).append(row)
    out: Dict[str, Any] = {}
    for name, items in by.items():
        def count(reason: str, spawn: Optional[bool] = None) -> int:
            n = 0
            for it in items:
                if it.get("end_reason") != reason:
                    continue
                if spawn is not None and bool(it.get("spawn_collision")) != spawn:
                    continue
                n += 1
            return n

        scored = [it for it in items if it.get("end_reason") not in ("reset_failed", "depth_failed")]
        prog = [float(it["progress"]) for it in scored]
        best = [float(it["best_progress"]) for it in scored]
        out[name] = {
            "n": len(items),
            "arrived": count("arrived"),
            "collision": count("collision", False),
            "spawn_collision": count("collision", True),
            "stuck": count("stuck"),
            "timeout": count("timeout"),
            "ground_lost": count("ground_lost"),
            "reset_failed": count("reset_failed"),
            "mean_progress": None if not prog else round(float(np.mean(prog)), 4),
            "mean_best_progress": None if not best else round(float(np.mean(best)), 4),
            "mean_steps": None if not scored else round(float(np.mean([it["steps"] for it in scored])), 1),
        }
        routes: Dict[str, Any] = {}
        for rid in sorted({it["route_id"] for it in items}):
            sub = [it for it in items if it["route_id"] == rid]
            routes[rid] = {
                "n": len(sub),
                "arrived": sum(it.get("end_reason") == "arrived" for it in sub),
                "collision": sum(it.get("end_reason") == "collision" and not it.get("spawn_collision") for it in sub),
                "stuck": sum(it.get("end_reason") == "stuck" for it in sub),
                "timeout": sum(it.get("end_reason") == "timeout" for it in sub),
            }
        out[name]["routes"] = routes
    return out


def write_results(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def _fit(frame: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    import cv2
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def grab_fpv(env: Any) -> Optional[np.ndarray]:
    """Policy-camera scene at the current pose. One RPC on the env's own client."""
    from experiments.aerial.rl.env.airsim_dog_env import decode_raw_scene

    airsim = env._airsim
    client = env._connect()
    resp = client.simGetImages([
        airsim.ImageRequest("front_custom", airsim.ImageType.Scene, False, False),
    ], **env._vk)[0]
    if not resp.width or not resp.height or not resp.image_data_uint8:
        return None
    rgb = decode_raw_scene(bytes(resp.image_data_uint8), int(resp.height), int(resp.width))
    return _fit(rgb)


def replay_chase(env: Any, poses: Sequence[Tuple[float, float, float, float]], tag: str) -> List[np.ndarray]:
    """Re-render the flown poses from behind and above, then put the policy camera back.

    The dog settings only have one working camera, so the chase view is a replay
    on that camera after the episode, not a second live view.
    """
    from experiments.aerial.rl.env.airsim_dog_env import decode_raw_scene

    airsim = env._airsim
    client = env._connect()
    home = airsim.Pose(airsim.Vector3r(0.35, 0.0, -0.15), airsim.to_quaternion(0.0, 0.0, 0.0))
    chase = airsim.Pose(airsim.Vector3r(-4.0, 0.0, -2.0), airsim.to_quaternion(math.radians(-18.0), 0.0, 0.0))
    frames: List[np.ndarray] = []
    try:
        client.simSetCameraPose("front_custom", chase, **env._vk)
        for i, (x, y, z_up, yaw) in enumerate(poses):
            client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(float(x), float(y), -float(z_up)),
                            airsim.to_quaternion(0.0, 0.0, float(yaw))),
                True, **env._vk)
            resp = client.simGetImages([
                airsim.ImageRequest("front_custom", airsim.ImageType.Scene, False, False),
            ], **env._vk)[0]
            if not resp.width or not resp.image_data_uint8:
                continue
            rgb = decode_raw_scene(bytes(resp.image_data_uint8), int(resp.height), int(resp.width))
            frames.append(_label(_fit(rgb), f"chase {tag} {i}"))
    finally:
        client.simSetCameraPose("front_custom", home, **env._vk)
    return frames


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return frame
    out = np.array(frame, copy=True)
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 9 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_mp4(frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w, h = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr[:h, :w], dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err[-500:]}")


def save_episode_video(video_dir: Path, row: Dict[str, Any], fpv: Sequence[np.ndarray],
                       chase: Sequence[np.ndarray], fps: float) -> None:
    stem = f"{row['model']}_{row['route_id']}_{Path(str(row['file'])).stem}"
    if fpv:
        write_mp4(fpv, video_dir / f"{stem}_fpv.mp4", fps)
    if chase:
        write_mp4(chase, video_dir / f"{stem}_chase.mp4", fps)
    n = min(len(fpv), len(chase))
    if n:
        dual = [np.concatenate([a, b], axis=1) for a, b in zip(fpv[:n], chase[:n])]
        write_mp4(dual, video_dir / f"{stem}_dual.mp4", fps)
    row["video"] = stem


def start_yaw(data_dir: Path, ep: Dict[str, Any]) -> float:
    with np.load(data_dir / "episodes" / ep["file"]) as z:
        return float(z["state"][0, 6])


def run_episode(env: Any, pred: Any, cm: Any, models: Sequence[Any], ep: Dict[str, Any],
                data_dir: Path, args: argparse.Namespace, device: str) -> Dict[str, Any]:
    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.collect_dog_consequence import is_stuck
    from experiments.aerial.rl.env.dog_dynamics import DogDynamicsParams

    row: Dict[str, Any] = {
        "model": args.model_name, "route_id": ep["route_id"], "file": ep["file"],
        "end_reason": "reset_failed", "steps": 0, "hit_object": "", "spawn_collision": False,
    }
    sx, sy = (float(x) for x in ep["spawn_xy"])
    gx, gy = float(ep["goal"][0]), float(ep["goal"][1])
    yaw0 = start_yaw(data_dir, ep)
    params = DogDynamicsParams(**{k: float(v) for k, v in ep["dog_dynamics"].items()})
    env.set_dynamics(params)
    ground = (float(ep["ground_z"]), float(ep.get("ground_spread") or 0.0))
    top = float(ep["goal"][2])
    try:
        obs = env.reset({"pos": [[sx, sy, top], [gx, gy, top]], "yaw": [yaw0, yaw0]}, ground=ground)
        # The altitude hold is still moving on the first frame. Sit again so both
        # models start from the same street height.
        client = env._connect()
        env._hold(client, float(obs.state[0]), float(obs.state[1]), -env._target_z_ned(), yaw0, 0.6)
        env.dyn.reset()
        obs = env.observe()
    except Exception as exc:  # noqa: BLE001 — one bad spawn should not kill the comparison
        row["error"] = str(exc)
        return row
    pred.reset()
    goal = np.array([gx, gy, float(obs.state[2])], dtype=np.float64)
    d0 = float(np.hypot(float(obs.state[0]) - gx, float(obs.state[1]) - gy))
    d_min = d0
    depth = pred.predict_depth(obs)
    if depth is None:
        row["end_reason"] = "depth_failed"
        return row
    horizon = int(models[0].horizon)
    rad = params.vyaw_max * params.yaw_gain / float(env.config.step_hz)
    rng = np.random.default_rng(episode_seed(args.seed, ep["file"]))
    prev = np.zeros(4, np.float32)
    motion = _Motion()
    motion.state.append(np.asarray(obs.state, np.float64))
    video_dir = Path(args.video_dir) if getattr(args, "video_dir", "") else None
    fpv_frames: List[np.ndarray] = []
    poses: List[Tuple[float, float, float, float]] = []

    def remember(step_i: int) -> None:
        if video_dir is None:
            return
        st = np.asarray(obs.state, np.float64)
        poses.append((float(st[0]), float(st[1]), float(st[2]), float(st[6])))
        try:
            frame = grab_fpv(env)
        except Exception as exc:  # noqa: BLE001 — a missed frame should not end the flight
            row["video_error"] = str(exc)
            return
        if frame is None:
            return
        tag = f"{args.model_name} {ep['route_id']} {ep['file']} {step_i}"
        fpv_frames.append(_label(frame, "fpv " + tag))

    remember(0)
    end = "timeout"
    hit_obj = ""
    surv_sum = prog_sum = 0.0
    nopass = 0
    families: Dict[str, int] = {}
    step = 0
    lost = 0
    dt = 1.0 / float(env.config.step_hz)
    for step in range(1, int(args.max_steps) + 1):
        d56 = pool_min(np.asarray(depth, np.float32))
        state = np.asarray(obs.state, np.float64)
        action, family, surv, pred_prog, n_pass = choose_action(
            cm, models, rng, d56, state, goal, prev, horizon=horizon, surv_min=args.surv_min,
            fov_deg=args.fov_max_deg, fwd_dx=args.fwd_dx, rad_per_step=rad, device=device)
        surv_sum += surv
        prog_sum += pred_prog
        nopass += int(n_pass == 0)
        families[family] = families.get(family, 0) + 1
        obs, info = env.step(action)
        prev = np.asarray(action, np.float32)
        motion.dog_vel.append(list(info.get("dog_vel") or [0.0, 0.0]))
        motion.state.append(np.asarray(obs.state, np.float64))
        dist = float(np.hypot(float(obs.state[0]) - gx, float(obs.state[1]) - gy))
        d_min = min(d_min, dist)
        remember(step)
        depth = pred.predict_depth(obs)
        if bool(obs.collided):
            end = "collision"
            hit_obj = str(info.get("hit_object") or "")
            break
        if dist <= float(args.arrive_m):
            end = "arrived"
            break
        cam_z = env.camera_z(float(obs.state[2]))
        if abs(cam_z - (env.ground_z + env.config.cam_height_m)) > float(args.height_tol_m):
            lost += 1
            if lost >= 3:
                end = "ground_lost"
                break
        else:
            lost = 0
        if is_stuck(motion, int(args.stuck_steps), dt):
            end = "stuck"
            break
        if depth is None:
            end = "depth_failed"
            break
    else:
        step = int(args.max_steps)
    dist = float(np.hypot(float(obs.state[0]) - gx, float(obs.state[1]) - gy))
    denom = max(d0, 1e-6)
    row.update({
        "end_reason": end,
        "steps": int(step),
        "hit_object": hit_obj,
        "spawn_collision": end == "collision" and int(step) <= 2,
        "d0": round(d0, 3),
        "d_final": round(dist, 3),
        "d_min": round(d_min, 3),
        "progress": round((d0 - dist) / denom, 4),
        "best_progress": round((d0 - d_min) / denom, 4),
        "yaw0": round(yaw0, 4),
        "reset_z": round(float(motion.state[0][2]), 3),
        "mean_surv": round(surv_sum / max(1, step), 4),
        "mean_pred_prog": round(prog_sum / max(1, step), 4),
        "nopass_steps": int(nopass),
        "families": families,
    })
    if video_dir is not None and (fpv_frames or poses):
        chase_frames: List[np.ndarray] = []
        try:
            chase_frames = replay_chase(env, poses, f"{args.model_name} {ep['route_id']} {ep['file']}")
        except Exception as exc:  # noqa: BLE001
            row["video_error"] = str(exc)
        save_episode_video(video_dir, row, fpv_frames, chase_frames, float(env.config.step_hz))
    return row


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--routes", default="od_train_07,od_train_17,od_train_25,od_train_31")
    ap.add_argument("--per-route", type=int, default=0, help="0 keeps every kept episode on the route")
    ap.add_argument("--model", action="append", required=True, help="name=ckpt[,ckpt...]")
    ap.add_argument("--depth-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=41481)
    ap.add_argument("--step-hz", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--arrive-m", type=float, default=1.0)
    ap.add_argument("--surv-min", type=float, default=0.75)
    ap.add_argument("--fov-max-deg", type=float, default=35.0)
    ap.add_argument("--fwd-dx", type=float, default=1.0)
    ap.add_argument("--stuck-steps", type=int, default=8)
    ap.add_argument("--height-tol-m", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-dir", default="", help="write fpv / chase / dual mp4 for each episode")
    args = ap.parse_args()

    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.airsim_dog_env import AirSimDogEnv, AirSimDogEnvConfig

    cm = load_consequence_module()
    routes = [r for r in args.routes.split(",") if r]
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    episodes = select_episodes(review, routes, int(args.per_route))
    if not episodes:
        raise SystemExit("no kept episodes matched --routes")
    specs = parse_models(args.model)
    ensembles = [(name, load_models(cm, paths, args.device)) for name, paths in specs]
    pred = DepthMinPredictor.from_checkpoint(args.depth_ckpt, device=args.device)
    env = AirSimDogEnv(AirSimDogEnvConfig(host=args.host, port=int(args.port), step_hz=float(args.step_hz)))

    out = Path(args.out)
    payload: Dict[str, Any] = {}
    if out.is_file():
        payload = json.loads(out.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = list(payload.get("episodes") or [])
    done = {(r["model"], r["file"]) for r in rows}
    payload["args"] = vars(args)
    payload["episodes"] = rows
    print(f"episodes {len(episodes)} models {[n for n, _ in ensembles]} done {len(done)}", flush=True)
    t0 = time.time()
    try:
        for ep in episodes:
            for name, models in ensembles:
                if (name, ep["file"]) in done:
                    continue
                args.model_name = name
                row = run_episode(env, pred, cm, models, ep, Path(args.data_dir), args, args.device)
                rows.append(row)
                done.add((name, ep["file"]))
                payload["summary"] = summarize(rows)
                write_results(out, payload)
                print(
                    f"{name} {ep['route_id']} {ep['file']} {row['end_reason']} "
                    f"steps={row['steps']} d={row.get('d_final')} prog={row.get('progress')} "
                    f"hit={row.get('hit_object') or '-'} {time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        env.close()
        payload["summary"] = summarize(rows)
        write_results(out, payload)
    print(json.dumps(payload["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
