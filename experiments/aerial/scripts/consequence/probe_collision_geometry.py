"""Re-create MPC collisions in AirSim to see what was hit and where it was relative to the camera.

contact cases: place the drone at the last pose before the contact (yaw recovered by matching the recorded
camera frame), replay the last motion, read AirSim's collision info (object, impact point) and express the
impact point in the body frame; also grab front/left/right/back/down views at that pose.
spawn cases: reset exactly like the eval, read the collision info right after reset and after hovering.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def col_info(client, vk):
    c = client.simGetCollisionInfo(**vk)
    return {"has_collided": bool(c.has_collided), "object_name": c.object_name, "time_stamp": int(c.time_stamp),
            "impact_ned": [c.impact_point.x_val, c.impact_point.y_val, c.impact_point.z_val],
            "normal": [c.normal.x_val, c.normal.y_val, c.normal.z_val], "penetration": float(c.penetration_depth),
            "pos_ned": [c.position.x_val, c.position.y_val, c.position.z_val]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    import airsim
    import yaml

    from experiments.aerial.rl.train_rl import _build_env

    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    out = Path(args.out)
    (out / "views").mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text())
    results = []
    client = None

    def grab(c, vk, cam):
        rs = c.simGetImages([airsim.ImageRequest(cam, airsim.ImageType.Scene, False, True),
                             airsim.ImageRequest(cam, airsim.ImageType.DepthPlanar, True, False)], **vk)
        rgb = cv2.imdecode(np.frombuffer(bytes(rs[0].image_data_uint8), np.uint8), cv2.IMREAD_COLOR)
        d = np.asarray(rs[1].image_data_float, np.float32).reshape(rs[1].height, rs[1].width)
        return cv2.resize(rgb, (224, 224)), d

    def place(c, vk, p_up, yaw, pitch=0.0):
        pose = airsim.Pose(airsim.Vector3r(float(p_up[0]), float(p_up[1]), -float(p_up[2])),
                           airsim.to_quaternion(float(pitch), 0.0, float(yaw)))
        c.simSetVehiclePose(pose, True, **vk)
        c.moveByVelocityAsync(0, 0, 0, 0.3, **vk)
        time.sleep(0.15)
        c.simSetVehiclePose(pose, True, **vk)
        time.sleep(0.05)

    def surface_check(c, vk, ci):
        """Camera 3 m out from the contact point on the drone's side of the contact surface, looking at it."""
        n = np.asarray(ci["normal"], np.float64)
        n = n / max(np.linalg.norm(n), 1e-6)
        imp = np.asarray(ci["impact_ned"], np.float64)
        if float(np.dot(n, np.asarray(ci["pos_ned"], np.float64) - imp)) < 0:
            n = -n
        eye = imp + 3.0 * n
        look = -n
        l_yaw = math.atan2(look[1], look[0])
        l_pitch = math.asin(float(np.clip(-look[2], -0.999, 0.999)))
        place(c, vk, [eye[0], eye[1], -eye[2]], l_yaw, max(-1.55, min(1.55, l_pitch)))
        rgb_s, d_s = grab(c, vk, env.config.camera)
        h, w = d_s.shape
        centre = float(np.nanmedian(d_s[h // 2 - h // 16:h // 2 + h // 16, w // 2 - w // 16:w // 2 + w // 16]))
        return centre, rgb_s, d_s

    for case in cases:
        rid = case["route_id"]
        if case["kind"] == "control":
            tag = f"control_{rid}_{case['seed']}"
            try:
                client = env._connect(); vk = env._vk
                p0 = np.asarray(case["pos_up"], np.float64)
                env.reset({"pos": [p0.tolist(), (p0 + [1, 0, 0]).tolist()], "yaw": [0.0, 0.0]})
                best = None
                for a in range(0, 360, 15):
                    place(client, vk, p0, math.radians(a))
                    _, d = grab(client, vk, env.config.camera)
                    h, w = d.shape
                    cd = float(np.nanmedian(d[h // 2 - h // 16:h // 2 + h // 16, w // 2 - w // 16:w // 2 + w // 16]))
                    if 3.0 <= cd <= 12.0 and float(np.nanmin(d)) > 1.5 and (best is None or cd < best[0]):
                        best = (cd, a)
                if best is None:
                    res = {"case": case, "error": "no wall 3-12 m ahead"}
                else:
                    yaw = math.radians(best[1])
                    place(client, vk, p0, yaw)
                    t0 = col_info(client, vk)["time_stamp"]
                    client.moveByVelocityAsync(math.cos(yaw), math.sin(yaw), 0.0, 15.0,
                                               yaw_mode=airsim.YawMode(False, best[1]), **vk)
                    hit = None
                    for _ in range(300):
                        time.sleep(0.05)
                        ci = col_info(client, vk)
                        if ci["has_collided"] and ci["time_stamp"] != t0:
                            hit = ci
                            break
                    client.moveByVelocityAsync(0, 0, 0, 0.1, **vk)
                    res = {"case": case, "wall_center_m": best[0], "yaw_deg": best[1], "hit": hit}
                    if hit:
                        centre, rgb_s, d_s = surface_check(client, vk, hit)
                        res.update({"surface_view_center_m": centre, "surface_visible": bool(abs(centre - 3.0) <= 1.0),
                                    "travel_m": float(np.linalg.norm(np.asarray(hit["pos_ned"]) * [1, 1, -1] - p0))})
                        cv2.imwrite(str(out / "views" / f"{tag}.jpg"), rgb_s)
            except Exception as exc:
                res = {"case": case, "error": repr(exc)}
            results.append(res)
            (out / "RESULTS.json").write_text(json.dumps(results, indent=1))
            print(tag, {k: res.get(k) for k in ("wall_center_m", "travel_m", "surface_view_center_m", "error")}, flush=True)
            continue
        tag = f"{case['group']}_{rid}_{case['seed']}"
        try:
            if case["kind"] == "spawn":
                routes = json.loads((root / case["annotation"]).read_text())["episodes"]
                r = next(x for x in routes if x["route_id"] == rid)
                pos = np.asarray(r["pos"], np.float64)
                start, goal = pos[0].copy(), pos[-1].copy()
                if case.get("use_spawn_z") and r.get("complexity_meta", {}).get("spawn_z_ok") is not None:
                    start[2] = goal[2] = float(r["complexity_meta"]["spawn_z_ok"])
                rows = []
                for rep in range(3):
                    client = env._connect(); vk = env._vk
                    before = col_info(client, vk)
                    obs = env.reset({"pos": [start.tolist(), goal.tolist()], "yaw": [float(r["yaw"][0])] * 2})
                    after_reset = col_info(client, vk)
                    obs1, _ = env.step(np.zeros(4))
                    after_hover = col_info(client, vk)
                    rows.append({"before": before, "after_reset": after_reset, "obs_collided_reset": bool(obs.collided),
                                 "after_hover": after_hover, "obs_collided_hover": bool(obs1.collided),
                                 "start_up": start.tolist(), "pos_after_reset_up": np.asarray(obs.position).tolist()})
                yaw = float(obs.state[6])
                views = {}
                for name, dy, pit in (("front", 0, 0), ("left", -90, 0), ("right", 90, 0), ("back", 180, 0), ("down", 0, -89)):
                    place(client, vk, start, yaw + math.radians(dy), math.radians(pit))
                    rgb, d = grab(client, vk, env.config.camera)
                    views[name] = (rgb, d)
                res = {"case": case, "reps": rows,
                       "view_min_m": {k: float(np.nanmin(v[1])) for k, v in views.items()}}
                ar = rows[-1]["after_reset"]
                if ar["has_collided"]:
                    centre, rgb_s, d_s = surface_check(client, vk, ar)
                    res.update({"surface_view_center_m": centre, "surface_visible": bool(abs(centre - 3.0) <= 1.0)})
                    views["at_surface_3m"] = (rgb_s, d_s)
            else:
                tr = np.load(root / case["dir"] / f"traj_{rid}.npz")["traj"].astype(np.float64)
                z = np.load(root / case["dir"] / f"last_frames_{rid}.npz")
                off = list(z["jpeg_offsets"])
                frames = [cv2.imdecode(z["jpeg"][off[i]:off[i + 1]], cv2.IMREAD_COLOR) for i in range(len(off) - 1)]
                ref = cv2.cvtColor(cv2.resize(frames[-2], (56, 56)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                p_pre, p_hit = tr[-2], tr[-1]
                client = env._connect(); vk = env._vk
                env.reset({"pos": [p_pre.tolist(), (p_pre + [1, 0, 0]).tolist()], "yaw": [0.0, 0.0]})

                def err(yaw):
                    place(client, vk, p_pre, yaw)
                    rgb, _ = grab(client, vk, env.config.camera)
                    g = cv2.cvtColor(cv2.resize(rgb, (56, 56)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    return float(np.mean(np.abs(g - ref)))
                grid = [(err(math.radians(a)), a) for a in range(0, 360, 10)]
                e0, a0 = min(grid)
                fine = [(err(math.radians(a)), a) for a in np.arange(a0 - 9, a0 + 10, 2)]
                e1, a1 = min(fine)
                yaw = math.radians(float(a1))
                views = {}
                for name, dy, pit in (("front", 0, 0), ("left", -90, 0), ("right", 90, 0), ("back", 180, 0), ("down", 0, -89)):
                    place(client, vk, p_pre, yaw + math.radians(dy), math.radians(pit))
                    views[name] = grab(client, vk, env.config.camera)
                # replay: from 2 steps before, fly the recorded displacement direction for 0.8 s
                hit, replay_dist = None, None
                for back in (2, 3, 4):
                    p_start = tr[-1 - back] if len(tr) > back else tr[0]
                    place(client, vk, p_start, yaw)
                    t0 = col_info(client, vk)["time_stamp"]
                    v = p_hit - p_start
                    sp = float(np.linalg.norm(v))
                    v = v / max(sp, 1e-6) * 1.0
                    client.moveByVelocityAsync(float(v[0]), float(v[1]), float(-v[2]), 2.0,
                                               yaw_mode=airsim.YawMode(False, math.degrees(yaw)), **vk)
                    for _ in range(40):
                        time.sleep(0.05)
                        ci = col_info(client, vk)
                        if ci["has_collided"] and ci["time_stamp"] != t0:
                            hit = ci
                            break
                    client.moveByVelocityAsync(0, 0, 0, 0.1, **vk)
                    if hit:
                        p_c = np.asarray(hit["pos_ned"]) * [1, 1, -1]
                        replay_dist = float(np.linalg.norm(p_c - p_hit))
                        break
                res = {"case": case, "yaw_deg": float(a1), "match_err": e1, "match_err_grid_median": float(np.median([g[0] for g in grid])),
                       "view_min_m": {k: float(np.nanmin(v_[1])) for k, v_ in views.items()}, "hit": hit,
                       "replay_contact_to_recorded_contact_m": replay_dist}
                if hit:
                    d = np.asarray(hit["impact_ned"]) - np.asarray(hit["pos_ned"])
                    fwd = d[0] * math.cos(yaw) + d[1] * math.sin(yaw)
                    right = -d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
                    up = -d[2]
                    az = math.degrees(math.atan2(right, fwd))
                    el = math.degrees(math.atan2(up, math.hypot(fwd, right)))
                    res.update({"impact_body_fwd_right_up": [fwd, right, up], "impact_az_deg": az, "impact_el_deg": el,
                                "impact_in_view": bool(fwd > 0 and abs(az) <= 45 and abs(el) <= 45)})
                    centre, rgb_s, d_s = surface_check(client, vk, hit)
                    res.update({"surface_view_center_m": centre, "surface_visible": bool(abs(centre - 3.0) <= 1.0)})
                    views["at_surface_3m"] = (rgb_s, d_s)
                views["recorded"] = (cv2.resize(frames[-2], (224, 224)), None)
            # composite: rgb row + depth row
            names = [n for n in ("recorded", "front", "left", "right", "back", "down", "at_surface_3m") if n in views]
            top = np.concatenate([views[n][0] for n in names], 1)
            bots = []
            for n in names:
                d = views[n][1]
                if d is None:
                    bots.append(np.zeros((224, 224, 3), np.uint8))
                    continue
                dd = cv2.resize(np.clip(d, 0, 20), (224, 224))
                col = cv2.applyColorMap((255 - dd / 20 * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.putText(col, f"min {np.nanmin(d):.1f}m", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                bots.append(col)
            bot = np.concatenate(bots, 1)
            for i, n in enumerate(names):
                cv2.putText(top, n, (5 + 224 * i, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(out / "views" / f"{tag}.jpg"), np.concatenate([top, bot], 0))
        except Exception as exc:  # keep going on sim errors
            res = {"case": case, "error": repr(exc)}
            try:
                env._connect().simPause(False)
            except Exception:
                pass
        results.append(res)
        (out / "RESULTS.json").write_text(json.dumps(results, indent=1))
        print(tag, {k: res.get(k) for k in ("yaw_deg", "match_err", "impact_az_deg", "impact_el_deg", "impact_in_view",
                                            "surface_view_center_m", "error")},
              (res.get("hit") or {}).get("object_name"), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
