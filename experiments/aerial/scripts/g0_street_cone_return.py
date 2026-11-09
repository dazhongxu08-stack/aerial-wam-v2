#!/usr/bin/env python3
"""15-step return on side-near poses, collision cost from the street depth head.

The world-model obstacle cost is not used. Each action is executed as a
kinematic pose, the camera is grabbed there, and forward/left/right come from
the street image head. Policy weights are not updated.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.aerial.rl.depth_geometry import approach_clearances, deploy_cones
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import (
    advance_goal_rel_body,
    analytic_progress,
    goal_rel_body,
)
from experiments.aerial.rl.reward import (
    RewardConfig,
    clearance_centering_bonus,
    clearance_risk_from_depth,
    efficiency_cost,
    path_shaping_terms,
    reward_terms,
)

ROOT = Path("/data/aerial-wam-v2")
TRAJ = (
    ROOT
    / "experiments/aerial/rl/artifacts/g0_leave_streetdepth_20261005/traj/route00.jsonl"
)
CFG = ROOT / "configs/_tmp_accept_soft_h125.yaml"
DEPTH = (
    ROOT
    / "experiments/aerial/rl/artifacts/da3_yaw_street_20261005/depth_yaw_street.pt"
)
OUT = ROOT / "experiments/aerial/rl/artifacts/g0_street_cone_return_20261005"
GOAL_XY = np.array([-952.6, 15.9], dtype=np.float64)
LIMITS = np.array([1.0, 0.4, 0.4, math.pi / 10.0], dtype=np.float64)
HORIZON = 15
YAW_STEPS = 5


def _wrap_deg(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _quat_yaw_deg(q) -> float:
    import airsim

    pitch, roll, yaw = airsim.to_eularian_angles(q)
    del pitch, roll
    return float(math.degrees(yaw))


def _lock_pose(client, x, y, z_up, yaw_deg):
    import airsim

    quat = airsim.to_quaternion(0.0, 0.0, math.radians(float(yaw_deg)))
    pose = airsim.Pose(airsim.Vector3r(float(x), float(y), -float(z_up)), quat)

    def _seat():
        client.simPause(True)
        client.simSetVehiclePose(pose, True)
        time.sleep(0.05)
        got = client.simGetVehiclePose()
        z_err = abs(float(got.position.z_val) - (-float(z_up)))
        yaw_err = abs(_wrap_deg(_quat_yaw_deg(got.orientation) - float(yaw_deg)))
        return z_err, yaw_err

    z_err, yaw_err = _seat()
    if z_err > 1.0 or yaw_err > 3.0:
        client.simPause(False)
        client.simSetVehiclePose(pose, True)
        time.sleep(0.05)
        z_err, yaw_err = _seat()
    if z_err > 1.0 or yaw_err > 3.0:
        raise RuntimeError(
            f"pose lock failed z_err={z_err:.2f} yaw_err={yaw_err:.1f}"
        )


def _grab_rgb(client):
    import airsim

    scene = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
    )[0]
    bgr = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(
        scene.height, scene.width, 3
    )
    return bgr[:, :, ::-1].copy()


def _grab_gt(client):
    import airsim

    depth_resp = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
    )[0]
    depth = airsim.list_to_2d_float_array(
        depth_resp.image_data_float, depth_resp.width, depth_resp.height
    )
    return np.asarray(depth, dtype=np.float32)


def _head_cones(pred, rgb):
    pred.reset()
    obs = Observation(rgb=rgb, state=np.zeros(7, dtype=np.float32))
    depth = pred.predict_depth(obs)
    if depth is None:
        raise RuntimeError("street head returned no depth")
    raw = approach_clearances(np.asarray(depth, dtype=np.float64))
    scaled = deploy_cones(np.asarray(depth, dtype=np.float64))
    return raw, scaled


def _rank_agree(a, b) -> bool:
    """True when both pairs agree which side is farther. Both-far is agreement."""
    la, ra = float(a[1]), float(a[2])
    lb, rb = float(b[1]), float(b[2])
    if min(la, ra, lb, rb) <= 0:
        return False
    return (la - ra) * (lb - rb) >= 0.0


def _apply(x, y, z, yaw_deg, action):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz, dyaw = (float(v) for v in action)
    return (
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        z + dz,
        _wrap_deg(yaw_deg + math.degrees(dyaw)),
    )


def _sequences(dy: float, open_sign: float):
    lim = LIMITS
    strafe = np.clip(np.array([0.0, dy, 0.0, 0.0]), -lim, lim)
    strafe[0] = max(0.0, float(strafe[0]))
    peel = np.clip(np.array([0.2, open_sign * 0.4, 0.0, 0.0]), -lim, lim)
    peel[0] = max(0.0, float(peel[0]))
    yaw = np.clip(np.array([0.0, 0.0, 0.0, open_sign * lim[3]]), -lim, lim)
    fwd = np.clip(np.array([0.7, 0.0, 0.0, 0.0]), -lim, lim)
    fwd[0] = max(0.0, float(fwd[0]))
    return {
        "strafe": [strafe] * HORIZON,
        "peel": [peel] * HORIZON,
        "yawgo": [yaw] * YAW_STEPS + [fwd] * (HORIZON - YAW_STEPS),
    }


def _select(rows):
    cands = []
    for row in rows:
        left, right, fwd = float(row["d_left"]), float(row["d_right"]), float(row["d_fwd"])
        near, far = min(left, right), max(left, right)
        if near <= 24.0 and far >= near + 30.0 and fwd >= near + 8.0:
            cands.append(row)
    if len(cands) < 3:
        raise RuntimeError(f"need 3 side-near poses, found {len(cands)}")
    mid = cands[len(cands) // 2]
    return [cands[0], mid, cands[-1]]


def _roll(client, pred, cfg, pose, actions, goal):
    x, y, z, yaw_deg = pose
    g = goal_rel_body(np.array([x, y, z]), math.radians(yaw_deg), goal)
    total = 0.0
    progress_sum = 0.0
    risk_sum = 0.0
    trace = []
    level_flags = []
    for t, action in enumerate(actions):
        _lock_pose(client, x, y, z, yaw_deg)
        rgb = _grab_rgb(client)
        gt = _grab_gt(client)
        raw, scaled = _head_cones(pred, rgb)
        gt_raw = approach_clearances(np.asarray(gt, dtype=np.float64))
        cones = {
            "forward": float(scaled["forward"]),
            "left": float(scaled["left"]),
            "right": float(scaled["right"]),
        }
        near = min(cones["forward"], cones["left"], cones["right"])
        risk = clearance_risk_from_depth(
            near,
            d_danger=float(cfg.near_miss_d_fwd_m),
            d_clear=float(cfg.near_miss_d_clear_m),
        )
        prog = analytic_progress(g, np.asarray(action[:3], dtype=np.float64))
        yaw_err = 0.0
        if float(np.hypot(g[0], g[1])) > 1e-6:
            yaw_err = float(math.atan2(float(g[1]), float(g[0])))
        eff = efficiency_cost(action, yaw_err_rad=yaw_err, ds_true_m=float(prog), cfg=cfg)
        shaping = path_shaping_terms(
            action, g, float(prog), hist_level=False, cfg=cfg,
        )
        center = clearance_centering_bonus(
            action,
            d_fwd=cones["forward"],
            d_left=cones["left"],
            d_right=cones["right"],
            progress=float(prog),
            cfg=cfg,
        )
        level_step = (
            abs(float(action[2])) <= float(cfg.level_dz_thr_m)
            and abs(float(action[3])) <= float(cfg.level_dyaw_thr_rad)
            and float(shaping["progress_eff"]) <= 0.0
        )
        level_flags.append(bool(level_step))
        wlen = max(1, int(cfg.level_window))
        hist = level_flags[-wlen:]
        if len(hist) >= wlen and all(hist):
            shaping = path_shaping_terms(
                action, g, float(prog), hist_level=True, cfg=cfg,
            )
        step_r = reward_terms(
            float(shaping["progress_eff"]),
            risk,
            float(np.linalg.norm(action)),
            cfg,
            efficiency_cost_val=float(eff["efficiency_cost"]),
            path_shaping_val=float(shaping["path_shaping"]),
            clearance_centering_val=float(center["clearance_centering"]),
        )["reward"]
        total += float(step_r)
        progress_sum += float(prog)
        risk_sum += float(risk)
        if t in (0, YAW_STEPS - 1, YAW_STEPS, HORIZON - 1):
            trace.append(
                {
                    "t": t,
                    "yaw_deg": round(yaw_deg, 1),
                    "fwd": round(cones["forward"], 2),
                    "left": round(cones["left"], 2),
                    "right": round(cones["right"], 2),
                    "gt_fwd": round(float(gt_raw[0]), 2),
                    "gt_left": round(float(gt_raw[1]), 2),
                    "gt_right": round(float(gt_raw[2]), 2),
                    "rank_agree": bool(_rank_agree(raw, gt_raw)),
                    "risk": round(float(risk), 3),
                    "prog": round(float(prog), 3),
                }
            )
        g = advance_goal_rel_body(g, action)
        x, y, z, yaw_deg = _apply(x, y, z, yaw_deg, action)
    return {
        "return": float(total),
        "progress": float(progress_sum),
        "risk": float(risk_sum),
        "trace": trace,
    }


def _measure_at(client, pred, x, y, z, yaw_deg):
    _lock_pose(client, x, y, z, yaw_deg)
    rgb = _grab_rgb(client)
    gt = _grab_gt(client)
    raw, scaled = _head_cones(pred, rgb)
    gt_raw = approach_clearances(np.asarray(gt, dtype=np.float64))
    return {
        "yaw_deg": round(float(yaw_deg), 1),
        "fwd": round(float(scaled["forward"]), 2),
        "left": round(float(scaled["left"]), 2),
        "right": round(float(scaled["right"]), 2),
        "gt_fwd": round(float(gt_raw[0]), 2),
        "gt_left": round(float(gt_raw[1]), 2),
        "gt_right": round(float(gt_raw[2]), 2),
        "rank_agree": bool(_rank_agree(raw, gt_raw)),
    }


def small_yaw() -> None:
    """In-place yaw of 0/18/36/54° toward the open side. No translation."""
    rows = [json.loads(line) for line in TRAJ.read_text().splitlines() if line.strip()]
    chosen = _select(rows)
    pred = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    results = []
    try:
        for row in chosen:
            pos = [float(v) for v in row["pos"]]
            z_up = float(pos[2]) if float(pos[2]) > 0.0 else -float(pos[2])
            yaw0 = float(row["yaw_deg"])
            left, right = float(row["d_left"]), float(row["d_right"])
            open_sign = 1.0 if left >= right else -1.0
            views = []
            for deg in (0.0, 18.0, 36.0, 54.0):
                yaw = _wrap_deg(yaw0 + open_sign * deg)
                view = _measure_at(client, pred, float(pos[0]), float(pos[1]), z_up, yaw)
                view["offset_deg"] = deg
                views.append(view)
                print(
                    f"step {int(row['step'])} off {deg:.0f} yaw {view['yaw_deg']:.1f} "
                    f"head F{view['fwd']:.1f} L{view['left']:.1f} R{view['right']:.1f} "
                    f"gt F{view['gt_fwd']:.1f} L{view['gt_left']:.1f} R{view['gt_right']:.1f} "
                    f"agree {view['rank_agree']}",
                    flush=True,
                )
            base_gt = float(views[0]["gt_fwd"])
            base_head = float(views[0]["fwd"])
            results.append(
                {
                    "step": int(row["step"]),
                    "open_sign": open_sign,
                    "views": views,
                    "gt_fwd_rose": any(float(v["gt_fwd"]) > base_gt + 2.0 for v in views[1:]),
                    "head_fwd_rose": any(float(v["fwd"]) > base_head + 2.0 for v in views[1:]),
                }
            )
    finally:
        try:
            client.simPause(False)
        except Exception:
            pass
        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "small_yaw.json").write_text(json.dumps(results, indent=2))
    rose = sum(1 for r in results if r["gt_fwd_rose"])
    print(f"GT_FWD_ROSE {rose}/{len(results)}", flush=True)


def main() -> None:
    rows = [json.loads(line) for line in TRAJ.read_text().splitlines() if line.strip()]
    chosen = _select(rows)
    reward_raw = yaml.safe_load(CFG.read_text()).get("reward") or {}
    fields = {f.name for f in dataclasses.fields(RewardConfig)}
    cfg = RewardConfig(**{k: v for k, v in reward_raw.items() if k in fields})
    cfg.feasible_side_progress = False
    pred = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    results = []
    try:
        for row in chosen:
            pos = [float(v) for v in row["pos"]]
            z_up = float(pos[2]) if float(pos[2]) > 0.0 else -float(pos[2])
            yaw_deg = float(row["yaw_deg"])
            left, right = float(row["d_left"]), float(row["d_right"])
            open_sign = 1.0 if left >= right else -1.0
            goal = np.array([GOAL_XY[0], GOAL_XY[1], z_up], dtype=np.float64)
            pose = (float(pos[0]), float(pos[1]), z_up, yaw_deg)
            seqs = _sequences(float(row["act_dy"]), open_sign)
            one = {
                "step": int(row["step"]),
                "logged_fwd": float(row["d_fwd"]),
                "logged_left": left,
                "logged_right": right,
                "open_sign": open_sign,
                "z_up": z_up,
            }
            for name, actions in seqs.items():
                one[name] = _roll(client, pred, cfg, pose, actions, goal)
                print(
                    f"step {one['step']} {name} return={one[name]['return']:.3f} "
                    f"prog={one[name]['progress']:.2f} risk={one[name]['risk']:.2f}",
                    flush=True,
                )
            yaw_r = one["yawgo"]["return"]
            str_r = one["strafe"]["return"]
            one["yaw_minus_strafe"] = float(yaw_r - str_r)
            one["yaw_wins"] = bool(yaw_r > str_r + 0.05)
            results.append(one)
    finally:
        try:
            client.simPause(False)
        except Exception:
            pass
        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    wins = sum(1 for r in results if r["yaw_wins"])
    summary = {
        "n": len(results),
        "yaw_wins": wins,
        "feasible_side_progress": False,
        "collision": "street head min(forward, left, right)",
        "poses": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"YAW_WINS {wins}/{len(results)}", flush=True)


if __name__ == "__main__":
    import sys

    if "--small-yaw" in sys.argv:
        small_yaw()
    else:
        main()
