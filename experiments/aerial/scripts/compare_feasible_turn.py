"""Read-only 15-step comparison on turnfix pre-wall states.

Rolls three fixed sequences through the frozen world model and scores them
with the feasible-side reward. Does not update weights and does not substitute
any action into a policy gradient.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from experiments.aerial.rl.env.action import body_delta_limits, forbid_backward_dx
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import advance_goal_rel_body, analytic_progress, goal_rel_body
from experiments.aerial.rl.reward import (
    RewardConfig,
    clearance_centering_bonus,
    clearance_m_along_action,
    clearance_risk_from_depth,
    efficiency_cost,
    feasible_side_progress,
    path_shaping_terms,
    reward_terms,
)
from experiments.aerial.rl.train_rl import load_torch_dynamics

HORIZON = 15
MAX_YAW = 8
ROOT = Path("/home/yao/aerial-wam-v2")
CFG_PATH = ROOT / "configs/_tmp_feasible_side_h125.yaml"
WM_PATH = ROOT / "experiments/aerial/rl/artifacts/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
TRAJ_DIR = ROOT / "experiments/aerial/rl/artifacts/feasible_side_turnfix_20260928/eval_noshield/traj"
OUT_PATH = (
    ROOT
    / "experiments/aerial/rl/artifacts/feasible_side_turnfix_20260928/turn_compare_15.json"
)


def load_reward(path: Path) -> RewardConfig:
    raw = yaml.safe_load(path.read_text())
    block = dict(raw.get("reward") or {})
    cfg = RewardConfig()
    for key, value in block.items():
        if not hasattr(cfg, key):
            continue
        cur = getattr(cfg, key)
        if isinstance(cur, bool) or isinstance(value, bool):
            setattr(cfg, key, bool(value))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            setattr(cfg, key, int(value))
        elif isinstance(cur, float):
            setattr(cfg, key, float(value))
        else:
            setattr(cfg, key, value)
    return cfg, raw


def _f(row: Dict[str, Any], key: str) -> Optional[float]:
    raw = row.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def building_ahead(fwd: float, left: float, right: float) -> bool:
    open_side = max(left, right)
    short_side = min(left, right)
    disparity = open_side >= short_side + 2.0
    return bool(disparity and ((fwd <= 18.0 and open_side >= fwd) or fwd <= 8.0))


def subsample(idxs: List[int], k: int) -> List[int]:
    if len(idxs) <= k:
        return list(idxs)
    pos = np.linspace(0, len(idxs) - 1, k)
    out: List[int] = []
    for p in pos:
        i = idxs[int(round(float(p)))]
        if i not in out:
            out.append(i)
    return out


def pick_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    street: List[int] = []
    ahead: List[int] = []
    last = max(0, len(rows) - 3)
    for i, row in enumerate(rows[:last]):
        fwd, left, right = _f(row, "d_fwd"), _f(row, "d_left"), _f(row, "d_right")
        if fwd is None or left is None or right is None:
            continue
        if building_ahead(fwd, left, right):
            ahead.append(i)
        elif fwd >= 18.0 and max(left, right) < fwd:
            street.append(i)
    chosen = subsample(ahead, 6)
    if street:
        chosen = [street[0]] + chosen
    picked = []
    for i in chosen:
        item = dict(rows[i])
        item["_kind"] = "street" if i in street[:1] and i == (street[0] if street else -1) and i not in ahead else "ahead"
        if i in ahead:
            item["_kind"] = "ahead"
        elif street and i == street[0]:
            item["_kind"] = "street"
        picked.append(item)
    return picked


def grab_rgb(client: Any, airsim: Any) -> np.ndarray:
    import cv2

    rq = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
    rs = client.simGetImages(rq)
    if not rs or rs[0].width <= 0 or not rs[0].image_data_uint8:
        raise RuntimeError("empty scene image")
    raw = np.frombuffer(rs[0].image_data_uint8, dtype=np.uint8).reshape(
        rs[0].height, rs[0].width, 3
    )
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def hats_of(dynamics: Any, z: np.ndarray) -> Optional[tuple]:
    fn = getattr(dynamics, "cone_hats", None)
    if not callable(fn):
        return None
    hats = fn(z)
    if hats is None:
        return None
    return float(hats[0]), float(hats[1]), float(hats[2])


def roll(
    dynamics: Any,
    z0: np.ndarray,
    g0: np.ndarray,
    actions: List[np.ndarray],
    cfg: RewardConfig,
) -> Dict[str, Any]:
    z = np.asarray(z0, dtype=np.float64).reshape(-1).copy()
    g = np.asarray(g0, dtype=np.float64).reshape(-1).copy()
    level_flags: List[bool] = []
    acc = {
        "return": 0.0,
        "sum_raw": 0.0,
        "sum_paid": 0.0,
        "sum_eff": 0.0,
        "sum_coll": 0.0,
        "sum_slide": 0.0,
        "sum_turn": 0.0,
        "sum_climb": 0.0,
        "sum_p": 0.0,
        "n": 0,
        "done_at": None,
    }
    end_hats = None
    for t, action in enumerate(actions):
        a = np.asarray(action, dtype=np.float64).reshape(4).copy()
        if bool(getattr(cfg, "forbid_backward_motion", False)):
            a = forbid_backward_dx(a)
        out = dynamics.step(z, a, goal_rel=g)
        prog = float(analytic_progress(g, a[:3]))
        cone_fwd = getattr(out, "d_fwd_cone_hat", None)
        if cone_fwd is not None and np.isfinite(float(cone_fwd)):
            side_fwd = float(cone_fwd)
        else:
            d_hat = getattr(out, "d_fwd_hat", None)
            side_fwd = float(d_hat) if d_hat is not None and np.isfinite(float(d_hat)) else None
        side = feasible_side_progress(
            prog,
            a,
            d_fwd=side_fwd,
            d_left=getattr(out, "d_left_hat", None),
            d_right=getattr(out, "d_right_hat", None),
            goal_rel=g,
            cfg=cfg,
        )
        prog_paid = float(side["progress"])
        level_step = (
            abs(float(a[2])) <= float(cfg.level_dz_thr_m)
            and abs(float(a[3])) <= float(cfg.level_dyaw_thr_rad)
            and prog_paid <= 0.0
        )
        level_flags.append(bool(level_step))
        wlen = max(1, int(cfg.level_window))
        if len(level_flags) > wlen:
            level_flags = level_flags[-wlen:]
        hist_level = len(level_flags) >= wlen and all(level_flags)
        shaping = path_shaping_terms(
            a,
            g,
            prog_paid,
            hist_level=hist_level,
            suppress_straight=bool(side["suppress_straight"]),
            turn_credit=float(side["feasible_side"]),
            cfg=cfg,
        )
        gxy = g[:2]
        yaw_err = float(np.arctan2(gxy[1], gxy[0])) if float(np.linalg.norm(gxy)) > 1e-6 else 0.0
        eff = efficiency_cost(a, yaw_err_rad=yaw_err, ds_true_m=float(prog), cfg=cfg)
        oc = getattr(out, "obstacle_cost", None)
        collision_risk = float(oc) if oc is not None and np.isfinite(float(oc)) else 0.0
        if bool(getattr(cfg, "blend_clearance_risk", False)):
            d_hat = getattr(out, "d_fwd_hat", None)
            cones = (
                {"forward": float(d_hat)}
                if d_hat is not None and np.isfinite(float(d_hat))
                else {}
            )
            d_along = clearance_m_along_action(a, cones, d_hat)
            if d_along is not None:
                clear_risk = clearance_risk_from_depth(
                    d_along,
                    d_danger=float(getattr(cfg, "near_miss_d_fwd_m", 3.0)),
                    d_clear=float(getattr(cfg, "near_miss_d_clear_m", 12.0)),
                )
                collision_risk = float(max(collision_risk, float(clear_risk)))
        if float(getattr(out, "p_coll", 0.0) or 0.0) >= 1.0 - 1e-6:
            collision_risk = max(collision_risk, float(getattr(cfg, "obstacle_cost_ceiling", 1.0)))
        d_hat = getattr(out, "d_fwd_hat", None)
        center = clearance_centering_bonus(
            a,
            d_fwd=float(d_hat) if d_hat is not None and np.isfinite(float(d_hat)) else None,
            d_left=getattr(out, "d_left_hat", None),
            d_right=getattr(out, "d_right_hat", None),
            progress=float(prog),
            cfg=cfg,
        )
        reward = reward_terms(
            float(shaping["progress_eff"]),
            collision_risk,
            float(np.linalg.norm(a)),
            cfg,
            efficiency_cost_val=float(eff["efficiency_cost"]),
            path_shaping_val=float(shaping["path_shaping"]),
            clearance_centering_val=float(center["clearance_centering"]),
            climb_cost=float(side["climb_cost"]),
            slide_cost=float(side["slide_cost"]),
        )["reward"]
        acc["return"] += reward
        acc["sum_raw"] += prog
        acc["sum_paid"] += prog_paid
        acc["sum_eff"] += float(shaping["progress_eff"])
        acc["sum_coll"] += collision_risk
        acc["sum_slide"] += float(side["slide_cost"])
        acc["sum_turn"] += float(side["feasible_side"])
        acc["sum_climb"] += float(side["climb_cost"])
        acc["sum_p"] += float(getattr(out, "p_coll", 0.0) or 0.0)
        acc["n"] += 1
        end_hats = hats_of(dynamics, np.asarray(out.z_next, dtype=np.float64).reshape(-1))
        z = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
        g = np.asarray(advance_goal_rel_body(g, a), dtype=np.float64).reshape(-1)
        if bool(getattr(out, "done", False)):
            acc["done_at"] = t
            break
    n = max(1, int(acc["n"]))
    acc["mean_p"] = acc["sum_p"] / n
    if end_hats is None:
        acc["end_fwd"] = acc["end_left"] = acc["end_right"] = None
    else:
        acc["end_fwd"], acc["end_left"], acc["end_right"] = end_hats
    acc["end_dist"] = float(g[3]) if g.size > 3 else float(np.linalg.norm(g[:3]))
    return acc


def sequences(
    limits: np.ndarray,
    dy: float,
    open_sign: float,
    dynamics: Any,
    z0: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
    strafe = np.clip(np.array([0.0, dy, 0.0, 0.0], dtype=np.float64), -lim, lim)
    yaw = np.clip(np.array([0.0, 0.0, 0.0, open_sign * 0.25], dtype=np.float64), -lim, lim)
    fwd = np.clip(np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -lim, lim)
    yaw_steps = HORIZON
    z = np.asarray(z0, dtype=np.float64).reshape(-1).copy()
    opened_at = None
    for t in range(MAX_YAW):
        out = dynamics.step(z, yaw)
        z = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
        hats = hats_of(dynamics, z)
        if hats is not None and hats[0] >= max(hats[1], hats[2]):
            opened_at = t + 1
            yaw_steps = opened_at
            break
    yawgo = [yaw.copy() for _ in range(yaw_steps)] + [fwd.copy() for _ in range(HORIZON - yaw_steps)]
    return {
        "strafe": [strafe.copy() for _ in range(HORIZON)],
        "yaw": [yaw.copy() for _ in range(HORIZON)],
        "yawgo": yawgo,
    }, opened_at


def main() -> None:
    import airsim

    cfg, raw = load_reward(CFG_PATH)
    limits = body_delta_limits(0.2)
    dynamics, _ = load_torch_dynamics(
        raw.get("world_model") or {},
        str(WM_PATH),
        device="cuda",
        freeze=True,
    )
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    results: List[Dict[str, Any]] = []
    try:
        for name in ("route00.jsonl", "route01.jsonl"):
            rows = [json.loads(line) for line in (TRAJ_DIR / name).read_text().splitlines() if line.strip()]
            picked = pick_rows(rows)
            print(f"{name} picked {len(picked)}", flush=True)
            for row in picked:
                pos = np.asarray(row["pos"], dtype=np.float64).reshape(3)
                yaw = float(np.radians(float(row["yaw_deg"])))
                intent = row.get("intent_target")
                if not isinstance(intent, (list, tuple)) or len(intent) < 3:
                    print(f"  skip step {row.get('step')} no intent", flush=True)
                    continue
                goal = np.asarray(intent[:3], dtype=np.float64)
                q = airsim.to_quaternion(0.0, 0.0, yaw)
                ned = airsim.Vector3r(float(pos[0]), float(pos[1]), float(-pos[2]))
                client.simSetVehiclePose(airsim.Pose(ned, q), ignore_collision=True)
                time.sleep(0.25)
                rgb = grab_rgb(client, airsim)
                state = np.array(
                    [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, yaw], dtype=np.float32
                )
                z0 = np.asarray(
                    dynamics.encode(Observation(rgb=rgb, state=state)), dtype=np.float64
                ).reshape(-1)
                g0 = np.asarray(goal_rel_body(pos, yaw, goal), dtype=np.float64).reshape(-1)
                start_hats = hats_of(dynamics, z0)
                left, right = _f(row, "d_left"), _f(row, "d_right")
                if left is None or right is None:
                    if start_hats is None:
                        print(f"  skip step {row.get('step')} no cones", flush=True)
                        continue
                    open_sign = 1.0 if start_hats[1] >= start_hats[2] else -1.0
                else:
                    open_sign = 1.0 if left >= right else -1.0
                dy = _f(row, "act_dy")
                if dy is None:
                    dy = 0.3 * open_sign
                seqs, opened_at = sequences(limits, dy, open_sign, dynamics, z0)
                scored = {
                    key: roll(dynamics, z0, g0, actions, cfg) for key, actions in seqs.items()
                }
                rec = {
                    "route": name,
                    "kind": row["_kind"],
                    "step": row.get("step"),
                    "pos": pos.tolist(),
                    "z": float(pos[2]),
                    "yaw_deg": float(row["yaw_deg"]),
                    "logged_fwd": _f(row, "d_fwd"),
                    "logged_left": left,
                    "logged_right": right,
                    "act_dy": dy,
                    "open_sign": open_sign,
                    "start_hats": None if start_hats is None else list(start_hats),
                    "yawgo_opened_at": opened_at,
                    "start_dist": float(g0[3]) if g0.size > 3 else None,
                    "scores": scored,
                }
                results.append(rec)
                s = scored
                print(
                    f"  {row['_kind']} step={row.get('step')} "
                    f"F={_f(row, 'd_fwd')} L={left} R={right} "
                    f"strafe={s['strafe']['return']:.3f} yaw={s['yaw']['return']:.3f} "
                    f"yawgo={s['yawgo']['return']:.3f} opened={opened_at}",
                    flush=True,
                )
    finally:
        try:
            client.enableApiControl(False)
        except Exception:
            pass
    OUT_PATH.write_text(json.dumps(results, indent=2))
    ahead = [r for r in results if r["kind"] == "ahead"]
    gaps = [r["scores"]["yawgo"]["return"] - r["scores"]["strafe"]["return"] for r in ahead]
    print(f"wrote {OUT_PATH} n_ahead={len(ahead)}", flush=True)
    if gaps:
        print(
            f"yawgo-strafe median={float(np.median(gaps)):.3f} "
            f"n_gt_0.05={sum(g > 0.05 for g in gaps)} "
            f"n_gt_0.5={sum(g > 0.5 for g in gaps)} n={len(gaps)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
