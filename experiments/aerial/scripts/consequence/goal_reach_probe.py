#!/usr/bin/env python3
"""Is the goal of a stalled route reachable in the simulator, regardless of the model?

For each route: (1) sit at the goal and record contact plus true depth over 8 headings; (2) from up to
``--n-starts`` positions the MPC flights actually reached within ``--near-m`` of the goal (spread at least
``--spacing-m`` apart), face the goal and fly turn-then-forward straight at it, no model involved, until
within ``--success-dist``, a collision, or ``--max-steps``. Records outcome, contact object and the minimum
true depth along the way. Writes GOAL_REACH.json in --out-dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--flight-dirs", required=True, help="comma list of eval out-dirs with traj_/steps_ files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--near-m", type=float, default=7.0)
    ap.add_argument("--spacing-m", type=float, default=1.5)
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--success-dist", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--speed", type=float, default=0.6)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import yaml

    from experiments.aerial.rl.collect_consequence_dataset import LIMITS, goal_bearing_body
    from experiments.aerial.rl.train_rl import _build_env

    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    routes = {r["route_id"]: r for r in json.loads((root / args.annotation).read_text())["episodes"]}
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(args), "routes": {}}

    def contact():
        return getattr(env, "collision_info", lambda: {})()

    for rid in args.routes.split(","):
        r = routes[rid]
        goal = np.asarray(r["pos"][-1], np.float64)
        z = r.get("complexity_meta", {}).get("spawn_z_ok")
        if z is not None:
            goal[2] = float(z)
        rec = {"goal": goal.tolist(), "at_goal": [], "approaches": []}
        for k in range(8):
            yaw = k * math.pi / 4
            try:
                obs = env.reset({"pos": [goal.tolist(), goal.tolist()], "yaw": [yaw, yaw]})
            except RuntimeError as exc:
                rec["at_goal"].append({"yaw_deg": k * 45, "reset_failed": str(exc)[:200]})
                continue
            dep = np.asarray(obs.depth, np.float32)
            c = contact()
            rec["at_goal"].append({"yaw_deg": k * 45, "depth_min": round(float(dep.min()), 2),
                                   "depth_center": round(float(np.median(dep[dep.shape[0] // 3: 2 * dep.shape[0] // 3,
                                                                          dep.shape[1] // 3: 2 * dep.shape[1] // 3])), 2),
                                   "contact": c.get("object_name") if c.get("has_collided") else None})
        pts = []
        for d in args.flight_dirs.split(","):
            for f in glob.glob(str(root / d / f"traj_{rid}.npz")):
                t = np.load(f)["traj"]
                for p in t:
                    if np.linalg.norm(p - goal) <= args.near_m and all(np.linalg.norm(p - q) >= args.spacing_m for q in pts):
                        pts.append(np.asarray(p, np.float64))
        pts = sorted(pts, key=lambda p: -np.linalg.norm(p - goal))
        if len(pts) > args.n_starts:
            pts = [pts[int(i)] for i in np.linspace(0, len(pts) - 1, args.n_starts)]
        for p in pts:
            yaw0 = math.atan2(goal[1] - p[1], goal[0] - p[0])
            try:
                obs = env.reset({"pos": [p.tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
            except RuntimeError as exc:
                rec["approaches"].append({"start": [round(float(x), 2) for x in p], "reset_failed": str(exc)[:200]})
                continue
            reset_c = contact()
            dmin, end, obj = float(np.asarray(obs.depth).min()), "max_steps", None
            for step in range(args.max_steps):
                st = np.asarray(obs.state, np.float64)
                b = goal_bearing_body(st[:3], float(st[6]), goal)
                yaw_rate = float(np.clip(b * 5.0, -LIMITS[3], LIMITS[3]))
                vx = args.speed if abs(b) < 0.35 else 0.1
                vz = float(np.clip(goal[2] - st[2], -0.4, 0.4))
                obs, _ = env.step(np.array([vx, 0.0, vz, yaw_rate]))
                dmin = min(dmin, float(np.asarray(obs.depth).min()))
                if bool(obs.collided):
                    end, obj = "collision", contact().get("object_name")
                    break
                if np.linalg.norm(np.asarray(obs.position) - goal) <= args.success_dist:
                    end = "arrived"
                    break
            rec["approaches"].append({"start": [round(float(x), 2) for x in p],
                                      "start_d": round(float(np.linalg.norm(p - goal)), 2),
                                      "reset_contact": reset_c.get("object_name") if reset_c.get("has_collided") else None,
                                      "end": end, "object": obj, "steps": step + 1,
                                      "d_final": round(float(np.linalg.norm(np.asarray(obs.position) - goal)), 2),
                                      "true_depth_min": round(dmin, 2)})
        res["routes"][rid] = rec
        (out / "GOAL_REACH.json").write_text(json.dumps(res, indent=1))
        print(rid, json.dumps({"at_goal_min_depth": min([a["depth_min"] for a in rec["at_goal"] if "depth_min" in a] or [None]),
                               "approaches": [(a.get("start_d"), a.get("end", "reset_failed"), a.get("object"), a.get("d_final"))
                                              for a in rec["approaches"]]}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
