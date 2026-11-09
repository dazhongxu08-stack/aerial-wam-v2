"""Replay saved flights through the one-second near-view clearance.

Does not train, fly, or delete a candidate. Prints the distance from each
flown step to the nearest surface in the depth frame that still sees the body.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.aerial.rl.body_surround import BODY_RADIUS_M
from experiments.aerial.rl.near_clearance import HISTORY_S, NearClearance


def _state(xyz: np.ndarray, yaw: float) -> np.ndarray:
    st = np.zeros(7, np.float64)
    st[:3] = xyz
    st[6] = yaw
    return st


def _action(state: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    dx, dy = float(nxt[0] - state[0]), float(nxt[1] - state[1])
    c, s = float(np.cos(state[6])), float(np.sin(state[6]))
    fwd = c * dx + s * dy
    left = s * dx - c * dy
    return np.array([fwd, -left, float(nxt[2] - state[2]), 0.0], np.float64)


def replay_flight(name: str, traj: np.ndarray, depth: np.ndarray, yaw: np.ndarray,
                  history_s: float) -> list[dict]:
    mem = NearClearance(history_s=history_s)
    n = min(len(yaw), len(depth) - 1, len(traj) - 1)
    rows = []
    for i in range(n):
        st = _state(traj[i], float(yaw[i]))
        mem.add(depth[i], st, i / 5.0)
        got = mem.step(st, _action(st, traj[i + 1]), i / 5.0)
        rows.append({
            "flight": name,
            "history_s": history_s,
            "step": i,
            "move_m": round(float(_action(st, traj[i + 1])[0]), 3),
            "distance_m": None if np.isinf(got.distance_m) else round(got.distance_m, 3),
            "inside_body": bool(got.seen and got.distance_m < BODY_RADIUS_M),
            "age_s": round(got.age_s, 2),
        })
    return rows


def flight_from_dir(root: Path, route: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = np.load(root / f"traj_{route}.npz")["traj"]
    depth = np.load(root / f"last_frames_{route}.npz")["da3_depth56"]
    steps = json.loads((root / f"steps_{route}.json").read_text())
    yaw = np.array([float(s["yaw"]) for s in steps], np.float64)
    return traj, depth, yaw


def episode_sample(limit: int = 12) -> dict:
    """Forward steps in a few kept training flights. Not a training set."""
    root = Path("experiments/aerial/rl/artifacts/consequence_data_20261004")
    review = json.loads((root / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    kept = [e for e in review["episodes"] if e.get("keep") and e.get("file")
            and str(e["route_id"]).startswith("od_train_")]
    out = {"episodes": 0, "forward": 0, "inside_body": 0, "inside_hit": 0,
           "clear": 0, "clear_hit": 0, "unseen": 0}
    for e in kept[:limit]:
        path = root / "episodes" / e["file"]
        z = np.load(path)
        state = np.asarray(z["state"])
        action = np.asarray(z["action"])
        collided = np.asarray(z["collided"]).astype(bool)
        depth = np.load(path.with_suffix(".da3.npz"))["depth56"]
        mem = NearClearance(history_s=HISTORY_S)
        n = min(len(state), len(action), len(collided), len(depth))
        out["episodes"] += 1
        for t in range(n - 1):
            mem.add(depth[t], state[t], t / 5.0)
            if float(action[t, 0]) < 0.5:
                continue
            got = mem.step(state[t], action[t], t / 5.0)
            out["forward"] += 1
            hit = bool(collided[t + 1])
            if not got.seen:
                out["unseen"] += 1
                continue
            if got.distance_m < BODY_RADIUS_M:
                out["inside_body"] += 1
                if hit:
                    out["inside_hit"] += 1
            else:
                out["clear"] += 1
                if hit:
                    out["clear_hit"] += 1
    return out


def main() -> None:
    base = Path("artifacts")
    flights = [
        ("frozen_c01", base / "hard_f35_trim_unseen_complex_20261029", "unseen_c_01"),
        ("bothlast_c01", base / "hard_f35_trim_unseen_c01_bothlast_20261102", "unseen_c_01"),
    ]
    rows = []
    for name, root, route in flights:
        traj, depth, yaw = flight_from_dir(root, route)
        for history_s in (HISTORY_S, 3.0):
            rows.extend(replay_flight(name, traj, depth, yaw, history_s))
    print(json.dumps(rows), flush=True)
    print(json.dumps(episode_sample()), flush=True)


if __name__ == "__main__":
    main()
