"""Count existing flights that already contain a surround-map window.

A window is a step whose 8 m body map holds a surface in the forward corridor
(between 0.8 m and 4 m ahead, within the 0.4 m body radius) while an in-place
yaw does not reach it. Val routes are reported apart from the train pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.aerial.rl.body_surround import (
    BODY_RADIUS_M,
    BodySurround,
    body_from_world,
    in_view,
)

VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
AHEAD = (0.8, 4.0)
GAP = 5


def thirds(depth: np.ndarray) -> tuple[float, float, float, float]:
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    rows = d[h // 4: h - h // 4]

    def near(block: np.ndarray) -> float:
        vals = block[np.isfinite(block) & (block > 0.05)]
        return float(vals.min()) if vals.size else 100.0

    left, right = near(rows[:, : w // 3]), near(rows[:, w - w // 3:])
    c0, c1 = w // 3, w - w // 3
    span = c1 - c0
    inner = near(rows[:, c0 + span // 4: c1 - span // 4])
    center = near(rows[:, c0:c1])
    return left, center, right, inner


def side_open(depth: np.ndarray) -> bool:
    left, center, right, inner = thirds(depth)
    if center > 4.0 or inner < 8.0:
        return False
    return (left <= 4.0 and right >= 8.0) or (right <= 4.0 and left >= 8.0)


def corridor(mem: BodySurround, state: np.ndarray) -> tuple[int, int]:
    """Surfaces in the nose corridor, split by whether the current view sees them."""
    mem.forget(state)
    if not mem.cells:
        return 0, 0
    pts = np.stack([p for p, _ in mem.cells.values()])
    fwd, left, dz = body_from_world(state, pts)
    use = (fwd > AHEAD[0]) & (fwd <= AHEAD[1]) & (np.abs(left) <= BODY_RADIUS_M) & (np.abs(dz) <= 8.0)
    if not np.any(use):
        return 0, 0
    seen = in_view(fwd[use], left[use])
    return int((~seen).sum()), int(seen.sum())


def blind_near(mem: BodySurround, state: np.ndarray) -> int:
    mem.forget(state)
    if not mem.cells:
        return 0
    pts = np.stack([p for p, _ in mem.cells.values()])
    fwd, left, dz = body_from_world(state, pts)
    rng = np.sqrt(fwd * fwd + left * left + dz * dz)
    use = (np.abs(fwd) <= 8.0) & (np.abs(left) <= 8.0) & (np.abs(dz) <= 8.0) & (rng <= 8.0)
    if not np.any(use):
        return 0
    return int((~in_view(fwd[use], left[use])).sum())


def main() -> None:
    root = Path("experiments/aerial/rl/artifacts/consequence_data_20261004")
    review = json.loads((root / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    kept = [e for e in review["episodes"] if e.get("keep") and e.get("file") and str(e["route_id"]).startswith("od_train_")]
    stats = {k: {"episodes": 0, "steps": 0, "blind": 0, "corridor": 0, "corridor_blind": 0,
                 "corridor_side": 0, "episodes_corridor": 0, "side_routes": {}} for k in ("train", "val")}
    for i, e in enumerate(kept):
        split = "val" if e["route_id"] in VAL else "train"
        st = stats[split]
        st["episodes"] += 1
        path = root / "episodes" / e["file"]
        z = np.load(path)
        state = np.asarray(z["state"])
        action = np.asarray(z["action"])
        collided = np.asarray(z["collided"]).astype(bool)
        depth = np.load(path.with_suffix(".da3.npz"))["depth56"]
        mem = BodySurround()
        cool = 0
        ep_hit = False
        n = min(len(state), len(depth))
        for t in range(n):
            mem.add_depth(depth[t], state[t], t / 5.0, stride=2)
            st["steps"] += 1
            if cool > 0:
                cool -= 1
                continue
            blind = blind_near(mem, state[t])
            out_view, in_frame = corridor(mem, state[t])
            hit = False
            if blind:
                st["blind"] += 1
                hit = True
                if split == "train" and t + 1 < len(collided):
                    flown = st.setdefault("flown", {"n": 0, "hit_next": 0, "hit_3": 0,
                                                    "hit_fwd": 0, "hit_turn": 0, "already": 0})
                    flown["n"] += 1
                    if bool(collided[t]):
                        flown["already"] += 1
                    dx = float(action[t, 0])
                    if bool(collided[t + 1]):
                        flown["hit_next"] += 1
                        if dx >= 0.5:
                            flown["hit_fwd"] += 1
                        if abs(float(action[t, 3])) >= 0.1 and dx < 0.25:
                            flown["hit_turn"] += 1
                    if bool(collided[t + 1:t + 4].any()):
                        flown["hit_3"] += 1
            if out_view or in_frame:
                st["corridor"] += 1
                hit = True
                ep_hit = True
            if out_view:
                st["corridor_blind"] += 1
            if (out_view or in_frame) and side_open(depth[t]):
                st["corridor_side"] += 1
                routes = st["side_routes"]
                routes[e["route_id"]] = routes.get(e["route_id"], 0) + 1
            if hit:
                cool = GAP
        if ep_hit:
            st["episodes_corridor"] += 1
        if (i + 1) % 40 == 0:
            print("scanned", i + 1, "/", len(kept), flush=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
