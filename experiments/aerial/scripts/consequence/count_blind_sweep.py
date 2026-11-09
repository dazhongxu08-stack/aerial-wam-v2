"""Count steps whose body sweeps a remembered surface outside the current view.

The nose corridor from 0.8 m to 4 m stays inside the 90 degree picture.
The blind spot is closer: a surface beside the nose, already outside the
picture, that this step's body still covers. Points behind the body are the
wake already flown over, and are counted separately. This does not train
and does not delete a candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.aerial.rl.body_surround import (
    BODY_RADIUS_M,
    CELL_M,
    HALF_M,
    BodySurround,
    body_from_world,
    in_view,
    step_cells,
)

VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
# Camera sits 1 m below the body. Keep the wall at that band; drop the ground.
WALL_DZ = (-1.2, 0.8)
NOSE_FWD_M = 0.15


def _segment_distance(fwd: np.ndarray, left: np.ndarray, end_f: float, end_l: float) -> np.ndarray:
    length2 = end_f * end_f + end_l * end_l
    if length2 < 1e-8:
        return np.hypot(fwd, left)
    scale = np.clip((fwd * end_f + left * end_l) / length2, 0.0, 1.0)
    return np.hypot(fwd - scale * end_f, left - scale * end_l)


def sweep_flags(mem: BodySurround, state: np.ndarray, action: np.ndarray) -> tuple[bool, bool]:
    """``(wake, nose_wall)`` for remembered points outside the current picture.

    ``wake`` is any covered point outside the picture, including the ground
    and surfaces already behind the body. ``nose_wall`` is a point in front
    of the nose, at body height, within 0.4 m of this step, and outside the
    90 degree picture.
    """
    mem.forget(state)
    cover = step_cells(state, action, BODY_RADIUS_M)
    if not cover or not mem.cells:
        return False, False
    pts = np.stack([p for p, _ in mem.cells.values()])
    fwd, left, dz = body_from_world(state, pts)
    i = np.floor((fwd + HALF_M) / CELL_M).astype(np.int64)
    j = np.floor((left + HALF_M) / CELL_M).astype(np.int64)
    mask = np.zeros((32, 32), np.bool_)
    for ii, jj in cover:
        if 0 <= ii < 32 and 0 <= jj < 32:
            mask[ii, jj] = True
    inside = (np.abs(dz) <= 8.0) & (i >= 0) & (i < 32) & (j >= 0) & (j < 32)
    if not np.any(inside):
        return False, False
    covered = np.zeros(len(fwd), np.bool_)
    covered[inside] = mask[i[inside], j[inside]]
    oov = covered & (~in_view(fwd, left))
    if not np.any(oov):
        return False, False
    action = np.asarray(action, np.float64)
    dist = _segment_distance(fwd[oov], left[oov], float(action[0]), -float(action[1]))
    wall = (fwd[oov] > NOSE_FWD_M) & (dist <= BODY_RADIUS_M)
    wall &= (dz[oov] >= WALL_DZ[0]) & (dz[oov] <= WALL_DZ[1])
    return True, bool(np.any(wall))


def _blank() -> dict:
    return {
        "episodes": 0,
        "steps": 0,
        "fwd_steps": 0,
        "hit_next": 0,
        "hit_fwd": 0,
        "wake_steps": 0,
        "wake_hit_next": 0,
        "wake_hit_fwd": 0,
        "nose_steps": 0,
        "nose_episodes": 0,
        "nose_hit_next": 0,
        "nose_hit_fwd": 0,
        "nose_hit_3": 0,
        "fwd_nose": 0,
        "fwd_plain": 0,
        "fwd_nose_hit": 0,
        "fwd_plain_hit": 0,
    }


def main() -> None:
    root = Path("experiments/aerial/rl/artifacts/consequence_data_20261004")
    review = json.loads((root / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    kept = [e for e in review["episodes"] if e.get("keep") and e.get("file") and str(e["route_id"]).startswith("od_train_")]
    stats = {k: _blank() for k in ("train", "val")}
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
        nose_ep = False
        n = min(len(state), len(action), len(collided), len(depth))
        for t in range(n - 1):
            mem.add_depth(depth[t], state[t], t / 5.0, stride=2)
            st["steps"] += 1
            dx = float(action[t, 0])
            forward = dx >= 0.5
            if forward:
                st["fwd_steps"] += 1
            wake, nose = sweep_flags(mem, state[t], action[t])
            hit = bool(collided[t + 1])
            if hit:
                st["hit_next"] += 1
                if forward:
                    st["hit_fwd"] += 1
            if wake:
                st["wake_steps"] += 1
                if hit:
                    st["wake_hit_next"] += 1
                    if forward:
                        st["wake_hit_fwd"] += 1
            if nose:
                nose_ep = True
                st["nose_steps"] += 1
                if hit:
                    st["nose_hit_next"] += 1
                    if forward:
                        st["nose_hit_fwd"] += 1
                if bool(collided[t + 1:t + 4].any()):
                    st["nose_hit_3"] += 1
            if forward:
                if nose:
                    st["fwd_nose"] += 1
                    if hit:
                        st["fwd_nose_hit"] += 1
                else:
                    st["fwd_plain"] += 1
                    if hit:
                        st["fwd_plain_hit"] += 1
        if nose_ep:
            st["nose_episodes"] += 1
        if (i + 1) % 40 == 0:
            print("scanned", i + 1, "/", len(kept), flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
