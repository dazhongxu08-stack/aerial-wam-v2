"""Trajectory grid and end-of-flight clips for an MPC run vs a reference run.

usage: vis_mpc_runs.py OUT_DIR NAME 'run_glob' REF_NAME 'ref_glob'   (ref_glob may be '')
Writes traj_grid.png, clips/<route>_s<seed>_<end>.mp4 (saved frames with the per-step
choice overlaid next to a top-down map) and sheets/<route>_s<seed>_<end>.jpg.
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SKIP = {"od_holdout_06", "od_holdout_13"}
COLOR = {"arrived": "tab:green", "collision": "tab:red", "stuck": "tab:orange", "max_steps": "tab:orange"}


def runs(pattern):
    out = []
    for d in sorted(glob.glob(pattern)):
        seed = d.rstrip("/").split("_s")[-1].split("_")[0]
        p = Path(d) / "MPC_EVAL.json"
        if not p.is_file():
            continue
        for e in json.load(open(p))["episodes"]:
            if "arrived" in e and e["route_id"] not in SKIP:
                out.append((Path(d), seed, e))
    return out


def frames(d, rid):
    z = np.load(d / f"last_frames_{rid}.npz")
    j, off = z["jpeg"], list(z["jpeg_offsets"])
    return [cv2.imdecode(j[off[i]:off[i + 1]], 1) for i in range(len(off) - 1)]


def main():
    out, name, pat, ref_name, ref_pat = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    main_runs, ref_runs = runs(pat), runs(ref_pat)
    by_route = defaultdict(lambda: {"main": [], "ref": []})
    for d, s, e in main_runs:
        by_route[e["route_id"]]["main"].append((d, s, e))
    for d, s, e in ref_runs:
        by_route[e["route_id"]]["ref"].append((d, s, e))

    routes = sorted(by_route)
    cols = 4
    rows = int(np.ceil(len(routes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows))
    for ax, rid in zip(axes.flat, routes):
        for d, s, e in by_route[rid]["ref"]:
            t = np.load(d / f"traj_{rid}.npz")["traj"]
            ax.plot(t[:, 0], t[:, 1], color="0.7", lw=1, ls="--")
        tag = []
        for d, s, e in by_route[rid]["main"]:
            z = np.load(d / f"traj_{rid}.npz")
            t, g = z["traj"], z["goal"]
            ax.plot(t[:, 0], t[:, 1], color=COLOR[e["end_reason"]], lw=1.6)
            ax.plot(t[-1, 0], t[-1, 1], "x" if e["end_reason"] == "collision" else "o", color=COLOR[e["end_reason"]], ms=7)
            ax.plot(t[0, 0], t[0, 1], "k^", ms=7)
            ax.plot(g[0], g[1], "b*", ms=12)
            tag.append(f"s{s}:{e['end_reason'][:4]}")
        ra = sum(e["arrived"] for *_, e in by_route[rid]["ref"])
        rc = sum(e["severe_collision"] for *_, e in by_route[rid]["ref"])
        ax.set_title(f"{rid[-2:]}  {name}: {' '.join(tag)}\n{ref_name} (grey) arr {ra} coll {rc}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(routes):]:
        ax.axis("off")
    fig.suptitle(f"{name} (green arrived / red collision x / orange stuck-timeout) vs {ref_name} (grey dashed). ^ start, * goal")
    fig.tight_layout()
    fig.savefig(out / "traj_grid.png", dpi=80)

    for d, s, e in main_runs:
        rid, end = e["route_id"], e["end_reason"]
        imgs = frames(d, rid)
        by_step = {x["step"]: x for x in json.load(open(d / f"steps_{rid}.json"))}
        z = np.load(d / f"traj_{rid}.npz")
        traj, goal = z["traj"], z["goal"]
        first = len(traj) - len(imgs)
        tag = f"{rid[-2:]}_s{s}_{end}"
        side = 448
        vw = cv2.VideoWriter(str(out / "clips" / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (2 * side, side))
        pts = np.concatenate([traj[:, :2], goal[None, :2]])
        lo, span = pts.min(0) - 5, max(float((pts.max(0) - pts.min(0)).max()) + 10, 1.0)
        to_px = lambda p: (int((p[0] - lo[0]) / span * (side - 1)), int(side - 1 - (p[1] - lo[1]) / span * (side - 1)))
        base = np.full((side, side, 3), 255, np.uint8)
        cv2.polylines(base, [np.array([to_px(p) for p in traj[:, :2]], np.int32)], False, (180, 180, 180), 1)
        cv2.drawMarker(base, to_px(goal[:2]), (255, 0, 0), cv2.MARKER_STAR, 18, 2)
        cv2.drawMarker(base, to_px(traj[0, :2]), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)
        tiles = []
        for k, im in enumerate(imgs):
            idx = first + k
            st = by_step.get(idx)
            im = cv2.resize(im, (side, side))
            lines = [f"{name} {rid} seed {s}  end={end}  step {idx}/{len(traj) - 1}"]
            if st:
                lines.append(f"pick {st['family']} surv {st['surv']:.2f} pass {st['n_pass']}/{len(st['cand_surv'])}")
                lines.append(f"d_goal {st['d_goal']:.1f}  z_off {st['z_off']:+.1f}")
            for i, txt in enumerate(lines):
                cv2.putText(im, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(im, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            m = base.copy()
            cv2.polylines(m, [np.array([to_px(p) for p in traj[:idx + 1, :2]], np.int32)], False, (0, 140, 0), 2)
            cv2.circle(m, to_px(traj[idx, :2]), 6, (0, 0, 255), -1)
            cv2.putText(m, f"z {traj[idx, 2]:.1f} m  goal z {goal[2]:.1f} m", (8, side - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            frame = np.concatenate([im, m], 1)
            vw.write(frame)
            tiles.append(cv2.resize(im, (224, 224)))
        vw.release()
        pick = [tiles[i] for i in np.linspace(0, len(tiles) - 1, 6).astype(int)]
        cv2.imwrite(str(out / "sheets" / f"{tag}.jpg"), np.concatenate(pick, 1))
    print("wrote", out)


if __name__ == "__main__":
    main()
