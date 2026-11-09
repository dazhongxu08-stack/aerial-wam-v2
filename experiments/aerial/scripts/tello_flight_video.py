#!/usr/bin/env python3
"""Render first-person, track and side-by-side videos of a Tello find flight.

Inputs: a ground-console recording (frames/, trajectory.csv) and the tello_find_object stdout log.
The find log's t_s=0 is the moment takeoff returned; it is aligned to the first recorder sample with
h >= --align-h-cm. Track x/y is the recorder's dead-reckoned odometry, rotated so the takeoff heading
points up (not ground truth).

  python3 experiments/aerial/scripts/tello_flight_video.py --rec artifacts/tello_gs/run_X --log tello_find.log \\
    --out artifacts/tello_gs/find_X
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

COL = {"advance": (230, 120, 30), "detour": (0, 140, 255), "clear": (150, 150, 150), "scan": (60, 180, 60),
       "hop": (200, 80, 160), "approach": (40, 40, 230), "final": (0, 0, 180), "return": (20, 20, 20)}
ZH = {"advance": "advance", "detour": "detour", "clear": "clear", "scan": "scan 360", "hop": "hop",
      "approach": "approach", "final": "final", "return": "return"}


def load_log(path: Path):
    steps, events = [], []
    for line in path.read_text(errors="replace").splitlines():
        k, _, j = line.partition(" ")
        if not j.startswith("{"):
            continue
        try:
            d = json.loads(j)
        except json.JSONDecodeError:
            continue
        if k == "step":
            steps.append(d)
        elif k in ("mode", "lock", "found", "land_on_target", "landing_spot", "search_area"):
            events.append((k, d))
    return steps, events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--align-h-cm", type=float, default=50.0)
    ap.add_argument("--goal-m", type=float, default=30.0)
    args = ap.parse_args()
    rec, out = Path(args.rec), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((rec / "trajectory.csv").open()))
    T = np.array([float(r["t_s"]) for r in rows])
    X = np.array([float(r["x_m"]) for r in rows])
    Y = np.array([float(r["y_m"]) for r in rows])
    YAW = np.array([float(r["yaw_deg"]) for r in rows])
    Hh = np.array([float(r["h_cm"] or 0) for r in rows])
    SEQ = [int(float(r["frame_seq"])) for r in rows]
    air = np.nonzero(Hh >= args.align_h_cm)[0]
    t_air = T[air[0]] if len(air) else 0.0
    t_end = T[air[-1]] + 3.0 if len(air) else T[-1]
    t_start = max(0.0, t_air - 6.0)

    y0 = math.radians(YAW[air[0]] if len(air) else YAW[0])
    fwd, right = np.array([math.cos(y0), math.sin(y0)]), np.array([math.cos(y0 + math.pi / 2), math.sin(y0 + math.pi / 2)])
    P = np.stack([X - X[air[0] if len(air) else 0], Y - Y[air[0] if len(air) else 0]], 1)
    PX, PY = P @ right, P @ fwd

    seq_t = {}
    for t, s in zip(T, SEQ):
        seq_t.setdefault(s, t)
    ks = sorted(seq_t)
    vs = [seq_t[k] for k in ks]
    frames = sorted((float(np.interp(int(f.stem), ks, vs)), f) for f in (rec / "frames").glob("*.jpg"))
    ft = [t for t, _ in frames]

    steps, events = load_log(Path(args.log))
    # Both logs integrate the same Tello vg: pick the time offset that best overlays the find odometry.
    a0 = air[0] if len(air) else 0
    so = np.array([s["t_s"] for s in steps])
    oxy = np.array([s["odom"] for s in steps], float)
    best = (1e9, 0.0)
    for off in np.arange(-8.0, 8.0, 0.05):
        tt = so + t_air + off
        ok = (tt >= T[0]) & (tt <= T[-1])
        if ok.sum() < 10:
            continue
        rx = np.interp(tt[ok], T, X - X[a0])
        ry = np.interp(tt[ok], T, Y - Y[a0])
        e = float(np.mean(np.hypot(rx - oxy[ok, 0], ry - oxy[ok, 1])))
        best = min(best, (e, float(off)))
    t_air += best[1]
    print(json.dumps({"align_offset_s": round(best[1], 2), "align_err_m": round(best[0], 3)}))
    st_t = [s["t_s"] + t_air for s in steps]
    ev = [(d.get("t_s", None), k, d) for k, d in events]
    mode_ch = [(d["t_s"] + t_air, d["to"]) for t, k, d in ev if k == "mode" and t is not None]
    # lock rows carry no t_s: take the first approach switch after each lock
    marks = [(tc, "lock") for tc, to in mode_ch if to in ("approach",)][:1]
    marks += [(d["t_s"] + t_air, k) for k, d in events if k in ("found", "land_on_target") and "t_s" in d]

    def mode_at(t: float) -> str:
        m = "advance"
        for tc, to in mode_ch:
            if tc <= t:
                m = to
        return m

    xmin, xmax = min(PX.min(), -3) - 2, max(PX.max(), 3) + 2
    ymin, ymax = min(PY.min(), -1) - 2, max(PY.max(), args.goal_m) + 3
    W2, H2 = 480, 720
    sc = min((W2 - 60) / (xmax - xmin), (H2 - 90) / (ymax - ymin))

    def px(x: float, y: float):
        return int(30 + (x - xmin) * sc), int(H2 - 50 - (y - ymin) * sc)

    base = np.full((H2, W2, 3), 255, np.uint8)
    for g in range(int(math.floor(ymin / 5)) * 5, int(ymax) + 1, 5):
        cv2.line(base, px(xmin, g), px(xmax, g), (235, 235, 235), 1)
        cv2.putText(base, f"{g}m", (2, px(0, g)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    gx, gy = px(0, args.goal_m)
    cv2.drawMarker(base, (gx, gy), (0, 0, 255), cv2.MARKER_STAR, 22, 2)
    cv2.putText(base, f"goal {args.goal_m:.0f} m", (gx + 10, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.drawMarker(base, px(0, 0), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    ly = 20
    for m, c in COL.items():
        cv2.line(base, (W2 - 110, ly), (W2 - 90, ly), c, 4)
        cv2.putText(base, ZH[m], (W2 - 85, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        ly += 16

    sizes = (("fpv", (640, 480)), ("track", (W2, H2)), ("combined", (1280, 480)))
    vw = {k: cv2.VideoWriter(str(out / f"{k}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), args.fps, sz) for k, sz in sizes}
    if not vw["fpv"].isOpened():
        vw = {k: cv2.VideoWriter(str(out / f"{k}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, sz) for k, sz in sizes}
    trail = base.copy()
    last_i = 0
    t = t_start
    n = 0
    while t <= t_end:
        i = max(0, bisect.bisect_right(T, t) - 1)
        for j in range(last_i + 1, i + 1):
            cv2.line(trail, px(PX[j - 1], PY[j - 1]), px(PX[j], PY[j]), COL.get(mode_at(T[j]), (0, 0, 0)), 3)
        last_i = max(last_i, i)
        trk = trail.copy()
        for tm, name in marks:
            if tm <= t:
                j = max(0, bisect.bisect_right(T, tm) - 1)
                cv2.circle(trk, px(PX[j], PY[j]), 7, (0, 0, 255), 2)
                cv2.putText(trk, name, (px(PX[j], PY[j])[0] + 9, px(PX[j], PY[j])[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1)
        hd = math.radians(YAW[i]) - y0
        cx, cy = px(PX[i], PY[i])
        tip = (int(cx + 16 * math.sin(hd)), int(cy - 16 * math.cos(hd)))
        cv2.circle(trk, (cx, cy), 6, (0, 0, 0), -1)
        cv2.arrowedLine(trk, (cx, cy), tip, (0, 0, 0), 2, tipLength=0.4)
        m = mode_at(t)
        el = t - t_air
        cv2.putText(trk, f"t={el:5.1f}s  {ZH.get(m, m)}", (10, H2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL.get(m, (0, 0, 0)), 2)
        cv2.putText(trk, f"fwd {PY[i]:5.1f} m  right {PX[i]:5.1f} m  h {Hh[i] / 100:.1f} m", (10, H2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        k = max(0, bisect.bisect_right(ft, t) - 1)
        fr = cv2.imread(str(frames[k][1])) if frames else np.zeros((480, 640, 3), np.uint8)
        fr = cv2.resize(fr, (640, 480))
        cv2.rectangle(fr, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(fr, f"Tello FPV  t={el:5.1f}s  mode: {ZH.get(m, m)}  h={Hh[i] / 100:.1f} m", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        si = bisect.bisect_right(st_t, t) - 1
        if 0 <= si < len(steps) and t - st_t[si] < 1.0:
            for tg in steps[si].get("targets") or []:
                b = [int(v) for v in tg["box"]]
                cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(fr, f"red stool {tg['conf']:.2f} {tg['range_m']:.1f}m", (b[0], max(36, b[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        tk = cv2.resize(trk, (W2 * 480 // H2, 480))
        comb = np.zeros((480, 1280, 3), np.uint8)
        comb[:, :640] = fr
        comb[:, 640:640 + tk.shape[1]] = tk
        comb[:, 640 + tk.shape[1]:] = 255
        vw["fpv"].write(fr)
        vw["track"].write(trk)
        vw["combined"].write(comb)
        t += 1.0 / args.fps
        n += 1
    for v in vw.values():
        v.release()
    cv2.imwrite(str(out / "track_final.png"), trk)
    print(json.dumps({"frames_out": n, "fpv_src_frames": len(frames), "dur_s": round(t_end - t_start, 1),
                      "path_m": round(float(np.hypot(np.diff(PX[air]), np.diff(PY[air])).sum()) if len(air) > 1 else 0, 1),
                      "end_fwd_right_m": [round(float(PY[air[-1]]), 1), round(float(PX[air[-1]]), 1)] if len(air) else None,
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
