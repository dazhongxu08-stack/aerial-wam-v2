#!/usr/bin/env python3
"""Fly the Tello toward a coloured object ahead and stop near it (runs on Orin).

YOLO-World is a TorchScript export with the class list baked in (``--model``); class 0 is the target.
Each frame runs the full image plus a 2x centre crop for far, small objects, then keeps boxes whose
pixels are mostly the target colour and whose box is not flat. A target is locked once 4 of the last 6
frames agree on a bearing within --max-off-deg of the search heading.
Motion is TelloConsequencePolicy (DA3 cones, hover / yaw / forward only); its goal bearing is the
takeoff heading until lock, then the locked object's world bearing. Stops at --stop-m.

Modes: advance (toward the goal point --approx-m ahead of takeoff, by odometry; when the goal distance stops
shrinking, detour via a point --detour-m away 75 deg off the goal bearing, sides alternating, then re-aim) -> scan (in-place 360 at the goal area: progress along the
search heading >= --approx-m or within --goal-r-m of the goal point, or the time budget / low battery) -> hop (fly to the next search point on
5 m / 10 m rings around the goal area, then scan again) ... -> return only when the points or the time /
battery run out (toward takeoff by odometry; land once fwd/left/right clearance >= --land-clear-m for ~1 s,
or after --return-max-s). A scan never starts with fwd/left/right clearance < --scan-clear-m: clear mode first
yaws to an open heading and creeps out (Tello drifts while rotating and cannot see sideways or behind);
after --clear-max-s without open space the scan here is skipped for the next search point. Any lock switches to approach; losing the target for --lost-s resumes the search.
Odometry integrates -vg*0.1 from the ground console status (yaw=0 world axes); it is dead reckoning. A lock in scan/return switches to approach; after found it also returns
to look for a landing spot. A frame is saved every second with the best raw target box in yellow.

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_find_object.py --base http://192.168.54.2:8090 \\
    --model ~/yw_stool.torchscript --classes stool,chair,table,barrel,bucket,box,person,shipping_container \\
    --color red --approx-m 30 --depth-ckpt ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

import experiments.aerial.scripts.tello_consequence_collect as tc
from experiments.aerial.rl.tello_consequence_policy import TelloConsequencePolicy, altitude_rc, decision_log, wrap

HFOV_DEG = 82.6
COLOR_HSV = {"red": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (180, 255, 255))]}


class Detector:
    def __init__(self, path: str, classes: List[str], device: str = "cuda", imgsz: int = 640) -> None:
        import torch

        self.torch = torch
        self.model = torch.jit.load(path, map_location=device).eval()
        self.classes, self.device, self.imgsz = classes, device, imgsz

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """rows x0 y0 x1 y1 conf cls in ``rgb`` pixels."""
        h, w = rgb.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        img = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        py, px = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        img[py:py + nh, px:px + nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        x = self.torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with self.torch.no_grad():
            y = self.model(x)[0].float().cpu().numpy()  # (4+nc, N)
        cls = y[4:].argmax(0)
        conf = y[4:].max(0)
        keep = conf >= 0.08
        if not keep.any():
            return np.zeros((0, 6), np.float32)
        b, conf, cls = y[:4, keep].T, conf[keep], cls[keep]
        x0 = (b[:, 0] - b[:, 2] / 2 - px) / s
        y0 = (b[:, 1] - b[:, 3] / 2 - py) / s
        x1 = (b[:, 0] + b[:, 2] / 2 - px) / s
        y1 = (b[:, 1] + b[:, 3] / 2 - py) / s
        return nms(np.stack([x0, y0, x1, y1, conf, cls], 1).astype(np.float32))

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        full = self._run(rgb)
        cx0, cy0 = w // 4, int(h * 0.2)
        crop = rgb[cy0:cy0 + h // 2, cx0:cx0 + w // 2]
        up = np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))
        c = self._run(up)
        if len(c):
            c[:, [0, 2]] = c[:, [0, 2]] / 2 + cx0
            c[:, [1, 3]] = c[:, [1, 3]] / 2 + cy0
        return nms(np.concatenate([full, c], 0)) if len(c) else full


def nms(d: np.ndarray, iou: float = 0.5) -> np.ndarray:
    if len(d) == 0:
        return d
    d = d[np.argsort(-d[:, 4])]
    out = []
    while len(d):
        out.append(d[0])
        a = d[0]
        xx0, yy0 = np.maximum(a[0], d[1:, 0]), np.maximum(a[1], d[1:, 1])
        xx1, yy1 = np.minimum(a[2], d[1:, 2]), np.minimum(a[3], d[1:, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
        d = d[1:][inter / (area(a) + area(d[1:]) - inter + 1e-6) < iou]
    return np.stack(out)


def color_frac(rgb: np.ndarray, box: np.ndarray, color: str) -> float:
    import cv2

    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(w, box[2])), int(min(h, box[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in COLOR_HSV[color]:
        m |= cv2.inRange(hsv, lo, hi) > 0
    return float(m.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://192.168.54.2:8090")
    ap.add_argument("--model", required=True)
    ap.add_argument("--classes", required=True, help="comma list in export order; first is the target")
    ap.add_argument("--color", default="red", choices=sorted(COLOR_HSV))
    ap.add_argument("--min-color-frac", type=float, default=0.4)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-aspect", type=float, default=0.6, help="box h/w; a barrel top seen edge-on is ~0.5")
    ap.add_argument("--max-off-deg", type=float, default=40.0, help="lock only within this of the search heading")
    ap.add_argument("--object-h-m", type=float, default=0.45, help="object height for range from box height")
    ap.add_argument("--approx-m", type=float, default=30.0)
    ap.add_argument("--bearing-deg", type=float, default=0.0, help="search heading, relative to nose at takeoff")
    ap.add_argument("--stop-m", type=float, default=2.5)
    ap.add_argument("--fwd-sticks", default="45,30,20", help="forward stick for clearance >=3 m, >=1.8 m, else")
    ap.add_argument("--near-m", type=float, default=0.9, help="policy turns in place below this forward clearance")
    ap.add_argument("--side-near-m", type=float, default=0.6, help="same for the left / right cones")
    ap.add_argument("--air-s", type=float, default=150.0)
    ap.add_argument("--return-s", type=float, default=70.0,
                    help="this long before --air-s: one look-around scan (if none yet), then return 25 s later")
    ap.add_argument("--return-bat", type=int, default=35)
    ap.add_argument("--return-max-s", type=float, default=40.0, help="land anyway after this long looking for a spot")
    ap.add_argument("--land-clear-m", type=float, default=2.0, help="landing spot needs this clearance fwd/left/right")
    ap.add_argument("--scan-stick", type=int, default=22)
    ap.add_argument("--scan-clear-m", type=float, default=2.0, help="min fwd/left/right clearance to rotate in place")
    ap.add_argument("--clear-max-s", type=float, default=15.0)
    ap.add_argument("--goal-r-m", type=float, default=4.0, help="search starts within this odometry distance of the goal point")
    ap.add_argument("--approach-grace-s", type=float, default=40.0)
    ap.add_argument("--stuck-s", type=float, default=20.0, help="blocked if the goal distance drops < --stuck-m in this window")
    ap.add_argument("--stuck-m", type=float, default=1.5)
    ap.add_argument("--detour-m", type=float, default=8.0, help="detour point distance, 75 deg off the goal bearing")
    ap.add_argument("--detour-max-s", type=float, default=20.0)
    ap.add_argument("--near-goal-m", type=float, default=10.0, help="within this of the goal point lock up to --near-off-deg")
    ap.add_argument("--near-off-deg", type=float, default=70.0)
    ap.add_argument("--steer-gain", type=float, default=1.0, help="yaw stick per degree of goal error while going forward")
    ap.add_argument("--steer-max", type=int, default=25)
    ap.add_argument("--land-on", action="store_true", help="after found: creep, then fly over the object and land on it")
    ap.add_argument("--final-stick", type=int, default=15)
    ap.add_argument("--base-frac", type=float, default=0.96, help="final creep stops once the box bottom reaches this")
    ap.add_argument("--cam-tilt-deg", type=float, default=0.0, help="camera pitch below horizontal")
    ap.add_argument("--seat-off-m", type=float, default=0.18, help="object front to its top centre")
    ap.add_argument("--final-lost-s", type=float, default=4.0)
    ap.add_argument("--hop-max-s", type=float, default=25.0, help="give up reaching a search point after this long")
    ap.add_argument("--lost-s", type=float, default=15.0, help="approach without seeing the target this long -> search")
    ap.add_argument("--depth-ckpt", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--det-device", default="", help="detector device (default --device); Mac: cpu, MPS lacks float64")
    ap.add_argument("--dry", action="store_true", help="detect on one frame, no flight")
    ap.add_argument("--out", default=str(Path.home() / "tello_find_runs"))
    args = ap.parse_args()
    tc.BASE = args.base
    classes = [c.replace("_", " ") for c in args.classes.split(",")]
    det = Detector(str(Path(args.model).expanduser()), classes, args.det_device or args.device)
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    def note(kind: str, **kw) -> None:
        row = {"t": round(time.time(), 3), "kind": kind, **kw}
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        print(kind, json.dumps(kw, ensure_ascii=False), flush=True)

    raw_best: dict = {}

    def targets(rgb: np.ndarray) -> List[dict]:
        h, w = rgb.shape[:2]
        fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
        res = []
        raw_best.clear()
        for r in det(rgb):
            if int(r[5]) != 0:
                continue
            cf = color_frac(rgb, r, args.color)
            bh = max(1.0, float(r[3] - r[1]))
            t = {"box": [round(float(v), 1) for v in r[:4]], "conf": round(float(r[4]), 3), "color": round(cf, 2),
                 "aspect": round(bh / max(1.0, float(r[2] - r[0])), 2),
                 "bearing_deg": round(math.degrees(math.atan(((r[0] + r[2]) / 2 - w / 2) / fx)), 1),
                 "range_m": round(fx * args.object_h_m / bh, 2)}
            if not raw_best or t["conf"] > raw_best["conf"]:
                raw_best.update(t)
            if r[4] >= args.min_conf and t["aspect"] >= args.min_aspect and cf >= args.min_color_frac:
                res.append(t)
        return res

    def save(rgb: np.ndarray, tg: List[dict], tag: str) -> None:
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        if raw_best:
            dr.rectangle(raw_best["box"], outline=(255, 255, 0), width=1)
            dr.text((raw_best["box"][0], raw_best["box"][3] + 2), f"raw {raw_best['conf']:.2f} c{raw_best['color']:.2f}",
                    fill=(255, 255, 0))
        for t in tg:
            dr.rectangle(t["box"], outline=(0, 255, 0), width=2)
            dr.text((t["box"][0], max(0, t["box"][1] - 12)), f"{t['conf']:.2f} {t['range_m']:.1f}m", fill=(0, 255, 0))
        im.save(out / f"{tag}.jpg", quality=85)

    rgb = tc.snapshot_rgb()
    tg = targets(rgb)
    save(rgb, tg, "preflight")
    note("preflight_det", targets=tg, raw=dict(raw_best), out=str(out))
    if args.dry:
        return 0

    pred = None
    if args.depth_ckpt:
        from experiments.aerial.rl.depth_predictor import DepthMinPredictor

        pred = DepthMinPredictor.from_checkpoint(tc.REPO / args.depth_ckpt, device=args.device)
        pred.reset()
    policy = TelloConsequencePolicy(device=args.device, near_m=args.near_m, side_near_m=args.side_near_m)
    fast, mid, slow = (int(v) for v in args.fwd_sticks.split(","))
    hist: deque = deque(maxlen=6)
    moves: deque = deque()
    search0 = math.radians(args.bearing_deg)
    goal0 = search0
    locked, found, n = False, None, 0
    mode, t_mode = "advance", 0.0
    scan_deg, prev_yaw, clear_n, scanned = 0.0, None, 0, False
    clr, det_n, goal_xy, t_prev = 99.0, 0, None, time.time()
    det_wp, det_side = (0.0, 0.0), 1
    unclipped, landed_on = None, False
    wps: List[tuple] = []
    wp = None
    t_seen = 0.0
    t_save = 0.0
    ox = oy = 0.0
    t_odom = None
    tc._req("POST", "/api/control_mode", {"mode": "auto"})
    try:
        tc._preflight()
        rec = tc._req("POST", "/api/record", {"on": True})
        tc.patch_meta(Path(str(rec.get("dir") or "")), {
            "goal_desc": f"{args.color} {classes[0]} ~{args.approx_m:.0f} m ahead", "goal_approx_m": args.approx_m,
            "goal_bearing_deg": args.bearing_deg, "control": "auto", "policy": "tello_find_object", "find_out": str(out)})
        note("takeoff", resp=tc.cmd("takeoff"))
        t0 = time.time()
        takeoff_yaw = None
        settled_since = None
        settled = False

        def set_mode(m: str, why: str) -> None:
            nonlocal mode, t_mode, scan_deg, clear_n, scanned, wps, det_n
            if m == "scan" and clr < args.scan_clear_m:
                m, why = "clear", f"{why}; clearance {clr:.1f} m < {args.scan_clear_m:.1f} m, find open space first"
            if m in ("scan", "clear") and not scanned:
                # Goal area: search waypoints on 5 m and 10 m rings around here.
                h0 = takeoff_yaw + search0
                ring = [(5.0, k * 60.0) for k in range(6)] + [(10.0, k * 45.0) for k in range(8)]
                wps = [(ox + r * math.cos(h0 + math.radians(a)), oy + r * math.sin(h0 + math.radians(a))) for r, a in ring]
                note("search_area", center=[round(ox, 2), round(oy, 2)], n_waypoints=len(wps))
            scanned = scanned or m in ("scan", "clear")
            note("mode", frm=mode, to=m, why=why, t_s=round(time.time() - t0, 1))
            mode, t_mode, scan_deg, clear_n = m, time.time(), 0.0, 0
            det_n += int(m == "detour")
            moves.clear()

        while True:
            tl = time.time()
            el = tl - t0
            st = tc.status()
            tc._gate(st)
            yaw = math.radians(float(st.get("yaw") or 0.0))
            takeoff_yaw = yaw if takeoff_yaw is None else takeoff_yaw
            hm = None if st.get("height_cm") is None else float(st["height_cm"]) / 100.0
            bat = int(st.get("battery") or 0)
            # Tello vg is dm/s in the yaw=0 world axes, sign-flipped. Dead reckoning, not ground truth.
            tv = st.get("tello") or {}
            if t_odom is not None and hm is not None and hm > 0.2:
                odt = min(0.5, tl - t_odom)
                ox -= float(tv.get("vgx") or 0.0) * 0.1 * odt
                oy -= float(tv.get("vgy") or 0.0) * 0.1 * odt
            t_odom = tl
            home_deg = math.degrees(wrap(math.atan2(-oy, -ox) - takeoff_yaw)) if math.hypot(ox, oy) > 0.5 else 180.0
            progress = ox * math.cos(takeoff_yaw + search0) + oy * math.sin(takeoff_yaw + search0)
            if goal_xy is None:
                goal_xy = (args.approx_m * math.cos(takeoff_yaw + search0), args.approx_m * math.sin(takeoff_yaw + search0))
            goal_d = math.hypot(goal_xy[0] - ox, goal_xy[1] - oy)
            if mode == "advance" and not locked and goal_d > 1.0:
                goal0 = wrap(math.atan2(goal_xy[1] - oy, goal_xy[0] - ox) - takeoff_yaw)
            if mode == "detour":
                goal0 = wrap(math.atan2(det_wp[1] - oy, det_wp[0] - ox) - takeoff_yaw)
            if not settled:
                if hm is not None and 0.85 <= hm <= 1.45:
                    settled_since = settled_since or time.time()
                    settled = time.time() - settled_since >= 2.0
                    t_mode = time.time()
                else:
                    settled_since = None
                tc.rc(0, 0, altitude_rc(hm), 0)
                time.sleep(max(0.0, 0.125 - (time.time() - tl)))
                continue
            if el >= args.air_s:
                note("air_budget", t_s=round(el, 1))
                break
            low_bat = bool(bat and bat < args.return_bat)
            if mode in ("advance", "detour") and not scanned and "tello" in st and (
                    progress >= args.approx_m or goal_d <= args.goal_r_m):
                set_mode("scan", f"odometry progress {progress:.1f} m, {goal_d:.1f} m from the goal point: "
                                 f"search for the target at the goal area")
            if mode in ("advance", "detour", "approach") and not scanned and (el >= args.air_s - args.return_s or low_bat):
                set_mode("scan", f"end t={el:.0f}s bat={bat}: rotate in place before returning")
            t_ret = args.air_s - args.return_s + 25.0
            # A locked target gets --approach-grace-s past the return time (and 10 % more battery) to be confirmed.
            approaching = mode == "approach" and locked and found is None
            if mode in ("advance", "detour", "approach", "hop", "clear") and (
                    (el >= t_ret + args.approach_grace_s or bool(bat and bat < args.return_bat - 10)) if approaching
                    else (el >= t_ret or low_bat)):
                goal0 = math.radians(home_deg)
                set_mode("return", f"budget t={el:.0f}s bat={bat}")
            rgb = tc.snapshot_rgb()
            tg = [t for t in targets(rgb) if t["range_m"] >= 1.0]
            best = min(tg, key=lambda t: abs(t["bearing_deg"]) + 0.2 * t["range_m"]) if tg else None
            wb_best = None if best is None else wrap(yaw + math.radians(best["bearing_deg"]))
            off = args.near_off_deg if goal_d <= args.near_goal_m else args.max_off_deg
            in_cone = wb_best is not None and (mode not in ("advance", "detour") or abs(
                wrap(wb_best - takeoff_yaw - search0)) <= math.radians(off))
            hist.append((wb_best, best["range_m"]) if in_cone else None)
            if best is not None:
                t_seen = tl
            seen = [h for h in hist if h is not None]
            if len(seen) >= 4 and mode != "land":
                wb = float(np.median([s[0] for s in seen]))
                if max(abs(wrap(s[0] - wb)) for s in seen) <= math.radians(8):
                    if not locked:
                        note("lock", bearing_world_deg=round(math.degrees(wrap(wb - takeoff_yaw)), 1),
                             range_m=float(np.median([s[1] for s in seen])), mode=mode)
                        save(rgb, tg, f"lock_{n:04d}")
                    locked = True
                    goal0 = wrap(wb - takeoff_yaw)
                    if mode in ("scan", "clear", "hop", "detour", "return") and found is None and not (
                            mode == "return" and (el >= t_ret + args.approach_grace_s or low_bat)):
                        set_mode("approach", "locked")
            if found is None and locked and best is not None and best["range_m"] <= args.stop_m and abs(best["bearing_deg"]) <= 10:
                found = best
                save(rgb, tg, "found")
                note("found", target=best, height_m=hm, yaw_from_takeoff_deg=round(math.degrees(wrap(yaw - takeoff_yaw)), 1),
                     t_s=round(el, 1))
                if args.land_on:
                    set_mode("final", "found; centre it and creep until its base reaches the frame bottom")
                    continue
                for _ in range(24):
                    hc = tc.status().get("height_cm")
                    tc.rc(0, 0, altitude_rc(None if hc is None else float(hc) / 100.0), 0)
                    time.sleep(0.125)
                goal0 = wrap(yaw - takeoff_yaw + math.pi)
                set_mode("return", "found; look for a landing spot")
                continue
            if tl - t_save >= 1.0:
                save(rgb, tg, f"f_{n:05d}")
                t_save = tl
            d56 = tc._depth56(pred, rgb)
            dec = policy.decide(rgb=rgb, height_m=hm, yaw_rad=yaw, takeoff_yaw_rad=float(takeoff_yaw),
                                goal_bearing0_rad=goal0, depth56=d56, **tc.imu_kwargs(st, None, 0))
            if dec.collision:
                note("collision", dec=decision_log(dec))
                tc.rc(0, 0, 0, 0)
                break
            a, b, c, d = dec.rc
            clr = min(dec.fwd_m, dec.left_m, dec.right_m)
            if mode == "final":
                # The object is the obstacle here: the policy is bypassed, only slow forward + centring yaw.
                c = altitude_rc(hm)
                if best is None:
                    a, b, d = 0, 0, 0
                    if tl - t_seen >= args.final_lost_s:
                        found, locked = None, False
                        hist.clear()
                        set_mode("scan", f"target lost for {args.final_lost_s:.0f} s during the final approach")
                else:
                    H, W = rgb.shape[:2]
                    y2 = best["box"][3] / H
                    err = best["bearing_deg"]
                    if y2 < 0.9:
                        unclipped = (best["range_m"], ox, oy)
                    d = int(np.clip(err * 1.5, -20, 20)) if abs(err) > 2.0 else 0
                    a, b = 0, (args.final_stick if abs(err) <= 6.0 else 0)
                    if y2 >= args.base_frac and abs(err) <= 4.0:
                        tc.rc(0, 0, 0, 0)
                        time.sleep(1.0)
                        hc = tc.status().get("height_cm")
                        hh = float(hc) / 100.0 if hc is not None else (hm or 1.0)
                        fpx = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
                        dep = math.atan((best["box"][3] - H / 2) / fpx) + math.radians(args.cam_tilt_deg)
                        est = {"geom": hh / math.tan(dep)}
                        if unclipped is not None:
                            est["size"] = unclipped[0] - math.hypot(ox - unclipped[1], oy - unclipped[2])
                        if d56 is not None:
                            x1, y1, x2, y2p = best["box"]
                            pr = d56[int(y1 / H * 56):max(int(y1 / H * 56) + 1, int(y2p / H * 56)),
                                     int(x1 / W * 56):max(int(x1 / W * 56) + 1, int(x2 / W * 56))]
                            est["da3"] = float(np.median(pr))
                        dist = float(np.median(list(est.values()))) + args.seat_off_m
                        cm = int(round(max(0.2, min(3.0, dist)) * 100))
                        save(rgb, tg, "final_base")
                        note("land_on_target", est={k: round(v, 2) for k, v in est.items()}, height_m=hh,
                             forward_cm=cm, bearing_deg=err, t_s=round(el, 1))
                        note("forward", resp=tc.cmd(f"forward {cm}"))
                        note("land_on", resp=tc.cmd("land"))
                        landed_on = True
                        break
            elif mode == "clear":
                if clr >= args.scan_clear_m:
                    set_mode("scan", f"open space (clearance {clr:.1f} m)")
                    a, b, d = 0, 0, 0
                elif tl - t_mode >= args.clear_max_s:
                    if wps and el < args.air_s - args.return_s + 25.0 and not low_bat:
                        wps.sort(key=lambda q: math.hypot(q[0] - ox, q[1] - oy))
                        wp = wps.pop(0)
                        set_mode("hop", f"no open space in {args.clear_max_s:.0f} s, skip this scan; "
                                        f"next search point {wp[0]:.1f},{wp[1]:.1f} ({len(wps)} left)")
                    else:
                        goal0 = math.radians(home_deg)
                        set_mode("return", "no open space to scan, budget")
                    a, b, d = 0, 0, 0
                else:
                    a, b, d = (0, slow, 0) if dec.fwd_m >= 3.0 else (0, 0, args.scan_stick)
            elif mode == "detour":
                ddet = math.hypot(det_wp[0] - ox, det_wp[1] - oy)
                if ddet <= 2.0 or tl - t_mode >= args.detour_max_s:
                    set_mode("approach" if locked else "advance",
                             f"detour point {'reached' if ddet <= 2.0 else 'timeout'}; head for the goal point again, "
                             f"{goal_d:.1f} m away")
            elif mode == "scan":
                if prev_yaw is not None:
                    scan_deg += abs(math.degrees(wrap(yaw - prev_yaw)))
                a, b, d = 0, 0, args.scan_stick
                if scan_deg >= 360.0:
                    if wps and el < args.air_s - args.return_s + 25.0 and not low_bat:
                        wps.sort(key=lambda q: math.hypot(q[0] - ox, q[1] - oy))
                        wp = wps.pop(0)
                        set_mode("hop", f"scan done, nothing found; next search point {wp[0]:.1f},{wp[1]:.1f} "
                                        f"({len(wps)} left)")
                    else:
                        goal0 = math.radians(home_deg)
                        set_mode("return", "search done or budget, nothing found")
            elif mode == "hop":
                dwp = math.hypot(wp[0] - ox, wp[1] - oy)
                goal0 = wrap(math.atan2(wp[1] - oy, wp[0] - ox) - takeoff_yaw)
                if dwp <= 1.5 or tl - t_mode >= args.hop_max_s:
                    set_mode("scan", f"at search point (d={dwp:.1f} m)" if dwp <= 1.5 else f"hop timeout d={dwp:.1f} m")
                    a, b, d = 0, 0, 0
            elif mode == "approach" and scanned and tl - t_seen >= args.lost_s:
                locked = False
                hist.clear()
                set_mode("scan", f"target lost for {args.lost_s:.0f} s")
                a, b, d = 0, 0, 0
            elif mode == "return":
                if found is None:
                    goal0 = math.radians(home_deg)
                ok = min(dec.fwd_m, dec.left_m, dec.right_m) >= args.land_clear_m
                clear_n = clear_n + 1 if ok else 0
                if (clear_n >= 8 and tl - t_mode >= 8.0) or tl - t_mode >= args.return_max_s:
                    note("landing_spot", clear=ok, fwd_m=round(dec.fwd_m, 2), left_m=round(dec.left_m, 2),
                         right_m=round(dec.right_m, 2), t_s=round(el, 1))
                    save(rgb, tg, "landing_spot")
                    break
            if b > 0 and mode not in ("clear", "final"):
                b = fast if dec.fwd_m >= 3.0 else mid if dec.fwd_m >= 1.8 else slow
                if locked and best is not None and best["range_m"] < 5.0:
                    b = slow
                # Curve toward the goal while moving instead of stop-turn-go.
                gerr = math.degrees(wrap(takeoff_yaw + goal0 - yaw))
                if d == 0 and abs(gerr) > 5.0:
                    d = int(np.clip(args.steer_gain * gerr, -args.steer_max, args.steer_max))
            if mode in ("advance", "approach"):
                # Progress toward the goal point (advance) or the locked target (approach, by box range).
                gd = goal_d if mode == "advance" else (best["range_m"] if best is not None else None)
                if gd is not None:
                    moves.append((tl, gd))
                while moves and tl - moves[0][0] > args.stuck_s:
                    moves.popleft()
                if tl - t_mode >= args.stuck_s and len(moves) > 8 and moves[0][1] - min(m[1] for m in moves) < args.stuck_m:
                    side = (1 if dec.right_m >= dec.left_m else -1) if det_n == 0 else -det_side
                    hd = takeoff_yaw + goal0 + side * math.radians(75.0)
                    det_wp = (ox + args.detour_m * math.cos(hd), oy + args.detour_m * math.sin(hd))
                    det_side = side
                    set_mode("detour", f"blocked: {moves[0][1]:.1f}->{moves[-1][1]:.1f} m in {args.stuck_s:.0f} s; go around "
                                       f"{'right' if side > 0 else 'left'} via {det_wp[0]:.1f},{det_wp[1]:.1f}, "
                                       f"{goal_d:.1f} m to the goal point")
            prev_yaw, t_prev = yaw, tl
            tc.rc(a, b, c, d)
            dec.rc = (a, b, c, d)
            if n % 4 == 0 or tg:
                note("step", t_s=round(el, 1), mode=mode, odom=[round(ox, 2), round(oy, 2)], progress=round(progress, 1),
                     h=hm, locked=locked, targets=tg, raw=dict(raw_best),
                     goal_deg=round(math.degrees(goal0), 1), rc=list(dec.rc), fam=dec.family, fwd_m=round(dec.fwd_m, 2),
                     bat=bat, temp=st.get("temp"))
            n += 1
            time.sleep(max(0.0, 0.125 - (time.time() - tl)))
        tc.rc(0, 0, 0, 0)
        note("end", found=found is not None, landed_on=landed_on, locked=locked, mode=mode, t_s=round(time.time() - t0, 1))
        tc.land_if_airborne()
        time.sleep(3)
        return 0 if found else 3
    except Exception as exc:  # noqa: BLE001
        note("abort", err=str(exc))
        try:
            tc.land_if_airborne()
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        for f in (lambda: tc.rc(0, 0, 0, 0), lambda: tc._req("POST", "/api/record", {"on": False}),
                  lambda: tc._req("POST", "/api/control_mode", {"mode": "manual"})):
            try:
                f()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
