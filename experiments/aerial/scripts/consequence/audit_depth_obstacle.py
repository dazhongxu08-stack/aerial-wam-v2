import glob
import json
import re

import numpy as np

from experiments.aerial.rl.depth_geometry import cone_clearances

D = "experiments/aerial/rl/artifacts/consequence_data_20261004"
F = 28.0  # 56 px, 90 deg HFOV
u = (np.arange(56) + 0.5 - 28.0) / F
YY, ZZ = np.meshgrid(u, u)  # col -> lateral, row -> vertical (down +)


def corridor(depth, r=1.5):
    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d > 0.05)
    lat = np.abs(YY * d)
    ver = np.abs(ZZ * d)
    m = ok & (lat <= r) & (ver <= r)
    return float(d[m].min()) if m.any() else np.inf


rows = []
press = []  # straight-press steps: clearances + collide-within-5
for p in sorted(glob.glob(f"{D}/episodes/episode_*.npz")):
    if not re.search(r"episode_\d+\.npz$", p):
        continue
    side = p.replace(".npz", ".da3.npz")
    try:
        z = np.load(p)
        a = np.load(side)["depth56"].astype(np.float64)
    except Exception:
        continue
    g = z["depth56"].astype(np.float64)
    coll = z["collided"]
    act = z["action"].astype(np.float64)
    state = z["state"].astype(np.float64)
    T = len(g)
    for t in range(T):
        cg = cone_clearances(g[t])
        ca = cone_clearances(a[t])
        # Straight-press outcome: travel until collision while actions stay forward-dominant.
        rows.append([
            cg["forward"], cg["left"], cg["right"], corridor(g[t]),
            ca["forward"], ca["left"], ca["right"], corridor(a[t]),
            float(np.nanmedian(np.where(g[t] > 0, g[t], np.nan))), float(np.nanmedian(a[t])),
        ])
        straight = lambda i: act[i, 0] > 0.6 and abs(act[i, 1]) < 0.05 and abs(act[i, 3]) < 0.03 and abs(act[i, 2]) < 0.05
        if t < T - 1 and straight(t):
            hit = 0
            for i in range(t, min(T - 1, t + 5)):
                if not straight(i):
                    break
                if coll[i + 1]:
                    hit = 1
                    break
            press.append([hit, cg["forward"], corridor(g[t]), ca["forward"], corridor(a[t])])
R = np.array(rows)
R[~np.isfinite(R)] = 200.0
R = np.clip(R, 0, 200)
gf, gl, gr, gc, af, al, ar, ac, gmed, amed = R.T
n = len(R)
print("frames", n)
print("image median depth: GT %.1f  DA3 %.1f  (DA3/GT median ratio %.2f)" % (np.median(gmed), np.median(amed), np.median(amed / np.maximum(gmed, 0.1))))
print("forward cone: GT median %.1f  DA3 median %.1f  ratio %.2f" % (np.median(gf), np.median(af), np.median(af / np.maximum(gf, 0.1))))
near = gf < 15
print("forward cone where GT<15m: DA3/GT ratio median %.2f  p10 %.2f p90 %.2f" % tuple(np.percentile(af[near] / np.maximum(gf[near], 0.1), [50, 10, 90])))
print("false alarm  DA3 fwd<3m while GT fwd>6m: %.1f%% of frames" % (100 * np.mean((af < 3) & (gf > 6))))
print("miss         DA3 fwd>6m while GT fwd<3m: %.1f%% of frames" % (100 * np.mean((af > 6) & (gf < 3))))
print("--- overlap of side cones with forward box (GT) ---")
tie = (np.abs(gl - gf) < 0.05) | (np.abs(gr - gf) < 0.05)
both = (np.abs(gl - gf) < 0.05) & (np.abs(gr - gf) < 0.05)
print("GT fwd<15: a side cone equals forward %.1f%%, BOTH sides equal forward %.1f%%" % (100 * tie[near].mean(), 100 * both[near].mean()))
print("GT fwd<15: |left-right|<1m (no side preference) %.1f%%" % (100 * np.mean(np.abs(gl - gr)[near] < 1.0)))
print("--- forward cone vs flight corridor (+-1.5 m tube, GT) ---")
print("GT fwd cone<8m: corridor median %.1f m; corridor>15m in %.1f%%" % (np.median(gc[gf < 8]), 100 * np.mean(gc[gf < 8] > 15)))
print("GT corridor<8m: fwd cone median %.1f m" % np.median(gf[gc < 8]))
print("DA3 corridor vs GT corridor (GT corr<15): ratio median %.2f" % np.median(ac[gc < 15] / np.maximum(gc[gc < 15], 0.1)))


def auc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum())


P = np.array(press, dtype=np.float64)
P[~np.isfinite(P)] = 200.0
y = P[:, 0].astype(int)
print("--- straight-press steps: predict collide within 5 steps (AUC, lower clearance = riskier) ---")
print("n", len(P), "positives", int(y.sum()))
for k, name in [(1, "GT forward cone"), (2, "GT corridor"), (3, "DA3 forward cone"), (4, "DA3 corridor")]:
    print("  %-17s AUC %.3f" % (name, auc(-P[:, k], y)))
for k, name in [(1, "GT forward cone"), (3, "DA3 forward cone")]:
    m = P[:, k] < 5
    print("  %s < 5 m on %d steps: actually collided within 5 steps %.1f%%" % (name, m.sum(), 100 * y[m].mean() if m.any() else float('nan')))
