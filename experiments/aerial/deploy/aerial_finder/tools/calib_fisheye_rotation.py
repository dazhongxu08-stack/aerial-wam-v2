"""Fisheye calibration from in-place turns, IMU yaw fixing the angular scale.

Kannala-Brandt intrinsics + per-pair rotations; matches must satisfy
x_j = proj(R_ij unproj(x_i)) and |R_ij| must equal the IMU yaw change.
Half the pairs (alternating) are held out to check the fitted model.
"""
import csv
import glob
import json
import re

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

W, H_FULL = 1280, 720
IMU_W = 4.0  # px of residual per degree of IMU disagreement


def run1():
    fs = sorted(glob.glob("/tmp/dogcal/runs/frames/20261019_161410/*.jpg"))
    out = []
    for f in fs:
        t = float(f.split("/")[-1][:7])
        if 139 <= t <= 157:
            out.append((f, float(re.search(r"yaw([+-]\d+)", f).group(1))))
    return out


def run2():
    rows = list(csv.DictReader(open("/tmp/dogcal/runs/rec_20261019_164228/control_imu.csv")))
    t = np.array([float(r["t"]) for r in rows])
    y = np.degrees(np.unwrap(np.radians([float(r["yaw"]) for r in rows])))
    fs = sorted(glob.glob("/tmp/dogcal/runs/rec_20261019_164228/frames/*/*.jpg"))
    out = []
    for f in fs:
        ft = float(f.split("/")[-1][:7])
        if 108 <= ft <= 200 or 248 <= ft <= 262:
            out.append((f, -float(np.interp(ft + 1.3, t, y))))
    return out


sift = cv2.SIFT_create(4000)
bf = cv2.BFMatcher()


def feat(f):
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    kp, d = sift.detectAndCompute(g, None)
    return np.float64([k.pt for k in kp]), d


def K_of(p):
    return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]]), np.array(p[3:7]).reshape(4, 1)


def unproj(x, p):
    K, D = K_of(p)
    u = cv2.fisheye.undistortPoints(x.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    v = np.c_[u, np.ones(len(u))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def proj(v, p):
    K, D = K_of(p)
    out = np.full((len(v), 2), 1e4)
    ok = v[:, 2] > 0.05
    if ok.any():
        out[ok] = cv2.fisheye.projectPoints(v[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return out


def kabsch(a, b):
    u, _, vt = np.linalg.svd(b.T @ a)
    return u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt


def ransac_rot(a, b, thr, rng=np.random.default_rng(0)):
    best = None
    for _ in range(400):
        idx = rng.choice(len(a), 3, replace=False)
        R = kabsch(a[idx], b[idx])
        inl = np.arccos(np.clip(np.sum((a @ R.T) * b, 1), -1, 1)) < thr
        if best is None or inl.sum() > best[1].sum():
            best = (R, inl)
    return kabsch(a[best[1]], b[best[1]]), best[1]


p0 = np.array([520.0, W / 2, H_FULL / 2, 0, 0, 0, 0])
pairs = []
for name, seq in (("run1", run1()), ("run2", run2())):
    F = [feat(f) for f, _ in seq]
    for i in range(len(seq)):
        for j in range(i + 1, min(len(seq), i + 12)):
            dy = abs((seq[j][1] - seq[i][1] + 180) % 360 - 180)
            if not 8 <= dy <= 45:
                continue
            m = bf.knnMatch(F[i][1], F[j][1], k=2)
            g = [a for a, b in m if a.distance < 0.7 * b.distance]
            if len(g) < 40:
                continue
            xi, xj = F[i][0][[a.queryIdx for a in g]], F[j][0][[a.trainIdx for a in g]]
            R, inl = ransac_rot(unproj(xi, p0), unproj(xj, p0), np.radians(2.0))
            if inl.sum() < 30:
                continue
            pairs.append(dict(run=name, ti=seq[i][0].split("/")[-1][:7], tj=seq[j][0].split("/")[-1][:7],
                              xi=xi[inl], xj=xj[inl], imu=dy, r0=Rotation.from_matrix(R).as_rotvec()))
            break  # one partner per frame keeps pairs roughly independent
fit = pairs[0::2]
hold = pairs[1::2]
print(f"pairs fit {len(fit)} hold {len(hold)}; matches {sum(len(q['xi']) for q in pairs)}")


def pair_res(p, rv, q):
    R = Rotation.from_rotvec(rv).as_matrix()
    a, b = unproj(q["xi"], p), unproj(q["xj"], p)
    return np.r_[(proj(a @ R.T, p) - q["xj"]).ravel(), (proj(b @ R, p) - q["xi"]).ravel()]


def resid(z, P, use_imu=True):
    p = z[:7]
    out = []
    for k, q in enumerate(P):
        rv = z[7 + 3 * k: 10 + 3 * k]
        out.append(pair_res(p, rv, q))
        if use_imu:
            out.append([IMU_W * (np.degrees(np.linalg.norm(rv)) - q["imu"]) * np.sqrt(len(q["xi"]))])
    return np.concatenate(out)


def solve(P, p_init, fix_p=None):
    if fix_p is None:
        z0 = np.concatenate([p_init] + [q["r0"] for q in P])
        lb = np.r_[[250, 540, 260, -1, -1, -1, -1], -np.inf * np.ones(3 * len(P))]
        ub = np.r_[[1100, 740, 460, 1, 1, 1, 1], np.inf * np.ones(3 * len(P))]
        s = least_squares(resid, z0, args=(P,), bounds=(lb, ub), loss="huber", f_scale=2.0, x_scale="jac", max_nfev=200)
        return s.x
    z0 = np.concatenate([q["r0"] for q in P])
    s = least_squares(lambda z: resid(np.r_[fix_p, z], P, use_imu=False), z0, loss="huber", f_scale=2.0, max_nfev=100)
    return np.r_[fix_p, s.x]


z = solve(fit, p0)
p = z[:7]
print("f=%.1f cx=%.1f cy=%.1f D=%s" % (p[0], p[1], p[2], np.round(p[3:], 4)))
zh = solve(hold, None, fix_p=p)  # held-out: rotations from images only
r_h = np.abs(resid(zh, hold, use_imu=False))
err = []
for k, q in enumerate(hold):
    ang = np.degrees(np.linalg.norm(zh[7 + 3 * k: 10 + 3 * k]))
    err.append(ang - q["imu"])
err = np.array(err)
print("held-out reprojection px: median %.2f p90 %.2f" % (np.median(r_h), np.percentile(r_h, 90)))
print("held-out image-vs-IMU angle: ratio median %.3f, abs err median %.2f deg, p90 %.2f deg"
      % (np.median([(e + q['imu']) / q['imu'] for e, q in zip(err, hold)]), np.median(np.abs(err)),
         np.percentile(np.abs(err), 90)))
hx = np.degrees(np.arccos(unproj(np.array([[0.0, p[2]], [W - 1.0, p[2]]]), p)[:, 2])).sum()
vx = np.degrees(np.arccos(unproj(np.array([[p[1], 0.0], [p[1], H_FULL - 1.0]]), p)[:, 2])).sum()
print("full 1280x720 frame: HFOV %.1f deg, VFOV %.1f deg" % (hx, vx))
K, D = K_of(p)
json.dump(dict(model="opencv_fisheye_kannala_brandt", width=W, height=H_FULL, K=K.tolist(), D=D.ravel().tolist(),
               hfov_deg=float(hx), vfov_deg=float(vx), pairs_fit=len(fit), pairs_holdout=len(hold),
               holdout_reproj_median_px=float(np.median(r_h)),
               holdout_angle_abs_err_median_deg=float(np.median(np.abs(err))),
               note="1280x720 full sensor frame; the recorded frames drop the bottom 72 rows, cy unchanged"),
          open("/tmp/dogcal/calib_fisheye.json", "w"), indent=1)
