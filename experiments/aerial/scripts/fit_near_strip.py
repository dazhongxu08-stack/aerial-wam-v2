#!/usr/bin/env python3
"""Freeze the depth network and fit a near-only correction.

Pixels already between ``lo`` and ``hi`` metres may be pulled closer.
Pixels outside that band, including the open floor near 2.1 m, stay put.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

from experiments.aerial.rl.consequence_model import (  # noqa: E402
    depth_input,
    episode_windows,
    load_checkpoint,
    load_episode,
    survival,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

ROOT = Path("/data/aerial-wam-v2")
MIX = Path("/data/simworld/run/indoor_office_mix")
ART = ROOT / "experiments/aerial/rl/artifacts"
CKPT_CANDIDATES = (
    ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt",
)
CONS = ART / "indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"
OUT = ART / "near_strip_20261102"


def _bump(depth: np.ndarray, alpha: float, lo: float, hi: float) -> np.ndarray:
    d = np.asarray(depth, np.float32)
    span = max(hi - lo, 1e-3)
    t = np.clip((d - lo) / span, 0.0, 1.0)
    bump = 4.0 * t * (1.0 - t)
    return d * (1.0 - float(alpha) * bump)


def correct(depth: np.ndarray, alpha: float, lo: float, hi: float) -> np.ndarray:
    return _bump(depth, alpha, lo, hi)


def thin_columns(depth: np.ndarray, lo: float, hi: float, max_run: int) -> np.ndarray:
    """Columns whose closest pixel sits in (lo, hi), in a run no wider than max_run."""
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    band = d[h // 4 : h - h // 4]
    near = (band.min(axis=0) > lo) & (band.min(axis=0) < hi)
    keep = np.zeros(w, dtype=bool)
    i = 0
    while i < w:
        if not near[i]:
            i += 1
            continue
        j = i + 1
        while j < w and near[j]:
            j += 1
        if (j - i) <= max_run:
            keep[i:j] = True
        i = j
    return keep


def correct_thin(depth: np.ndarray, alpha: float, lo: float, hi: float, max_run: int) -> np.ndarray:
    d = np.asarray(depth, np.float32)
    single = d.ndim == 2
    if single:
        d = d[None]
    out = np.empty_like(d)
    for i in range(len(d)):
        pulled = _bump(d[i], alpha, lo, hi)
        cols = thin_columns(d[i], lo, hi, max_run)
        out[i] = np.where(cols[None, :], pulled, d[i])
    return out[0] if single else out


def under_right(depth: np.ndarray, limit: float = 1.3) -> float:
    d = np.asarray(depth, np.float32)
    h, w = d.shape[-2:]
    band = d[..., h // 4 : h - h // 4, :]
    right = band[..., w - w // 4 :]
    return float((right < limit).mean())


def predict_episode(model, path: Path) -> np.ndarray:
    z = np.load(path)
    rgb = ft.decode_frames(z)
    z.close()
    maps = []
    for i in range(0, len(rgb), 8):
        chunk = np.stack(rgb[i : i + 8])
        with torch.no_grad():
            maps.append(ft.forward_maps(model, chunk).detach().float().cpu().numpy())
    return np.concatenate(maps).astype(np.float32)


def rates(cons, episodes, corrected) -> tuple[float, float, float, float]:
    ps, ys = [], []
    near_da, open_da = [], []
    for fname, _route, ep in episodes:
        ep = dict(ep)
        ep["da3_depth56"] = corrected[fname]
        w = episode_windows(ep, 10)
        if len(w["depth"]) == 0:
            continue
        gt = np.stack([ft.cones_np(d)[0] for d in ep["depth56"][: len(w["depth"])]])
        da = np.stack([ft.cones_np(d)[0] for d in corrected[fname][: len(w["depth"])]])
        near_da.extend(float(a) for g, a in zip(gt, da) if g < 1.2)
        open_da.extend(float(a) for g, a in zip(gt, da) if g >= 2.2)
        with torch.no_grad():
            b = {k: torch.tensor(w[k], dtype=torch.float32) for k in ("depth", "goal", "prev_action", "actions", "alive", "hit")}
            out = cons(b["depth"], b["goal"], b["prev_action"], b["actions"])
            last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
            p = (1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)).numpy()
        y = w["hit"].sum(-1) > 0
        ps.append(p)
        ys.append(y)
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    safe = float(p[~y].mean()) if (~y).any() else 1.0
    hit = float(p[y].mean()) if y.any() else 0.0
    return safe, hit, float(np.median(near_da)) if near_da else 9.0, float(np.median(open_da)) if open_da else 0.0


def main() -> None:
    ckpt = next(p for p in CKPT_CANDIDATES if p.is_file())
    print("depth", ckpt, flush=True)
    pred = DepthMinPredictor.from_checkpoint(str(ckpt), device="cuda")
    model = pred._model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    wanted = []
    for row in review["episodes"]:
        if row.get("route_id") in ("office_val", "office_yaw_val") and row.get("file"):
            wanted.append((Path(row["file"]).name, row["route_id"]))
    cache = {}
    episodes = []
    for fname, route in wanted:
        path = MIX / "episodes" / fname
        raw = predict_episode(model, path)
        ep = load_episode(path)
        if len(raw) < len(ep["state"]):
            raise RuntimeError(f"{fname} frames {len(raw)} < state {len(ep['state'])}")
        cache[fname] = raw[: len(ep["state"])]
        episodes.append((fname, route, ep))
        print(f"cached {fname} {len(raw)} rss {ft.rss_mb():.0f}", flush=True)
    del pred, model
    torch.cuda.empty_cache()
    cons = load_checkpoint(CONS, "cpu")
    raw4 = cache["episode_20030.npz"][4]
    h, w = raw4.shape
    band = raw4[h // 4 : h - h // 4]
    print("frame4 col_min", " ".join(f"{v:.2f}" for v in band.min(axis=0)), flush=True)
    lo, hi = 1.25, 1.65
    runs = (3, 5, 8)
    alphas = (0.0, 0.20, 0.35, 0.50)
    print("run alpha under13 p4 p5 p6 safe open yaw_open near straight keep", flush=True)
    best = None
    base_safe = None
    val_eps = [item for item in episodes if item[1] == "office_val"]
    ep30 = next(item[2] for item in episodes if item[0] == "episode_20030.npz")
    for max_run in runs:
        for alpha in alphas:
            if alpha == 0.0 and max_run != runs[0]:
                continue
            corrected = {
                k: correct_thin(v, alpha, lo, hi, max_run) for k, v in cache.items()
            }
            safe, hit, near_med, open_med = rates(cons, val_eps, corrected)
            if base_safe is None:
                base_safe = safe
            yaw_open = []
            for fname, route, ep in episodes:
                if route != "office_yaw_val":
                    continue
                gt = np.stack([ft.cones_np(d)[0] for d in ep["depth56"]])
                da = np.stack([ft.cones_np(d)[0] for d in corrected[fname][: len(gt)]])
                yaw_open.extend(float(a) for g, a in zip(gt, da) if g >= 2.2)
            yaw_open_med = float(np.median(yaw_open)) if yaw_open else 0.0
            ep = dict(ep30)
            ep["da3_depth56"] = corrected["episode_20030.npz"]
            win = episode_windows(ep, 10)
            with torch.no_grad():
                b = {k: torch.tensor(win[k], dtype=torch.float32) for k in ("depth", "goal", "prev_action", "actions", "alive")}
                out = cons(b["depth"], b["goal"], b["prev_action"], b["actions"])
                last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
                p = (1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)).numpy()
            under = under_right(corrected["episode_20030.npz"][4])
            d4 = depth_input(corrected["episode_20030.npz"][4])
            g4 = win["goal"][4]
            prev4 = win["prev_action"][4]
            straight = np.zeros((10, 4), np.float32)
            straight[:, 0] = 0.8
            keep = np.repeat(win["actions"][4, 0][None], 10, axis=0)
            p_straight = _full_p(cons, d4, g4, prev4, straight)
            p_keep = _full_p(cons, d4, g4, prev4, keep)
            row = {
                "mode": "thin", "alpha": alpha, "lo": lo, "hi": hi, "max_run": max_run,
                "under13": under, "p4": float(p[4]), "p5": float(p[5]), "p6": float(p[6]),
                "safe": safe, "open": open_med, "yaw_open": yaw_open_med,
                "near": near_med, "hit": hit, "straight": p_straight, "keep": p_keep,
                "base_safe": base_safe,
            }
            print(
                f"{max_run:d} {alpha:.2f} {under:.3f} {row['p4']:.3f} {row['p5']:.3f} "
                f"{row['p6']:.3f} {safe:.3f} {open_med:.2f} {yaw_open_med:.2f} "
                f"{near_med:.2f} {p_straight:.3f} {p_keep:.3f}",
                flush=True,
            )
            ok = (
                safe <= base_safe + 0.01 and open_med >= 2.05 and yaw_open_med >= 2.0
                and 0.75 <= near_med <= 1.3 and row["p4"] >= safe + 0.15
            )
            if ok and (best is None or row["p4"] > best["p4"]):
                best = row
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"base": str(ckpt), "best": best}
    (OUT / "near_strip.json").write_text(json.dumps(payload, indent=1))
    print("BEST", json.dumps(best), flush=True)


def _full_p(cons, depth_in, goal_f, prev, actions) -> float:
    with torch.no_grad():
        out = cons(
            torch.tensor(depth_in[None], dtype=torch.float32),
            torch.tensor(np.asarray(goal_f, np.float32)[None]),
            torch.tensor(np.asarray(prev, np.float32)[None]),
            torch.tensor(np.asarray(actions, np.float32)[None]),
        )
        return float(1.0 - survival(out["hazard_logit"])[0, -1])


if __name__ == "__main__":
    main()
