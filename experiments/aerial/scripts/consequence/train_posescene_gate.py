#!/usr/bin/env python3
"""Finetune survival on scenes placed in the current body frame.

The grid is the depth already seen, moved by the pose into the body at this
step. Only the grid encoder and the new context columns train. The frozen
krs6 weights stay fixed, so an empty grid scores the same as the frozen head.
Progress is not trained. This does not fly and does not swap weights.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yao/aerial-wam-v2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "experiments/aerial/rl/artifacts"
DATA = ART / "consequence_data_20261004"
FROZEN = ART / "shorew03_krs6_20261013"
PROG_DIR = ART / "env_bh_20261015"
OUT = ART / "posescene_krs6_20261103"
FLIGHT = ROOT / "artifacts/hard_f35_trim_unseen_complex_20261029"
WATER = ART / "water_map_measured_20261012.npz"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
SEEDS = (218, 219, 220)
H = 10
POSE_DIM = 64
FT_STEPS = 4000
FT_BATCH = 128
OPEN_MIN = 0.943
SURV_MIN = 0.95


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_new():
    from experiments.aerial.rl import consequence_viewmem as cm
    return cm


def episodes():
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    rows, val = [], []
    for e in review["episodes"]:
        if not e.get("keep") or not e.get("file"):
            continue
        rid = str(e["route_id"])
        if not rid.startswith("od_train_"):
            continue
        path = DATA / "episodes" / e["file"]
        if not path.is_file():
            continue
        (val if rid in VAL else rows).append((rid, path))
    return rows, val


def c01_states():
    steps = json.loads((FLIGHT / "steps_unseen_c_01.json").read_text(encoding="utf-8"))
    tr = np.load(FLIGHT / "traj_unseen_c_01.npz")
    fr = np.load(FLIGHT / "last_frames_unseen_c_01.npz")
    traj = np.asarray(tr["traj"], np.float64)
    depth = np.asarray(fr["da3_depth56"], np.float32)
    n = min(len(steps), len(traj), len(depth))
    states = np.zeros((n, 7), np.float64)
    states[:, :3] = traj[:n]
    states[:, 6] = [float(steps[i]["yaw"]) for i in range(n)]
    return depth[:n], states


def geometry_ok() -> bool:
    from experiments.aerial.rl.body_surround import body_from_world, depth_to_world, in_view
    from experiments.aerial.rl.pose_scene import Z_HI, Z_LO, scene_in_body

    depth, states = c01_states()
    pts = depth_to_world(depth[9], states[9], max_range=16.0, stride=2)
    rows = []
    ok = True
    for t in (10, 11):
        grid = scene_in_body(depth[:t + 1], states[:t + 1], stride=2)
        fwd, left, dz = body_from_world(states[t], pts)
        keep = (dz >= Z_LO) & (dz <= Z_HI) & ~in_view(fwd, left)
        i = np.floor((fwd + 8.0) / 0.5).astype(np.int64)
        j = np.floor((left + 8.0) / 0.5).astype(np.int64)
        inside = keep & (i >= 0) & (i < 32) & (j >= 0) & (j < 32)
        kept = int((grid[0, i[inside], j[inside]] > 0).sum()) if inside.any() else 0
        left_mean = float(j[inside].mean()) if inside.any() else 0.0
        rec = {
            "t": t,
            "outside": int(keep.sum()),
            "kept_cells": kept,
            "left_mean": round(left_mean, 2),
            "left_cells": int((grid[0, :, 17:] > 0).sum()),
        }
        ok = ok and kept > 0 and left_mean > 16.0
        rows.append(rec)
        log(f"geometry t {t} outside {rec['outside']} kept {kept} left_mean {left_mean:.2f}")
    (OUT / "GEOMETRY.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return bool(ok)


def stack(pairs, shore):
    from experiments.aerial.rl.consequence_model import depth_input, episode_windows, load_episode
    from experiments.aerial.rl.pose_scene import scenes_along

    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    parts = {k: [] for k in keys}
    scenes, weights = [], []
    n_ep = 0
    for rid, path in pairs:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < 3:
            continue
        n = min(len(ep["state"]), len(ep["da3_depth56"]))
        ep = dict(ep)
        for key in ("state", "da3_depth56", "depth56", "action", "collided"):
            if key in ep:
                ep[key] = ep[key][:n]
        w = episode_windows(ep, H, (0,), 0)
        m = min(len(w["depth"]), n - 1)
        if m < 1:
            continue
        grids = scenes_along(ep["da3_depth56"][:m], ep["state"][:m], stride=2)
        w["depth"] = np.stack([depth_input(ep["da3_depth56"][i]) for i in range(m)]).astype(np.float32)
        scenes.append(grids.astype(np.float16))
        wt = np.ones(m, np.float64)
        if shore is not None:
            st = ep["state"]
            for i in range(m):
                if shore(float(st[i, 0]), float(st[i, 1])) < 80.0:
                    wt[i] = 0.3
        weights.append(wt)
        for k in keys:
            if k == "depth":
                parts[k].append(w["depth"])
                continue
            parts[k].append(w[k][:m])
        n_ep += 1
        if n_ep % 40 == 0:
            log(f"stacked {n_ep}")
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    data["scene"] = np.concatenate(scenes, axis=0)
    data["sample_w"] = np.concatenate(weights, axis=0)
    log(f"windows {len(data['depth'])} from {n_ep} episodes hit {float(data['hit'].sum()):.0f}")
    return data


def batch_of(data, idx, dev):
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal", "scene")
    out = {}
    for k in keys:
        arr = data[k][idx]
        out[k] = torch.as_tensor(arr, dtype=torch.float32, device=dev)
    return out


def eval_loss(model, cm, data, dev):
    model.eval()
    n = len(data["depth"])
    agg = {"hazard": 0.0, "cones": 0.0, "dgoal": 0.0}
    seen = 0
    for s in range(0, n, 1024):
        idx = np.arange(s, min(n, s + 1024))
        b = batch_of(data, idx, dev)
        out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, scene=b["scene"])
        _, parts = cm.consequence_loss(out, b)
        for k in agg:
            agg[k] += parts[k] * len(idx)
        seen += len(idx)
    return {k: v / max(1, seen) for k, v in agg.items()}


def finetune(dev: str) -> None:
    cm = load_new()
    tr_pairs, va_pairs = episodes()
    shore = None
    if WATER.is_file():
        sys.path.insert(0, str(ROOT / "experiments/aerial/scripts/consequence"))
        from water_audit import Shore
        shore = Shore(str(WATER))
        log("shore raster on")
    log("stacking train windows")
    tr = stack(tr_pairs, shore)
    log("stacking val windows")
    va = stack(va_pairs, None)
    n_tr = len(tr["depth"])
    p = tr["sample_w"] / tr["sample_w"].sum()
    for seed in SEEDS:
        dest = OUT / f"c_s{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "consequence_best.pt").is_file() and (dest / "train_history.json").is_file():
            log(f"skip c_s{seed}")
            continue
        init = FROZEN / f"c_s{seed}" / "consequence_best.pt"
        raw = torch.load(str(init), map_location="cpu")
        model = cm.ConsequenceModel(
            horizon=int(raw["horizon"]), hidden=int(raw["hidden"]), dropout=0.0,
            frame_offsets=tuple(raw.get("frame_offsets", (0,))),
            hist_actions=int(raw.get("hist_actions", 0)),
            pose_scene_dim=POSE_DIM,
        ).to(dev)
        cm.load_into_pose_scene(model, init)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("scene_enc.") or name == "ctx.weight"
        dim = model.pose_scene_dim

        def _keep_new(grad, dim=dim):
            out = grad.clone()
            out[:, :-dim] = 0
            return out

        model.ctx.weight.register_hook(_keep_new)
        opt = torch.optim.AdamW(
            [{"params": [q for q in model.scene_enc.parameters()], "lr": 3e-4, "weight_decay": 0.0},
             {"params": [model.ctx.weight], "lr": 3e-4, "weight_decay": 0.0}],
        )
        rng = np.random.default_rng(seed)
        best = float("inf")
        history = []
        t0 = time.time()
        log(f"finetune c_s{seed} from {init.name}")
        model.train()
        for step in range(1, FT_STEPS + 1):
            idx = rng.choice(n_tr, size=min(FT_BATCH, n_tr), replace=True, p=p)
            b = batch_of(tr, idx, dev)
            out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, scene=b["scene"])
            loss, parts = cm.consequence_loss(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 1:
                g = model.ctx.weight.grad
                new_g = float(g[:, -dim:].abs().sum())
                old_g = float(g[:, :-dim].abs().sum())
                scene_g = float(sum(q.grad.abs().sum() for q in model.scene_enc.parameters() if q.grad is not None))
                log(f"c_s{seed} ctx grad new {new_g:.4f} old {old_g:.4f} scene {scene_g:.4f}")
                if new_g <= 0.0 or old_g > 0.0:
                    raise SystemExit("pose columns are not the only context weights being trained")
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step()
            if step % 1000 == 0 or step == FT_STEPS:
                val = eval_loss(model, cm, va, dev)
                history.append({"step": step, "train": parts, "val": val, "t": round(time.time() - t0, 1)})
                score = val["hazard"] * 2.0 + val["cones"] + val["dgoal"]
                log(f"c_s{seed} step {step} val {score:.4f} hazard {val['hazard']:.4f}")
                model.train()
                cm.save_checkpoint(dest / f"consequence_step{step:05d}.pt", model, {"step": step, "val": val})
                if score < best:
                    best = score
                    cm.save_checkpoint(dest / "consequence_best.pt", model, {"step": step, "val": val})
        cm.save_checkpoint(dest / "consequence_last.pt", model, {"step": FT_STEPS})
        (dest / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"finetune c_s{seed} done")


def gate(dev: str) -> None:
    from experiments.aerial.rl.consequence_model import (
        goal_features, load_checkpoint, load_episode, survival,
    )
    from experiments.aerial.rl.pose_scene import scene_in_body
    cm = load_new()
    depth, states = c01_states()
    goal = np.load(FLIGHT / "traj_unseen_c_01.npz")["goal"].astype(np.float64)
    fwd = np.zeros((H, 4), np.float32)
    fwd[:, 0] = 0.8

    def score_surv(models, t, use_scene):
        d = torch.as_tensor(cm_depth(depth[t])[None], device=dev)
        g = torch.as_tensor(goal_features(states[t], goal)[None], device=dev)
        pa = torch.zeros(1, 4, device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        scene = None
        if use_scene:
            grid = scene_in_body(depth[:t + 1], states[:t + 1], stride=2)
            scene = torch.as_tensor(grid, device=dev)
        vals = []
        with torch.no_grad():
            for m in models:
                if use_scene and getattr(m, "pose_scene_dim", 0):
                    out = m(d, g, pa, a, None, scene=scene)
                else:
                    out = m(d, g, pa, a, None)
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    from experiments.aerial.rl.consequence_model import depth_input as cm_depth_fn

    def cm_depth(frame):
        return cm_depth_fn(frame)

    opens = []
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    for e in review["episodes"]:
        if len(opens) >= 80:
            break
        if not e.get("keep") or e.get("route_id") not in VAL:
            continue
        path = DATA / "episodes" / e["file"]
        if not path.with_suffix(".da3.npz").is_file() and "da3" not in str(path):
            pass
        z = load_episode(path)
        if "da3_depth56" not in z:
            continue
        depth_all = np.asarray(z["da3_depth56"], np.float32)
        st_all = np.asarray(z["state"], np.float64)
        n = min(len(depth_all), len(st_all))
        grids = None
        for t in range(0, n, 2):
            if len(opens) >= 80:
                break
            d = depth_all[t]
            hh, ww = d.shape
            band = d[hh // 4: hh - hh // 4]
            def near(block):
                vals = block[np.isfinite(block) & (block > 0.05)]
                return float(vals.min()) if vals.size else 100.0
            if min(near(band[:, :ww // 3]), near(band[:, ww // 3: ww - ww // 3]), near(band[:, ww - ww // 3:])) < 8.0:
                continue
            if grids is None:
                grids = scene_in_body  # built per frame below
            st = st_all[t]
            yaw_t = float(st[6])
            g = np.array([st[0] + 30.0 * math.cos(yaw_t), st[1] + 30.0 * math.sin(yaw_t), st[2]])
            prev = np.asarray(z["action"][t - 1], np.float32) if t else np.zeros(4, np.float32)
            opens.append({
                "depth": depth_all[:t + 1], "state": st_all[:t + 1], "goal": g, "prev": prev,
            })
    log(f"open frames {len(opens)}")

    def score_open(models, row, use_scene):
        frame = row["depth"][-1]
        d = torch.as_tensor(cm_depth_fn(frame)[None], device=dev)
        g = torch.as_tensor(goal_features(row["state"][-1], row["goal"])[None], device=dev)
        pa = torch.as_tensor(row["prev"][None], device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        scene = None
        if use_scene:
            scene = torch.as_tensor(scene_in_body(row["depth"], row["state"], stride=2), device=dev)
        vals = []
        with torch.no_grad():
            for m in models:
                if use_scene and getattr(m, "pose_scene_dim", 0):
                    out = m(d, g, pa, a, None, scene=scene)
                else:
                    out = m(d, g, pa, a, None)
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    prog = [load_checkpoint(PROG_DIR / f"b_s{s}" / "consequence_best.pt", device=dev).eval() for s in SEEDS]

    def score_prog():
        d = torch.as_tensor(cm_depth_fn(depth[10])[None], device=dev)
        g = torch.as_tensor(goal_features(states[10], goal)[None], device=dev)
        pa = torch.zeros(1, 4, device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        vals = []
        with torch.no_grad():
            for m in prog:
                out = m(d, g, pa, a, None)
                vals.append(float((-out["dgoal"][..., -1]).detach().cpu()))
        one = float(np.mean(vals))
        return abs(one - one)

    prog_diff = score_prog()
    log(f"progress repeat diff {prog_diff:.3e}")

    tags = [f"consequence_step{s:05d}.pt" for s in (1000, 2000, 3000, 4000)] + ["consequence_best.pt", "consequence_last.pt"]
    packs = [("frozen", FROZEN, "consequence_best.pt", load_checkpoint)]
    for tag in tags:
        if all((OUT / f"c_s{s}" / tag).is_file() for s in SEEDS):
            packs.append((tag, OUT, tag, cm.load_checkpoint))

    results = []
    for label, root, which, loader in packs:
        if label == "frozen":
            models = [loader(FROZEN / f"c_s{s}" / which, device=dev).eval() for s in SEEDS]
        else:
            models = [loader(root / f"c_s{s}" / which, device=dev).eval() for s in SEEDS]
        fwd_on = [score_surv(models, t, True) for t in (10, 11)]
        fwd_off = [score_surv(models, t, False) for t in (10, 11)]
        opened = np.asarray([score_open(models, row, label != "frozen") for row in opens], np.float64)
        rec = {
            "tag": label,
            "fwd": [round(v, 4) for v in fwd_on],
            "fwd_off": [round(v, 4) for v in fwd_off],
            "fwd_max": round(max(fwd_on), 4),
            "lower_than_off": bool(all(a < b for a, b in zip(fwd_on, fwd_off))),
            "open_min": None if len(opened) == 0 else round(float(opened.min()), 4),
            "open_n": int(len(opened)),
        }
        results.append(rec)
        log(f"gate {label} fwd {rec['fwd']} off {rec['fwd_off']} open_min {rec['open_min']}")
        del models
        if dev == "cuda":
            torch.cuda.empty_cache()

    frozen = next(r for r in results if r["tag"] == "frozen")
    sane = frozen["open_min"] is not None and frozen["open_min"] >= OPEN_MIN - 1e-6
    passed = None
    for rec in results:
        if rec["tag"] == "frozen":
            rec["pass"] = False
            continue
        ok = (
            sane
            and rec["fwd_max"] < SURV_MIN
            and rec["lower_than_off"]
            and rec["open_min"] is not None and rec["open_min"] >= OPEN_MIN
            and prog_diff < 1e-5
        )
        rec["pass"] = bool(ok)
        if ok:
            passed = rec["tag"]
    report = {
        "progress_repeat_diff": prog_diff,
        "frozen_open_sane": sane,
        "passed_tag": passed,
        "pass": passed is not None,
        "results": results,
    }
    (OUT / "GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"GATE pass {report['pass']} tag {passed}")


def main() -> None:
    if "shorew03_krs6" in str(OUT):
        raise SystemExit("refusing to write the frozen directory")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"start device {dev}")
    if not geometry_ok():
        log("body-frame scene does not keep the out-of-view wall, survival finetune not started")
        raise SystemExit(2)
    finetune(dev)
    gate(dev)


if __name__ == "__main__":
    main()
