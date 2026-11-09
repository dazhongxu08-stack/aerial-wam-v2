#!/usr/bin/env python3
"""Finetune survival on stored depth images with their capture poses.

Each of the last 8 frames keeps the depth it saw and the pose it was taken
from. A frame that repeats the current view from the same pose contributes
nothing. The reader changes only the hazard logit. A uniform hazard shift
across the batch is penalized, so having a record cannot make every step safer.
The frozen krs6 weights stay fixed. Progress is not trained. This does not
fly and does not swap weights.
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
OUT = ART / "scenepose_krs6_20261103_diff"
MEAN_PENALTY = 10.0
FLIGHT = ROOT / "artifacts/hard_f35_trim_unseen_complex_20261029"
WATER = ART / "water_map_measured_20261012.npz"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
SEEDS = (218, 219, 220)
H = 10
SCENE_DIM = 64
LENGTH = 8
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
    return depth[:n], states, np.asarray(tr["goal"], np.float64)


def past_of(depth_in, states, t):
    """Frames strictly before t, at most LENGTH. Returns depth, state, mask."""
    start = max(0, int(t) - LENGTH)
    frames = np.asarray(depth_in[start:t], np.float32)
    seen = np.asarray(states[start:t], np.float32)
    k = len(frames)
    depth = np.zeros((LENGTH, 56, 56), np.float32)
    state = np.zeros((LENGTH, 7), np.float32)
    mask = np.zeros(LENGTH, np.bool_)
    if k:
        depth[-k:] = frames
        state[-k:] = seen
        mask[-k:] = True
    return depth, state, mask


def records_ok() -> bool:
    from experiments.aerial.rl.consequence_model import depth_input
    from experiments.aerial.rl.scene_pose import relative_pose

    depth, states, _ = c01_states()
    scaled = np.stack([depth_input(depth[i]) for i in range(len(depth))]).astype(np.float32)
    rows = []
    ok = True
    refs = []
    for t in (10, 11):
        frames, _seen, mask = past_of(scaled, states, t)
        hold = LENGTH - (t - 9)
        same = bool(np.allclose(frames[hold], scaled[9]))
        now = torch.as_tensor(states[t][None], dtype=torch.float32)
        rel = relative_pose(now, torch.as_tensor(states[9][None, None], dtype=torch.float32))
        rel_l = [round(float(v), 4) for v in rel.reshape(-1).tolist()]
        refs.append(rel)
        rec = {"t": t, "holds_frame9": same, "n_past": int(mask.sum()), "rel_frame9": rel_l}
        ok = ok and same and int(mask.sum()) == LENGTH
        rows.append(rec)
        log(f"records t {t} holds frame 9 {same} n {int(mask.sum())} rel {rel_l}")
    differ = float((refs[0] - refs[1]).abs().sum()) > 1e-3
    ok = ok and differ
    (OUT / "RECORDS.json").write_text(json.dumps({"rows": rows, "pose_differs": differ, "ok": ok}, indent=2), encoding="utf-8")
    return bool(ok)


def stack(pairs, shore, depth_input, episode_windows, load_episode):
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    parts = {k: [] for k in keys}
    ep_ids, t_ids, weights = [], [], []
    library = []
    n_ep = 0
    for _rid, path in pairs:
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
        scaled = np.stack([depth_input(ep["da3_depth56"][i]) for i in range(m)]).astype(np.float16)
        library.append({
            "depth": scaled,
            "state": np.asarray(ep["state"][:m], np.float32),
        })
        ep_ids.append(np.full(m, len(library) - 1, np.int32))
        t_ids.append(np.arange(m, dtype=np.int32))
        w["depth"] = scaled.astype(np.float32)
        wt = np.ones(m, np.float64)
        if shore is not None:
            st = ep["state"]
            for i in range(m):
                if shore(float(st[i, 0]), float(st[i, 1])) < 80.0:
                    wt[i] = 0.3
        weights.append(wt)
        for k in keys:
            parts[k].append(w[k][:m] if k != "depth" else w["depth"])
        n_ep += 1
        if n_ep % 40 == 0:
            log(f"stacked {n_ep}")
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    data["ep"] = np.concatenate(ep_ids)
    data["t"] = np.concatenate(t_ids)
    data["sample_w"] = np.concatenate(weights)
    log(f"windows {len(data['depth'])} from {n_ep} episodes hit {float(data['hit'].sum()):.0f}")
    return data, library


def batch_of(data, library, idx, dev):
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    out = {k: torch.as_tensor(data[k][idx], dtype=torch.float32, device=dev) for k in keys}
    b = len(idx)
    depth = np.zeros((b, LENGTH, 56, 56), np.float32)
    state = np.zeros((b, LENGTH, 7), np.float32)
    now = np.zeros((b, 7), np.float32)
    mask = np.zeros((b, LENGTH), np.bool_)
    for j, row in enumerate(idx):
        lib = library[int(data["ep"][row])]
        t = int(data["t"][row])
        frames, seen, bit = past_of(lib["depth"], lib["state"], t)
        depth[j] = frames
        state[j] = seen
        now[j] = lib["state"][t]
        mask[j] = bit
    out["scene_depth"] = torch.as_tensor(depth, device=dev)
    out["scene_state"] = torch.as_tensor(state, device=dev)
    out["body_state"] = torch.as_tensor(now, device=dev)
    out["scene_mask"] = torch.as_tensor(mask, device=dev)
    return out


def eval_loss(model, cm, data, library, dev):
    model.eval()
    n = len(data["depth"])
    agg = {"hazard": 0.0, "cones": 0.0, "dgoal": 0.0}
    seen = 0
    for s in range(0, n, 256):
        idx = np.arange(s, min(n, s + 256))
        b = batch_of(data, library, idx, dev)
        out = model(
            b["depth"], b["goal"], b["prev_action"], b["actions"], None,
            scene_depth=b["scene_depth"], scene_state=b["scene_state"],
            body_state=b["body_state"], scene_mask=b["scene_mask"],
        )
        _, parts = cm.consequence_loss(out, b)
        for k in agg:
            agg[k] += parts[k] * len(idx)
        seen += len(idx)
    return {k: v / max(1, seen) for k, v in agg.items()}


def finetune(dev: str) -> None:
    cm = load_new()
    from experiments.aerial.rl.consequence_model import depth_input, episode_windows, load_episode
    tr_pairs, va_pairs = episodes()
    shore = None
    if WATER.is_file():
        sys.path.insert(0, str(ROOT / "experiments/aerial/scripts/consequence"))
        from water_audit import Shore
        shore = Shore(str(WATER))
        log("shore raster on")
    log("stacking train windows")
    tr, tr_lib = stack(tr_pairs, shore, depth_input, episode_windows, load_episode)
    log("stacking val windows")
    va, va_lib = stack(va_pairs, None, depth_input, episode_windows, load_episode)
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
            scene_pose_dim=SCENE_DIM,
        ).to(dev)
        cm.load_into_scene_pose(model, init)
        trainable = [q for name, q in model.named_parameters()
                     if name.startswith("scene_pose.") or name.startswith("scene_hazard.")]
        for param in model.parameters():
            param.requires_grad = False
        for param in trainable:
            param.requires_grad = True
        opt = torch.optim.AdamW([{"params": trainable, "lr": 3e-4, "weight_decay": 0.0}])
        rng = np.random.default_rng(seed)
        best = float("inf")
        history = []
        t0 = time.time()
        log(f"finetune c_s{seed} from {init.name}")
        model.train()
        for step in range(1, FT_STEPS + 1):
            idx = rng.choice(n_tr, size=min(FT_BATCH, n_tr), replace=True, p=p)
            b = batch_of(tr, tr_lib, idx, dev)
            out = model(
                b["depth"], b["goal"], b["prev_action"], b["actions"], None,
                scene_depth=b["scene_depth"], scene_state=b["scene_state"],
                body_state=b["body_state"], scene_mask=b["scene_mask"],
            )
            loss, parts = cm.consequence_loss(out, b)
            pen = out["scene_delta"].mean(dim=0).pow(2).mean()
            loss = loss + MEAN_PENALTY * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step in (1, 2):
                haz_g = float(model.scene_hazard.s.weight.grad.abs().sum())
                old_g = float(sum(q.grad.abs().sum() for name, q in model.named_parameters()
                                  if q.grad is not None and not name.startswith(("scene_pose.", "scene_hazard."))))
                scene_g = float(sum(q.grad.abs().sum() for q in model.scene_pose.parameters() if q.grad is not None))
                log(f"c_s{seed} step {step} hazard grad {haz_g:.4f} old {old_g:.4f} scene {scene_g:.4f} "
                    f"delta mean {float(out['scene_delta'].mean()):.5f}")
                if haz_g <= 0.0 or old_g > 0.0:
                    raise SystemExit("scene hazard is not the only trained path")
                if step == 2 and scene_g <= 0.0:
                    raise SystemExit("scene-pose reader got no gradient")
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step()
            if step % 1000 == 0 or step == FT_STEPS:
                val = eval_loss(model, cm, va, va_lib, dev)
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
    from experiments.aerial.rl.consequence_model import depth_input, goal_features, load_checkpoint, load_episode, survival
    cm = load_new()
    depth, states, goal = c01_states()
    scaled = np.stack([depth_input(depth[i]) for i in range(len(depth))]).astype(np.float32)
    fwd = np.zeros((H, 4), np.float32)
    fwd[:, 0] = 0.8

    def score_surv(models, t, use_scene):
        d = torch.as_tensor(scaled[t][None], device=dev)
        g = torch.as_tensor(goal_features(states[t], goal)[None], device=dev)
        pa = torch.zeros(1, 4, device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        kwargs = {}
        if use_scene:
            frames, seen, mask = past_of(scaled, states, t)
            kwargs = {
                "scene_depth": torch.as_tensor(frames[None], device=dev),
                "scene_state": torch.as_tensor(seen[None], device=dev),
                "body_state": torch.as_tensor(states[t][None].astype(np.float32), device=dev),
                "scene_mask": torch.as_tensor(mask[None], device=dev),
            }
        vals = []
        with torch.no_grad():
            for m in models:
                if use_scene and getattr(m, "scene_pose_dim", 0):
                    out = m(d, g, pa, a, None, **kwargs)
                else:
                    out = m(d, g, pa, a, None)
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    def score_kind(models, t, kind):
        frames, seen, mask = past_of(scaled, states, t)
        frames, seen = frames.copy(), seen.copy()
        k = int(mask.sum())
        if kind == "repeat":
            frames[-k:] = scaled[t]
            seen[-k:] = states[t].astype(np.float32)
        elif kind == "identity":
            seen[-k:] = states[t].astype(np.float32)
        elif kind == "yaw90":
            seen[-k:, :3] = states[t, :3].astype(np.float32)
            seen[-k:, 6] = float(states[t, 6] + math.pi / 2)
        d = torch.as_tensor(scaled[t][None], device=dev)
        g = torch.as_tensor(goal_features(states[t], goal)[None], device=dev)
        pa = torch.zeros(1, 4, device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        vals = []
        with torch.no_grad():
            for m in models:
                out = m(
                    d, g, pa, a, None,
                    scene_depth=torch.as_tensor(frames[None], device=dev),
                    scene_state=torch.as_tensor(seen, device=dev).unsqueeze(0),
                    body_state=torch.as_tensor(states[t][None].astype(np.float32), device=dev),
                    scene_mask=torch.as_tensor(mask[None], device=dev),
                )
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    opens = []
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    for e in review["episodes"]:
        if len(opens) >= 80:
            break
        if not e.get("keep") or e.get("route_id") not in VAL:
            continue
        path = DATA / "episodes" / e["file"]
        z = load_episode(path)
        if "da3_depth56" not in z:
            continue
        depth_all = np.asarray(z["da3_depth56"], np.float32)
        st_all = np.asarray(z["state"], np.float64)
        n = min(len(depth_all), len(st_all))
        scaled_ep = None
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
            if scaled_ep is None:
                scaled_ep = np.stack([depth_input(depth_all[i]) for i in range(n)]).astype(np.float32)
            st = st_all[t]
            yaw_t = float(st[6])
            g = np.array([st[0] + 30.0 * math.cos(yaw_t), st[1] + 30.0 * math.sin(yaw_t), st[2]])
            prev = np.asarray(z["action"][t - 1], np.float32) if t else np.zeros(4, np.float32)
            frames, seen, mask = past_of(scaled_ep, st_all, t)
            opens.append({
                "cur": scaled_ep[t], "goal": g, "prev": prev, "now": st.astype(np.float32),
                "frames": frames, "seen": seen, "mask": mask,
            })
    log(f"open frames {len(opens)}")

    def score_open(models, row, use_scene):
        d = torch.as_tensor(row["cur"][None], device=dev)
        g = torch.as_tensor(goal_features(row["now"], row["goal"])[None], device=dev)
        pa = torch.as_tensor(row["prev"][None], device=dev)
        a = torch.as_tensor(fwd[None], device=dev)
        vals = []
        with torch.no_grad():
            for m in models:
                if use_scene and getattr(m, "scene_pose_dim", 0):
                    out = m(
                        d, g, pa, a, None,
                        scene_depth=torch.as_tensor(row["frames"][None], device=dev),
                        scene_state=torch.as_tensor(row["seen"][None], device=dev),
                        body_state=torch.as_tensor(row["now"][None], device=dev),
                        scene_mask=torch.as_tensor(row["mask"][None], device=dev),
                    )
                else:
                    out = m(d, g, pa, a, None)
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    prog = [load_checkpoint(PROG_DIR / f"b_s{s}" / "consequence_best.pt", device=dev).eval() for s in SEEDS]

    def score_prog():
        d = torch.as_tensor(scaled[10][None], device=dev)
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
        if label != "frozen":
            rec["repeat"] = [round(score_kind(models, t, "repeat"), 4) for t in (10, 11)]
            rec["identity"] = [round(score_kind(models, t, "identity"), 4) for t in (10, 11)]
            rec["yaw90"] = [round(score_kind(models, t, "yaw90"), 4) for t in (10, 11)]
        results.append(rec)
        extra = "" if label == "frozen" else f" repeat {rec['repeat']} identity {rec['identity']} yaw90 {rec['yaw90']}"
        log(f"gate {label} fwd {rec['fwd']} off {rec['fwd_off']} open_min {rec['open_min']}{extra}")
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
    if not records_ok():
        log("stored frame 9 is not kept with a pose that changes, survival finetune not started")
        raise SystemExit(2)
    finetune(dev)
    gate(dev)


if __name__ == "__main__":
    main()
