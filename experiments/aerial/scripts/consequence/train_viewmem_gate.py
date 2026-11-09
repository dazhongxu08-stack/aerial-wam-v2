#!/usr/bin/env python3
"""Train short view memory, then finetune survival from frozen krs6.

The decoder must redraw the frame from 10 steps earlier on a real flight where
that frame's near wall is now outside the 90 degree view. Only then is survival
finetuned, in a new directory, from the frozen shorew03 seeds. Progress is not
trained. This does not fly and does not swap weights.
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
OUT = ART / "viewmem_krs6_20261103_recent"
FLIGHT = ROOT / "artifacts/hard_f35_trim_unseen_complex_20261029"
DEPTH_CKPT = ART / "depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
WATER = ART / "water_map_measured_20261012.npz"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
SEEDS = (218, 219, 220)
SPAN = 10
LATENT = 256
H = 10
MIN_LEN = SPAN + 1
MAX_LEN = 32
MEM_STEPS = 8000
MEM_BATCH = 16
FT_STEPS = 4000
FT_BATCH = 128
OPEN_MIN = 0.943
SURV_MIN = 0.95
NEAR_M = 4.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_vm():
    from experiments.aerial.rl import consequence_viewmem as cm
    from experiments.aerial.rl.view_memory import ViewMemory
    return cm, ViewMemory


def episodes():
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    rows = []
    for e in review["episodes"]:
        if not e.get("keep") or not e.get("file"):
            continue
        rid = str(e["route_id"])
        if not rid.startswith("od_train_") or rid in VAL:
            continue
        path = DATA / "episodes" / e["file"]
        if path.is_file():
            rows.append((rid, path))
    val = []
    for e in review["episodes"]:
        if not e.get("keep") or not e.get("file"):
            continue
        rid = str(e["route_id"])
        if rid in VAL:
            path = DATA / "episodes" / e["file"]
            if path.is_file():
                val.append((rid, path))
    return rows, val


def depth_input_seq(depth_m: np.ndarray) -> np.ndarray:
    from experiments.aerial.rl.consequence_model import depth_input
    return np.stack([depth_input(d) for d in depth_m]).astype(np.float32)


def cache_split(pairs):
    from experiments.aerial.rl.consequence_model import load_episode
    seqs = []
    kept = []
    for rid, path in pairs:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < MIN_LEN or len(ep["da3_depth56"]) < MIN_LEN:
            continue
        n = min(len(ep["state"]), len(ep["da3_depth56"]))
        seqs.append(np.asarray(ep["da3_depth56"][:n], np.float16))
        kept.append((rid, path, ep["state"][:n].astype(np.float32), ep["da3_depth56"][:n].astype(np.float16)))
    return seqs, kept


def sample_seq(seqs, rng, length: int) -> np.ndarray:
    for _ in range(20):
        seq = seqs[int(rng.integers(0, len(seqs)))]
        if len(seq) >= length:
            i = int(rng.integers(0, len(seq) - length + 1))
            return np.asarray(seq[i:i + length], np.float32)
    raise RuntimeError("no sequence long enough")


def sample_batch(seqs, rng, batch: int, length: int | None = None):
    """One length per batch. Depth is meters in ``seqs``; the state starts at zero."""
    from experiments.aerial.rl.view_memory import recent_past_index

    if length is None:
        length = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    crops, idxs = [], []
    for _ in range(batch):
        meters = sample_seq(seqs, rng, length)
        crops.append(depth_input_seq(meters))
        idxs.append(recent_past_index(meters))
    return np.stack(crops), np.stack(idxs)


def train_memory(dev: str) -> Path:
    cm, ViewMemory = load_vm()
    ckpt = OUT / "view_memory_best.pt"
    if ckpt.is_file() and (OUT / "memory_history.json").is_file():
        log(f"memory checkpoint exists {ckpt}")
        return ckpt
    tr_pairs, va_pairs = episodes()
    log(f"loading depth train {len(tr_pairs)} val {len(va_pairs)}")
    tr, _ = cache_split(tr_pairs)
    va, _ = cache_split(va_pairs)
    log(f"memory sequences train {len(tr)} val {len(va)}")
    if len(tr) < 10 or len(va) < 3:
        raise SystemExit("not enough sequences")
    vm = ViewMemory(LATENT, SPAN).to(dev)
    opt = torch.optim.AdamW(vm.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    best = float("inf")
    history = []
    t0 = time.time()
    for step in range(1, MEM_STEPS + 1):
        batch, past = sample_batch(tr, rng, MEM_BATCH)
        vm.train()
        loss, parts = vm.loss(torch.as_tensor(batch, device=dev), torch.as_tensor(past, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == MEM_STEPS:
            vm.eval()
            with torch.no_grad():
                short, short_past = sample_batch(va, rng, MEM_BATCH, MIN_LEN)
                _, vparts = vm.loss(
                    torch.as_tensor(short, device=dev), torch.as_tensor(short_past, device=dev))
            rec = float(vparts["rec"])
            history.append({"step": step, "train_rec": float(parts["rec"]), "val_rec": rec, "t": round(time.time() - t0, 1)})
            log(f"memory step {step} train_rec {float(parts['rec']):.4f} val11_rec {rec:.4f}")
            if rec < best:
                best = rec
                torch.save({"model": vm.state_dict(), "span": SPAN, "latent": LATENT, "step": step, "val_rec": rec}, ckpt)
    (OUT / "memory_history.json").write_text(json.dumps(history), encoding="utf-8")
    log(f"memory best val_rec {best:.4f}")
    return ckpt


def unproject(depth, xyz, yaw, stride=2, max_range=12.0):
    h, w = depth.shape
    fx = (w / 2) / math.tan(math.radians(45.0))
    fy = (h / 2) / math.tan(math.radians(58.7 / 2))
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = np.asarray(depth, np.float64)[::stride, ::stride]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    fwd = d[ok]
    right = (u[ok] + 0.5 - w / 2) / fx * fwd
    down = (v[ok] + 0.5 - h / 2) / fy * fwd
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.stack([
        xyz[0] + fwd * c - right * s,
        xyz[1] + fwd * s + right * c,
        xyz[2] - 1.0 - down,
    ], -1)


def outside_count(depth_m, xyz, yaw, past: int, now: int) -> int:
    pts = unproject(depth_m[past], xyz[past], yaw[past])
    if len(pts) == 0:
        return 0
    c, s = math.cos(float(yaw[now])), math.sin(float(yaw[now]))
    dx, dy = pts[:, 0] - xyz[now][0], pts[:, 1] - xyz[now][1]
    fwd = c * dx + s * dy
    left = s * dx - c * dy
    dz = pts[:, 2] - xyz[now][2]
    dist = np.hypot(fwd, left)
    ang = np.degrees(np.arctan2(np.abs(left), np.maximum(fwd, 1e-6)))
    out = (dist <= NEAR_M) & (dist > 0.2) & (np.abs(dz) < 2.0) & ~((fwd > 0.05) & (ang <= 45.0))
    return int(out.sum())


def draw_past(vm, frames, dev):
    vm.eval()
    with torch.no_grad():
        seq = torch.as_tensor(np.asarray(frames, np.float32), device=dev)
        state, _, _ = vm._states(seq.unsqueeze(0))
        drawn = vm._draw(state[:, -1])[0, 1].detach().float().cpu().numpy()
    return drawn


def one_case(vm, frames, depth_m, t, dev):
    past_i = t - SPAN
    drawn = draw_past(vm, frames[:t + 1], dev)
    old = np.asarray(frames[past_i], np.float32)
    cur = np.asarray(frames[t], np.float32)
    near = (np.asarray(depth_m[past_i], np.float32) < NEAR_M) & (np.asarray(depth_m[past_i], np.float32) > 0.05)
    rec = {
        "t": int(t),
        "mse_old": float(np.mean((drawn - old) ** 2)),
        "mse_cur": float(np.mean((drawn - cur) ** 2)),
        "near_n": int(near.sum()),
    }
    if near.sum() >= 10:
        rec["near_mae_old"] = float(np.mean(np.abs(drawn[near] - old[near])))
        rec["near_mae_cur"] = float(np.mean(np.abs(cur[near] - old[near])))
    else:
        rec["near_mae_old"] = None
        rec["near_mae_cur"] = None
    rec["closer"] = bool(rec["mse_old"] < rec["mse_cur"])
    rec["wall"] = bool(
        rec["near_mae_old"] is not None
        and rec["near_mae_cur"] is not None
        and rec["near_mae_old"] + 0.02 < rec["near_mae_cur"]
    )
    return rec


def recon_check(dev: str) -> bool:
    done = OUT / "RECON.json"
    if done.is_file():
        report = json.loads(done.read_text(encoding="utf-8"))
        log(f"recon already {report.get('pass')}")
        return bool(report.get("pass"))
    _, ViewMemory = load_vm()
    vm = ViewMemory(LATENT, SPAN).to(dev)
    raw = torch.load(OUT / "view_memory_best.pt", map_location="cpu")
    vm.load_state_dict(raw["model"])
    vm.eval()
    from experiments.aerial.rl.view_memory import recent_past_index

    _, va_pairs = episodes()
    _, kept = cache_split(va_pairs)
    rng = np.random.default_rng(1)
    val_rows = []
    for _ in range(80):
        _, _, _, depth_m = kept[int(rng.integers(0, len(kept)))]
        depth_m = np.asarray(depth_m, np.float32)
        if len(depth_m) < MIN_LEN:
            continue
        length = int(rng.integers(MIN_LEN, min(MAX_LEN, len(depth_m)) + 1))
        start = int(rng.integers(0, len(depth_m) - length + 1))
        crop = depth_m[start:start + length]
        past_i = int(recent_past_index(crop)[-1])
        frames = depth_input_seq(crop)
        drawn = draw_past(vm, frames, dev)
        old, cur = frames[past_i], frames[-1]
        val_rows.append({
            "past": past_i,
            "mse_old": float(np.mean((drawn - old) ** 2)),
            "mse_cur": float(np.mean((drawn - cur) ** 2)),
            "closer": bool(np.mean((drawn - old) ** 2) < np.mean((drawn - cur) ** 2)),
        })
    flight = c01_recent(vm, dev)
    report = {
        "memory_step": int(raw.get("step", -1)),
        "memory_val_rec": raw.get("val_rec"),
        "val": {
            "n": len(val_rows),
            "frac_closer": float(np.mean([r["closer"] for r in val_rows])) if val_rows else 0.0,
            "median_mse_old": float(np.median([r["mse_old"] for r in val_rows])) if val_rows else None,
            "median_mse_cur": float(np.median([r["mse_cur"] for r in val_rows])) if val_rows else None,
        },
        "c01": flight,
    }
    val_ok = report["val"]["n"] >= 40 and report["val"]["frac_closer"] >= 0.8
    c01_ok = (
        len(flight) == 2
        and all(r["past"] == 9 and r["closer"] for r in flight)
    )
    report["pass"] = bool(val_ok and c01_ok)
    done.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"recon pass {report['pass']} val closer {report['val']['frac_closer']:.2f} "
        f"c01 {[(r['t'], r['past'], r['closer']) for r in flight]}"
    )
    return bool(report["pass"])


def c01_recent(vm, dev):
    from experiments.aerial.rl.view_memory import recent_past_index

    fr = np.load(FLIGHT / "last_frames_unseen_c_01.npz")
    depth = np.asarray(fr["da3_depth56"], np.float32)
    out = []
    for t in (10, 11):
        prefix = depth[:t + 1]
        past_i = int(recent_past_index(prefix)[t])
        frames = depth_input_seq(prefix)
        drawn = draw_past(vm, frames, dev)
        old, cur = frames[past_i], frames[t]
        mse_old = float(np.mean((drawn - old) ** 2))
        mse_cur = float(np.mean((drawn - cur) ** 2))
        out.append({
            "t": t,
            "past": past_i,
            "mse_old": mse_old,
            "mse_cur": mse_cur,
            "closer": bool(mse_old < mse_cur),
        })
    return out


def c01_cases(vm, dev):
    steps = json.loads((FLIGHT / "steps_unseen_c_01.json").read_text(encoding="utf-8"))
    tr = np.load(FLIGHT / "traj_unseen_c_01.npz")
    fr = np.load(FLIGHT / "last_frames_unseen_c_01.npz")
    traj = np.asarray(tr["traj"], np.float64)
    depth = np.asarray(fr["da3_depth56"], np.float32)
    n = min(len(steps), len(traj), len(depth))
    xyz = traj[:n]
    yaw = np.array([float(steps[i]["yaw"]) for i in range(n)], np.float64)
    frames = depth_input_seq(depth[:n])
    out = []
    for t in range(SPAN, n):
        count = outside_count(depth[:n], xyz, yaw, t - SPAN, t)
        if count < 30:
            continue
        rec = one_case(vm, frames, depth[:n], t, dev)
        rec["outside"] = count
        rec["family"] = steps[t]["family"]
        out.append(rec)
    return out


def memory_module(dev: str):
    _, ViewMemory = load_vm()
    vm = ViewMemory(LATENT, SPAN).to(dev).eval()
    raw = torch.load(OUT / "view_memory_best.pt", map_location="cpu")
    vm.load_state_dict(raw["model"])
    for p in vm.parameters():
        p.requires_grad = False
    return vm


@torch.no_grad()
def encode_episode(vm, frames, dev):
    seq = torch.as_tensor(np.asarray(frames, np.float32), device=dev).unsqueeze(0)
    state, _, _ = vm._states(seq)
    return state[0].float().cpu().numpy()


def stack_windows(pairs, vm, dev, shore):
    from experiments.aerial.rl.consequence_model import episode_windows, load_episode
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    parts = {k: [] for k in keys}
    mems = []
    weights = []
    n_ep = 0
    for rid, path in pairs:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < SPAN + 2:
            continue
        n = min(len(ep["state"]), len(ep["da3_depth56"]))
        ep = dict(ep)
        ep["state"] = ep["state"][:n]
        ep["da3_depth56"] = ep["da3_depth56"][:n]
        if "depth56" in ep:
            ep["depth56"] = ep["depth56"][:n]
        if "action" in ep:
            ep["action"] = ep["action"][:n]
        if "collided" in ep:
            ep["collided"] = ep["collided"][:n]
        w = episode_windows(ep, H, (0,), 0)
        frames = depth_input_seq(ep["da3_depth56"])
        h = encode_episode(vm, frames, dev)
        m = min(len(w["depth"]), len(h), n - 1)
        if m < 1:
            continue
        w["depth"] = frames[:m].astype(np.float32)
        mems.append(h[:m].astype(np.float16))
        wt = np.ones(m, np.float64)
        if shore is not None:
            st = ep["state"]
            for i in range(m):
                if shore(float(st[i, 0]), float(st[i, 1])) < 80.0:
                    wt[i] = 0.3
        weights.append(wt)
        for k in keys:
            if k == "depth":
                continue
            parts[k].append(w[k][:m])
        parts["depth"].append(w["depth"])
        n_ep += 1
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    data["memory"] = np.concatenate(mems, axis=0)
    data["sample_w"] = np.concatenate(weights, axis=0)
    log(f"windows {len(data['depth'])} from {n_ep} episodes hit {float(data['hit'].sum()):.0f}")
    return data


def batch_of(data, idx, dev):
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal", "memory")
    return {k: torch.as_tensor(data[k][idx], dtype=torch.float32, device=dev) for k in keys}


def finetune(dev: str) -> None:
    cm, _ = load_vm()
    vm = memory_module(dev)
    tr_pairs, va_pairs = episodes()
    shore = None
    if WATER.is_file():
        sys.path.insert(0, str(ROOT / "experiments/aerial/scripts/consequence"))
        from water_audit import Shore
        shore = Shore(str(WATER))
        log("shore raster on")
    log("stacking train windows")
    tr = stack_windows(tr_pairs, vm, dev, shore)
    log("stacking val windows")
    va = stack_windows(va_pairs, vm, dev, None)
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
            horizon=int(raw["horizon"]), hidden=int(raw["hidden"]), dropout=0.1,
            frame_offsets=tuple(raw.get("frame_offsets", (0,))),
            hist_actions=int(raw.get("hist_actions", 0)),
            view_memory_dim=LATENT, view_span=SPAN,
        ).to(dev)
        cm.load_into_view_memory(model, init)
        model.view_memory.load_state_dict(vm.state_dict())
        for p_ in model.view_memory.parameters():
            p_.requires_grad = False
        dim = model.view_memory_dim
        old, new = [], []
        for name, param in model.named_parameters():
            if name.startswith("view_memory."):
                param.requires_grad = False
            elif name == "ctx.weight":
                param.requires_grad = True
                new.append(param)
            else:
                param.requires_grad = True
                old.append(param)
        def _keep_new(grad, dim=dim):
            out = grad.clone()
            out[:, :-dim] = 0
            return out
        model.ctx.weight.register_hook(_keep_new)
        opt = torch.optim.AdamW(
            [{"params": old, "lr": 3e-5, "weight_decay": 1e-4},
             {"params": new, "lr": 3e-4, "weight_decay": 0.0}],
        )
        rng = np.random.default_rng(seed)
        best = float("inf")
        history = []
        t0 = time.time()
        log(f"finetune c_s{seed} from {init.name}")
        for step in range(1, FT_STEPS + 1):
            idx = rng.choice(n_tr, size=min(FT_BATCH, n_tr), replace=True, p=p)
            model.train()
            model.view_memory.eval()
            b = batch_of(tr, idx, dev)
            out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, memory=b["memory"])
            loss, parts = cm.consequence_loss(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 1:
                g = model.ctx.weight.grad
                new_g = float(g[:, -dim:].abs().sum())
                old_g = float(g[:, :-dim].abs().sum())
                log(f"c_s{seed} ctx grad new {new_g:.4f} old {old_g:.4f}")
                if new_g <= 0.0 or old_g > 0.0:
                    raise SystemExit("memory columns are not the only context weights being trained")
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step()
            if step % 1000 == 0 or step == FT_STEPS:
                val = eval_loss(model, cm, va, dev)
                history.append({"step": step, "train": parts, "val": val, "t": round(time.time() - t0, 1)})
                score = val["hazard"] * 2.0 + val["cones"] + val["dgoal"]
                log(f"c_s{seed} step {step} val {score:.4f} hazard {val['hazard']:.4f}")
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


@torch.no_grad()
def eval_loss(model, cm, data, dev):
    model.eval()
    n = len(data["depth"])
    agg = {"hazard": 0.0, "cones": 0.0, "dgoal": 0.0}
    seen = 0
    for s in range(0, n, 1024):
        idx = np.arange(s, min(n, s + 1024))
        b = batch_of(data, idx, dev)
        out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, memory=b["memory"])
        _, parts = cm.consequence_loss(out, b)
        for k in agg:
            agg[k] += parts[k] * len(idx)
        seen += len(idx)
    model.train()
    model.view_memory.eval()
    return {k: v / max(1, seen) for k, v in agg.items()}


def gate(dev: str) -> None:
    from experiments.aerial.rl.collect_consequence_dataset import goal_bearing_body, sample_macro
    from experiments.aerial.rl.consequence_model import (
        altitude_allowed, altitude_band_allowed, goal_features, history_inputs,
        load_checkpoint, select_candidate, survival, visible_motion_allowed,
    )
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.mpc_consequence_eval import (
        choose_align, cruise_yaw_cap, goal_arc, hold_straight, smooth_turn,
    )
    from experiments.aerial.rl.precompute_env_consequence import env_pair
    from experiments.aerial.rl.probe_counterfactual_airsim import FAMILIES
    cm, _ = load_vm()

    def build_candidates(rng, bearing):
        cands, names = [], []
        for fam in FAMILIES:
            name, seq = sample_macro(rng, H, bearing, name=fam)
            if fam == "fwd":
                seq[:, 0] = 0.8
            if name in ("yaw_fwd", "goal_turn_fwd"):
                seq = smooth_turn(seq, 0.6, 0.10)
            cands.append(seq)
            names.append(name)
        cap = cruise_yaw_cap(bearing, 0.10, 0.04, math.radians(25.0))
        cands.append(goal_arc(bearing, H, 0.8, cap))
        names.append("goal_arc")
        return np.stack(cands), names

    def allow(cands, state, goal, z_lo, z_hi):
        allowed = altitude_allowed(cands, state, goal)
        allowed = allowed & altitude_allowed(cands[:, :1], state, goal)
        banded = allowed & altitude_band_allowed(cands, float(state[2]), z_lo, z_hi, 1, 0.0)
        allowed = banded if banded.any() else allowed
        seen = allowed & visible_motion_allowed(cands, 1, 35.0)
        return seen if seen.any() else allowed

    def flown_index(names, rec):
        logged = np.asarray(rec["cand_surv"], np.float64)
        target = float(rec["surv"])
        best = None
        for i, name in enumerate(names):
            if name != rec["family"]:
                continue
            err = abs(float(logged[i]) - target)
            if best is None or err < best[0]:
                best = (err, i)
        if best is None:
            return None
        return int(best[1])

    steps = json.loads((FLIGHT / "steps_unseen_c_01.json").read_text(encoding="utf-8"))
    tr = np.load(FLIGHT / "traj_unseen_c_01.npz")
    fr = np.load(FLIGHT / "last_frames_unseen_c_01.npz")
    traj = np.asarray(tr["traj"], np.float64)
    goal = np.asarray(tr["goal"], np.float64)
    depth = np.asarray(fr["da3_depth56"], np.float32)
    pred = DepthMinPredictor.from_checkpoint(str(DEPTH_CKPT), device=dev)
    rgb = []
    import cv2
    blob, offsets = fr["jpeg"], fr["jpeg_offsets"]
    for i in range(len(offsets) - 1):
        buf = np.frombuffer(blob[int(offsets[i]):int(offsets[i + 1])], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        rgb.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    env, _ = env_pair(pred._model, rgb)
    env = np.asarray(env, np.float32)
    z_ref = float(traj[0, 2])
    z_lo, z_hi = min(z_ref, float(goal[2])) - 1.0, max(z_ref, float(goal[2])) + 1.0
    rng = np.random.default_rng(0)
    rows = []
    n = min(len(steps), len(traj), len(depth), len(env))
    for t, rec in enumerate(steps[:n]):
        state = np.zeros(7, np.float64)
        state[:3] = traj[t]
        state[6] = float(rec["yaw"])
        bearing = goal_bearing_body(state[:3], float(rec["yaw"]), goal)
        cands, names = build_candidates(rng, float(bearing))
        idx = flown_index(names, rec)
        rows.append({
            "t": t, "state": state, "depth": depth[:t + 1], "env": env[t], "goal": goal,
            "names": names, "cands": cands, "allowed": allow(cands, state, goal, z_lo, z_hi),
            "prev": np.zeros(4, np.float32), "bearing": float(bearing), "flown": idx,
            "family": rec["family"], "logged": float(rec["surv"]),
        })
    xyz = traj[:n]
    yaw = np.array([float(steps[i]["yaw"]) for i in range(n)])
    blind = []
    for row in rows:
        t = row["t"]
        if t < SPAN:
            continue
        if row["flown"] is None:
            continue
        count = outside_count(depth[:n], xyz, yaw, t - SPAN, t)
        a0 = row["cands"][row["flown"], 0]
        if count >= 30 and float(a0[0]) >= 0.5:
            blind.append(t)
    log(f"blind forward steps {blind}")

    def load_ens(root, which, loader):
        models = []
        for s in SEEDS:
            path = root / f"{'b_s' if loader is load_checkpoint and 'env_bh' in str(root) else 'c_s'}{s}" / which
            models.append(loader(path, device=dev).eval())
        return models

    prog = []
    for s in SEEDS:
        prog.append(load_checkpoint(PROG_DIR / f"b_s{s}" / "consequence_best.pt", device=dev).eval())

    def score_surv(models, row, action, use_memory=True):
        frames = depth_input_seq(row["depth"])
        d = torch.as_tensor(frames[-1][None], device=dev)
        g = torch.as_tensor(goal_features(row["state"], row["goal"])[None], device=dev)
        pa = torch.as_tensor(row["prev"][None], device=dev)
        a = torch.as_tensor(np.asarray(action, np.float32)[None], device=dev)
        vals = []
        with torch.no_grad():
            for m in models:
                if use_memory and getattr(m, "view_memory_dim", 0):
                    mem = m.remember(torch.as_tensor(frames, device=dev))
                    out = m(d, g, pa, a, None, memory=mem)
                else:
                    out = m(d, g, pa, a, None)
                vals.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(vals))

    def score_prog(row, action):
        d_k, _ = history_inputs(list(row["depth"][-1:]), [], prog[0].frame_offsets, prog[0].hist_actions)
        d_k = d_k[0] if len(prog[0].frame_offsets) == 1 else d_k
        d = torch.as_tensor(d_k[None], device=dev)
        g = torch.as_tensor(goal_features(row["state"], row["goal"])[None], device=dev)
        pa = torch.as_tensor(row["prev"][None], device=dev)
        a = torch.as_tensor(np.asarray(action, np.float32)[None], device=dev)
        ev = torch.as_tensor(row["env"][None], device=dev)
        vals = []
        with torch.no_grad():
            for m in prog:
                out = m(d, g, pa, a, None, env=ev) if m.env_dim else m(d, g, pa, a, None)
                vals.append(float((-out["dgoal"][..., -1]).detach().cpu()))
        return float(np.mean(vals))

    # Progress does not read memory. Two calls on the same inputs must match.
    prog_diff = 0.0
    for row in rows:
        if row["flown"] is None:
            continue
        a = row["cands"][row["flown"]]
        prog_diff = max(prog_diff, abs(score_prog(row, a) - score_prog(row, a)))
    log(f"progress repeat diff {prog_diff:.3e}")

    opens = []
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    from experiments.aerial.rl.consequence_model import load_episode
    for e in review["episodes"]:
        if len(opens) >= 80:
            break
        if not e.get("keep") or e.get("route_id") not in VAL:
            continue
        path = DATA / "episodes" / e["file"]
        side = path.with_suffix(".da3.npz")
        if not side.is_file():
            continue
        z = load_episode(path)
        depth_all = np.asarray(z["da3_depth56"], np.float32)
        n = min(len(depth_all), len(z["state"]))
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
            st = np.asarray(z["state"][t], np.float64)
            yaw_t = float(st[6])
            g = np.array([st[0] + 30.0 * math.cos(yaw_t), st[1] + 30.0 * math.sin(yaw_t), st[2]])
            prev = np.asarray(z["action"][t - 1], np.float32) if t else np.zeros(4, np.float32)
            act = np.zeros((H, 4), np.float32)
            act[:, 0] = 0.8
            opens.append({"depth": depth_all[:t + 1], "state": st, "goal": g, "prev": prev, "cands": act[None], "flown": 0, "env": np.zeros(1, np.float32)})
    log(f"open frames {len(opens)}")

    tags = [f"consequence_step{s:05d}.pt" for s in (1000, 2000, 3000, 4000)] + ["consequence_best.pt", "consequence_last.pt"]
    packs = [("frozen", FROZEN, "consequence_best.pt", load_checkpoint)]
    for tag in tags:
        if all((OUT / f"c_s{s}" / tag).is_file() for s in SEEDS):
            packs.append((tag, OUT, tag, cm.load_checkpoint))

    results = []
    for label, root, which, loader in packs:
        if label == "frozen":
            models = [load_checkpoint(FROZEN / f"c_s{s}" / which, device=dev).eval() for s in SEEDS]
        else:
            models = [loader(root / f"c_s{s}" / which, device=dev).eval() for s in SEEDS]
        fwd = np.zeros((H, 4), np.float32)
        fwd[:, 0] = 0.8
        fwd_on, fwd_off = [], []
        for t in (10, 11):
            row = rows[t]
            fwd_on.append(score_surv(models, row, fwd, True))
            fwd_off.append(score_surv(models, row, fwd, False))
        arr = np.asarray([score_surv(models, row, row["cands"][0]) for row in opens], np.float64)
        rec = {
            "tag": label,
            "fwd": [round(v, 4) for v in fwd_on],
            "fwd_off": [round(v, 4) for v in fwd_off],
            "fwd_max": round(max(fwd_on), 4),
            "lower_than_off": bool(all(a < b for a, b in zip(fwd_on, fwd_off))),
            "open_min": None if len(arr) == 0 else round(float(arr.min()), 4),
            "open_n": int(len(arr)),
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
        "blind_steps": blind,
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
    train_memory(dev)
    if not recon_check(dev):
        log("reconstruction failed, survival finetune not started")
        raise SystemExit(2)
    finetune(dev)
    gate(dev)


if __name__ == "__main__":
    main()
