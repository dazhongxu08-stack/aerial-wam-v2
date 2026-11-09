#!/usr/bin/env python3
"""Finetune a sim consequence checkpoint with real Tello windows.

Real windows (``tello_posttrain_windows.py``) carry hazard labels only
(``valid`` = 0 masks cones/dgoal). Every step mixes a sim batch built exactly
like ``train_consequence`` (episodes + branch probes, same shore weighting)
with a real batch, so the sim heads and the candidate ranking keep their
supervision. Whole real runs are held out for validation.

Checkpoint selection: lowest real-val hazard among evals whose sim-val score
(2*hazard + cones + dgoal) stays within ``--sim-tol`` of the init checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger("posttrain_consequence_real")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0], score[label <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--real-windows", required=True)
    ap.add_argument("--real-val-runs", required=True, help="comma list of run dir names held out")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-routes", default="od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21")
    ap.add_argument("--branch-dirs", default="")
    ap.add_argument("--branch-groups", type=int, default=16)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--water-raster", default="")
    ap.add_argument("--min-inland-m", type=float, default=80.0)
    ap.add_argument("--shore-weight", type=float, default=0.0)
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--real-batch", type=int, default=64)
    ap.add_argument("--real-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sim-tol", type=float, default=0.03, help="max relative sim-val score increase")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch

    from experiments.aerial.rl.consequence_model import (
        consequence_loss, group_ranking_loss, load_checkpoint, mirror_batch, save_checkpoint, survival,
    )
    from experiments.aerial.rl.train_consequence import stack_branches, stack_windows

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    model = load_checkpoint(root / args.init_ckpt, device=args.device).to(dev)
    H = int(model.horizon)

    data_dir = root / args.data_dir
    review = json.loads((data_dir / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    val_routes = set(r for r in args.val_routes.split(",") if r)
    eps = sorted(p for p in (data_dir / "episodes").glob("episode_*.npz")
                 if re.fullmatch(r"episode_\d+\.npz", p.name) and p.name in route_of)
    shore = None
    if args.water_raster:
        sys.path.insert(0, str(root / "experiments" / "aerial" / "scripts" / "consequence"))
        from water_audit import Shore
        shore = Shore(str(root / args.water_raster))
    tr = stack_windows([p for p in eps if route_of[p.name] not in val_routes], H, "da3", (0,), 0,
                       shore, args.min_inland_m, args.shore_weight)
    va = stack_windows([p for p in eps if route_of[p.name] in val_routes], H, "da3", (0,), 0,
                       shore, args.min_inland_m)
    br: Dict[str, np.ndarray] = {}
    if args.branch_dirs:
        br = stack_branches([root / d for d in args.branch_dirs.split(",") if d], H, root / args.depth_ckpt,
                            args.device, "da3", frozenset(), (0,), 0, shore=shore,
                            min_inland_m=args.min_inland_m, shore_weight=args.shore_weight)
    br_groups = np.unique(br["group"]) if br else np.zeros(0, np.int64)
    br_index = {int(g): np.flatnonzero(br["group"] == g) for g in br_groups} if br else {}
    br_p = None
    if br:
        gw = np.array([br["group_w"][br_index[int(g)][0]] for g in br_groups])
        br_p = gw / gw.sum()
    tr_p = tr["sample_w"] / tr["sample_w"].sum() if "sample_w" in tr else None

    z = np.load(root / args.real_windows)
    names = [str(n) for n in z["runs"]]
    held = set(r for r in args.real_val_runs.split(",") if r)
    missing = held - set(names)
    if missing:
        raise SystemExit(f"unknown --real-val-runs: {sorted(missing)}")
    held_idx = {i for i, n in enumerate(names) if n in held}
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    real = {k: z[k] for k in keys}
    is_val = np.isin(z["run"], list(held_idx))
    rtr = {k: v[~is_val] for k, v in real.items()}
    rva = {k: v[is_val] for k, v in real.items()}
    rtr_hit = rtr["hit"].sum(-1) > 0
    # Contact windows are rare; sample them at a fixed share of the real batch.
    pos_idx, neg_idx = np.flatnonzero(rtr_hit), np.flatnonzero(~rtr_hit)
    logger.info("sim train %d val %d branch %d | real train %d (hit %d) val %d (hit %d)",
                len(tr["depth"]), len(va["depth"]), len(br.get("depth", [])), len(rtr["depth"]),
                len(pos_idx), len(rva["depth"]), int((rva["hit"].sum(-1) > 0).sum()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def run(b):
        return model(b["depth"], b["goal"], b["prev_action"], b["actions"], None)

    def to_batch(d, idx):
        return {k: torch.as_tensor(d[k][idx], dtype=torch.float32, device=dev) for k in keys}

    @torch.no_grad()
    def evaluate(d) -> Dict[str, float]:
        model.eval()
        n = len(d["depth"])
        agg = {"hazard": 0.0, "cones": 0.0, "dgoal": 0.0}
        p_all, did_all = [], []
        for s in range(0, n, 1024):
            idx = np.arange(s, min(n, s + 1024))
            b = to_batch(d, idx)
            out = run(b)
            _, parts = consequence_loss(out, b)
            for k in agg:
                agg[k] += parts[k] * len(idx)
            last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
            p_all.append((1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)).cpu().numpy())
            did_all.append(b["hit"].sum(-1).cpu().numpy())
        model.train()
        p, did = np.concatenate(p_all), np.concatenate(did_all)
        res = {k: v / max(1, n) for k, v in agg.items()}
        res["collide_abs_err"] = float(np.abs(p - did).mean())
        res["auc"] = auc(p, did)
        res["p_hit_pos"] = float(p[did > 0].mean()) if (did > 0).any() else float("nan")
        res["p_hit_neg"] = float(p[did <= 0].mean())
        return res

    def sim_score(r):
        return 2.0 * r["hazard"] + r["cones"] + r["dgoal"]

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    init = {"step": 0, "sim_val": evaluate(va), "real_val": evaluate(rva), "real_train": evaluate(rtr)}
    sim0 = sim_score(init["sim_val"])
    logger.info("%s", json.dumps(init))
    history = [init]
    best = (init["real_val"]["hazard"], 0)
    save_checkpoint(out_dir / "consequence_best.pt", model, {"step": 0, "init": args.init_ckpt})
    t0 = time.time()
    n_tr = len(tr["depth"])
    n_pos = min(len(pos_idx), max(1, args.real_batch // 8)) if len(pos_idx) else 0
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n_tr, size=args.batch) if tr_p is None else rng.choice(n_tr, size=args.batch, p=tr_p)
        b = to_batch(tr, idx)
        if rng.random() < 0.5:
            b = mirror_batch(b)
        loss, parts = consequence_loss(run(b), b)
        ridx = np.concatenate([rng.choice(pos_idx, size=n_pos) if n_pos else np.zeros(0, np.int64),
                               rng.choice(neg_idx, size=args.real_batch - n_pos)])
        rb = to_batch(rtr, ridx)
        if rng.random() < 0.5:
            rb = mirror_batch(rb)
        rloss, rparts = consequence_loss(run(rb), rb)
        loss = loss + float(args.real_weight) * rloss
        parts["real_hazard"] = rparts["hazard"]
        if br:
            gsel = rng.choice(br_groups, size=min(len(br_groups), args.branch_groups), replace=False, p=br_p)
            bidx = np.concatenate([br_index[int(g)] for g in gsel])
            bb = to_batch(br, bidx)
            if rng.random() < 0.5:
                bb = mirror_batch(bb)
            bout = run(bb)
            bloss, _ = consequence_loss(bout, bb)
            rank = group_ranking_loss(bout, bb, torch.as_tensor(br["group"][bidx], device=dev))
            loss = loss + bloss + float(args.rank_weight) * rank
            parts["branch"] = float(bloss.detach())
            parts["rank"] = float(rank.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "train": parts, "sim_val": evaluate(va), "real_val": evaluate(rva),
                   "t": round(time.time() - t0, 1)}
            ok = sim_score(rec["sim_val"]) <= sim0 * (1.0 + args.sim_tol)
            rec["sim_ok"] = bool(ok)
            if ok and rec["real_val"]["hazard"] < best[0]:
                best = (rec["real_val"]["hazard"], step)
                save_checkpoint(out_dir / "consequence_best.pt", model, {"step": step, "init": args.init_ckpt,
                                                                         "real_val": rec["real_val"]})
            history.append(rec)
            logger.info("%s", json.dumps(rec))
    rec_final = {"best_step": best[1], "real_train_final": evaluate(rtr)}
    save_checkpoint(out_dir / "consequence_last.pt", model, {"step": args.steps, "init": args.init_ckpt})
    (out_dir / "train_history.json").write_text(json.dumps(
        {"args": vars(args), "history": history, **rec_final}, indent=1))
    logger.info("best step %d real-val hazard %.4f", best[1], best[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
