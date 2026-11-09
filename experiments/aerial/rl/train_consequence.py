#!/usr/bin/env python3
"""Train the action-sequence -> consequence predictor on self-collected data.

Only episodes with a DA3 sidecar are used. A few train-region routes are held
out by route id as validation (overfit check inside the train region); the
real generalisation test is the counterfactual probe on the held-out region
(``probe_counterfactual_airsim`` + ``eval_consequence_gate``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("train_consequence")


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(y), np.float64)
    ranks[order] = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def mark_visible_contact(w: Dict[str, np.ndarray], ep: Dict, visible_m: float) -> Dict[str, np.ndarray]:
    """Keep a collision only when that step's forward clearance is under ``visible_m``.

    A logged collision whose clearance stays above that is masked out of the
    hazard loss: the camera never saw a wall, so it is neither a positive nor
    a safe negative. ``hit_phys`` keeps the original collision flags.
    ``cone_now`` is the clearance at the frame the action is chosen.
    """
    from experiments.aerial.rl.consequence_model import cones3_from_depth56

    depth = ep["depth56"]
    t_n = len(w["depth"])
    w["cone_now"] = np.stack([cones3_from_depth56(depth[t]) for t in range(t_n)]).astype(np.float32)
    phys = np.array(w["hit"], copy=True)
    w["hit_phys"] = phys
    fwd = np.exp(np.asarray(w["log_cones"][..., 0], np.float64))
    alive = np.asarray(w["alive"]) > 0
    visible = alive & (phys > 0) & (fwd < float(visible_m))
    invisible = alive & (phys > 0) & (fwd >= float(visible_m))
    keep = alive & ~invisible
    w["hit"] = visible.astype(np.float32)
    w["alive"] = keep.astype(np.float32)
    w["valid"] = (np.asarray(w["valid"]) * keep).astype(np.float32)
    return w


def add_avoidance_windows(tr: Dict[str, np.ndarray], margin_m: float = 0.3,
                          step_m: float = 0.175, yaw_limit: float = 0.3141592653589793,
                          ) -> Dict[str, np.ndarray]:
    """Same frame, yaw-only or a short creep: the wall is not reached, so the hazard is 0.

    Cone and goal losses are masked. These copies exist so a close wall is not
    tied to the single logged push.
    """
    if "cone_now" not in tr or len(tr.get("depth", [])) == 0:
        return tr
    fwd = np.asarray(tr["cone_now"][:, 0], np.float64)
    left = np.asarray(tr["cone_now"][:, 1], np.float64)
    right = np.asarray(tr["cone_now"][:, 2], np.float64)
    h = int(tr["actions"].shape[1])
    yaw_ok = (fwd > margin_m) & (fwd < 4.0)
    slow_a = 0.3
    slow_ok = (fwd > slow_a * step_m * h + margin_m) & (fwd < 4.0)
    parts = [tr]
    specs = (
        (yaw_ok, "yaw"),
        (slow_ok, "slow"),
    )
    for mask, kind in specs:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        part = {k: np.array(v[idx], copy=True) for k, v in tr.items()}
        acts = np.zeros_like(part["actions"])
        if kind == "yaw":
            sign = np.where(left[idx] > right[idx], 1.0, -1.0).astype(np.float32)
            acts[..., 3] = sign[:, None] * np.float32(yaw_limit)
        else:
            acts[..., 0] = np.float32(slow_a)
        part["actions"] = acts
        part["hit"][:] = 0
        if "hit_phys" in part:
            part["hit_phys"][:] = 0
        part["alive"][:] = 1
        part["valid"][:] = 0
        if "sample_w" in part:
            part["sample_w"] = part["sample_w"] * 0.5
        else:
            part["sample_w"] = np.full(len(idx), 0.5, np.float64)
        parts.append(part)
        logger.info("avoidance %s windows %d (forward %.2f–%.2f m)", kind, len(idx),
                    float(fwd[idx].min()) if len(idx) else 0.0, float(fwd[idx].max()) if len(idx) else 0.0)
    if len(parts) == 1:
        return tr
    keys = [k for k in tr if all(k in p for p in parts)]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def relabel_path_goal(ep: Dict, w: Dict[str, np.ndarray], lookahead_m: float) -> Dict[str, np.ndarray]:
    """Point each window at a spot on the path the dog actually walked.

    The stored route goal is often the other side of the street, so the old
    label said that walking made the goal farther. The new target is the
    pose about ``lookahead_m`` further along this episode.
    """
    from experiments.aerial.rl.consequence_model import goal_features

    st = np.asarray(ep["state"], np.float64)
    T, H = w["dgoal"].shape
    pose = np.zeros((T, 7), np.float32)
    goals = np.zeros((T, 4), np.float32)
    dg = np.zeros((T, H), np.float32)
    xy = st[:, :2]
    for t in range(min(T, len(st) - 1)):
        acc = 0.0
        j = t
        while j < len(st) - 1 and acc < lookahead_m:
            j += 1
            acc += float(np.hypot(xy[j, 0] - xy[j - 1, 0], xy[j, 1] - xy[j - 1, 1]))
        g = st[j, :3]
        pose[t] = st[t, :7]
        goals[t] = goal_features(st[t], g)
        d0 = float(np.linalg.norm(st[t, :3] - g))
        for k in range(H):
            if w["alive"][t, k] <= 0:
                break
            i = min(t + k + 1, len(st) - 1)
            dg[t, k] = float(np.linalg.norm(st[i, :3] - g) - d0)
    w["goal"] = goals
    w["dgoal"] = dg
    w["pose"] = pose
    w["dgoal_valid"] = np.array(w["valid"], copy=True)
    return w


def add_goal_contrast(tr: Dict[str, np.ndarray], lookahead_m: float,
                      yaw_limit: float = 0.3141592653589793) -> Dict[str, np.ndarray]:
    """Same picture, three answers the logged walk never showed.

    Standing still closes nothing. A goal 8 m to the side is closed by turning
    toward it and then walking, and made farther by pushing the way the dog is
    already facing. Those two side cases only supervise progress: there is no
    logged collision for an action the dog did not take.
    """
    import math

    from experiments.aerial.rl.consequence_model import goal_features

    if "pose" not in tr or len(tr.get("depth", [])) == 0:
        return tr
    H = int(tr["actions"].shape[1])
    pose = np.asarray(tr["pose"], np.float64)
    n = len(pose)
    base_keys = [k for k in tr if k != "pose"]
    parts = [{k: tr[k] for k in base_keys}]

    still = {k: np.array(v, copy=True) for k, v in parts[0].items()}
    still["actions"][:] = 0
    still["prev_action"][:] = 0
    still["dgoal"][:] = 0
    still["hit"][:] = 0
    still["alive"][:] = 1
    still["valid"][:] = 1
    still["dgoal_valid"][:] = 1
    still["log_cones"][:] = still["log_cones"][:, :1, :]
    parts.append(still)

    n_turn = int(min(H - 1, max(1, round((math.pi / 2) / yaw_limit))))
    for sign in (1.0, -1.0):
        bearing = sign * math.pi / 2
        turn_acts = np.zeros((n, H, 4), np.float32)
        turn_acts[:, :n_turn, 3] = np.float32(sign * yaw_limit)
        turn_acts[:, n_turn:, 0] = 1.0
        push_acts = np.zeros((n, H, 4), np.float32)
        push_acts[:, :, 0] = 1.0
        for acts in (turn_acts, push_acts):
            part = {k: np.array(v, copy=True) for k, v in parts[0].items()}
            gf = np.zeros((n, 4), np.float32)
            dg = np.zeros((n, H), np.float32)
            for i in range(n):
                x, y, z, yaw = pose[i, 0], pose[i, 1], pose[i, 2], float(pose[i, 6])
                gx = x + lookahead_m * math.cos(yaw + bearing)
                gy = y + lookahead_m * math.sin(yaw + bearing)
                g = np.array([gx, gy, z], np.float64)
                gf[i] = goal_features(pose[i], g)
                px, py, psi = x, y, yaw
                for k in range(H):
                    psi += float(acts[i, k, 3])
                    step = max(0.0, float(acts[i, k, 0])) * 1.0
                    px += step * math.cos(psi)
                    py += step * math.sin(psi)
                    dg[i, k] = math.hypot(gx - px, gy - py) - lookahead_m
            part["actions"] = acts
            part["prev_action"][:] = 0
            part["goal"] = gf
            part["dgoal"] = dg.astype(np.float32)
            part["hit"][:] = 0
            part["alive"][:] = 0
            part["valid"][:] = 0
            part["dgoal_valid"][:] = 1
            parts.append(part)
    logger.info("path-goal contrast: logged %d + still %d + side %d", n, n, 4 * n)
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def stack_windows(paths: List[Path], horizon: int, input_depth: str = "da3",
                  frame_offsets: tuple = (0,), hist_actions: int = 0,
                  shore=None, min_inland_m: float = 0.0, shore_weight: float = 0.0,
                  relabel=None, need_env: bool = False,
                  path_weight: Optional[Dict[str, float]] = None,
                  route_of: Optional[Dict[str, str]] = None,
                  visible_m: float = 0.0) -> Dict[str, np.ndarray]:
    from experiments.aerial.rl.consequence_model import episode_windows, load_episode

    parts: Dict[str, List[np.ndarray]] = {}
    n_keep = n_drop = n_no_env = 0
    for p in paths:
        ep = load_episode(p)
        if "da3_depth56" not in ep or len(ep["state"]) < 2:
            continue
        if need_env and "env" not in ep:
            n_no_env += 1
            continue
        if not need_env:
            ep.pop("env", None)
            ep.pop("env_flip", None)
        if input_depth == "gt":
            ep["da3_depth56"] = ep["depth56"]
        w = episode_windows(ep, horizon, frame_offsets, hist_actions)
        if visible_m > 0 and len(w["depth"]):
            w = mark_visible_contact(w, ep, visible_m)
        if relabel is not None:
            w = relabel(p, ep, w)
            if w is None:
                continue
        if path_weight is not None and len(w["depth"]):
            wt = float(path_weight.get(p.name, 1.0))
            if "sample_w" in w:
                w["sample_w"] = np.asarray(w["sample_w"], np.float64) * wt
            else:
                w["sample_w"] = np.full(len(w["depth"]), wt, np.float64)
        if route_of is not None and len(w["depth"]):
            w["route_name"] = np.full(len(w["depth"]), route_of[p.name])
        if shore is not None and min_inland_m > 0 and len(w["depth"]):
            st = np.asarray(ep["state"])
            mask = np.array([shore(float(st[t, 0]), float(st[t, 1])) >= min_inland_m
                             for t in range(len(w["depth"]))], dtype=bool)
            n_keep += int(mask.sum())
            n_drop += int((~mask).sum())
            if shore_weight > 0:
                w["sample_w"] = np.where(mask, 1.0, shore_weight)
            else:
                if not mask.any():
                    continue
                w = {k: v[mask] for k, v in w.items()}
        else:
            n_keep += len(w["depth"])
        for k, v in w.items():
            parts.setdefault(k, []).append(v)
    if shore is not None:
        logger.info("episode windows keep %d %s %d", n_keep,
                    f"shore_weight={shore_weight}" if shore_weight > 0 else "drop_shore", n_drop)
    if n_no_env:
        logger.warning("dropped %d episodes without an env sidecar", n_no_env)
    return {k: np.concatenate(v) for k, v in parts.items()}


def stack_branches(
    branch_dirs: List[Path],
    horizon: int,
    depth_ckpt: Path,
    device: str,
    input_depth: str = "da3",
    exclude_routes: frozenset = frozenset(),
    frame_offsets: tuple = (0,),
    hist_actions: int = 0,
    only_routes: frozenset | None = None,
    shore=None,
    min_inland_m: float = 0.0,
    shore_weight: float = 0.0,
    need_env: bool = False,
) -> Dict[str, np.ndarray]:
    """Candidate windows from train-region branching probes, with a group id per start state."""
    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.consequence_model import branch_windows
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from experiments.aerial.rl.precompute_da3_consequence import decode_frames
    from experiments.aerial.rl.precompute_env_consequence import env_pair

    pred = DepthMinPredictor.from_checkpoint(depth_ckpt, device=device)
    env_head = pred._model.float() if need_env else None
    parts: Dict[str, List[np.ndarray]] = {}
    gid = 0
    for bd in branch_dirs:
        review = json.loads((bd / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
        for rec in review["probes"]:
            if rec.get("train_keep") is False:
                continue
            if rec["route_id"] in exclude_routes or (only_routes is not None and rec["route_id"] not in only_routes):
                continue
            z = np.load(bd / "probes" / rec["file"])
            if "step_depth56" not in z.files:
                continue
            gw = 1.0
            if shore is not None and min_inland_m > 0:
                s = np.asarray(z["state"]).reshape(-1)
                if shore(float(s[0]), float(s[1])) < min_inland_m:
                    if shore_weight <= 0:
                        continue
                    gw = shore_weight
            frames = None
            if input_depth == "gt":
                d56 = z["gt_depth56"]
            else:
                frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
                pred.reset()
                d56 = []
                for rgb in frames:
                    d = pred.predict_depth(Observation(rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False))
                    d56.append(pool_min(d))
            env = None
            if env_head is not None:
                if frames is None:
                    frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
                e, ef = env_pair(env_head, frames[-1:])
                env = (e[0], ef[0])
            w = branch_windows(z, d56, horizon, frame_offsets, hist_actions, env)
            w["group"] = np.full(len(w["depth"]), gid, np.int64)
            w["group_w"] = np.full(len(w["depth"]), gw, np.float64)
            gid += 1
            for k, v in w.items():
                parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()} if parts else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-routes", default="od_train_03,od_train_17,od_train_25,od_train_33")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also write consequence_stepNNNNN.pt on these steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--branch-dirs", default="", help="comma list of train-split branching probe dirs")
    ap.add_argument("--branch-groups", type=int, default=16, help="branch start states per batch")
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--input-depth", default="da3", choices=("da3", "gt"),
                    help="gt = diagnostic only (GT depth is not available on the drone)")
    ap.add_argument("--branch-val", action="store_true",
                    help="hold out branch states on --val-routes and add their hazard loss to checkpoint selection "
                         "(implies --branch-exclude-val)")
    ap.add_argument("--branch-exclude-val", action="store_true",
                    help="drop branch states on --val-routes so they can test ranking in-region")
    ap.add_argument("--frame-offsets", default="0",
                    help="comma list of past-frame offsets fed as depth channels, e.g. 0,2,5")
    ap.add_argument("--hist-actions", type=int, default=0, help="number of past executed actions fed as context")
    ap.add_argument("--water-raster", default="",
                    help="measured water npz; drop episode/branch samples closer than --min-inland-m")
    ap.add_argument("--min-inland-m", type=float, default=80.0)
    ap.add_argument("--shore-weight", type=float, default=0.0,
                    help="0 drops samples closer than --min-inland-m; >0 keeps them at this relative sampling weight")
    ap.add_argument("--geo-top-dir", default="",
                    help="dir of top_<route_id>.npz (topdown_height.py): episode dgoal labels become geodesic "
                         "(around-buildings) distance change; steps without a geodesic value are masked (valid=0); "
                         "episodes on routes without a map are dropped")
    ap.add_argument("--geo-clear-m", type=float, default=1.0)
    ap.add_argument("--geo-inflate-m", type=float, default=1.0)
    ap.add_argument("--init-ckpt", default="", help="start from this checkpoint (same horizon/hidden/history)")
    ap.add_argument("--sem-keepout", action="store_true",
                    help="add the keep-out mask input + keep-out head, trained on synthetic objects "
                         "(keepout_synth); --init-ckpt may be a depth-only checkpoint")
    ap.add_argument("--ko-prob", type=float, default=0.5, help="fraction of windows that get a virtual object")
    ap.add_argument("--ko-weight", type=float, default=2.0)
    ap.add_argument("--ko-radius", default="0.3,1.0", help="keep radius range sampled per window (m)")
    ap.add_argument("--sim-hfov-deg", type=float, default=90.0)
    ap.add_argument("--env-feat", action="store_true",
                    help="feed the depth head's DA3 encoder scene vector (episode_*.env.npz from "
                         "precompute_env_consequence) as context; --init-ckpt may be a model without it")
    ap.add_argument("--env-hidden", type=int, default=64)
    ap.add_argument("--env-drop", type=float, default=0.2, help="fraction of samples whose env vector is dropped")
    ap.add_argument("--max-per-route", type=int, default=0,
                    help="cap train episodes per route (0 keeps all); validation routes are not capped")
    ap.add_argument("--balance-routes", action="store_true",
                    help="give every train route the same total sampling weight")
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0,
                    help="BCE weight on collision steps; 1 keeps the unweighted loss")
    ap.add_argument("--depth-noise", type=float, default=0.0,
                    help="gaussian noise added to log-depth during training")
    ap.add_argument("--near-m", type=float, default=0.0,
                    help="if >0, upweight windows with a forward cone inside this range, or a collision")
    ap.add_argument("--near-weight", type=float, default=4.0)
    ap.add_argument("--horizon-weight", type=float, default=0.0,
                    help="extra loss on the horizon collision probability; 0 keeps the per-step loss")
    ap.add_argument("--select", choices=("loss", "auc", "horizon"), default="loss",
                    help="checkpoint score: loss is the weighted val objective; auc is collision ranking; "
                         "horizon prefers a low safe-window collision probability")
    ap.add_argument("--visible-m", type=float, default=0.0,
                    help="if >0, a collision counts only when forward clearance is under this many meters; "
                         "farther collisions are masked")
    ap.add_argument("--avoid-aug", action="store_true",
                    help="add yaw-only and short-creep copies of close frames, labeled as no contact")
    ap.add_argument("--path-goal-m", type=float, default=0.0,
                    help="label progress against the pose this far ahead on the walked path "
                         "(a0=1 is 1 m), and add still / side-turn copies")
    args = ap.parse_args()
    frame_offsets = tuple(int(x) for x in args.frame_offsets.split(",") if x != "")
    if args.input_depth == "gt" and (frame_offsets != (0,) or args.hist_actions):
        raise SystemExit("history inputs need per-frame depth; branch probes keep GT depth for the last frame only")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch

    from experiments.aerial.rl.consequence_model import (
        ConsequenceModel, consequence_loss, group_ranking_loss, mirror_batch, save_checkpoint, survival,
    )

    torch.manual_seed(args.seed)
    data_dir = root / args.data_dir
    review = json.loads((data_dir / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    val_routes = set(r for r in args.val_routes.split(",") if r)
    eps = sorted(p for p in (data_dir / "episodes").glob("episode_*.npz")
                 if re.fullmatch(r"episode_\d+\.npz", p.name) and p.name in route_of)
    tr_paths = [p for p in eps if route_of[p.name] not in val_routes]
    va_paths = [p for p in eps if route_of[p.name] in val_routes]
    if args.max_per_route > 0:
        cap_rng = np.random.default_rng(args.seed)
        grouped: Dict[str, List[Path]] = {}
        for p in tr_paths:
            grouped.setdefault(route_of[p.name], []).append(p)
        capped: List[Path] = []
        for rid, ps in sorted(grouped.items()):
            if len(ps) > args.max_per_route:
                pick = cap_rng.choice(len(ps), size=args.max_per_route, replace=False)
                ps = [ps[int(i)] for i in sorted(pick)]
            capped.extend(ps)
        logger.info("capped train episodes %d -> %d (max %d per route, %d routes)",
                    len(tr_paths), len(capped), args.max_per_route, len(grouped))
        tr_paths = capped
    shore = None
    if args.water_raster:
        sys.path.insert(0, str(root / "experiments" / "aerial" / "scripts" / "consequence"))
        from water_audit import Shore
        shore = Shore(str(root / args.water_raster))
        logger.info("water raster %s min_inland_m %.1f", args.water_raster, args.min_inland_m)
    relabel = None
    if args.geo_top_dir:
        sys.path.insert(0, str(root / "experiments" / "aerial" / "scripts" / "consequence"))
        from geo_field import GeoField
        fields: Dict[tuple, "GeoField"] = {}
        geo_n = {"steps": 0, "masked": 0, "eps_no_map": 0}

        def relabel(p: Path, ep: Dict, w: Dict[str, np.ndarray]):
            rid = route_of[p.name]
            top_path = root / args.geo_top_dir / f"top_{rid}.npz"
            if not top_path.is_file():
                geo_n["eps_no_map"] += 1
                return None
            st = np.asarray(ep["state"], np.float64)
            goal = np.asarray(ep["meta"]["goal"], np.float64)
            key = (rid, round(float(np.median(st[:, 2]))))
            if key not in fields:
                fields[key] = GeoField(dict(np.load(top_path)), goal[:2], float(key[1]),
                                       args.geo_clear_m, args.geo_inflate_m)
            g = fields[key].lookup(st[:, :2])
            T, H = w["dgoal"].shape
            idx = np.arange(T)[:, None] + np.arange(1, H + 1)[None, :]
            g_end = g[np.minimum(idx, len(g) - 1)]
            dg = g_end - g[:T, None]
            ok = np.isfinite(dg) & (w["alive"] > 0)
            geo_n["steps"] += int((w["alive"] > 0).sum())
            geo_n["masked"] += int(((w["alive"] > 0) & ~ok).sum())
            w["dgoal"] = np.where(ok, dg, 0.0).astype(np.float32)
            w["valid"] = (w["valid"] * ok).astype(np.float32)
            return w

    if args.path_goal_m > 0:
        geo_relabel = relabel

        def relabel(p: Path, ep: Dict, w: Dict[str, np.ndarray], _geo=geo_relabel, _m=float(args.path_goal_m)):
            if _geo is not None:
                w = _geo(p, ep, w)
                if w is None:
                    return None
            return relabel_path_goal(ep, w, _m)

    tr = stack_windows(tr_paths, args.horizon, args.input_depth, frame_offsets, args.hist_actions,
                       shore, args.min_inland_m, args.shore_weight, relabel, need_env=args.env_feat,
                       route_of=route_of if args.balance_routes else None, visible_m=args.visible_m)
    va = stack_windows(va_paths, args.horizon, args.input_depth, frame_offsets, args.hist_actions,
                       shore, args.min_inland_m, relabel=relabel, need_env=args.env_feat,
                       visible_m=args.visible_m) if va_paths else None
    if args.balance_routes and "route_name" in tr:
        names = tr.pop("route_name")
        _, inv, counts = np.unique(names, return_inverse=True, return_counts=True)
        bal = 1.0 / counts[inv].astype(np.float64)
        tr["sample_w"] = tr["sample_w"] * bal if "sample_w" in tr else bal
        logger.info("balanced %d train routes (windows per route min %d max %d)",
                    len(counts), int(counts.min()), int(counts.max()))
    if args.near_m > 0 and len(tr.get("depth", [])):
        fwd = np.exp(tr["log_cones"][:, 0, 0])
        focus = np.maximum(
            np.where(fwd <= args.near_m, args.near_weight, 1.0),
            np.where(tr["hit"].sum(-1) > 0, args.near_weight, 1.0),
        ).astype(np.float64)
        tr["sample_w"] = (tr["sample_w"] if "sample_w" in tr else np.ones(len(focus))) * focus
        logger.info("near focus <= %.1fm weight %.1f on %.0f%% of windows",
                    args.near_m, args.near_weight, 100.0 * float((focus > 1).mean()))
    if args.avoid_aug:
        before = len(tr["depth"])
        tr = add_avoidance_windows(tr)
        logger.info("avoidance copies %d (windows %d -> %d)", len(tr["depth"]) - before, before, len(tr["depth"]))
    if args.path_goal_m > 0:
        alive = tr["dgoal_valid"] > 0
        logger.info("path goal %.1fm logged dgoal med %+.2f frac closer %.2f",
                    args.path_goal_m, float(np.median(tr["dgoal"][alive])),
                    float((tr["dgoal"][alive] < 0).mean()))
        tr = add_goal_contrast(tr, float(args.path_goal_m))
        if va is not None and "pose" in va:
            del va["pose"]
    if args.geo_top_dir:
        logger.info("geodesic labels: %s fields %d", geo_n, len(fields))
    n_tr = len(tr["depth"])
    logger.info("train windows %d (%d eps) val windows %d (%d eps) hit rate %.3f",
                n_tr, len(tr_paths), 0 if va is None else len(va["depth"]), len(va_paths),
                float(tr["hit"].sum() / max(1.0, tr["alive"].sum())))
    br: Dict[str, np.ndarray] = {}
    br_va: Dict[str, np.ndarray] = {}
    if args.branch_dirs:
        bdirs = [root / d for d in args.branch_dirs.split(",") if d]
        br = stack_branches(bdirs, args.horizon, root / args.depth_ckpt, args.device, args.input_depth,
                            frozenset(val_routes) if (args.branch_exclude_val or args.branch_val) else frozenset(),
                            frame_offsets, args.hist_actions, shore=shore, min_inland_m=args.min_inland_m,
                            shore_weight=args.shore_weight, need_env=args.env_feat)
        if args.branch_val:
            br_va = stack_branches(bdirs, args.horizon, root / args.depth_ckpt, args.device, args.input_depth,
                                   frozenset(), frame_offsets, args.hist_actions,
                                   only_routes=frozenset(val_routes), shore=shore, min_inland_m=args.min_inland_m,
                                   need_env=args.env_feat)
            logger.info("held-out branch windows %d", len(br_va.get("depth", [])))
    br_groups = np.unique(br["group"]) if br else np.zeros(0, np.int64)
    br_index = {int(g): np.flatnonzero(br["group"] == g) for g in br_groups} if br else {}
    br_p = None
    if br:
        gw = np.array([br["group_w"][br_index[int(g)][0]] for g in br_groups])
        br_p = gw / gw.sum()
    tr_p = tr["sample_w"] / tr["sample_w"].sum() if "sample_w" in tr else None
    if br:
        logger.info("branch states %d windows %d collided candidates %d",
                    len(br_groups), len(br["depth"]), int((br["hit"].sum(-1) > 0).sum()))

    dev = torch.device(args.device)
    model = ConsequenceModel(horizon=args.horizon, hidden=args.hidden, dropout=args.dropout,
                             frame_offsets=frame_offsets, hist_actions=args.hist_actions,
                             sem_channels=1 if args.sem_keepout else 0,
                             env_dim=int(tr["env"].shape[1]) if args.env_feat else 0,
                             env_hidden=args.env_hidden, env_drop=args.env_drop).to(dev)
    if args.init_ckpt:
        from experiments.aerial.rl.consequence_model import load_into_semantic
        load_into_semantic(model, root / args.init_ckpt)
        logger.info("init from %s", args.init_ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    if "dgoal_valid" in tr:
        keys = keys + ("dgoal_valid",)
    if args.hist_actions:
        keys = keys + ("hist_act",)
    if args.env_feat:
        keys = keys + ("env", "env_flip")
    ko_cfg = None
    if args.sem_keepout:
        import math

        from experiments.aerial.rl.keepout_synth import KeepoutSynthConfig, keepout_side_check, synth_keepout
        keys = keys + ("fut_xy",)
        lo, hi = (float(x) for x in args.ko_radius.split(","))
        ko_cfg = KeepoutSynthConfig(p_object=args.ko_prob, keep_radius_m=(lo, hi),
                                    hfov_rad=math.radians(args.sim_hfov_deg))
        ko_gen = torch.Generator(device=dev).manual_seed(args.seed)
        side = keepout_side_check(torch.as_tensor(tr["depth"][:200000], dtype=torch.float32),
                                  torch.as_tensor(tr["fut_xy"][:200000]), torch.as_tensor(tr["hit"][:200000]),
                                  hfov_rad=ko_cfg.hfov_rad)
        logger.info("keep-out side check (body +y = image right): %s", side)
        if side["n"] >= 50 and side["agree"] < 0.55:
            raise SystemExit(f"keep-out side check failed {side}: body y / image column convention is wrong")

    def run(b: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        return model(b["depth"], b["goal"], b["prev_action"], b["actions"], b.get("hist_act"),
                     b.get("sem"), b.get("keep_radius"), b.get("env"))

    def to_batch(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, "torch.Tensor"]:
        use = [k for k in keys if k in d]
        return {k: torch.as_tensor(d[k][idx], dtype=torch.float32, device=dev) for k in use}

    def with_keepout(b: Dict[str, "torch.Tensor"], gen=None) -> Dict[str, "torch.Tensor"]:
        return synth_keepout(b, gen if gen is not None else ko_gen, ko_cfg) if ko_cfg is not None else b

    @torch.no_grad()
    def evaluate(d: Dict[str, np.ndarray]) -> Dict[str, float]:
        model.eval()
        n = len(d["depth"])
        agg = {"hazard": 0.0, "cones": 0.0, "dgoal": 0.0}
        if ko_cfg is not None:
            agg["keepout"] = 0.0
            val_gen = torch.Generator(device=dev).manual_seed(12345)
        surv_err, ko_err, nb = 0.0, 0.0, 0
        p_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        phys_chunks: List[np.ndarray] = []
        for s in range(0, n, 1024):
            idx = np.arange(s, min(n, s + 1024))
            b = to_batch(d, idx)
            if ko_cfg is not None:
                b = with_keepout(b, val_gen)
            out = run(b)
            _, parts = consequence_loss(out, b, args.ko_weight, args.hazard_pos_weight)
            for k in agg:
                agg[k] += parts[k] * len(idx)
            # P(collide within the logged horizon) vs whether it did.
            last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
            p_hit = 1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)
            did = b["hit"].sum(-1)
            surv_err += float(torch.abs(p_hit - did).sum())
            p_chunks.append(p_hit.detach().float().cpu().numpy())
            y_chunks.append(did.detach().float().cpu().numpy())
            if "hit_phys" in d:
                phys_chunks.append((d["hit_phys"][idx].sum(-1) > 0).astype(np.float32))
            if ko_cfg is not None:
                k_last = (b["ko_alive"].sum(-1) - 1).clamp(min=0).long()
                p_ko = 1.0 - torch.cumprod(1.0 - torch.sigmoid(out["keepout_logit"]), -1).gather(1, k_last[:, None]).squeeze(1)
                ko_err += float(torch.abs(p_ko - b["ko_hit"].sum(-1)).sum())
            nb += len(idx)
        model.train()
        res = {k: v / max(1, nb) for k, v in agg.items()}
        res["collide_abs_err"] = surv_err / max(1, nb)
        if p_chunks:
            p = np.concatenate(p_chunks)
            y = np.concatenate(y_chunks)
            yb = (y > 0).astype(np.int64)
            res["auc"] = _auc(yb, p)
            if yb.any():
                res["p_when_hit"] = float(p[yb == 1].mean())
            if (yb == 0).any():
                res["p_when_safe"] = float(p[yb == 0].mean())
            if phys_chunks:
                res["auc_phys"] = _auc(np.concatenate(phys_chunks).astype(np.int64), p)
        if ko_cfg is not None:
            res["keepout_abs_err"] = ko_err / max(1, nb)
        return res

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n_tr, size=args.batch) if tr_p is None else rng.choice(n_tr, size=args.batch, p=tr_p)
        b = to_batch(tr, idx)
        if rng.random() < 0.5:
            b = mirror_batch(b)
        if args.depth_noise > 0:
            b["depth"] = b["depth"] + torch.randn_like(b["depth"]) * float(args.depth_noise)
        b = with_keepout(b)
        out = run(b)
        loss, parts = consequence_loss(out, b, args.ko_weight, args.hazard_pos_weight)
        if br:
            gsel = rng.choice(br_groups, size=min(len(br_groups), args.branch_groups), replace=False, p=br_p)
            bidx = np.concatenate([br_index[int(g)] for g in gsel])
            bb = to_batch(br, bidx)
            if rng.random() < 0.5:
                bb = mirror_batch(bb)
            bb = with_keepout(bb)
            bout = run(bb)
            bloss, _ = consequence_loss(bout, bb, args.ko_weight, args.hazard_pos_weight)
            rank = group_ranking_loss(bout, bb, torch.as_tensor(br["group"][bidx], device=dev))
            loss = loss + bloss + float(args.rank_weight) * rank
            parts["branch"] = float(bloss.detach())
            parts["rank"] = float(rank.detach())
        if args.horizon_weight > 0:
            last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
            p_h = 1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)
            y = (b["hit"].sum(-1) > 0).float()
            n_safe = (1.0 - y).sum().clamp(min=1.0)
            n_hit = y.sum().clamp(min=1.0)
            # Safe windows have to stay under the 0.05 budget. Hit windows stay the likely class.
            hloss = (p_h * (1.0 - y)).sum() / n_safe + 2.0 * ((1.0 - p_h) * y).sum() / n_hit
            loss = loss + float(args.horizon_weight) * hloss
            parts["horizon"] = float(hloss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "train": parts, "t": round(time.time() - t0, 1)}
            if va is not None:
                rec["val"] = evaluate(va)
                if args.select == "auc":
                    v = -float(rec["val"].get("auc", 0.0))
                elif args.select == "horizon":
                    safe_p = float(rec["val"].get("p_when_safe", 1.0))
                    hit_p = float(rec["val"].get("p_when_hit", 0.0))
                    v = safe_p + max(0.0, 0.50 - hit_p)
                else:
                    v = rec["val"]["hazard"] * 2.0 + rec["val"]["cones"] + rec["val"]["dgoal"]
                    v += rec["val"].get("keepout", 0.0) * args.ko_weight
                if br_va:
                    rec["branch_val"] = evaluate(br_va)
                    v += rec["branch_val"]["hazard"] * 2.0
                if v < best:
                    best = v
                    save_checkpoint(out_dir / "consequence_best.pt", model, {"step": step, "val": rec["val"]})
            history.append(rec)
            logger.info("%s", json.dumps(rec))
            if args.ckpt_every and step % args.ckpt_every == 0:
                save_checkpoint(out_dir / f"consequence_step{step:05d}.pt", model,
                                {"step": step, "val": rec.get("val")})
    save_checkpoint(out_dir / "consequence_last.pt", model, {"step": args.steps})
    if va is None:
        save_checkpoint(out_dir / "consequence_best.pt", model, {"step": args.steps})
    (out_dir / "train_history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
