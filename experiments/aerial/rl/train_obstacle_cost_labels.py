#!/usr/bin/env python3
"""Offline directional obstacle-cost labels, pack, train, and gate.

Plan 2026-09-21 tasks 2 / 4 / 5. Labels use simulator GT depth only (never
DA3 / S-8j). Online imagine / planner must not call wedge helpers here.

Workflow::

    # 1) GT depth frames → clearance labels (optional intermediate)
    python -m experiments.aerial.rl.train_obstacle_cost_labels label \\
        --depth-npz frames.npz --out labels_raw.npz

    # 2) RGB (+ optional depth) → packed [h‖z] + labels (task 4 input)
    python -m experiments.aerial.rl.train_obstacle_cost_labels pack \\
        --wm-ckpt .../wm_step_8500.pt --frames frames.npz --out labels.npz

    # 3) Freeze backbone; train obstacle_cost_head; save with trained flag
    python -m experiments.aerial.rl.train_obstacle_cost_labels train \\
        --wm-ckpt .../wm_step_8500.pt --labels labels.npz --out-ckpt .../wm_obs.pt

    # 4) Task-5 three sorts (exit 1 if any fail)
    python -m experiments.aerial.rl.train_obstacle_cost_labels gate \\
        --wm-ckpt .../wm_obs.pt --labels labels.npz --report gate.json

``frames.npz`` keys for pack:
  ``rgb`` [N,H,W,3] uint8, ``action_xyz`` [N,3] or ``action`` [N,4],
  optional ``depth`` [N,H,W] (required to auto-label), ``proprio`` [N,4],
  ``group`` string array (``fwd_empty`` / ``fwd_near`` / ``left_near``).

GT corpus convention: ``experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from experiments.aerial.rl.depth_geometry import (
    clearance_to_obstacle_label,
    directional_clearance_m,
)
from experiments.aerial.rl.reward import (
    OBSTACLE_COST_EMPTY,
    OBSTACLE_COST_NEAR,
    STRAIGHT_WEIGHT,
    directional_oa_reward_cfg,
    path_shaping_terms,
    reward_terms,
)

logger = logging.getLogger(__name__)

GT_DEPTH_CORPUS_REL = (
    "experiments/aerial/rl/artifacts/obstacle_cost_gt_depth"
)

# Candidate body deltas for task-5 sorts (metres|rad per step @ 5 Hz).
_CAND = {
    "fwd": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "side": np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float64),
    "climb": np.array([0.7, 0.0, 0.7, 0.0], dtype=np.float64),
    "hover": np.zeros(4, dtype=np.float64),
    "away": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "left": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
    "empty_close": np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float64),
}


def label_frame(
    depth: np.ndarray,
    action_xyz: np.ndarray,
    *,
    percentile: float = 5.0,
    d_near: float = 3.0,
    d_far: float = 22.0,
) -> Dict[str, float]:
    """GT depth + body (dx,dy,dz) → clearance_m and [0,1] obstacle label."""
    c = directional_clearance_m(
        depth, action_xyz, percentile=float(percentile)
    )
    y = clearance_to_obstacle_label(c, d_near=float(d_near), d_far=float(d_far))
    return {"clearance_m": float(c), "obstacle_label": float(y)}


def label_batch(
    depths: np.ndarray,
    actions: np.ndarray,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    depths = np.asarray(depths)
    actions = np.asarray(actions, dtype=np.float64)
    n = int(depths.shape[0])
    clear = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out = label_frame(depths[i], actions[i, :3], **kwargs)
        clear[i] = out["clearance_m"]
        labels[i] = out["obstacle_label"]
    return {"clearance_m": clear, "obstacle_label": labels}


def score_candidate(
    action: np.ndarray,
    goal_rel: np.ndarray,
    progress: float,
    obstacle_cost: float,
    *,
    cfg=None,
) -> float:
    """Same scalar used by real/imagined reward (task-3 coefficients)."""
    cfg = cfg or directional_oa_reward_cfg()
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    shaping = path_shaping_terms(a, goal_rel, float(progress), cfg=cfg)
    prog_eff = float(shaping.get("progress_eff", progress))
    return float(
        reward_terms(
            prog_eff,
            float(obstacle_cost),
            float(np.linalg.norm(a)),
            cfg,
            path_shaping_val=float(shaping["path_shaping"]),
        )["reward"]
    )


def gate_sort_checks(
    *,
    r_fwd_empty: float,
    r_side_empty: float,
    r_hover: float,
    r_away: float,
    r_fwd_near: float,
    r_empty_approach: float,
    r_fwd_left_near: float,
    r_left_left_near: float,
) -> Dict[str, bool]:
    """Task-5 three sorts (constants or learned costs already folded in)."""
    return {
        "fwd_empty_wins": bool(
            r_fwd_empty > r_side_empty
            and r_fwd_empty > r_hover
            and r_fwd_empty > r_away
        ),
        "fwd_near_loses": bool(r_empty_approach > r_fwd_near),
        "left_near_prefers_fwd": bool(
            r_fwd_left_near > r_left_left_near
            and r_fwd_left_near > r_hover
            and r_fwd_left_near > r_away
        ),
    }


def _pad_action4(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = a.shape[0]
    if a.shape[-1] >= 4:
        return a[:, :4].astype(np.float32, copy=False)
    pad = np.zeros((n, 4 - a.shape[-1]), dtype=np.float32)
    return np.concatenate([a.astype(np.float32), pad], axis=-1)


def _progress_along_goal(action: np.ndarray, goal_rel: np.ndarray) -> float:
    """Analytic Δ‖g‖ for one body delta (matches NavigationReward carrot path)."""
    from experiments.aerial.rl.goal_features import advance_goal_rel_body

    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    g0 = float(g[3]) if g.size > 3 else float(np.linalg.norm(g[:3]))
    g1 = float(advance_goal_rel_body(g, action)[3])
    return g0 - g1


def pack_features_from_frames(
    dyn: Any,
    rgbs: np.ndarray,
    actions: np.ndarray,
    *,
    depths: Optional[np.ndarray] = None,
    proprios: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    percentile: float = 5.0,
    d_near: float = 3.0,
    d_far: float = 22.0,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """RGB → packed ``feature=[h‖z]`` + action + obstacle labels.

    If ``labels`` is given, depth is optional. Otherwise ``depths`` is required.
    """
    from experiments.aerial.rl.env.obs import Observation

    rgbs = np.asarray(rgbs)
    n = int(rgbs.shape[0])
    act4 = _pad_action4(actions)
    if labels is None:
        if depths is None:
            raise ValueError("pack needs depths=... or labels=...")
        lab = label_batch(
            depths, act4, percentile=percentile, d_near=d_near, d_far=d_far
        )
        clear = lab["clearance_m"]
        y = lab["obstacle_label"]
    else:
        y = np.asarray(labels, dtype=np.float64).reshape(n)
        clear = np.full(n, np.nan, dtype=np.float64)

    if proprios is None:
        proprios = np.zeros((n, 4), dtype=np.float32)
    else:
        proprios = np.asarray(proprios, dtype=np.float32).reshape(n, 4)

    feats = []
    for i in range(n):
        # state = [x,y,z,vx,vy,vz,yaw]
        st = np.zeros(7, dtype=np.float32)
        st[0], st[1], st[2], st[6] = proprios[i]
        obs = Observation(rgb=rgbs[i], state=st)
        feats.append(np.asarray(dyn.encode(obs), dtype=np.float32))
    feat = np.stack(feats, axis=0)
    out: Dict[str, np.ndarray] = {
        "feature": feat,
        "action": act4,
        "obstacle_label": y.astype(np.float32),
        "clearance_m": clear.astype(np.float64),
    }
    if groups is not None:
        out["group"] = np.asarray(groups).astype(str)
    return out


def _freeze_backbone_train_obstacle_head(dyn: Any) -> None:
    for p in dyn.parameters():
        p.requires_grad = False
    for p in dyn.obstacle_cost_head.parameters():
        p.requires_grad = True


def train_obstacle_cost_head(
    dyn: Any,
    features: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 200,
    batch: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Freeze encoder/RSSM; fit softplus head to [0,1] labels. Marks trained."""
    import torch
    import torch.nn.functional as F

    _freeze_backbone_train_obstacle_head(dyn)
    opt = torch.optim.AdamW(
        [p for p in dyn.obstacle_cost_head.parameters() if p.requires_grad],
        lr=float(lr),
    )
    rng = np.random.default_rng(int(seed))
    feat = np.asarray(features, dtype=np.float32)
    act = _pad_action4(actions)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    n = int(feat.shape[0])
    if n == 0:
        raise ValueError("empty training set")
    last_loss = 0.0
    dyn.train()
    for _step in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch), n))
        f_t = torch.as_tensor(feat[idx], device=dyn.device, dtype=dyn.torch_dtype)
        a_t = torch.as_tensor(
            act[idx, : int(dyn.action_dim)], device=dyn.device, dtype=dyn.torch_dtype
        )
        y_t = torch.as_tensor(y[idx], device=dyn.device, dtype=dyn.torch_dtype)
        if int(f_t.shape[-1]) != int(dyn.latent_dim):
            raise ValueError(
                f"feature width {int(f_t.shape[-1])} != latent_dim {dyn.latent_dim}"
            )
        pred = F.softplus(
            dyn.obstacle_cost_head(torch.cat([f_t, a_t], dim=-1)).squeeze(-1)
        )
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    dyn.mark_obstacle_cost_trained(True)
    dyn.eval()
    return {"loss": last_loss, "n": float(n), "steps": float(steps)}


def calibrate_scale_report(
    dyn: Any,
    features: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Mean predictions for empty / near / left_near vs STRAIGHT_WEIGHT."""
    preds = []
    for i in range(len(labels)):
        preds.append(float(dyn.predict_obstacle_cost(features[i], actions[i])))
    preds_a = np.asarray(preds, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.float64)
    out: Dict[str, float] = {
        "pred_mean": float(np.mean(preds_a)),
        "label_mean": float(np.mean(lab)),
        "straight_w": float(STRAIGHT_WEIGHT),
    }
    if groups is not None:
        g = np.asarray(groups).astype(str)
        for name in ("fwd_empty", "fwd_near", "left_near"):
            m = g == name
            if np.any(m):
                out[f"pred_{name}"] = float(np.mean(preds_a[m]))
    empty = lab <= 0.05
    near = lab >= 0.95
    if np.any(empty):
        out["pred_empty"] = float(np.mean(preds_a[empty]))
    if np.any(near):
        out["pred_near"] = float(np.mean(preds_a[near]))
    return out


def _mean_score_on_features(
    dyn: Any,
    features: np.ndarray,
    action: np.ndarray,
    goal_rel: np.ndarray,
    *,
    cfg,
) -> float:
    a = np.asarray(action, dtype=np.float64).reshape(4)
    prog = _progress_along_goal(a, goal_rel)
    scores = []
    for i in range(len(features)):
        oc = float(dyn.predict_obstacle_cost(features[i], a))
        scores.append(score_candidate(a, goal_rel, prog, oc, cfg=cfg))
    return float(np.mean(scores)) if scores else float("nan")


def run_task5_gate(
    dyn: Any,
    features: np.ndarray,
    groups: np.ndarray,
    *,
    goal_rel: Optional[np.ndarray] = None,
    cfg=None,
) -> Dict[str, Any]:
    """Score three plan sorts on grouped features; return numbers + pass flags."""
    cfg = cfg or directional_oa_reward_cfg()
    g_rel = (
        np.asarray(goal_rel, dtype=np.float64)
        if goal_rel is not None
        else np.array([20.0, 0.0, 0.0, 20.0], dtype=np.float64)
    )
    g = np.asarray(groups).astype(str)
    feat = np.asarray(features)

    def _idx(name: str) -> np.ndarray:
        m = np.where(g == name)[0]
        if m.size == 0:
            raise ValueError(f"gate group {name!r} has 0 frames")
        return m

    i_empty = _idx("fwd_empty")
    i_near = _idx("fwd_near")
    i_left = _idx("left_near")

    r_fwd_empty = _mean_score_on_features(
        dyn, feat[i_empty], _CAND["fwd"], g_rel, cfg=cfg
    )
    r_side_empty = _mean_score_on_features(
        dyn, feat[i_empty], _CAND["side"], g_rel, cfg=cfg
    )
    r_hover = _mean_score_on_features(
        dyn, feat[i_empty], _CAND["hover"], g_rel, cfg=cfg
    )
    r_away = _mean_score_on_features(
        dyn, feat[i_empty], _CAND["away"], g_rel, cfg=cfg
    )
    r_fwd_near = _mean_score_on_features(
        dyn, feat[i_near], _CAND["fwd"], g_rel, cfg=cfg
    )
    # "更空、仍在靠近"：侧向/爬升绕行（同帧上动作代价应更低），不是同向前的 0.8 m。
    r_side_near = _mean_score_on_features(
        dyn, feat[i_near], _CAND["side"], g_rel, cfg=cfg
    )
    r_climb_near = _mean_score_on_features(
        dyn, feat[i_near], _CAND["climb"], g_rel, cfg=cfg
    )
    r_empty_approach = float(max(r_side_near, r_climb_near))
    # Left-near: forward should beat left / hover / away on the same features.
    r_fwd_left_near = _mean_score_on_features(
        dyn, feat[i_left], _CAND["fwd"], g_rel, cfg=cfg
    )
    r_left_left_near = _mean_score_on_features(
        dyn, feat[i_left], _CAND["left"], g_rel, cfg=cfg
    )
    r_hover_ln = _mean_score_on_features(
        dyn, feat[i_left], _CAND["hover"], g_rel, cfg=cfg
    )
    r_away_ln = _mean_score_on_features(
        dyn, feat[i_left], _CAND["away"], g_rel, cfg=cfg
    )

    numbers = {
        "r_fwd_empty": r_fwd_empty,
        "r_side_empty": r_side_empty,
        "r_hover": r_hover,
        "r_away": r_away,
        "r_fwd_near": r_fwd_near,
        "r_empty_approach": r_empty_approach,
        "r_fwd_left_near": r_fwd_left_near,
        "r_left_left_near": r_left_left_near,
        "r_hover_left_near": r_hover_ln,
        "r_away_left_near": r_away_ln,
        "n_fwd_empty": int(i_empty.size),
        "n_fwd_near": int(i_near.size),
        "n_left_near": int(i_left.size),
    }
    checks = gate_sort_checks(
        r_fwd_empty=r_fwd_empty,
        r_side_empty=r_side_empty,
        r_hover=r_hover,
        r_away=r_away,
        r_fwd_near=r_fwd_near,
        r_empty_approach=r_empty_approach,
        r_fwd_left_near=r_fwd_left_near,
        r_left_left_near=r_left_left_near,
    )
    # Third group also vs hover/away on left-near frames.
    checks["left_near_prefers_fwd"] = bool(
        checks["left_near_prefers_fwd"]
        and r_fwd_left_near > r_hover_ln
        and r_fwd_left_near > r_away_ln
    )
    return {"numbers": numbers, "checks": checks, "passed": all(checks.values())}


def _load_wm(ckpt: str, device: str) -> Any:
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    dyn = TorchRSSMDynamics(device=str(device))
    if ckpt:
        payload = dyn.load_checkpoint(ckpt)
        logger.info(
            "loaded wm obstacle_cost_trained=%s keys=%s",
            payload.get("obstacle_cost_trained"),
            payload.get("obstacle_cost_keys_loaded"),
        )
    return dyn


def _cmd_label(args: argparse.Namespace) -> int:
    data = np.load(args.depth_npz)
    if "depth" not in data or "action_xyz" not in data:
        raise SystemExit("npz must contain 'depth' [N,H,W] and 'action_xyz' [N,3]")
    out = label_batch(
        data["depth"],
        data["action_xyz"],
        percentile=args.percentile,
        d_near=args.d_near,
        d_far=args.d_far,
    )
    payload = dict(out)
    if "group" in data:
        payload["group"] = data["group"]
    payload["meta_straight_w"] = np.array([STRAIGHT_WEIGHT])
    payload["meta_empty"] = np.array([OBSTACLE_COST_EMPTY])
    payload["meta_near"] = np.array([OBSTACLE_COST_NEAR])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(
        f"wrote {args.out}: n={len(out['obstacle_label'])} "
        f"label_mean={float(np.nanmean(out['obstacle_label'])):.3f}"
    )
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    data = np.load(args.frames)
    if "rgb" not in data:
        raise SystemExit("frames npz must contain 'rgb' [N,H,W,3]")
    if "action_xyz" in data:
        actions = data["action_xyz"]
    elif "action" in data:
        actions = data["action"]
    else:
        raise SystemExit("frames need 'action_xyz' [N,3] or 'action' [N,4]")
    dyn = _load_wm(args.wm_ckpt, args.device)
    pre_labels = data["obstacle_label"] if "obstacle_label" in data else None
    depths = data["depth"] if "depth" in data else None
    if pre_labels is None and depths is None:
        raise SystemExit("pack needs depth=... or obstacle_label=...")
    out = pack_features_from_frames(
        dyn,
        data["rgb"],
        actions,
        depths=depths,
        proprios=data["proprio"] if "proprio" in data else None,
        groups=data["group"] if "group" in data else None,
        percentile=args.percentile,
        d_near=args.d_near,
        d_far=args.d_far,
        labels=pre_labels,
    )
    out["meta_straight_w"] = np.array([STRAIGHT_WEIGHT])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(
        f"wrote {args.out}: n={out['feature'].shape[0]} "
        f"feat_dim={out['feature'].shape[1]} "
        f"label_mean={float(np.nanmean(out['obstacle_label'])):.3f}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    data = np.load(args.labels)
    for k in ("feature", "action", "obstacle_label"):
        if k not in data:
            raise SystemExit(f"labels npz missing '{k}'")
    dyn = _load_wm(args.wm_ckpt, args.device)
    dyn.mark_obstacle_cost_trained(False)
    stats = train_obstacle_cost_head(
        dyn,
        data["feature"],
        data["action"],
        data["obstacle_label"],
        steps=int(args.steps),
        batch=int(args.batch),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    groups = data["group"] if "group" in data else None
    calib = calibrate_scale_report(
        dyn, data["feature"], data["action"], data["obstacle_label"], groups=groups
    )
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    dyn.save_checkpoint(args.out_ckpt, step=int(args.steps))
    print(f"train stats={stats}")
    print(f"calib={calib}")
    print(
        f"wrote {args.out_ckpt} obstacle_cost_trained="
        f"{bool(dyn.obstacle_cost_trained)}"
    )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    data = np.load(args.labels)
    if "feature" not in data or "group" not in data:
        raise SystemExit("gate needs labels with 'feature' and 'group'")
    dyn = _load_wm(args.wm_ckpt, args.device)
    if not bool(getattr(dyn, "obstacle_cost_trained", False)):
        raise SystemExit(
            "refuse gate: obstacle_cost_trained=False "
            "(run train and save with explicit flag first)"
        )
    report = run_task5_gate(dyn, data["feature"], data["group"])
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        print("GATE FAILED — do not start task 6")
        return 1
    print("GATE PASSED")
    return 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("label", help="GT depth → obstacle labels")
    pl.add_argument("--depth-npz", type=str, required=True)
    pl.add_argument("--out", type=str, required=True)
    pl.add_argument("--percentile", type=float, default=5.0)
    pl.add_argument("--d-near", type=float, default=3.0)
    pl.add_argument("--d-far", type=float, default=22.0)

    pp = sub.add_parser("pack", help="RGB → feature=[h‖z] + labels")
    pp.add_argument("--wm-ckpt", type=str, required=True)
    pp.add_argument("--frames", type=str, required=True)
    pp.add_argument("--out", type=str, required=True)
    pp.add_argument("--device", type=str, default="cpu")
    pp.add_argument("--percentile", type=float, default=5.0)
    pp.add_argument("--d-near", type=float, default=3.0)
    pp.add_argument("--d-far", type=float, default=22.0)

    pt = sub.add_parser("train", help="freeze backbone; train obstacle_cost_head")
    pt.add_argument("--wm-ckpt", type=str, default="")
    pt.add_argument("--labels", type=str, required=True)
    pt.add_argument("--out-ckpt", type=str, required=True)
    pt.add_argument("--steps", type=int, default=500)
    pt.add_argument("--batch", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--device", type=str, default="cpu")

    pg = sub.add_parser("gate", help="task-5 three sorts; exit 1 if fail")
    pg.add_argument("--wm-ckpt", type=str, required=True)
    pg.add_argument("--labels", type=str, required=True)
    pg.add_argument("--report", type=str, default="")
    pg.add_argument("--device", type=str, default="cpu")

    args = p.parse_args(argv)
    if args.cmd == "label":
        return _cmd_label(args)
    if args.cmd == "pack":
        return _cmd_pack(args)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "gate":
        return _cmd_gate(args)
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(_main())
