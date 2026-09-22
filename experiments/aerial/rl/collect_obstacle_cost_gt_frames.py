#!/usr/bin/env python3
"""Collect GT depth frames for directional obstacle-cost labels (plan 2026-09-21).

Interior/inland urban only. At each spawn (+ a few steps), grab RGB+GT depth and
emit one row per probe action (fwd / left / right / up) with a gate ``group``:

  fwd_empty  — forward clearance ≥ d_far
  fwd_near   — forward clearance ≤ d_near
  left_near  — left ≤ d_near and forward ≥ d_far * 0.7

Writes ``frames.npz`` for ``train_obstacle_cost_labels pack``.

    python -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \\
      --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json \\
      --out experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames.npz \\
      --episodes 12 --steps-per-ep 8 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from experiments.aerial.rl.depth_geometry import directional_clearance_m
from experiments.aerial.rl.train_rl import build_from_config

logger = logging.getLogger(__name__)

_PROBES = (
    ("fwd", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    # Pure ±90° left/right is outside default 90° HFOV (±45°) → empty wedge.
    # Use forward-oblique bearings that still stress body-left / body-right.
    ("left", np.array([1.0, 1.0, 0.0], dtype=np.float64)),
    ("right", np.array([1.0, -1.0, 0.0], dtype=np.float64)),
    ("up", np.array([1.0, 0.0, 1.0], dtype=np.float64)),
)


def _classify_probe(
    name: str,
    clear: Dict[str, float],
    *,
    d_near: float,
    d_far: float,
) -> Optional[str]:
    fwd = float(clear.get("fwd", np.inf))
    left = float(clear.get("left", np.inf))
    if name == "fwd":
        if np.isfinite(fwd) and fwd <= d_near:
            return "fwd_near"
        if np.isfinite(fwd) and fwd >= d_far:
            return "fwd_empty"
    elif name in ("left", "right"):
        # Side wall with relatively open forward (task-5 group 3).
        # Right-near maps into the same gate bucket (symmetric failure mode).
        side = left if name == "left" else float(clear.get("right", np.inf))
        if np.isfinite(side) and side <= max(d_near, 6.0) and (
            not np.isfinite(fwd) or fwd >= max(6.0, side + 2.0)
        ):
            return "left_near"
    return None


def _load_episodes(annotation: str, max_eps: int) -> List[dict]:
    path = Path(annotation)
    raw = yaml.safe_load(path.read_text()) if path.suffix in (".yaml", ".yml") else None
    if raw is None:
        import json

        raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "routes" in raw:
        eps = raw["routes"]
    elif isinstance(raw, dict) and "episodes" in raw:
        eps = raw["episodes"]
    elif isinstance(raw, list):
        eps = raw
    else:
        raise SystemExit(f"unrecognized annotation schema: {annotation}")
    # Prefer inland/interior tags when present.
    scored: List[Tuple[int, dict]] = []
    for i, e in enumerate(eps):
        tags = e.get("tags") or e.get("geo") or {}
        inland = bool(tags.get("in_interior") or tags.get("inland") or e.get("in_interior"))
        water = bool(tags.get("in_water") or e.get("in_water"))
        score = int(inland) - int(water)
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    out = [e for _, e in scored[: max(1, int(max_eps))]]
    logger.info("selected %d episodes from %s", len(out), annotation)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/aerial_rl_urban_complex_p2c.yaml")
    p.add_argument(
        "--annotation",
        default=(
            "experiments/aerial/phase3_unified/annotations/"
            "outdoor_complex_inland_patched.json"
        ),
    )
    p.add_argument(
        "--out",
        default="experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames.npz",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=41451)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps-per-ep", type=int, default=10)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--d-near", type=float, default=3.0)
    p.add_argument("--d-far", type=float, default=22.0)
    p.add_argument("--min-per-group", type=int, default=40)
    p.add_argument("--max-frames", type=int, default=800)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((repo / args.config).read_text())
    cfg.setdefault("env", {})
    cfg["env"]["backend"] = "airsim"
    cfg["env"]["host"] = str(args.host)
    cfg["env"]["port"] = int(args.port)
    cfg["env"]["grab_depth"] = True
    cfg["env"]["step_hz"] = float(args.step_hz)
    cfg.setdefault("safety", {})["kind"] = "null"
    cfg.setdefault("planner", {})["enable"] = False
    cfg.setdefault("world_model", {}).setdefault("depth_head", {})["enable"] = False
    cfg.setdefault("tau_predictor", {})["enable"] = False
    cfg.setdefault("corrector", {})["episodes_per_iter"] = 0

    loop = build_from_config(cfg)
    env = loop.collector.env
    episodes = _load_episodes(str(repo / args.annotation), int(args.episodes))

    rgbs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    acts: List[np.ndarray] = []
    proprios: List[np.ndarray] = []
    groups: List[str] = []
    counts = {"fwd_empty": 0, "fwd_near": 0, "left_near": 0}

    for ei, ep in enumerate(episodes):
        if sum(counts.values()) >= int(args.max_frames):
            break
        try:
            obs = env.reset(ep)
        except Exception as exc:
            logger.warning("reset failed ep=%d: %s", ei, exc)
            continue
        if getattr(obs, "depth", None) is None:
            logger.warning("no GT depth on reset ep=%d — skip", ei)
            continue
        for si in range(int(args.steps_per_ep)):
            d = np.asarray(obs.depth, dtype=np.float64)
            if d.ndim != 2 or not np.isfinite(d).any():
                break
            clear = {
                name: directional_clearance_m(d, xyz)
                for name, xyz in _PROBES
            }
            # Emit all probe rows that map to a gate group (reuse same rgb/depth).
            for name, xyz in _PROBES:
                grp = _classify_probe(
                    name, clear, d_near=float(args.d_near), d_far=float(args.d_far)
                )
                if grp is None:
                    continue
                rgbs.append(np.asarray(obs.rgb, dtype=np.uint8).copy())
                depths.append(d.copy())
                acts.append(xyz.astype(np.float32))
                proprios.append(np.asarray(obs.proprio4(), dtype=np.float32).copy())
                groups.append(grp)
                counts[grp] = counts.get(grp, 0) + 1
            # Alternate forward / left-strafe / yaw so we catch side walls.
            phase = si % 5
            if clear.get("fwd", 99.0) <= float(args.d_near) * 1.2:
                # Wall ahead → yaw CW so it lands on body +left (group left_near).
                cmd = np.array([0.0, 0.0, 0.0, -0.95], dtype=np.float64)
            elif phase == 0:
                cmd = np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float64)
            elif phase == 1:
                cmd = np.array([0.15, 0.55, 0.0, 0.0], dtype=np.float64)
            elif phase == 2:
                cmd = np.array([0.0, 0.0, 0.0, 0.45], dtype=np.float64)
            elif phase == 3:
                cmd = np.array([0.35, -0.45, 0.0, 0.0], dtype=np.float64)
            else:
                cmd = np.array([0.4, 0.0, 0.0, -0.45], dtype=np.float64)
            try:
                obs, _ = env.step(cmd)
            except Exception as exc:
                logger.warning("step failed ep=%d si=%d: %s", ei, si, exc)
                break
            if getattr(obs, "depth", None) is None:
                break
        logger.info(
            "ep=%d counts=%s total=%d", ei, counts, sum(counts.values())
        )
        if all(counts.get(k, 0) >= int(args.min_per_group) for k in ("fwd_empty", "fwd_near", "left_near")):
            logger.info("min-per-group met — stop early")
            break

    if not rgbs:
        raise SystemExit("collected 0 frames — check AirSim grab_depth / annotation")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb=np.stack(rgbs, axis=0),
        depth=np.stack(depths, axis=0),
        action_xyz=np.stack(acts, axis=0),
        proprio=np.stack(proprios, axis=0),
        group=np.asarray(groups, dtype=str),
    )
    meta = {
        "n_frames": len(rgbs),
        "counts": counts,
        "out": str(out_path),
        "d_near": float(args.d_near),
        "d_far": float(args.d_far),
    }
    (out_path.parent / "COLLECT_META.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n"
    )
    print(f"wrote {out_path} n={len(rgbs)} counts={counts}")
    missing = [k for k in ("fwd_empty", "fwd_near", "left_near") if counts.get(k, 0) < 5]
    if missing:
        logger.warning("sparse groups %s — gate may fail; collect more", missing)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
