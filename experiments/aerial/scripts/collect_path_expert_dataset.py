"""Densify OpenFly paths via PathExpert closed-loop @ step_hz (real RGB).

Uses the annotated polyline (A* / human waypoints) as a geometric teacher —
NOT Heuristic straight-line, NOT ImaginationPlanner. Renders every control step
so AC/BC get dense RGB + body-delta labels.

    # mock smoke (no renderer):
    python -m experiments.aerial.rl.collect_path_expert_dataset \\
      --backend mock --episodes 2 --max-steps 80 \\
      --annotation experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json \\
      --out experiments/aerial/rl/artifacts/dataset_path_expert_mock

    # 125 / AirSim (keep full OpenFly waypoints — do NOT approach-bias):
    source experiments/aerial/scripts/env_4090.sh
    $AERIAL_PY -m experiments.aerial.rl.collect_path_expert_dataset \\
      --backend airsim --host 127.0.0.1 --step-hz 5.0 --grab-depth \\
      --episodes 32 --max-steps 400 \\
      --annotation artifacts/seen_airsim16_m1a20.json \\
      --out experiments/aerial/rl/artifacts/dataset_v0_path_expert_openfly_20260827
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from experiments.aerial.path_expert import PathExpertPolicy
from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.env.action import DEFAULT_STEP_HZ
from experiments.aerial.rl.reward import DEFAULT_ONLINE_SUCCESS_DIST_M, RewardConfig
from experiments.aerial.rl.safety import NullSafetyShield
from experiments.aerial.rl.train_rl import _build_env, _load_episodes

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _episode_arrived(transitions: list, success_dist_m: float) -> bool:
    if not transitions:
        return False
    last = transitions[-1]
    goal = None
    for bag in (last.info, getattr(last.obs, "info", {}) or {}):
        if isinstance(bag, dict) and bag.get("goal") is not None:
            goal = np.asarray(bag["goal"], dtype=np.float64).reshape(3)
            break
    if goal is None:
        return False
    pos = np.asarray(last.next_obs.position if last.next_obs is not None else last.obs.position)
    collided = bool(
        (last.next_obs.collided if last.next_obs is not None else False) or last.obs.collided
    )
    return (not collided) and float(np.linalg.norm(pos - goal)) < float(success_dist_m)


def _write_episode_npz(
    out_dir: Path,
    index: int,
    transitions: list,
    *,
    arrived: bool,
    n_waypoints: int,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{index:05d}.npz"
    arrays = ds.episode_arrays(transitions)
    arrays["arrived"] = np.asarray(bool(arrived))
    arrays["n_waypoints"] = np.asarray(int(n_waypoints), dtype=np.int32)
    np.savez_compressed(path, **arrays)
    return path


def _lift_episode_z(ep: Dict[str, Any], min_spawn_z: float) -> Dict[str, Any]:
    """Uniform z-lift so spawn meets min fly altitude (teacher collect only)."""
    if min_spawn_z <= 0:
        return ep
    out = dict(ep)
    pts = np.asarray(out.get("pos", []), dtype=np.float64).reshape(-1, 3).copy()
    if len(pts) == 0:
        return out
    if float(pts[0, 2]) < float(min_spawn_z):
        pts[:, 2] += float(min_spawn_z) - float(pts[0, 2])
    out["pos"] = [[float(x), float(y), float(z)] for x, y, z in pts]
    return out


def build_collector(args: argparse.Namespace) -> RolloutCollector:
    env_cfg: Dict[str, Any] = {
        "backend": args.backend,
        "step_hz": float(args.step_hz),
        "width": 224,
        "height": 224,
        "seed": int(args.seed),
    }
    if args.backend == "airsim":
        env_cfg.update(
            host=args.host,
            port=args.port,
            camera=args.camera,
            vehicle=args.vehicle,
            grab_depth=bool(args.grab_depth),
        )
    env = _build_env(env_cfg)
    reward_cfg = RewardConfig(success_dist_m=float(args.success_dist_m))
    buffer = ReplayBuffer(capacity_episodes=max(8, int(args.episodes)))
    return RolloutCollector(
        env,
        PathExpertPolicy(),
        buffer,
        reward_cfg=reward_cfg,
        safety=NullSafetyShield(),  # clean teacher; shield is for deploy eval
        max_steps=int(args.max_steps),
        target_hz=float(args.step_hz),
        planner=None,
        depth_predictor=None,
        tau_predictor=None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mock", "airsim"), default="mock")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=41451)
    p.add_argument("--camera", default="0")
    p.add_argument("--vehicle", default="")
    p.add_argument("--grab-depth", action="store_true")
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--annotation",
        required=True,
        help="OpenFly annotation JSON (full waypoint paths; no approach-bias).",
    )
    p.add_argument(
        "--out",
        default="experiments/aerial/rl/artifacts/dataset_v0_path_expert_openfly",
    )
    p.add_argument(
        "--success-dist-m",
        type=float,
        default=DEFAULT_ONLINE_SUCCESS_DIST_M,
        help="Arrival radius (V4 online default 3 m; OpenFly VLN uses 20 m).",
    )
    p.add_argument(
        "--keep-failed",
        action="store_true",
        help="Also write non-arrived / collided eps (marked arrived=false).",
    )
    p.add_argument(
        "--min-spawn-z",
        type=float,
        default=0.0,
        help="Lift episode z uniformly so spawn z >= this (teacher flyability).",
    )
    p.add_argument(
        "--spawn-z-retry-m",
        type=float,
        default=2.0,
        help="On reset spawn collision, retry once with +this many meters (0=disable).",
    )
    p.add_argument(
        "--min-usable-path-m",
        type=float,
        default=0.0,
        help="Count episode usable (and retain with --keep-failed) if path_length_m >= this.",
    )
    p.add_argument(
        "--min-usable-steps",
        type=int,
        default=40,
        help="With --min-usable-path-m, also require at least this many steps.",
    )
    p.add_argument("--config", default="", help="Optional yaml overlay (unused keys ok).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[path-expert-collect] %(message)s")
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"annotation": args.annotation, "max_episodes": int(args.episodes)}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        if y.get("annotation"):
            cfg["annotation"] = y["annotation"]
    episodes = _load_episodes(cfg)
    if not episodes:
        print("[path-expert-collect] FAIL: no episodes from annotation", file=sys.stderr)
        return 1

    collector = build_collector(args)
    manifest: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    n_arrived = 0
    n_written = 0
    n_skipped_fail = 0

    try:
        for i, ep in enumerate(episodes[: int(args.episodes)]):
            ep = _lift_episode_z(ep, float(args.min_spawn_z))
            if hasattr(collector.policy, "bind_episode"):
                collector.policy.bind_episode(ep)
            transitions, stats = collector.collect_episode(ep)
            if stats.skipped and float(args.spawn_z_retry_m) > 0:
                pts = np.asarray(ep["pos"], dtype=np.float64).copy()
                pts[:, 2] += float(args.spawn_z_retry_m)
                ep_retry = dict(ep)
                ep_retry["pos"] = [[float(x), float(y), float(z)] for x, y, z in pts]
                if hasattr(collector.policy, "bind_episode"):
                    collector.policy.bind_episode(ep_retry)
                transitions, stats = collector.collect_episode(ep_retry)
                ep = ep_retry
            if stats.skipped:
                logger.warning("ep %d: skipped (spawn collision)", i)
                continue
            if not transitions:
                continue
            n_wp = len(np.asarray(ep.get("pos", [])).reshape(-1, 3))
            arrived = _episode_arrived(transitions, args.success_dist_m)
            if arrived:
                n_arrived += 1
            elif not args.keep_failed:
                n_skipped_fail += 1
                logger.info(
                    "ep %d: not arrived — skip (pass --keep-failed to retain)", i
                )
                continue
            path = _write_episode_npz(
                out_dir, n_written, transitions, arrived=arrived, n_waypoints=n_wp
            )
            rep = ds.quality_report(transitions)
            reports.append(rep)
            path_m = float(rep.get("path_length_m") or 0.0)
            min_path = float(args.min_usable_path_m)
            usable = (bool(arrived) and not bool(rep.get("quarantined"))) or (
                min_path > 0
                and path_m >= min_path
                and len(transitions) >= int(args.min_usable_steps)
                and not bool(rep.get("quarantined"))
            )
            manifest.append(
                {
                    "file": path.name,
                    "steps": len(transitions),
                    "arrived": bool(arrived),
                    "n_waypoints": int(n_wp),
                    "path_length_m": path_m,
                    "return": float(sum(t.reward for t in transitions)),
                    "usable": bool(usable),
                    "source": "openfly_path_expert_densify",
                }
            )
            n_written += 1
            logger.info(
                "wrote %s steps=%d arrived=%s waypoints=%d",
                path.name,
                len(transitions),
                arrived,
                n_wp,
            )
    finally:
        close = getattr(collector.env, "close", None)
        if callable(close):
            close()

    meta = {
        "kind": "path_expert_openfly_densify",
        "backend": args.backend,
        "step_hz": float(args.step_hz),
        "max_steps": int(args.max_steps),
        "success_dist_m": float(args.success_dist_m),
        "grab_depth": bool(args.grab_depth),
        "annotation": str(args.annotation),
        "n_requested": int(args.episodes),
        "n_written": n_written,
        "n_arrived": n_arrived,
        "n_skipped_not_arrived": n_skipped_fail,
        "planner": False,
        "shield": "null",
        "note": (
            "OpenFly polyline chased by PathExpert @ step_hz; "
            "real render each step; no approach-bias; no ImaginationPlanner"
        ),
    }
    ds.write_manifest(out_dir, manifest, meta=meta)
    (out_dir / "path_expert_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if reports:
        ds.write_quality_summary(out_dir, reports)

    usable = sum(1 for m in manifest if m.get("usable"))
    print(
        f"[path-expert-collect] wrote={n_written} arrived={n_arrived} "
        f"usable={usable} skipped_fail={n_skipped_fail} out={out_dir}"
    )
    min_ok = max(1, int(round(0.5 * int(args.episodes))))
    if usable == 0:
        print("[path-expert-collect] FAIL: 0 usable episodes", file=sys.stderr)
        return 1
    if usable < min_ok:
        print(
            f"[path-expert-collect] FAIL: usable={usable} < min_ok={min_ok}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
