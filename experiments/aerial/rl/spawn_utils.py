"""Uniform z-lift and spawn-collision retries for annotated episodes."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def lift_episode_z(ep: Dict[str, Any], min_spawn_z: float) -> Dict[str, Any]:
    """Uniform z-lift so spawn meets min fly altitude."""
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


def nudge_episode_z(ep: Dict[str, Any], delta_m: float) -> Dict[str, Any]:
    """Raise entire polyline by delta_m (spawn-collision retries)."""
    if delta_m <= 0:
        return ep
    out = dict(ep)
    pts = np.asarray(out.get("pos", []), dtype=np.float64).reshape(-1, 3).copy()
    if len(pts) == 0:
        return out
    pts[:, 2] += float(delta_m)
    out["pos"] = [[float(x), float(y), float(z)] for x, y, z in pts]
    return out


def nudge_episode_xy(ep: Dict[str, Any], dx_m: float, dy_m: float) -> Dict[str, Any]:
    """Translate entire polyline in world XY (spawn-collision lateral retries)."""
    if abs(float(dx_m)) < 1e-9 and abs(float(dy_m)) < 1e-9:
        return ep
    out = dict(ep)
    pts = np.asarray(out.get("pos", []), dtype=np.float64).reshape(-1, 3).copy()
    if len(pts) == 0:
        return out
    pts[:, 0] += float(dx_m)
    pts[:, 1] += float(dy_m)
    out["pos"] = [[float(x), float(y), float(z)] for x, y, z in pts]
    return out


def prepare_episode_spawn(
    ep: Dict[str, Any],
    *,
    min_spawn_z: float = 0.0,
    extra_z_nudge_m: float = 0.0,
) -> Dict[str, Any]:
    """Apply base z-lift then optional extra nudge (retries stack on lifted ep)."""
    out = lift_episode_z(ep, float(min_spawn_z))
    if float(extra_z_nudge_m) > 0:
        out = nudge_episode_z(out, float(extra_z_nudge_m))
    return out


def spawn_retry_plan(
    ep: Dict[str, Any],
    *,
    min_spawn_z: float,
    spawn_z_retry_m: float,
    spawn_z_max_retries: int,
    spawn_xy_nudge_m: float = 12.0,
) -> list[Dict[str, Any]]:
    """Episode variants: lifted base, +z retries, then XY offsets at max z."""
    base = lift_episode_z(ep, float(min_spawn_z))
    tries = [base]
    if float(spawn_z_retry_m) > 0 and int(spawn_z_max_retries) > 0:
        for attempt in range(1, 1 + int(spawn_z_max_retries)):
            tries.append(nudge_episode_z(base, float(spawn_z_retry_m) * float(attempt)))
    # After exhausting pure z lifts, try lateral shifts on the highest z variant
    # so we escape building footprints that z-only cannot clear.
    xy = float(spawn_xy_nudge_m)
    if xy > 0:
        top = tries[-1]
        offsets = [
            (xy, 0.0), (-xy, 0.0), (0.0, xy), (0.0, -xy),
            (xy, xy), (-xy, -xy), (xy, -xy), (-xy, xy),
            (2 * xy, 0.0), (-2 * xy, 0.0), (0.0, 2 * xy), (0.0, -2 * xy),
            (3 * xy, 0.0), (-3 * xy, 0.0), (0.0, 3 * xy), (0.0, -3 * xy),
            (2 * xy, xy), (-2 * xy, -xy), (xy, 2 * xy), (-xy, -2 * xy),
        ]
        for dx, dy in offsets:
            tries.append(nudge_episode_xy(top, dx, dy))
    return tries


def collect_episode_with_spawn_retries(
    collector: Any,
    ep: Dict[str, Any],
    *,
    min_spawn_z: float = 0.0,
    spawn_z_retry_m: float = 0.0,
    spawn_z_max_retries: int = 0,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Try collect; on spawn collision, lift z and retry."""
    variants = spawn_retry_plan(
        ep,
        min_spawn_z=float(min_spawn_z),
        spawn_z_retry_m=float(spawn_z_retry_m),
        spawn_z_max_retries=int(spawn_z_max_retries),
    )
    transitions = []
    stats = None
    ep_try = variants[0]
    for attempt, ep_try in enumerate(variants):
        if hasattr(collector.policy, "bind_episode"):
            collector.policy.bind_episode(ep_try)
        transitions, stats = collector.collect_episode(ep_try)
        if stats is None or not stats.skipped:
            if attempt > 0:
                z0 = float(np.asarray(ep_try["pos"], dtype=np.float64).reshape(-1, 3)[0, 2])
                logger.info("spawn retry %d ok at z=%.1f", attempt, z0)
            return ep_try, transitions, stats
        if attempt + 1 < len(variants):
            z0 = float(np.asarray(variants[attempt + 1]["pos"], dtype=np.float64).reshape(-1, 3)[0, 2])
            logger.info("spawn collision — retry %d at z=%.1f", attempt + 1, z0)
    return ep_try, transitions, stats


def reset_with_spawn_retries(
    env: Any,
    ep: Dict[str, Any],
    *,
    min_spawn_z: float = 0.0,
    spawn_z_retry_m: float = 0.0,
    spawn_z_max_retries: int = 0,
    spawn_tol_m: float = 12.0,
    mock: bool = False,
    min_spawn_clear_m: float = 0.0,
) -> Tuple[Dict[str, Any], Any, float, bool]:
    """Reset env; retry with z-lift on collision, pose error, or tight clearance.

    When ``min_spawn_clear_m > 0`` and GT ``obs.depth`` is present, reject spawns
    whose forward directional clearance is below the threshold (avoids starting
    already inside a wall FOV with --no-shield).

    Returns (episode_used, observation, spawn_err_m, spawn_failed).
    """
    variants = spawn_retry_plan(
        ep,
        min_spawn_z=float(min_spawn_z),
        spawn_z_retry_m=float(spawn_z_retry_m),
        spawn_z_max_retries=int(spawn_z_max_retries),
    )
    last_err = float("inf")
    last_obs = None
    ep_try = variants[0]
    need_clear = float(min_spawn_clear_m)
    for attempt, ep_try in enumerate(variants):
        obs = env.reset(ep_try)
        last_obs = obs
        pts = np.asarray(ep_try.get("pos", []), dtype=np.float64).reshape(-1, 3)
        start_pos = pts[0] if len(pts) else np.zeros(3)
        p_curr = np.asarray(obs.position, dtype=np.float64)
        spawn_err = float(np.linalg.norm(p_curr - start_pos))
        collided = bool(getattr(obs, "collided", False))
        last_err = spawn_err
        clear_ok = True
        d_fwd = None
        if need_clear > 0.0 and getattr(obs, "depth", None) is not None:
            from experiments.aerial.rl.depth_geometry import directional_clearance_m

            d_fwd = directional_clearance_m(
                np.asarray(obs.depth, dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            if np.isfinite(float(d_fwd)) and float(d_fwd) < need_clear:
                clear_ok = False
        ok = mock or (
            not collided and spawn_err <= float(spawn_tol_m) and clear_ok
        )
        if ok:
            if attempt > 0:
                logger.info(
                    "eval spawn retry %d ok at z=%.1f err=%.1fm d_fwd=%s",
                    attempt,
                    float(start_pos[2]),
                    spawn_err,
                    f"{float(d_fwd):.1f}" if d_fwd is not None else "n/a",
                )
            return ep_try, obs, spawn_err, False
        if attempt + 1 < len(variants):
            z_next = float(
                np.asarray(variants[attempt + 1]["pos"], dtype=np.float64).reshape(-1, 3)[0, 2]
            )
            logger.warning(
                "eval spawn fail err=%.1fm collided=%s d_fwd=%s — retry at z=%.1f",
                spawn_err,
                collided,
                f"{float(d_fwd):.1f}" if d_fwd is not None else "n/a",
                z_next,
            )
    return ep_try, last_obs, last_err, True
