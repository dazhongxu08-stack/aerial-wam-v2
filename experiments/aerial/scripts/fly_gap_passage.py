"""Fly the measured side gap and save the episode if it reaches the goal.

Teacher only. Routes 0 and 1 of the inland annotation. Altitude is held at
the clear spawn height.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.dataset import write_episode
from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig
from experiments.aerial.rl.gap_passage import gap_action
from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

logger = logging.getLogger("fly_gap_passage")


def _load_learner(config_path: str, wm_ckpt: str, actor_ckpt: str):
    import yaml

    from experiments.aerial.rl.actor_critic import LatentActorCritic, LatentActorDeployPolicy
    from experiments.aerial.rl.train_rl import load_torch_dynamics

    cfg = yaml.safe_load(open(config_path))
    dynamics, _payload = load_torch_dynamics(
        cfg.get("world_model") or {},
        wm_ckpt,
        device="cuda",
        freeze=True,
    )
    actor = LatentActorCritic.load_from_checkpoint(actor_ckpt, device="cuda")
    return LatentActorDeployPolicy(dynamics, actor, deterministic=True)


def _load_routes(path: Path, routes: list[int]) -> list[dict]:
    data = json.loads(path.read_text())
    episodes = data["episodes"] if isinstance(data, dict) else data
    by_idx = {int(ep.get("route_idx", i)): ep for i, ep in enumerate(episodes)}
    return [by_idx[i] for i in routes]


def _forward_m(obs) -> float | None:
    depth = getattr(obs, "depth", None)
    if depth is None:
        return None
    from experiments.aerial.rl.depth_geometry import approach_clearances

    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    fwd, _, _ = approach_clearances(arr, center_frac=0.12)
    return float(fwd)


def fly_one(
    env,
    episode: dict,
    *,
    max_steps: int,
    success_dist: float,
    policy=None,
    intervene_m: float = 0.0,
    record_executed: bool = False,
    early_west: bool = False,
    correction=None,
) -> dict:
    ep_used, obs, err, failed = reset_with_spawn_retries(
        env,
        episode,
        min_spawn_z=24.0,
        spawn_z_retry_m=5.0,
        spawn_z_max_retries=6,
        min_spawn_clear_m=5.0,
    )
    goal = np.asarray(ep_used["pos"], dtype=np.float64).reshape(-1, 3)[-1]
    if failed or obs is None:
        return {"ok": False, "reason": "spawn", "spawn_err": err, "transitions": []}
    # The west gap and the east crossing were open at 29 m. A lower clear
    # spawn on route 1 flew into the next row.
    z_hold = max(float(obs.position[2]), 29.0)
    obs.info["goal"] = goal.tolist()
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    transitions = []
    min_d = float("inf")
    blocked_run = 0
    n_intervene = 0
    phase = "west"
    prev_y = None
    latched = False
    gap_y = None
    for step in range(max_steps):
        pos = np.asarray(obs.position, dtype=np.float64)
        dist = float(np.linalg.norm(pos - goal))
        min_d = min(min_d, dist)
        if dist <= success_dist:
            break
        label, phase = gap_action(pos, float(obs.yaw), goal, z_hold)
        fwd = _forward_m(obs)
        intervened = False
        if policy is None:
            action = label
            store = True
        else:
            learner = np.asarray(policy.act(obs), dtype=np.float64).reshape(4)
            if early_west:
                x = float(pos[0])
                y = float(pos[1])
                if latched and x <= -960.0 and gap_y is None:
                    gap_y = y
                    logger.info("at-gap step=%d xy=(%.1f,%.1f) yaw=%.2f", step, x, y, float(obs.yaw))
                yaw_err = (float(obs.yaw) + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi
                facing_south = abs(yaw_err) < 0.45
                if (
                    latched
                    and x >= -948.0
                    and y <= 26.0
                    and facing_south
                ):
                    latched = False
                    logger.info(
                        "release step=%d xy=(%.1f,%.1f) yaw=%.2f",
                        step, x, y, float(obs.yaw),
                    )
                elif (
                    not latched
                    and y > 60.0
                    and x > -960.0
                    and prev_y is not None
                    and y < prev_y - 0.3
                ):
                    latched = True
                    logger.info("early-west step=%d xy=(%.1f,%.1f)", step, x, y)
                prev_y = y
                if latched:
                    action = label
                    intervened = True
                    store = bool(record_executed)
                else:
                    action = learner
                    store = bool(record_executed)
            else:
                about_to_hit = (
                    float(intervene_m) > 0.0
                    and fwd is not None
                    and fwd < float(intervene_m)
                    and float(learner[0]) > 0.25
                )
                if about_to_hit:
                    action = label
                    intervened = True
                    store = True
                    logger.info(
                        "intervene step=%d xy=(%.1f,%.1f) fwd=%.1f phase=%s",
                        step, pos[0], pos[1], float(fwd), phase,
                    )
                elif float(intervene_m) > 0.0:
                    action = learner
                    store = bool(record_executed)
                else:
                    action = learner
                    store = True
        if policy is None and fwd is not None and fwd < 6.0 and float(action[0]) > 0.5:
            blocked_run += 1
        else:
            blocked_run = 0
        if blocked_run >= 8:
            logger.warning(
                "stop route forward blocked phase=%s pos=%s fwd=%.1f",
                phase, pos[:2].round(1).tolist(), fwd or -1,
            )
            break
        if correction is not None and not early_west:
            action = correction.apply(action, pos, goal, float(obs.yaw))
        next_obs, _info = env.step(action)
        if store and policy is not None and float(intervene_m) <= 0.0 and not record_executed:
            action = label
        next_obs.info["goal"] = goal.tolist()
        done = bool(getattr(next_obs, "collided", False)) or dist <= success_dist
        if intervened and not store:
            n_intervene += 1
        if store:
            n_intervene += int(intervened or policy is None)
            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=0.0,
                    done=done,
                    next_obs=next_obs,
                    info={"goal": goal.tolist(), "phase": phase},
                )
            )
        if step % 20 == 0:
            logger.info(
                "step=%d phase=%s hold=%s xy=(%.1f,%.1f) z=%.1f yaw=%.2f d=%.1f fwd=%s",
                step, phase, int(latched), pos[0], pos[1], pos[2], float(obs.yaw), dist,
                None if fwd is None else round(fwd, 1),
            )
        obs = next_obs
        if done:
            break
    end = np.asarray(obs.position, dtype=np.float64)
    d_final = float(np.linalg.norm(end - goal))
    arrived = d_final <= success_dist and not bool(getattr(obs, "collided", False))
    return {
        "ok": arrived,
        "arrived": arrived,
        "d_final": d_final,
        "min_d": min_d,
        "steps": len(transitions),
        "n_intervene": int(n_intervene) if policy is not None else 0,
        "z_hold": z_hold,
        "end_xy": end[:2].tolist(),
        "collided": bool(getattr(obs, "collided", False)),
        "transitions": transitions,
        "goal": goal.tolist(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", required=True)
    p.add_argument("--routes", default="0,1")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--success-dist", type=float, default=3.0)
    p.add_argument("--learner-ckpt", default="")
    p.add_argument("--wm-ckpt", default="")
    p.add_argument("--config", default="")
    p.add_argument("--intervene-m", type=float, default=0.0)
    p.add_argument("--record-executed", action="store_true")
    p.add_argument("--early-west", action="store_true")
    p.add_argument("--correction", default="")
    args = p.parse_args()
    policy = None
    if args.learner_ckpt:
        policy = _load_learner(args.config, args.wm_ckpt, args.learner_ckpt)
    correction = None
    if args.correction:
        from experiments.aerial.rl.gap_correction import GapCorrection

        correction = GapCorrection.load(args.correction, device="cpu")
    routes = [int(x) for x in args.routes.split(",") if x.strip()]
    episodes = _load_routes(Path(args.annotation), routes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = AirSimDroneEnv(AirSimEnvConfig(step_hz=5.0, grab_depth=True, health_check=False))
    saved = 0
    summary = []
    try:
        for ep in episodes:
            idx = int(ep.get("route_idx", -1))
            logger.info("route %s", idx)
            result = fly_one(
                env, ep, max_steps=args.max_steps, success_dist=args.success_dist,
                policy=policy, intervene_m=args.intervene_m,
                record_executed=args.record_executed,
                early_west=args.early_west,
                correction=correction,
            )
            transitions = result.pop("transitions")
            result["route_idx"] = idx
            logger.info("route %s result %s", idx, result)
            (out / f"route{idx:02d}_summary.json").write_text(json.dumps(result, indent=2))
            keep = bool(transitions) and (
                (args.record_executed and result.get("arrived"))
                or (
                    not args.record_executed
                    and (
                        result.get("arrived")
                        or (policy is not None and args.intervene_m <= 0.0)
                        or (args.intervene_m > 0.0 and result.get("n_intervene", 0) > 0)
                    )
                )
            )
            if keep:
                write_episode(out, saved, transitions)
                saved += 1
            summary.append(result)
    finally:
        env.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("saved_episodes=%d", saved)


if __name__ == "__main__":
    main()
