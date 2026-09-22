"""``SerialCorrectorLoop`` — Plan-A orchestration (collect → WM → imagine-RL).

One serial pass per iteration:

    collector.collect(...)                 # V0: real, runnable now
    [GATE V1] dynamics.update(windows)     # world-model training — no-op stub
    [GATE V4] imagine + policy/value update # RL in imagination — no-op stub

The two learning stages are real methods guarded by ``enable_wm_update`` /
``enable_policy_update`` flags that default OFF (spec ladder: "未过关不叠加下一阶段").
Running the loop today exercises the full V0 collection path and cleanly no-ops
the gated stages, logging why. Flip a flag (once its milestone passes) and the
insertion point is already wired: WM training consumes buffer windows; the RL
update consumes ``imagine(...)`` trajectories.

``smoke=True`` runs exactly one collection pass and returns its stats — the
on-4090 smoke test entrypoint.
"""
from __future__ import annotations

import logging
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import CollectStats, RolloutCollector
from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.imagination import imagine
from experiments.aerial.rl.reward import maneuver_weight_at

logger = logging.getLogger(__name__)


def _save_actor_ckpt(
    ckpt_dir: str,
    actor_critic: Any,
    iter_idx: int,
    *,
    also_best: bool = False,
) -> None:
    from pathlib import Path

    import torch

    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor_critic._actor.state_dict(),
        "critic": actor_critic._critic.state_dict(),
        "log_std": actor_critic._log_std.detach().cpu(),
        "config": actor_critic.config.__dict__,
        "iter_idx": int(iter_idx),
    }
    torch.save(payload, out / "v4_ac_latest.pt")
    torch.save(payload, out / f"v4_ac_iter_{iter_idx:04d}.pt")
    if also_best:
        torch.save(payload, out / "v4_ac_best.pt")
        logger.info("wrote BEST ckpt iter %d -> %s", iter_idx, out / "v4_ac_best.pt")
    logger.info("wrote ckpt iter %d -> %s", iter_idx, out / "v4_ac_latest.pt")


@dataclass
class CorrectorConfig:
    iterations: int = 10
    episodes_per_iter: int = 1
    # V1/V4 gates — OFF until each milestone passes.
    enable_wm_update: bool = False
    enable_policy_update: bool = False
    strict_gates: bool = False        # raise (vs skip+log) if a gate is off
    # WM-training window sampling (used once enable_wm_update flips on).
    wm_batch: int = 32
    wm_window: int = 8
    # Imagination-RL params (used once enable_policy_update flips on).
    imagine_batch: int = 64
    imagine_horizon: int = 10
    # Behavior cloning on expert transitions (optional; default OFF).
    enable_bc_update: bool = False
    bc_batch: int = 64
    bc_updates_per_iter: int = 4
    bc_loss_scale: float = 1.0
    #: Harvest planner≠actor (or climb/escape) steps from online collect into BC pool.
    enable_online_planner_bc: bool = False
    online_bc_diff_thr: float = 0.08
    online_bc_max_pool: int = 8000
    smoke: bool = False
    start_iter: int = 0
    ckpt_dir: Optional[str] = None
    save_every_iter: bool = False
    #: Ignore ultra-short collects when updating v4_ac_best (spawn noise).
    min_steps_for_best: int = 20
    # Periodic AirSim renderer restart (125 long online runs). 0 = disabled.
    renderer_restart_every: int = 0
    renderer_restart_script: Optional[str] = None
    renderer_restart_scene: str = "outdoor"
    renderer_restart_wait_s: float = 30.0
    renderer_host: str = "127.0.0.1"
    renderer_port: int = 41451


@dataclass
class IterationReport:
    collect: CollectStats
    wm: Dict[str, Any] = field(default_factory=dict)
    rl: Dict[str, Any] = field(default_factory=dict)


class SerialCorrectorLoop:
    def __init__(
        self,
        collector: RolloutCollector,
        buffer: ReplayBuffer,
        dynamics: LatentDynamics,
        *,
        imagination_policy: Optional[Any] = None,
        actor_critic: Optional[Any] = None,
        config: Optional[CorrectorConfig] = None,
        episodes: Optional[List[Dict[str, Any]]] = None,
        expert_transitions: Optional[List[Any]] = None,
    ) -> None:
        self.collector = collector
        self.buffer = buffer
        self.dynamics = dynamics
        self.imagination_policy = imagination_policy
        self.actor_critic = actor_critic
        self.config = config or CorrectorConfig()
        self.episodes = episodes
        # Flat expert Transition list for BC (DepthScene / PathExpert demos).
        self.expert_transitions = list(expert_transitions or [])
        # Snapshot the base maneuver weight ONCE: the curriculum rewrites
        # ``collector.reward_cfg.w_maneuver`` each iteration, so the schedule must
        # ramp from this immutable start, never from its own last output.
        self._w_maneuver_start = float(getattr(self.collector.reward_cfg, "w_maneuver", 0.0))
        self._best_collect_return = float("-inf")

    def run(self) -> List[IterationReport]:
        # Own the env lifecycle: whatever happens, release the single-consumer
        # renderer so a crash mid-run never leaves it armed/occupied.
        try:
            if self.config.smoke:
                stats = self.collector.collect(1, episodes=self.episodes)
                logger.info("smoke collect: %d steps @ %.1f Hz", stats.steps, stats.achieved_hz)
                return [IterationReport(collect=stats, wm={"skipped": True}, rl={"skipped": True})]

            reports: List[IterationReport] = []
            start = max(0, int(self.config.start_iter))
            for it in range(start, self.config.iterations):
                if self._should_restart_renderer(it):
                    self._restart_renderer()
                stats = self.collector.collect(
                    self.config.episodes_per_iter,
                    episodes=self.episodes,
                    episode_offset=it,
                )
                if int(stats.steps) > 0:
                    n_bc_h = self._harvest_online_planner_bc()
                    if n_bc_h:
                        logger.info(
                            "online planner-BC harvest +%d (pool=%d)",
                            n_bc_h,
                            len(self.expert_transitions),
                        )
                self._apply_maneuver_curriculum(stats)
                # Empty / all-spawn-collision iters must not RL-update on stale buffer
                # (was poisoning hard014 with 0-step "updated" noise).
                if int(stats.steps) <= 0:
                    wm = {"status": "skipped", "reason": "empty collect"}
                    rl = {"status": "skipped", "reason": "empty collect"}
                else:
                    wm = self._update_world_model()
                    rl = self._update_policy()
                logger.info(
                    "iter %d: %d steps @ %.1f Hz | wm=%s | rl=%s | w_man=%.4g",
                    it, stats.steps, stats.achieved_hz, wm.get("status", "?"), rl.get("status", "?"),
                    self.collector.reward_cfg.w_maneuver,
                )
                reports.append(IterationReport(collect=stats, wm=wm, rl=rl))
                if self.config.save_every_iter and self.config.ckpt_dir and self.actor_critic is not None:
                    mean_ret = (
                        float(np.mean(stats.returns)) if stats.returns else float("-inf")
                    )
                    min_best = int(getattr(self.config, "min_steps_for_best", 20) or 0)
                    is_best = bool(
                        int(stats.steps) >= max(1, min_best)
                        and mean_ret > self._best_collect_return
                    )
                    if is_best:
                        self._best_collect_return = mean_ret
                    _save_actor_ckpt(
                        self.config.ckpt_dir,
                        self.actor_critic,
                        it,
                        also_best=is_best,
                    )
            return reports
        finally:
            close = getattr(getattr(self.collector, "env", None), "close", None)
            if callable(close):
                close()

    def _should_restart_renderer(self, iter_idx: int) -> bool:
        every = int(self.config.renderer_restart_every)
        if every <= 0 or iter_idx <= 0:
            return False
        return iter_idx % every == 0

    def _restart_renderer(self) -> None:
        script = self.config.renderer_restart_script
        if not script:
            logger.warning("renderer restart requested but renderer_restart_script is unset — skipping")
            return
        scene_sh = Path(script).expanduser()
        if not scene_sh.is_file():
            alt = Path.home() / "aerial-indoor-wam/experiments/aerial/scripts/recover_renderer_scene.sh"
            if alt.is_file():
                scene_sh = alt
            else:
                raise FileNotFoundError(f"renderer restart script missing: {script}")

        close = getattr(getattr(self.collector, "env", None), "close", None)
        if callable(close):
            close()

        scene = str(self.config.renderer_restart_scene)
        logger.info("restarting AirSim renderer (scene=%s) via %s", scene, scene_sh)
        subprocess.run(["bash", str(scene_sh), scene], check=True)
        time.sleep(float(self.config.renderer_restart_wait_s))

        host = str(self.config.renderer_host)
        port = int(self.config.renderer_port)
        for attempt in range(1, 37):
            try:
                socket.create_connection((host, port), 3).close()
                logger.info("AirSim reachable at %s:%d (try %d)", host, port, attempt)
                return
            except OSError:
                time.sleep(5)
        raise RuntimeError(f"AirSim not reachable at {host}:{port} after renderer restart")

    # -- maneuver-penalty curriculum (design doc §2.4) -------------------
    def _apply_maneuver_curriculum(self, stats: CollectStats) -> None:
        """Ramp ``reward_cfg.w_maneuver`` by competence (mean episode return).

        Applied AFTER each iteration's collect, so the ramped weight takes effect
        on the next iteration's episodes (``NavigationReward`` is rebuilt from
        ``collector.reward_cfg`` per episode). No-op when the curriculum is
        unconfigured (``w_maneuver_final == w_maneuver``) or a fully-skipped
        iteration yields no returns to score.
        """
        if not stats.returns:
            return
        metric = float(np.mean(stats.returns))
        self.collector.reward_cfg.w_maneuver = maneuver_weight_at(
            metric, self.collector.reward_cfg, w_start=self._w_maneuver_start,
        )

    def _harvest_online_planner_bc(self) -> int:
        """Append planner-selected steps that diverge from the actor into BC pool."""
        if not bool(self.config.enable_online_planner_bc):
            return 0
        if self.buffer.num_episodes <= 0:
            return 0
        ep = list(self.buffer._episodes)[-1]
        thr = float(self.config.online_bc_diff_thr)
        added = 0
        for t in ep:
            info = t.info if isinstance(t.info, dict) else {}
            if not info.get("planner_meta") and info.get("offer_escape") is None:
                continue
            keep = bool(info.get("chose_climb")) or bool(info.get("offer_escape"))
            actor_a = info.get("action_actor")
            if actor_a is not None:
                aa = np.asarray(actor_a, dtype=np.float64).reshape(-1)
                pa = np.asarray(t.action, dtype=np.float64).reshape(-1)
                if aa.size == pa.size and float(np.linalg.norm(pa - aa)) >= thr:
                    keep = True
            if not keep:
                continue
            self.expert_transitions.append(t)
            added += 1
        cap = int(self.config.online_bc_max_pool)
        if cap > 0 and len(self.expert_transitions) > cap:
            self.expert_transitions = self.expert_transitions[-cap:]
        return added

    # -- GATE V1: world-model training -----------------------------------
    def _update_world_model(self) -> Dict[str, Any]:
        if not self.config.enable_wm_update:
            msg = "world-model training is V1-gated (enable_wm_update=False)"
            if self.config.strict_gates:
                raise RuntimeError(msg)
            return {"status": "skipped", "reason": msg}
        try:
            windows = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window)
        except ValueError as exc:
            return {"status": "skipped", "reason": f"insufficient data: {exc}"}
        result = self.dynamics.update(windows)
        if result.get("skipped"):
            # Gate flipped ON but the dynamics has no real training step yet (the
            # V1 fast-WM has not landed): report "noop", never "updated". Flipping
            # a boolean must not fabricate progress.
            return {
                "status": "noop",
                "reason": result.get("reason", "dynamics.update is a stub"),
                **result,
            }
        return {"status": "updated", **result}

    # -- GATE V4: imagination actor-critic update ------------------------
    def _update_policy(self) -> Dict[str, Any]:
        if not self.config.enable_policy_update:
            if bool(self.config.enable_bc_update):
                bc_out = self._update_bc()
                if bc_out.get("status") == "updated":
                    return {"status": "bc_only", **bc_out}
                return {
                    "status": "skipped",
                    "reason": "bc-only produced no update",
                    "bc": bc_out,
                }
            msg = "imagination RL update is V4-gated (enable_policy_update=False)"
            if self.config.strict_gates:
                raise RuntimeError(msg)
            return {"status": "skipped", "reason": msg}
        if self.imagination_policy is None:
            return {"status": "skipped", "reason": "no imagination_policy provided"}
        # Encode start states, imagine forward, then AC-update the rollout.
        try:
            transitions = self.buffer.sample(self.config.imagine_batch)
        except ValueError as exc:
            return {"status": "skipped", "reason": f"insufficient data: {exc}"}
        # Point imagination at the current episode goal so imagined progress /
        # arrival track the real task (a stub with goal=None yields progress≡0,
        # a reward misaligned with the real collector).
        set_goal = getattr(self.dynamics, "set_goal", None)
        if callable(set_goal):
            set_goal(getattr(getattr(self.collector, "env", None), "goal", None))
        from experiments.aerial.rl.goal_features import body_vel_from_obs, goal_rel_from_obs

        goal_rel0 = np.stack([goal_rel_from_obs(t.obs) for t in transitions], axis=0)
        body_vel0 = np.stack([body_vel_from_obs(t.obs) for t in transitions], axis=0)
        z0 = np.stack([self.dynamics.encode(t.obs) for t in transitions], axis=0)
        # Imagine inside the DEPLOYED action set (C2, 2026-08-18): the AC owns the
        # box (= ``body_delta_limits(1/step_hz)``) and its sampling law already
        # respects it, so this is a no-op guard whose ``n_action_clipped`` counter
        # is what proves the two spaces agree.
        ac = getattr(self, "actor_critic", None)
        act_limits = getattr(ac, "action_limits", None) if ac is not None else None
        rollout = imagine(
            self.dynamics, self.imagination_policy, z0, self.config.imagine_horizon,
            reward_cfg=getattr(self.collector, "reward_cfg", None),  # match real weights + bonus
            goal_rel0=goal_rel0,
            body_vel0=body_vel0,
            action_limits=act_limits,
        )
        if rollout.n_action_clipped:
            logger.warning(
                "imagined actions left the deployed box %d times (policy_class=%s) — "
                "imagined and deployed action spaces disagree",
                int(rollout.n_action_clipped),
                getattr(getattr(ac, "config", None), "policy_class", "?"),
            )
        mean_abs_goal_rel = float(np.mean(np.abs(goal_rel0)))
        mean_progress = float(rollout.progress.mean())
        if ac is not None:
            ac_out = ac.update(rollout)
            out = {
                "status": "updated",
                "batch": int(z0.shape[0]),
                "horizon": int(self.config.imagine_horizon),
                "mean_return": float(rollout.returns.mean()),
                "mean_abs_goal_rel": mean_abs_goal_rel,
                "mean_progress": mean_progress,
                **{k: v for k, v in ac_out.items() if k != "status"},
            }
            bc_out = self._update_bc()
            if bc_out:
                out["bc"] = bc_out
            return out
        # >>> V4 INSERTION POINT: actor_critic.update(rollout) <<<
        return {
            "status": "imagined",
            "batch": int(z0.shape[0]),
            "horizon": int(self.config.imagine_horizon),
            "mean_return": float(rollout.returns.mean()),
            "mean_abs_goal_rel": mean_abs_goal_rel,
            "mean_progress": mean_progress,
            "note": "trajectories produced; wire actor_critic for AC update",
        }

    def _update_bc(self) -> Dict[str, Any]:
        """Optional BC step on ``expert_transitions`` after the AC update."""
        if not bool(self.config.enable_bc_update):
            return {}
        ac = self.actor_critic
        if ac is None or not hasattr(ac, "update_bc"):
            return {"status": "skipped", "reason": "no actor_critic.update_bc"}
        pool = self.expert_transitions
        if not pool:
            return {"status": "skipped", "reason": "no expert_transitions"}
        from experiments.aerial.rl.goal_features import goal_rel_from_obs

        n_updates = max(1, int(self.config.bc_updates_per_iter))
        batch = max(1, int(self.config.bc_batch))
        losses: List[float] = []
        maes: List[float] = []
        rng = np.random.default_rng()
        for _ in range(n_updates):
            idx = rng.integers(0, len(pool), size=min(batch, len(pool)))
            chosen = [pool[int(i)] for i in idx]
            z = np.stack([self.dynamics.encode(t.obs) for t in chosen], axis=0)
            acts = np.stack([np.asarray(t.action, dtype=np.float64) for t in chosen], axis=0)
            goal_rel = np.stack([goal_rel_from_obs(t.obs) for t in chosen], axis=0)
            bc = ac.update_bc(
                z, acts, goal_rel=goal_rel, loss_scale=float(self.config.bc_loss_scale)
            )
            if bc.get("status") == "updated":
                losses.append(float(bc["bc_loss"]))
                maes.append(float(bc["bc_mae"]))
        if not losses:
            return {"status": "skipped", "reason": "bc updates produced no loss"}
        return {
            "status": "updated",
            "n_updates": int(n_updates),
            "batch": int(batch),
            "bc_loss": float(sum(losses) / len(losses)),
            "bc_mae": float(sum(maes) / len(maes)),
            "n_expert": int(len(pool)),
        }
