"""``RolloutCollector`` — the serial real-env worker (Plan A).

One env instance (the renderer is single-consumer) driven at ~``step_hz``: reset
→ loop {policy → step → reward} → push a full episode to the ``ReplayBuffer``.
This is the only place that touches the real renderer; sample *volume* for
learning comes from imagination, not from parallel envs.

Policy dispatch is duck-typed (mirrors ``collect_dagger._predict_delta``):

  1. ``policy.act(obs) -> [4] body delta``            (RL / continuous policy)
  2. ``policy.predict_delta(rgb, state, instr)``      (delta-native policy)
  3. ``policy.predict_primitive(rgb, state, instr)``  → ``primitive_to_delta``
     (the existing ``FastWAMAerialPolicy`` / ``ReplayPolicy`` primitive path)

Achieved Hz is measured and logged every episode; a warning fires if it drops
below the configured target so the ~30 Hz Plan-A assumption is validated on real
hardware rather than assumed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from experiments.aerial.openfly_actions import primitive_to_delta
from experiments.aerial.rl.buffer import Episode, ReplayBuffer, Transition
from experiments.aerial.rl.env.action import DEFAULT_STEP_HZ, body_delta_limits, clip_body_delta
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import body_vel_from_obs, goal_rel_from_obs
from experiments.aerial.rl.reward import NavigationReward, RewardConfig
from experiments.aerial.rl.safety import SafetyShield
from experiments.aerial.rl.scene_profile import (
    SceneProfile,
    apply_episode_scene_profile,
    restore_scene_profile_context,
)

logger = logging.getLogger(__name__)


def act_delta(
    policy: Any,
    obs: Observation,
    instruction: str,
    limits: Optional[np.ndarray] = None,
    *,
    forbid_backward: bool = False,
) -> np.ndarray:
    """Resolve any supported policy to a finite, clipped 4-D body delta.

    ``limits`` is the per-step displacement cap for the env's control rate
    (``body_delta_limits(dt)``); defaults to the 30 Hz continuous cap. NOTE: a
    discrete-primitive policy returns a macro-sized delta (e.g. fwd 9 m) which
    this clips to a single per-step increment — driving macro primitives
    faithfully needs a multi-step executor, out of scope for the V0 skeleton.
    """
    act = getattr(policy, "act", None)
    if callable(act):
        # Hand the RGB-only view, never the full Observation: depth/IMU/velocity/
        # collision GT must not reach the policy graph (spec §1.2 boundary).
        # Pass stamped local goal (toward_g) so PolicyObservation.goal matches
        # obs.info["goal"] — Phase2CollectionPolicy may also set it, but the
        # view must carry the carrot even for inner deploy policies alone.
        goal = None
        info = getattr(obs, "info", None)
        if isinstance(info, dict) and info.get("goal") is not None:
            goal = np.asarray(info["goal"], dtype=np.float64).reshape(3)
        raw = act(obs.policy_view(goal=goal))
    else:
        predict_delta = getattr(policy, "predict_delta", None)
        if callable(predict_delta):
            raw = predict_delta(obs.rgb, obs.proprio4(), instruction)
        else:
            primitive = int(policy.predict_primitive(obs.rgb, obs.proprio4(), instruction))
            raw = primitive_to_delta(primitive)
    return clip_body_delta(
        np.asarray(raw, dtype=np.float64), limits, forbid_backward=forbid_backward
    )


@dataclass
class CollectStats:
    episodes: int = 0
    steps: int = 0
    seconds: float = 0.0
    #: Any shield change (hard brake OR soft TTI governor). Diagnostic IR.
    interventions: int = 0
    #: Exclusion / τ·p_coll emergency only (matches w_intervention charge).
    hard_brakes: int = 0
    #: Soft TTI forward/lateral speed caps only.
    governor_caps: int = 0
    # Episodes dropped at reset because the vehicle spawned already colliding
    # (spawn-inside-geometry). Not counted in `episodes`; never reach the buffer.
    skipped: int = 0
    returns: List[float] = field(default_factory=list)

    @property
    def achieved_hz(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    @property
    def hard_brake_rate(self) -> float:
        return self.hard_brakes / self.steps if self.steps > 0 else 0.0

    @property
    def governor_cap_rate(self) -> float:
        return self.governor_caps / self.steps if self.steps > 0 else 0.0

    @property
    def intervention_rate(self) -> float:
        return self.interventions / self.steps if self.steps > 0 else 0.0


class RolloutCollector:
    def __init__(
        self,
        env: Any,
        policy: Any,
        buffer: ReplayBuffer,
        *,
        reward_cfg: Optional[RewardConfig] = None,
        safety: Optional[SafetyShield] = None,
        max_steps: int = 200,
        target_hz: float = 30.0,
        on_episode: Optional[Callable[[Episode, CollectStats], None]] = None,
        skip_reset_collision: bool = True,
        min_spawn_z: float = 0.0,
        spawn_z_retry_m: float = 0.0,
        spawn_z_max_retries: int = 0,
        depth_predictor: Optional[Any] = None,
        tau_predictor: Optional[Any] = None,
        planner: Optional[Any] = None,
        dynamics: Optional[Any] = None,
        scene_profiles: Optional[Dict[str, SceneProfile]] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.reward_cfg = reward_cfg or RewardConfig()
        self.safety = safety
        self.max_steps = int(max_steps)
        self.target_hz = float(target_hz)
        # Drop episodes whose reset spawns the vehicle already in collision
        # (inside geometry): no action has been taken, so it's a spawn artifact,
        # not a learnable trajectory. Skipped before any step / buffer write.
        self.skip_reset_collision = bool(skip_reset_collision)
        self.min_spawn_z = float(min_spawn_z)
        self.spawn_z_retry_m = float(spawn_z_retry_m)
        self.spawn_z_max_retries = int(spawn_z_max_retries)
        # Optional sink invoked with every completed episode (e.g. persist to
        # disk). None -> collector stays purely in-memory (offline tests / V0).
        self.on_episode = on_episode
        # Frozen §4 ④: produce ``obs.info['depth_min_pred']`` BEFORE the shield
        # runs. ``DepthMinPredictor`` (or any object with ``predict_min`` /
        # optional ``reset``). None → leave info empty (default V0 posture).
        self.depth_predictor = depth_predictor
        # V1b [1d]: τ independent of D̂ — ``predict_tau(obs)`` → obs.info['tau_pred'].
        self.tau_predictor = tau_predictor
        # V1b: optional short-horizon imagination planner (scores candidates).
        self.planner = planner
        # V4 P2: online WM for live p_coll → should_override(obs, wm_out=...).
        self.dynamics = dynamics
        self.scene_profiles = scene_profiles
        #: Streaming latent for planner / shield. When collect policy is a
        #: ``LatentActorDeployPolicy`` (possibly under Phase2CollectionPolicy),
        #: this MUST track the same posterior stream as ``policy._latent`` —
        #: not a separate prior roll (2026-09-21 dual-latent audit).
        self._latent: Optional[np.ndarray] = None

    def _streaming_deploy_policy(self) -> Optional[Any]:
        """Unwrap Phase2CollectionPolicy → LatentActorDeployPolicy if present."""
        p = self.policy
        inner = getattr(p, "_inner", None)
        cand = inner if inner is not None else p
        if getattr(cand, "stream_latent", False) and hasattr(cand, "_latent"):
            return cand
        return None

    def _sync_latent_from_policy(self) -> None:
        sp = self._streaming_deploy_policy()
        if sp is not None and getattr(sp, "_latent", None) is not None:
            self._latent = np.asarray(sp._latent, dtype=np.float64).copy()

    def collect_episode(self, episode: Optional[Dict[str, Any]] = None) -> tuple[Episode, CollectStats]:
        instruction = str((episode or {}).get("gpt_instruction", ""))
        obs = self.env.reset(episode)
        # Entry guard: a vehicle already colliding at reset spawned inside
        # geometry. Skip before any step so it never pollutes the buffer/dataset
        # as a 1-step instant crash. (`collided` is populated at reset by both
        # backends — airsim_env.observe() / mock bounds check.)
        if self.skip_reset_collision and bool(getattr(obs, "collided", False)):
            logger.warning(
                "reset spawned in collision — skipping episode "
                "(spawn-inside-geometry; start pose may need resampling)"
            )
            return [], CollectStats(episodes=0, skipped=1)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        reset_pred = getattr(self.depth_predictor, "reset", None)
        if callable(reset_pred):
            reset_pred()
        if self.dynamics is not None:
            self._latent = np.asarray(self.dynamics.encode(obs), dtype=np.float64)

        reward = NavigationReward(getattr(self.env, "goal", None), self.reward_cfg)
        reward.reset(getattr(self.env, "goal", None), obs.position)
        env_goal = getattr(self.env, "goal", None)
        goal_xyz = (
            None if env_goal is None
            else np.asarray(env_goal, dtype=np.float32).reshape(3)
        )

        transitions: List[Transition] = []
        stats = CollectStats(episodes=1)
        # Per-step displacement cap for this env's control rate (keeps the clip
        # consistent with what env.step will apply).
        step_hz = float(getattr(getattr(self.env, "config", None), "step_hz", DEFAULT_STEP_HZ))
        scene_ctx = apply_episode_scene_profile(
            episode,
            self.reward_cfg,
            self.safety,
            step_hz=step_hz,
            profiles=self.scene_profiles,
        )
        limits = scene_ctx.limits
        # Variable-cs training: explicit cruise_speed overrides scene default.
        ep_cs = float((episode or {}).get("cruise_speed", 0.0))
        _prev_shield_cs: Optional[float] = None
        if ep_cs > 0.0:
            limits = np.asarray(limits, dtype=np.float64).copy()
            limits[0] = ep_cs / step_hz
            if self.safety is not None:
                zone = getattr(self.safety, "zone", None)
                if zone is not None and hasattr(zone, "v_cruise_m_s"):
                    _prev_shield_cs = float(zone.v_cruise_m_s)
                    zone.v_cruise_m_s = ep_cs
        ep_scene = str(episode.get("scene", "")).strip() if episode else ""
        if ep_scene:
            obs.info["scene"] = ep_scene
        ep_pose = str(episode.get("pose_source", "")).strip() if episode else ""
        if ep_pose:
            obs.info["pose_source"] = ep_pose
        if episode:
            for key in ("map_id", "handover_id", "leg"):
                val = episode.get(key)
                if val:
                    obs.info[key] = str(val)
        t_start = time.perf_counter()

        for _ in range(self.max_steps):
            # Depth/τ before policy so depth-conditioned experts (and eval-aligned
            # scene intent) see the same obs.info as shield / long_eval.
            if self.depth_predictor is not None:
                pred_azimuth = getattr(self.depth_predictor, "predict_min_cones_and_azimuth", None)
                pred_both = getattr(self.depth_predictor, "predict_min_and_cones", None)
                if callable(pred_azimuth):
                    # Single depth-head pass: min + cones + N-bin azimuth. Must
                    # be ONE call (not min_and_cones + a separate azimuth call)
                    # to avoid double-feeding the RGB history window.
                    d_min, cones, azimuth = pred_azimuth(obs)
                    if d_min is not None:
                        obs.info["depth_min_pred"] = float(d_min)
                    if isinstance(cones, dict):
                        obs.info["depth_cones_pred"] = {
                            k: (float(v) if v is not None else None)
                            for k, v in cones.items()
                        }
                    if isinstance(azimuth, dict):
                        obs.info["depth_azimuth_pred"] = azimuth
                elif callable(pred_both):
                    d_min, cones = pred_both(obs)
                    if d_min is not None:
                        obs.info["depth_min_pred"] = float(d_min)
                    if isinstance(cones, dict):
                        obs.info["depth_cones_pred"] = {
                            k: (float(v) if v is not None else None)
                            for k, v in cones.items()
                        }
                else:
                    d_min = self.depth_predictor.predict_min(obs)
                    if d_min is not None:
                        obs.info["depth_min_pred"] = float(d_min)
                    pred_cones = getattr(self.depth_predictor, "predict_cones", None)
                    if callable(pred_cones):
                        cones = pred_cones(obs)
                        if isinstance(cones, dict):
                            obs.info["depth_cones_pred"] = {
                                k: (float(v) if v is not None else None)
                                for k, v in cones.items()
                            }
            if self.tau_predictor is not None:
                tau = self.tau_predictor.predict_tau(obs)
                if tau is not None:
                    obs.info["tau_pred"] = float(tau)

            # toward_g (Phase2): stamp clipped carrot onto live obs.info BEFORE
            # act_delta / planner / WM probe — matches wam_phase2_long_eval order.
            stamp = getattr(self.policy, "stamp_local_goal", None)
            if callable(stamp):
                stamp(obs)

            # Per-step dx cap from TowardGoalIntent.safe_speed_limit (eval-aligned).
            step_limits = np.asarray(limits, dtype=np.float64).reshape(-1).copy()
            if isinstance(getattr(obs, "info", None), dict):
                safe_v = obs.info.get("safe_speed_limit")
                if safe_v is not None:
                    try:
                        sv = float(safe_v)
                        if np.isfinite(sv) and sv > 0.0:
                            step_limits[0] = sv / float(step_hz)
                    except (TypeError, ValueError):
                        pass

            action = act_delta(
                self.policy,
                obs,
                instruction,
                step_limits,
                forbid_backward=bool(
                    getattr(self.reward_cfg, "forbid_backward_motion", False)
                ),
            )
            # Keep planner/shield on the deploy policy's posterior stream.
            self._sync_latent_from_policy()
            feature_t = (
                np.asarray(self._latent, dtype=np.float64).copy()
                if self._latent is not None
                else None
            )
            if self.planner is not None:
                set_goal = getattr(self.planner, "set_goal", None)
                if callable(set_goal):
                    # Phase-2 toward_g: policy sees clipped subgoal in obs.info["goal"];
                    # planner must track the same target as eval (not raw episode G).
                    planner_goal = getattr(self.env, "goal", None)
                    info = getattr(obs, "info", None)
                    if isinstance(info, dict) and info.get("goal") is not None:
                        planner_goal = np.asarray(info["goal"], dtype=np.float64).reshape(3)
                    set_goal(planner_goal)
                action = np.asarray(
                    self.planner.plan(obs, action, latent=self._latent),
                    dtype=np.float64,
                ).reshape(4)
                action = clip_body_delta(
                    action,
                    step_limits,
                    forbid_backward=bool(
                        getattr(self.reward_cfg, "forbid_backward_motion", False)
                    ),
                )
            intervened = False
            # Safety shield sits ABOVE the learned policy (spec §2#6).
            # Probe p_coll on the *pre-shield* action (policy/planner intent) so
            # the real reward charges the action the actor proposed, not the
            # already-sanitized override (2026-09-21 audit).
            wm_out = None
            if self.dynamics is not None and self._latent is not None:
                wm_out = self.dynamics.step(
                    self._latent,
                    action,
                    goal_rel=goal_rel_from_obs(obs),
                    body_vel=body_vel_from_obs(obs),
                )
            if self.safety is not None:
                apply_fn = getattr(self.safety, "apply_action", None)
                if callable(apply_fn):
                    action, intervened = apply_fn(
                        action, obs, wm_out=wm_out, limits=step_limits
                    )
                elif self.safety.should_override(obs, wm_out=wm_out):
                    action = clip_body_delta(
                        self.safety.override_action(obs),
                        step_limits,
                        forbid_backward=bool(
                            getattr(self.reward_cfg, "forbid_backward_motion", False)
                        ),
                    )
                    intervened = True
                    if isinstance(obs.info, dict):
                        obs.info["shield_hard_brake"] = True

            # Streaming deploy policy recorded its own proposal as ``_prev_act``.
            # Rewrite to the *executed* (post-shield) action so the next
            # ``observe_and_advance`` matches real dynamics.
            if bool(getattr(self.reward_cfg, "forbid_backward_motion", False)):
                from experiments.aerial.rl.env.action import forbid_backward_dx

                action = forbid_backward_dx(action)
            sp = self._streaming_deploy_policy()
            if sp is not None:
                sp._prev_act = np.asarray(action, dtype=np.float64).copy()

            next_obs, info = self.env.step(action)
            # Soft collision risk from the pre-shield WM probe (intent).
            p_coll_real = None
            if wm_out is not None:
                p_coll_real = float(getattr(wm_out, "p_coll", 0.0))
            # Latent ownership: streaming deploy policy advances on the next
            # ``act()`` via observe_and_advance — do NOT also prior-roll
            # collector._latent (that was the dual-stream bug). Non-streaming
            # policies still get a prior advance for shield p_coll continuity.
            if sp is None and self.dynamics is not None and self._latent is not None:
                dyn_out = self.dynamics.step(
                    self._latent,
                    action,
                    goal_rel=goal_rel_from_obs(obs),
                    body_vel=body_vel_from_obs(obs),
                )
                self._latent = np.asarray(dyn_out.z_next, dtype=np.float64)
                if p_coll_real is None:
                    p_coll_real = float(getattr(dyn_out, "p_coll", 0.0))
            # Carry decision-time depth/τ onto next_obs so NavigationReward can
            # fold clearance_risk (env.observe() rebuilds info with only goal).
            if isinstance(getattr(obs, "info", None), dict) and isinstance(
                getattr(next_obs, "info", None), dict
            ):
                for k in (
                    "depth_min_pred",
                    "depth_cones_pred",
                    "depth_azimuth_pred",
                    "tau_pred",
                    "shield_hard_brake",
                    "shield_governor_cap",
                    "shield_channels",
                    "safe_speed_limit",
                    "yaw_err_rad",
                ):
                    if k in obs.info and k not in next_obs.info:
                        next_obs.info[k] = obs.info[k]
            # Directional obstacle cost on pre-step feature + *executed* action.
            if (
                feature_t is not None
                and self.dynamics is not None
                and bool(getattr(self.dynamics, "obstacle_cost_trained", False))
                and hasattr(self.dynamics, "predict_obstacle_cost")
            ):
                # Do not swallow errors: with use_learned_obstacle_cost, a silent
                # miss becomes free OA (risk=0). Fail loud when the head is live.
                oc = float(
                    self.dynamics.predict_obstacle_cost(feature_t, action)
                )
                if isinstance(getattr(next_obs, "info", None), dict):
                    next_obs.info["obstacle_cost"] = oc
            if isinstance(getattr(next_obs, "info", None), dict):
                next_obs.info.setdefault("goal_rel", goal_rel_from_obs(obs))
            r, done, terms = reward.step(
                next_obs, action, p_coll=p_coll_real, intervened=intervened,
            )
            ep_info = {**info, **terms, "intervention": intervened}
            # Prefer stamped local goal (toward_g carrot) over terminal env.goal.
            stamped = None
            if isinstance(obs.info, dict) and obs.info.get("goal") is not None:
                stamped = np.asarray(obs.info["goal"], dtype=np.float32).reshape(3)
            if stamped is not None:
                ep_info["goal"] = stamped.copy()
            elif goal_xyz is not None:
                ep_info["goal"] = goal_xyz.copy()
            if isinstance(obs.info, dict):
                if "scene" in obs.info:
                    ep_info["scene"] = obs.info["scene"]
                if "pose_source" in obs.info:
                    ep_info["pose_source"] = obs.info["pose_source"]
                for key in ("map_id", "handover_id", "leg"):
                    if key in obs.info:
                        ep_info[key] = obs.info[key]
            # ATTR / P7: persist shield inputs onto transition.info
            if isinstance(obs.info, dict):
                for k in (
                    "depth_min_pred",
                    "depth_cones_pred",
                    "depth_azimuth_pred",
                    "tau_pred",
                    "shield_channels",
                    "shield_hard_brake",
                    "shield_governor_cap",
                    "safe_speed_limit",
                    "yaw_err_rad",
                    "three_zone_speed_cap_m_s",
                    "tii_speed_cap_m_s",
                ):
                    if k in obs.info and k not in ep_info:
                        ep_info[k] = obs.info[k]
            transitions.append(
                Transition(
                    obs=obs, action=action, reward=r, done=done,
                    next_obs=next_obs,
                    info=ep_info,
                )
            )
            stats.steps += 1
            stats.interventions += int(intervened)
            if isinstance(getattr(obs, "info", None), dict):
                stats.hard_brakes += int(bool(obs.info.get("shield_hard_brake")))
                stats.governor_caps += int(bool(obs.info.get("shield_governor_cap")))
            obs = next_obs
            if done:
                break

        stats.seconds = time.perf_counter() - t_start
        stats.returns.append(float(sum(t.reward for t in transitions)))
        if self.target_hz > 0 and stats.achieved_hz < self.target_hz * 0.8:
            logger.warning(
                "collector achieved %.1f Hz (< %.1f Hz target) over %d steps",
                stats.achieved_hz, self.target_hz, stats.steps,
            )
        restore_scene_profile_context(scene_ctx, self.reward_cfg, self.safety)
        # Restore cruise_speed override when scene profile did not replace the zone.
        if _prev_shield_cs is not None and scene_ctx.prev_shield_zone is None and self.safety is not None:
            from experiments.aerial.rl.three_zone import ThreeZoneSpec

            zone = getattr(self.safety, "zone", None)
            if zone is not None:
                self.safety.zone = ThreeZoneSpec(
                    **{k: getattr(zone, k) for k in zone.__dataclass_fields__},
                    v_cruise_m_s=_prev_shield_cs,
                )
        self.buffer.add_episode(transitions)
        if self.on_episode is not None:
            self.on_episode(transitions, stats)
        return transitions, stats

    def collect(
        self,
        num_episodes: int = 1,
        episodes: Optional[List[Dict[str, Any]]] = None,
        episode_offset: int = 0,
    ) -> CollectStats:
        from experiments.aerial.rl.spawn_utils import collect_episode_with_spawn_retries

        total = CollectStats()
        for i in range(int(num_episodes)):
            ep = None
            if episodes:
                ep = episodes[(int(episode_offset) + i) % len(episodes)]
            if ep is not None and (
                self.min_spawn_z > 0
                or (self.spawn_z_retry_m > 0 and self.spawn_z_max_retries > 0)
            ):
                _, _, s = collect_episode_with_spawn_retries(
                    self,
                    ep,
                    min_spawn_z=self.min_spawn_z,
                    spawn_z_retry_m=self.spawn_z_retry_m,
                    spawn_z_max_retries=self.spawn_z_max_retries,
                )
            else:
                _, s = self.collect_episode(ep)
            total.episodes += s.episodes
            total.steps += s.steps
            total.seconds += s.seconds
            total.interventions += s.interventions
            total.hard_brakes += s.hard_brakes
            total.governor_caps += s.governor_caps
            total.skipped += s.skipped
            total.returns.extend(s.returns)
        return total
