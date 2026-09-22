"""Persist collected episodes + score their quality (V1 WM-training input).

The V0 collector holds episodes only in the in-memory ``ReplayBuffer``; the
world-model milestone (V1) needs them on disk. This module turns an episode
(list of ``Transition``) into:

  * a compressed ``.npz`` — the heavy per-frame tensors (RGB stack, proprio,
    actions, rewards, done/collided masks, optional depth). One file per episode.
  * a ``quality_report`` — cheap statistics that answer the only question a raw
    collection can't assume: *is this data non-trivial?* A frozen renderer
    (identical frames), a drone that never moved (API control silently off), or
    an all-zero reward channel are all invisible to "it ran without error" but
    fatal to WM training. ``assert_nontrivial`` turns the dangerous ones into
    hard failures.

Pure numpy + json; no torch / cv2 / AirSim, so it is fully unit-testable and the
writer runs anywhere the collector does.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.env.obs import Observation, depth_sanity_detail
from experiments.aerial.rl.goal_features import attach_goal

# A frozen renderer / dead API-control run is the failure we most need to catch.
MIN_FRAME_VARIATION = 1e-3   # mean |Δ| between consecutive RGB frames (uint8 scale)
MIN_RGB_STD = 1.0            # pixel std across the episode (all-black/constant guard)
MIN_PATH_LENGTH_M = 1e-3     # summed |Δposition| — did the vehicle move at all?

# An episode that ends in a collision after <=this many steps is an instant
# crash — a bad spawn / start pose, not a learnable trajectory. Quarantined
# (excluded from the usable set) rather than hard-failed: for steps<=1 the path
# length is structurally 0 (a lone position can't be differenced), so `collided`
# is the ONLY valid discriminator here — the frozen/moved path checks can't see it.
SPAWN_COLLISION_MAX_STEPS = 2
# Run level: a handful of instant crashes is expected, but if more than this
# fraction of a collection is quarantined the start-pose distribution is broken
# and the whole run should fail.
MAX_QUARANTINE_FRACTION = 0.2


def _imu_row(imu: Dict[str, Any], key: str) -> np.ndarray:
    """One IMU triple (``ang_vel`` / ``lin_acc``) as [3] f32, or NaN if absent.

    The real env's ``_grab_imu`` returns ``{}`` on an RPC miss; a NaN row keeps
    the on-disk array dense and rectangular while the paired ``imu_present`` mask
    records which frames actually carried inertial data (schema v2, spec §4.1c —
    VIO supervision needs the raw IMU that the 4-D proprio dropped).
    """
    vec = imu.get(key) if isinstance(imu, dict) else None
    if vec is None:
        return np.full(3, np.nan, dtype=np.float32)
    return np.asarray(vec, dtype=np.float32).reshape(3)


def episode_arrays(transitions: Sequence[Transition]) -> Dict[str, np.ndarray]:
    """Stack an episode's transitions into per-field arrays for ``.npz`` storage."""
    if not transitions:
        raise ValueError("cannot serialize an empty episode")
    arrays: Dict[str, np.ndarray] = {
        "rgb": np.stack([t.obs.rgb for t in transitions]),                 # [N,H,W,3] u8
        "proprio": np.stack([t.obs.proprio4() for t in transitions]),      # [N,4] f32
        "actions": np.stack([np.asarray(t.action, np.float32) for t in transitions]),  # [N,4]
        "rewards": np.asarray([t.reward for t in transitions], np.float32),
        "dones": np.asarray([t.done for t in transitions], np.bool_),
        # Post-step collision (next_obs): the flag that ends an episode lives on
        # the observation AFTER the action. Using obs.collided alone misses the
        # instant-crash case (1-step: pre-step obs is clean, post-step is not).
        "collided": np.asarray(
            [
                bool(
                    (t.next_obs.collided if t.next_obs is not None else False)
                    or t.obs.collided
                )
                for t in transitions
            ],
            np.bool_,
        ),
        # Schema v2 (spec §4.1c / §7 V0): the signals the 4-D policy proprio drops
        # but the perception pillars need. Velocity (state[3:6]) + per-frame IMU
        # feed the [1c] windowed VIO trainer; timestamps give the real dt for
        # inertial integration + depth-rate provenance. Supervision-only — never
        # fed to the policy/WM (the RGB+proprio4 boundary lives in obs.py).
        "vel": np.stack([np.asarray(t.obs.velocity, np.float32) for t in transitions]),  # [N,3]
        "imu_ang_vel": np.stack([_imu_row(t.obs.imu, "ang_vel") for t in transitions]),  # [N,3]
        "imu_lin_acc": np.stack([_imu_row(t.obs.imu, "lin_acc") for t in transitions]),  # [N,3]
        "imu_present": np.asarray([bool(t.obs.imu) for t in transitions], np.bool_),      # [N]
        "timestamps": np.asarray([float(t.obs.t) for t in transitions], np.float32),      # [N]
    }
    # Depth is optional per-step (grab_depth); store it only if every frame has it.
    if all(t.obs.depth is not None for t in transitions):
        arrays["depth"] = np.stack([np.asarray(t.obs.depth, np.float32) for t in transitions])
    # Episode goal for V1-② reward features (optional; legacy corpora omit it).
    goal = None
    for tr in transitions:
        for bag in (tr.info, getattr(tr.obs, "info", {}) or {}):
            if isinstance(bag, dict) and bag.get("goal") is not None:
                goal = np.asarray(bag["goal"], dtype=np.float32).reshape(3)
                break
        if goal is not None:
            break
    if goal is None:
        # Write path: only persist an explicit collector-stamped goal (or none).
        # Do not invent end-proprio proxies into training corpora.
        pass
    else:
        arrays["goal"] = np.asarray(goal, dtype=np.float32).reshape(3)
    return arrays


def write_episode(out_dir: Path, index: int, transitions: Sequence[Transition]) -> Path:
    """Write one episode to ``out_dir/episode_{index:05d}.npz``; return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{index:05d}.npz"
    np.savez_compressed(path, **episode_arrays(transitions))
    return path


def load_episode(path: Path) -> List[Transition]:
    """Rehydrate an episode written by ``write_episode``.

    Schema v2 restores what v1 dropped: when ``vel`` is present the full 7-D
    ``state`` is reconstructed (v1 npz pad ``vx,vy,vz = 0``); when ``imu_*`` /
    ``timestamps`` are present they repopulate ``obs.imu`` / ``obs.t``. Every new
    key is guarded by ``in raw.files`` so legacy ``dataset_v0`` / ``dataset_v1_rgb``
    npz still load with the old fallbacks. ``next_obs`` is the following frame's
    obs (last step duplicates obs) — enough for buffer window sampling + the
    perception/WM trainers; not a bit-exact clone of the live ``Transition`` graph.
    """
    path = Path(path)
    raw = np.load(path)
    rgb = raw["rgb"]
    proprio = raw["proprio"]
    actions = raw["actions"]
    rewards = raw["rewards"]
    dones = raw["dones"]
    collided = raw["collided"]
    depth = raw["depth"] if "depth" in raw.files else None
    vel = raw["vel"] if "vel" in raw.files else None
    if vel is None:
        # Legacy v1 npz: velocity was not stored, so state[3:6] is zero-FILLED,
        # not zero-MEASURED. A VIO / perception consumer must not read these
        # zeros as ground-truth motion. v2 always writes 'vel'; a v2 file missing
        # it is corrupt. Surface the lossy path instead of failing silently.
        warnings.warn(
            f"{path.name}: no 'vel' field (legacy v1); velocity is zero-filled, "
            "not measured — do not train VIO/scale on it",
            stacklevel=2,
        )
    imu_ang_vel = raw["imu_ang_vel"] if "imu_ang_vel" in raw.files else None
    imu_lin_acc = raw["imu_lin_acc"] if "imu_lin_acc" in raw.files else None
    imu_present = raw["imu_present"] if "imu_present" in raw.files else None
    timestamps = raw["timestamps"] if "timestamps" in raw.files else None
    goal_npz = (
        np.asarray(raw["goal"], dtype=np.float32).reshape(3)
        if "goal" in raw.files else None
    )
    goals_per_step = (
        np.asarray(raw["goals"], dtype=np.float32).reshape(-1, 3)
        if "goals" in raw.files else None
    )
    n = int(rgb.shape[0])
    if n == 0:
        raise ValueError(f"empty episode file: {path}")

    def _obs_at(i: int, collided_flag: bool) -> Observation:
        x, y, z, yaw = (float(v) for v in proprio[i])
        # v2: recover the velocity triple; v1: pad zeros (documented lossy path).
        if vel is not None:
            vx, vy, vz = (float(v) for v in vel[i])
        else:
            vx = vy = vz = 0.0
        state = np.array([x, y, z, vx, vy, vz, yaw], dtype=np.float32)
        d = None if depth is None else np.asarray(depth[i], dtype=np.float32)
        imu: Dict[str, Any] = {}
        # Only repopulate IMU for frames the mask marks present (NaN rows are the
        # RPC-miss sentinel — reconstructing them as [nan,nan,nan] would poison
        # sanity.imu_ok / the VIO trainer).
        if (
            imu_ang_vel is not None
            and imu_lin_acc is not None
            and (imu_present is None or bool(imu_present[i]))
        ):
            imu = {
                "ang_vel": np.asarray(imu_ang_vel[i], np.float32).tolist(),
                "lin_acc": np.asarray(imu_lin_acc[i], np.float32).tolist(),
            }
        t = float(timestamps[i]) if timestamps is not None else 0.0
        return Observation(
            rgb=np.asarray(rgb[i], dtype=np.uint8),
            state=state,
            collided=bool(collided_flag),
            depth=d,
            imu=imu,
            t=t,
        )

    # ``collided[i]`` is a POST-step flag (written as next_obs[i].collided): the
    # contact results from action i and lives on the frame AFTER it. So frame i's
    # PRE-step collided state is the previous step's post flag (collided[i-1]),
    # and False on the first frame. Assigning collided[i] to obs[i] (the old
    # reload) smeared the terminal post-step contact onto the pre-step obs.
    transitions: List[Transition] = []
    for i in range(n):
        pre = bool(collided[i - 1]) if i >= 1 else False
        obs = _obs_at(i, pre)
        if i + 1 < n:
            # next_obs is the following frame, whose pre-step state == collided[i].
            next_obs = _obs_at(i + 1, bool(collided[i]))
        else:
            # Terminal: no stored frame n+1 — synthesize a distinct post-step obs
            # carrying collided[i] so quarantine_reasons / ④ still see the crash,
            # without marking the terminal pre-step obs as collided.
            next_obs = _obs_at(i, bool(collided[i]))
        transitions.append(
            Transition(
                obs=obs,
                action=np.asarray(actions[i], dtype=np.float32),
                reward=float(rewards[i]),
                done=bool(dones[i]),
                next_obs=next_obs,
            )
        )
    # Stamp goal for reward-head features (V1-②). Stored npz key only — do not
    # invent end-proprio / reward-fit proxies here (r60 end-proxy and fit both
    # mislead; use ``backfill_episode_goals`` against the OpenFly annotation).
    if goals_per_step is not None:
        from experiments.aerial.rl.goal_features import attach_goals_per_step

        attach_goals_per_step(transitions, goals_per_step)
    elif goal_npz is not None:
        attach_goal(transitions, goal_npz)
    return transitions


def load_dataset(
    out_dir: Path,
    *,
    skip_quarantined: bool = True,
) -> List[List[Transition]]:
    """Load every ``episode_*.npz`` under ``out_dir`` (sorted by name).

    When ``skip_quarantined`` is true, instant-crash episodes are omitted so a
    V0 smoke corpus can still exercise the load→buffer→stub-WM path without
    feeding spawn-collision junk into window sampling.
    """
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz under {out_dir}")
    episodes: List[List[Transition]] = []
    for p in paths:
        ep = load_episode(p)
        if skip_quarantined and quarantine_reasons(quality_report(ep)):
            continue
        episodes.append(ep)
    return episodes


def quality_report(transitions: Sequence[Transition]) -> Dict[str, Any]:
    """Cheap non-triviality statistics over one episode (JSON-serializable)."""
    arr = episode_arrays(transitions)
    rgb = arr["rgb"].astype(np.float32)
    pos = arr["proprio"][:, :3].astype(np.float64)
    rewards = arr["rewards"].astype(np.float64)

    frame_var = (
        float(np.mean(np.abs(np.diff(rgb, axis=0)))) if rgb.shape[0] > 1 else 0.0
    )
    path_len = (
        float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        if pos.shape[0] > 1 else 0.0
    )
    report: Dict[str, Any] = {
        "steps": int(rgb.shape[0]),
        "rgb_frame_variation": frame_var,      # ~0 => renderer frozen
        "rgb_std": float(rgb.std()),           # ~0 => constant/black image
        "path_length_m": path_len,             # ~0 => vehicle never moved
        "pos_min": pos.min(axis=0).tolist(),
        "pos_max": pos.max(axis=0).tolist(),
        "reward_sum": float(rewards.sum()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_nonzero_frac": float(np.mean(rewards != 0.0)),
        "collisions": int(arr["collided"].sum()),
        "has_depth": "depth" in arr,
        # Schema v2: fraction of frames carrying IMU (0.0 on legacy npz that never
        # stored it) — lets QUALITY_SUMMARY confirm the perception dataset is
        # VIO-trainable, not just collision-safe.
        "has_imu": "imu_present" in arr,
        "imu_present_frac": (
            float(np.mean(arr["imu_present"])) if "imu_present" in arr else 0.0
        ),
    }
    if "depth" in arr:
        # Reuse the validated depth gate on the mean frame.
        report["depth"] = depth_sanity_detail(arr["depth"].mean(axis=0))
    return report


def assert_nontrivial(report: Dict[str, Any]) -> List[str]:
    """Return a list of HARD failures (empty == data is usable for WM training).

    These are the silent-but-fatal conditions: a frozen renderer, a constant
    image, or a vehicle that never moved. A flat reward channel is a *warning*
    (a legitimately stationary episode can be all-zero), not a hard failure.
    """
    failures: List[str] = []
    if report["steps"] <= 0:
        failures.append("empty episode (0 steps)")
    if report["rgb_frame_variation"] < MIN_FRAME_VARIATION and report["steps"] > 1:
        failures.append(
            f"RGB frames barely change ({report['rgb_frame_variation']:.2e} "
            f"< {MIN_FRAME_VARIATION}) — renderer may be frozen"
        )
    if report["rgb_std"] < MIN_RGB_STD:
        failures.append(
            f"RGB nearly constant (std {report['rgb_std']:.2e} < {MIN_RGB_STD}) "
            "— black / blank frames"
        )
    if report["path_length_m"] < MIN_PATH_LENGTH_M and report["steps"] > 1:
        failures.append(
            f"vehicle did not move (path {report['path_length_m']:.2e} m) "
            "— API control may be off"
        )
    return failures


def quarantine_reasons(report: Dict[str, Any]) -> List[str]:
    """Soft exclusions: episodes that ran cleanly but aren't usable training data.

    Distinct from ``assert_nontrivial`` (silent-but-fatal, always a hard fail): a
    quarantined episode's collection succeeded — it's just untrustworthy. The
    canonical case is the *instant crash*: a 1–2 step episode ending in a
    collision (spawn-inside-geometry or a start pose facing a wall). The path
    check in ``assert_nontrivial`` can't catch these — a 1-step episode has a
    structurally-zero path — so ``collided`` is the discriminator. These are
    excluded per-episode; the *run-level* gate (``MAX_QUARANTINE_FRACTION``)
    decides whether there are enough of them to condemn the whole collection.

    Also treat ultra-short episodes (``steps <= SPAWN_COLLISION_MAX_STEPS``) with a
    large negative return as crashes: legacy ``dataset_v0`` npz stored pre-step
    ``obs.collided`` (always false on the first frame), so ``collisions`` alone
    under-counts on that corpus.
    """
    reasons: List[str] = []
    short = 0 < report["steps"] <= SPAWN_COLLISION_MAX_STEPS
    if short and report["collisions"] > 0:
        reasons.append(
            f"instant crash: collision within {report['steps']} step(s) "
            "— suspected bad spawn / start pose"
        )
    elif short and report["reward_sum"] <= -5.0:
        reasons.append(
            f"instant crash: {report['steps']} step(s) with reward_sum="
            f"{report['reward_sum']:.2f} — suspected bad spawn / start pose"
        )
    return reasons


def write_manifest(out_dir: Path, entries: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write the dataset index (``manifest.json``) listing every episode + meta."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    payload = {"meta": meta or {}, "episodes": entries}
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_quality_summary(out_dir: Path, per_episode: List[Dict[str, Any]]) -> Path:
    """Aggregate per-episode reports into ``QUALITY_SUMMARY.json`` (tracked in git)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "QUALITY_SUMMARY.json"

    def _agg(key: str) -> Dict[str, float]:
        vals = [float(r[key]) for r in per_episode]
        return {"min": min(vals), "max": max(vals), "mean": float(np.mean(vals))}

    quarantined = int(sum(1 for r in per_episode if quarantine_reasons(r)))
    summary = {
        "episodes": len(per_episode),
        "quarantined": quarantined,           # instant-crash / bad-spawn episodes
        "usable": len(per_episode) - quarantined,
        "total_steps": int(sum(r["steps"] for r in per_episode)),
        "total_collisions": int(sum(r["collisions"] for r in per_episode)),
        "rgb_frame_variation": _agg("rgb_frame_variation"),
        "path_length_m": _agg("path_length_m"),
        "reward_sum": _agg("reward_sum"),
        "any_depth": any(r["has_depth"] for r in per_episode),
        "any_imu": any(r.get("has_imu", False) for r in per_episode),
        "all_imu": all(r.get("has_imu", False) for r in per_episode),
    } if per_episode else {"episodes": 0}
    path.write_text(json.dumps(summary, indent=2))
    return path
