"""Unit tests for DepthSceneExpertPolicy."""
from __future__ import annotations

import numpy as np

from experiments.aerial.rl.depth_scene_expert import DepthSceneExpertPolicy
from experiments.aerial.rl.env.obs import Observation


def _obs(pos, yaw=0.0, d_fwd=50.0, left=50.0, right=50.0):
    return Observation(
        rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        state=np.array([pos[0], pos[1], pos[2], 0, 0, 0, yaw], dtype=np.float32),
        t=0.0,
        info={
            "depth_min_pred": float(d_fwd),
            "depth_cones_pred": {
                "forward": float(d_fwd),
                "left": float(left),
                "right": float(right),
                "up": 20.0,
                "down": 20.0,
            },
        },
    )


def test_depth_scene_expert_heads_toward_goal_when_clear():
    goal = np.array([40.0, 0.0, 10.0])
    pol = DepthSceneExpertPolicy(lambda: goal, r_m=25.0, step_m=1.0)
    pol.reset()
    a = pol.act(_obs([0.0, 0.0, 10.0], yaw=0.0, d_fwd=80.0))
    assert a[0] > 0.5  # forward
    assert abs(a[3]) < 0.05  # little yaw when goal ahead


def test_depth_scene_expert_peels_when_forward_blocked():
    goal = np.array([40.0, 0.0, 10.0])
    pol = DepthSceneExpertPolicy(lambda: goal, r_m=25.0, step_m=1.0, d_danger=3.0, d_clear=40.0)
    pol.reset()
    # Forward blocked, left open → expect off-axis / leftish command
    a = pol.act(_obs([0.0, 0.0, 10.0], yaw=0.0, d_fwd=2.0, left=40.0, right=5.0))
    assert pol.last_info.get("chosen_idx", 0) != 0 or a[1] != 0.0 or abs(a[3]) > 0.01
    assert "target_world" in pol.last_info


def test_depth_scene_expert_writes_subgoal_into_obs_info():
    goal = np.array([30.0, 10.0, 5.0])
    pol = DepthSceneExpertPolicy(lambda: goal, r_m=20.0)
    pol.reset()
    obs = _obs([0.0, 0.0, 5.0])
    pol.act(obs)
    assert "goal" in obs.info
    assert len(obs.info["goal"]) == 3
