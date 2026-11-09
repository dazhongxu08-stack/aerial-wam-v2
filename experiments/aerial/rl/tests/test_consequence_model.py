from __future__ import annotations

import math

import numpy as np
import torch

from experiments.aerial.rl.consequence_model import (
    ConsequenceModel,
    branch_windows,
    cones3_from_depth56,
    consequence_loss,
    episode_windows,
    goal_features,
    group_ranking_loss,
    load_checkpoint,
    mirror_batch,
    score_candidates,
)


def _episode(n_states: int, collide_at: int | None) -> dict:
    state = np.zeros((n_states, 7), np.float32)
    state[:, 0] = np.arange(n_states, dtype=np.float32)
    coll = np.zeros(n_states, bool)
    if collide_at is not None:
        coll[collide_at] = True
    cones = np.full((n_states, 5), 20.0, np.float32)
    return {
        "state": state,
        "action": np.tile(np.array([1.0, 0, 0, 0], np.float32), (n_states, 1)),
        "collided": coll,
        "cones": cones,
        "macro": np.zeros(n_states, np.int16),
        "depth56": np.full((n_states, 56, 56), 20.0, np.float16),
        "da3_depth56": np.full((n_states, 56, 56), 30.0, np.float16),
        "meta": {"goal": [100.0, 0.0, 0.0]},
    }


def test_windows_stop_at_collision_and_episode_end():
    w = episode_windows(_episode(6, collide_at=4), horizon=4)
    assert w["depth"].shape == (5, 56, 56)
    # Start t=1: steps move 1->2, 2->3, 3->4 (collides) -> alive 3, hit on the third.
    np.testing.assert_array_equal(w["alive"][1], [1, 1, 1, 0])
    np.testing.assert_array_equal(w["hit"][1], [0, 0, 1, 0])
    # Start t=4: only one transition left, no collision on it.
    np.testing.assert_array_equal(w["alive"][4], [1, 0, 0, 0])
    assert w["hit"][4].sum() == 0
    # Moving +x toward the goal at x=100 shrinks the distance by 1 m per step.
    np.testing.assert_allclose(w["dgoal"][0][:3], [-1, -2, -3], atol=1e-5)


def test_goal_features_body_frame():
    g = goal_features(np.array([0, 0, 20, 0, 0, 0, math.pi / 2]), np.array([0.0, 10.0, 20.0]))
    assert abs(g[0]) < 1e-6 and abs(g[1] - 1.0) < 1e-6


def test_mirror_swaps_sides_and_signs():
    b = {
        "depth": torch.arange(56 * 56, dtype=torch.float32).reshape(1, 56, 56),
        "goal": torch.tensor([[0.5, 0.8, 0.1, 0.0]]),
        "prev_action": torch.tensor([[0.5, 0.1, 0.0, 0.2]]),
        "actions": torch.tensor([[[0.5, 0.1, 0.2, 0.3]]]),
        "log_cones": torch.tensor([[[1.0, 2.0, 3.0]]]),
    }
    m = mirror_batch(b)
    assert torch.equal(m["depth"][0, :, 0], b["depth"][0, :, -1])
    np.testing.assert_allclose(m["actions"][0, 0].numpy(), [0.5, -0.1, 0.2, -0.3], atol=1e-6)
    assert m["goal"][0, 0].item() == -0.5
    assert m["log_cones"][0, 0].tolist() == [1.0, 3.0, 2.0]


def test_side_cones_do_not_see_a_wall_dead_ahead():
    d = np.full((56, 56), 40.0, np.float32)
    d[20:36, 20:36] = 3.0
    fwd, left, right = cones3_from_depth56(d)
    assert fwd == 3.0 and left == 40.0 and right == 40.0


def _probe(M=3, H=4):
    dist = np.full((M, H + 1), np.nan, np.float32)
    dist[:, 0] = 50.0
    dist[0, 1:] = [49, 48, 47, 46]
    dist[1, 1:3] = [49, 48]
    dist[2, 1:] = [50, 50, 50, 50]
    return {
        "candidates": np.zeros((M, H, 4), np.float32),
        "coll_step": np.array([-1, 1, -1]),
        "dist": dist,
        "step_depth56": np.full((M, H + 1, 56, 56), 10.0, np.float16),
        "hist_action": np.zeros((3, 4), np.float32),
        "state": np.array([0, 0, 24, 0, 0, 0, 0], np.float32),
        "goal": np.array([50, 0, 24], np.float32),
    }


def test_branch_windows_mark_the_collision_step():
    w = branch_windows(_probe(), np.full((56, 56), 9.0, np.float16), horizon=4)
    np.testing.assert_array_equal(w["alive"][1], [1, 1, 0, 0])
    np.testing.assert_array_equal(w["hit"][1], [0, 1, 0, 0])
    np.testing.assert_array_equal(w["alive"][0], [1, 1, 1, 1])
    np.testing.assert_allclose(w["dgoal"][0], [-1, -2, -3, -4])


def test_group_ranking_loss_prefers_survivor():
    w = branch_windows(_probe(), np.full((56, 56), 9.0, np.float16), horizon=4)
    b = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in w.items()}
    group = torch.zeros(3, dtype=torch.long)
    good = {"hazard_logit": torch.tensor([[-5.0] * 4, [3.0] * 4, [-5.0] * 4])}
    bad = {"hazard_logit": torch.tensor([[3.0] * 4, [-5.0] * 4, [3.0] * 4])}
    assert float(group_ranking_loss(good, b, group)) < float(group_ranking_loss(bad, b, group))


def test_loads_v0_checkpoint_without_dropout(tmp_path):
    m = ConsequenceModel(horizon=5, hidden=32)
    sd = {k.replace("enc.10.", "enc.9.", 1) if k.startswith("enc.10.") else k: v for k, v in m.state_dict().items()}
    torch.save({"model": sd, "horizon": 5, "hidden": 32}, str(tmp_path / "v0.pt"))
    loaded = load_checkpoint(tmp_path / "v0.pt")
    assert torch.equal(loaded.enc[10].weight, m.enc[10].weight)


def test_model_shapes_and_loss_backprop():
    w = episode_windows(_episode(12, collide_at=9), horizon=5)
    b = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in w.items() if k != "macro"}
    model = ConsequenceModel(horizon=5, hidden=32)
    out = model(b["depth"], b["goal"], b["prev_action"], b["actions"])
    assert out["hazard_logit"].shape == (11, 5)
    assert out["log_cones"].shape == (11, 5, 3)
    loss, _ = consequence_loss(out, b)
    loss.backward()
    assert score_candidates(out).shape == (11,)


def test_select_candidate_prefers_progress_among_survivors():
    from experiments.aerial.rl.consequence_model import select_candidate

    surv = np.array([0.95, 0.75, 0.4, 0.99])
    prog = np.array([0.2, 3.0, 9.0, 1.0])
    allowed = np.array([True, True, True, False])
    assert select_candidate(surv, prog, allowed, 0.7) == 1
    assert select_candidate(surv, prog, allowed, 0.96) == 0


def test_altitude_band_blocks_receding_horizon_drift():
    from experiments.aerial.rl.consequence_model import altitude_band_allowed

    c = np.zeros((3, 10, 4), np.float32)
    c[0, :, 2] = 0.3
    c[1, :, 2] = -0.3
    ok = altitude_band_allowed(c, 25.0, 23.0, 25.0, 1)
    assert ok.tolist() == [False, True, True]
    ok = altitude_band_allowed(c, 30.0, 23.0, 25.0, 1)
    assert ok.tolist() == [False, True, True]
    ok = altitude_band_allowed(c, 22.0, 23.0, 25.0, 1)
    assert ok.tolist() == [True, False, True]
    ok = altitude_band_allowed(c, 30.0, 23.0, 25.0, 1, return_per_step=0.1)
    assert ok.tolist() == [False, True, False]
    ok = altitude_band_allowed(c, 22.0, 23.0, 25.0, 1, return_per_step=0.1)
    assert ok.tolist() == [True, False, False]
    ok = altitude_band_allowed(c, 24.0, 23.0, 25.0, 1, return_per_step=0.1)
    assert ok.tolist() == [True, True, True]


def test_visible_motion_blocks_sideways_translation():
    from experiments.aerial.rl.consequence_model import visible_motion_allowed

    c = np.zeros((5, 10, 4), np.float32)
    c[0, :, 0] = 0.8                      # forward
    c[1, :, 0], c[1, :, 1] = 0.2, 0.4     # strafe, ~63 deg off the nose
    c[2, :, 0], c[2, :, 1] = 0.8, 0.3     # forward with drift, ~21 deg
    c[3, :, 3] = 0.3                      # spin in place
    c[4, 0, 0], c[4, 1:, 1] = 0.5, 0.4    # sideways only after the executed step
    assert visible_motion_allowed(c, 1, 35.0).tolist() == [True, False, True, True, True]
    assert visible_motion_allowed(c, 2, 35.0).tolist() == [True, False, True, True, False]


def test_history_inputs_clamps_to_flight_start_and_pads_actions():
    from experiments.aerial.rl.consequence_model import depth_input, history_inputs

    seq = [np.full((56, 56), float(i + 1), np.float32) for i in range(3)]
    acts = [np.full(4, float(i + 1), np.float32) for i in range(2)]
    d, a = history_inputs(seq, acts, (0, 2, 5), 4)
    assert d.shape == (3, 56, 56) and a.shape == (4, 4)
    np.testing.assert_allclose(d[0], depth_input(seq[2]))
    np.testing.assert_allclose(d[1], depth_input(seq[0]))
    np.testing.assert_allclose(d[2], depth_input(seq[0]))
    assert a[:2].sum() == 0 and a[2, 0] == 1 and a[3, 0] == 2


def test_history_model_roundtrip(tmp_path):
    from experiments.aerial.rl.consequence_model import (
        ConsequenceModel, load_checkpoint, mirror_batch, save_checkpoint,
    )

    m = ConsequenceModel(horizon=10, hidden=32, frame_offsets=(0, 2, 5), hist_actions=5).eval()
    B = 3
    d = torch.rand(B, 3, 56, 56)
    g, pa, ha = torch.zeros(B, 4), torch.zeros(B, 4), torch.rand(B, 5, 4)
    a = torch.zeros(B, 10, 4)
    o = m(d, g, pa, a, ha)
    assert o["hazard_logit"].shape == (B, 10)
    save_checkpoint(tmp_path / "c.pt", m)
    m2 = load_checkpoint(tmp_path / "c.pt")
    assert m2.frame_offsets == (0, 2, 5) and m2.hist_actions == 5
    torch.testing.assert_close(m2(d, g, pa, a, ha)["hazard_logit"], o["hazard_logit"])
    mb = mirror_batch({"depth": d, "actions": a, "prev_action": pa, "hist_act": ha, "goal": g,
                       "log_cones": torch.zeros(B, 10, 3)})
    assert torch.equal(mb["hist_act"][..., 1], -ha[..., 1])
