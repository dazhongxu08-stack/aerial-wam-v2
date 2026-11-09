import torch

from experiments.aerial.rl.depth_policy import DepthPolicy


def test_depth_policy_reads_a_map_and_emits_a_continuous_stick():
    policy = DepthPolicy()
    depth = torch.rand(2, 1, 64, 64) * 20.0
    goal = torch.tensor(
        [[20.0, 4.0, 0.0, 21.0], [8.0, -6.0, 0.0, 10.0]],
        dtype=torch.float32,
    )
    action = policy(depth, goal)
    assert action.shape == (2, 4)
    assert torch.all(action[:, 0] >= 0.0) and torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] == 0.0) and torch.all(action[:, 2] == 0.0)
