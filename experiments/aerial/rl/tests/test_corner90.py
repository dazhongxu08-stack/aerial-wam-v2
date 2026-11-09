"""Frozen-encoder corner heads: holdout split and a yaw the inverse head can read."""
import math

import torch

from experiments.aerial.rl.corner90 import (
    CHUNK,
    Corner90Heads,
    chunk_policy_loss,
    corner90_losses,
    demo_routes,
)


def test_holdout_corners_stay_out_of_the_demos():
    trained = demo_routes()
    assert 2 not in trained and 6 not in trained and 10 not in trained
    assert 0 in trained and 4 in trained


def test_inverse_loss_sees_yaw():
    torch.manual_seed(0)
    dim = 8
    heads = Corner90Heads(dim)
    h = torch.randn(16, dim)
    action = torch.zeros(16, 4)
    action[:, 3] = torch.tensor([0.3, -0.3] * 8)
    h_next = h + action[:, 3:4]
    goal = torch.zeros(16, 4)
    goal[:, 0] = 10.0
    log_fwd = torch.zeros(16)
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        opt.zero_grad()
        losses = corner90_losses(heads, h, h_next, action, goal, log_fwd)
        if first is None:
            first = float(losses["inv"].detach())
        losses["loss"].backward()
        opt.step()
    with torch.no_grad():
        pred = heads.predict_action(h, h_next)
    assert float(losses["inv"].detach()) < first
    assert torch.sign(pred[:, 3]).tolist() == torch.sign(action[:, 3]).tolist()


def _fly_latch(start, yaw0_deg, goal, n=250):
    from experiments.aerial.rl.corner90 import BearingLatch

    latch = BearingLatch()
    x, y = start
    yaw = math.radians(yaw0_deg)
    for _ in range(n):
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = goal[0] - x, goal[1] - y
        if math.hypot(wx, wy) < 3.0:
            return math.hypot(wx, wy), math.degrees(yaw)
        act = latch.act_bearing(math.atan2(-s * wx + c * wy, c * wx + s * wy))
        x += c * float(act[0])
        y += s * float(act[0])
        yaw += float(act[3])
    return math.hypot(goal[0] - x, goal[1] - y), math.degrees(yaw)


def test_latch_holds_the_street_until_the_bearing_opens():
    from experiments.aerial.rl.corner90 import BearingLatch

    latch = BearingLatch()
    acts = [latch.act_bearing(math.radians(d)) for d in (35, 60, 75, 82, 40, -8)]
    assert all(a[3] == 0.0 and a[0] == 1.0 for a in acts[:3])
    assert acts[3][3] > 0.0 and acts[3][0] < 0.3
    assert acts[4][3] > 0.0
    assert acts[5][3] < 0.0
    d2, _ = _fly_latch((-784.0, -32.0), 90.0, (-824.0, 48.0))
    d6, _ = _fly_latch((-1040.0, 216.0), 0.0, (-960.0, 272.0))
    assert d2 < 3.0 and d6 < 3.0
    torch.manual_seed(0)
    from experiments.aerial.rl.corner90 import BearingTurnHead

    head = BearingTurnHead()
    opt = torch.optim.Adam(head.parameters(), lr=5e-2)
    bearing = torch.tensor([0.3, 0.6, 0.9, 1.5, -1.5])
    goal = torch.stack(
        [
            torch.cos(bearing),
            torch.sin(bearing),
            torch.zeros_like(bearing),
            torch.full_like(bearing, 50.0),
        ],
        dim=-1,
    )
    target = torch.zeros(5)
    target[3] = 0.2
    target[4] = -0.2
    for _ in range(200):
        opt.zero_grad()
        (head(goal)[:, 3] - target).abs().mean().backward()
        opt.step()
    with torch.no_grad():
        pred = head(goal)[:, 3]
    assert float(pred[:3].abs().max()) < 0.05
    assert float(pred[3]) > 0.1 and float(pred[4]) < -0.1
    assert float(head.bias.detach()) > 1.0
    torch.manual_seed(2)
    dim = 8
    heads = Corner90Heads(dim)
    h = torch.randn(16, dim)
    chunk = torch.zeros(16, CHUNK, 4)
    chunk[:, :, 0] = 1.0
    goal = torch.zeros(16, 4)
    goal[:, 1] = 12.0
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    for _ in range(80):
        opt.zero_grad()
        chunk_policy_loss(heads, h, chunk, goal).backward()
        opt.step()
    with torch.no_grad():
        pred = heads.chunk(h, goal)
    assert float(pred[:, :, 3].abs().mean()) < 0.05
    assert float(pred[:, 0, 0].mean()) > 0.5
    torch.manual_seed(1)
    dim = 8
    heads = Corner90Heads(dim)
    h = torch.randn(12, dim)
    chunk = torch.zeros(12, CHUNK, 4)
    chunk[:, :, 3] = 0.15
    goal = torch.zeros(12, 4)
    goal[:, 1] = 8.0
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        chunk_policy_loss(heads, h, chunk, goal).backward()
        opt.step()
    with torch.no_grad():
        pred = heads.chunk(h, goal)
    assert float(pred[:, :, 3].sum(dim=-1).mean()) > 0.5
