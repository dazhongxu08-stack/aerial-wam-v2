"""View memory redraws an earlier depth frame. Default survival checkpoints stay unchanged."""
from __future__ import annotations

import torch

from experiments.aerial.rl.consequence_model import (
    ConsequenceModel,
    load_checkpoint,
    load_into_view_memory,
    save_checkpoint,
)
from experiments.aerial.rl.view_memory import ViewMemory


def _batch():
    depth = torch.rand(4, 56, 56)
    goal = torch.randn(4, 4)
    prev = torch.zeros(4, 4)
    actions = torch.zeros(4, 5, 4)
    return depth, goal, prev, actions


def test_default_model_has_no_view_memory(tmp_path):
    base = ConsequenceModel(horizon=5, hidden=32)
    assert base.view_memory_dim == 0
    depth, goal, prev, actions = _batch()
    base(depth, goal, prev, actions)
    save_checkpoint(tmp_path / "base.pt", base)
    loaded = load_checkpoint(tmp_path / "base.pt")
    assert loaded.view_memory_dim == 0
    assert not any(k.startswith("view_memory.") for k in loaded.state_dict())


def test_zero_memory_matches_the_frozen_head(tmp_path):
    torch.manual_seed(0)
    base = ConsequenceModel(horizon=5, hidden=32)
    save_checkpoint(tmp_path / "base.pt", base)
    torch.manual_seed(1)
    mem = ConsequenceModel(horizon=5, hidden=32, view_memory_dim=8, view_span=4)
    load_into_view_memory(mem, tmp_path / "base.pt")
    depth, goal, prev, actions = _batch()
    with torch.no_grad():
        a = base(depth, goal, prev, actions)["hazard_logit"]
        b = mem(depth, goal, prev, actions, memory=torch.zeros(4, 8))["hazard_logit"]
        c = mem(depth, goal, prev, actions, memory=torch.randn(4, 8))["hazard_logit"]
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(b, c, atol=1e-5)
    mem.ctx.weight.requires_grad_(True)
    mem(depth, goal, prev, actions, memory=torch.randn(4, 8))["hazard_logit"].sum().backward()
    assert float(mem.ctx.weight.grad[:, -8:].abs().sum()) > 0.0


def test_recent_past_keeps_the_near_center_not_the_open_start():
    import numpy as np

    from experiments.aerial.rl.view_memory import recent_past_index

    depth = np.full((12, 56, 56), 8.0, np.float32)
    depth[8] = 3.3
    depth[9] = 2.6
    depth[10] = 9.6
    idx = recent_past_index(depth)
    assert int(idx[10]) == 9
    assert int(idx[11]) == 9
    assert int(idx[1]) == 0


def test_state_redraws_the_earlier_frame_not_only_the_current_one():
    torch.manual_seed(0)
    span = 4
    seq = torch.zeros(2, span + 1, 56, 56)
    seq[:, 0, 8:24, 8:24] = 1.0
    seq[:, -1, 32:48, 32:48] = 1.0
    past_at = torch.full((seq.shape[1],), -1, dtype=torch.long)
    past_at[-1] = 0
    mem = ViewMemory(latent=32, span=span)
    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    mem.train()
    for _ in range(60):
        opt.zero_grad()
        loss, _ = mem.loss(seq, past_at)
        loss.backward()
        opt.step()
    mem.eval()
    with torch.no_grad():
        _, parts = mem.loss(seq, past_at)
        past = parts["drawn"][:, 1]
    assert float(torch.mean((past - seq[:, 0]) ** 2)) < float(torch.mean((past - seq[:, -1]) ** 2))
    assert float(torch.mean((past - seq[:, 0]) ** 2)) < 0.05
