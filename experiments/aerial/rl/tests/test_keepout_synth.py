"""Semantic keep-out inside the consequence model: labels, init, and that it learns to read the mask."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from experiments.aerial.rl.consequence_model import (
    ConsequenceModel,
    body_xy,
    consequence_loss,
    dead_reckon_xy,
    depth_input,
    load_into_semantic,
    mirror_batch,
    save_checkpoint,
    survival,
)
from experiments.aerial.rl.keepout_synth import KeepoutSynthConfig, keepout_side_check, synth_keepout

H = 8


def _batch(B: int, lateral: float = 0.0, seed: int = 0) -> dict:
    """Straight 0.5 m/step flights over a far, flat scene."""
    g = torch.Generator().manual_seed(seed)
    fut = torch.zeros(B, H, 2)
    fut[..., 0] = 0.5 * torch.arange(1, H + 1)
    fut[..., 1] = lateral
    depth = torch.full((B, 56, 56), float(depth_input(np.array([[30.0]]))[0, 0]))
    depth = depth + 0.01 * torch.rand(B, 56, 56, generator=g)
    acts = torch.zeros(B, H, 4)
    acts[..., 0] = 0.5
    return {
        "depth": depth, "goal": torch.tensor([[0.0, 1.0, 0.5, 0.0]]).repeat(B, 1),
        "prev_action": torch.zeros(B, 4), "actions": acts, "fut_xy": fut,
        "hit": torch.zeros(B, H), "alive": torch.ones(B, H), "valid": torch.ones(B, H),
        "log_cones": torch.zeros(B, H, 3), "dgoal": torch.zeros(B, H),
    }


def test_body_frame_and_dead_reckoning_agree() -> None:
    # NED yaw +90 deg faces east; a point further east is straight ahead, one south is to the right.
    p = body_xy(np.array([[0.0, 2.0], [-1.0, 0.0]]), np.zeros(2), math.pi / 2)
    np.testing.assert_allclose(p, [[2.0, 0.0], [0.0, 1.0]], atol=1e-9)
    # Yaw right (+dyaw) then forward ends up on the right (+y).
    xy = dead_reckon_xy(np.array([[0, 0, 0, math.pi / 2], [1.0, 0, 0, 0]]))
    np.testing.assert_allclose(xy[-1], [0.0, 1.0], atol=1e-6)


def test_object_on_path_is_entered_and_off_path_is_not() -> None:
    cfg = KeepoutSynthConfig(p_object=1.0, p_offpath=0.0, lateral_m=0.0, keep_radius_m=(0.5, 0.5),
                             obj_radius_m=(0.2, 0.2))
    out = synth_keepout(_batch(64), torch.Generator().manual_seed(1), cfg)
    obj = out["ko_obj"] > 0
    assert obj.float().mean() > 0.5
    assert torch.all(out["ko_hit"][obj].sum(-1) == 1)
    assert torch.all(out["sem"][obj].sum((1, 2)) >= 3)
    # Pasted pixels read nearer than the far scene and are centred (object straight ahead).
    cols = out["sem"][obj].sum(1)
    centre = (cols * torch.arange(56)).sum(-1) / cols.sum(-1)
    assert torch.all((centre - 27.5).abs() < 4)
    assert torch.all(out["depth"][obj][out["sem"][obj] > 0] < _batch(1)["depth"].min())
    # Steps after the first entry are not alive for keep-out.
    k = out["ko_hit"][obj].argmax(-1)
    for i, kk in enumerate(k.tolist()):
        assert out["ko_alive"][obj][i, kk + 1:].sum() == 0


def test_entered_iff_path_comes_within_reach() -> None:
    out = synth_keepout(_batch(256, lateral=0.3), torch.Generator().manual_seed(4), KeepoutSynthConfig(p_object=1.0))
    obj = out["ko_obj"] > 0
    d = (out["fut_xy"] - out["ko_center"][:, None]).norm(dim=-1).amin(-1)
    entered = out["ko_hit"].sum(-1) > 0
    assert torch.equal(entered[obj], d[obj] < out["ko_reach"][obj])
    assert entered[obj].any() and (~entered[obj]).any()
    assert not entered[~obj].any() and out["sem"][~obj].sum() == 0


def test_object_to_the_right_lands_in_right_columns_and_mirrors() -> None:
    cfg = KeepoutSynthConfig(p_object=1.0, p_offpath=0.0, lateral_m=0.0, keep_radius_m=(0.4, 0.4))
    out = synth_keepout(_batch(32, lateral=1.2), torch.Generator().manual_seed(2), cfg)
    obj = out["ko_obj"] > 0
    cols = out["sem"][obj].sum(1)
    assert torch.all((cols[:, 28:].sum(-1)) > cols[:, :28].sum(-1))
    m = mirror_batch(_batch(32, lateral=1.2))
    out_m = synth_keepout(m, torch.Generator().manual_seed(2), cfg)
    cols_m = out_m["sem"][out_m["ko_obj"] > 0].sum(1)
    assert torch.all(cols_m[:, :28].sum(-1) > cols_m[:, 28:].sum(-1))


def test_loss_stops_physics_at_keepout_entry() -> None:
    model = ConsequenceModel(horizon=H, hidden=32, sem_channels=1)
    b = synth_keepout(_batch(16), torch.Generator().manual_seed(3),
                      KeepoutSynthConfig(p_object=1.0, p_offpath=0.0, lateral_m=0.0))
    out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, b["sem"], b["keep_radius"])
    loss, parts = consequence_loss(out, b)
    assert torch.isfinite(loss) and "keepout" in parts
    loss.backward()


def test_semantic_model_from_depth_only_checkpoint_keeps_old_outputs(tmp_path) -> None:
    torch.manual_seed(0)
    old = ConsequenceModel(horizon=H, hidden=32).eval()
    p = tmp_path / "old.pt"
    save_checkpoint(p, old)
    new = ConsequenceModel(horizon=H, hidden=32, sem_channels=1).eval()
    load_into_semantic(new, p)
    b = _batch(4)
    o0 = old(b["depth"], b["goal"], b["prev_action"], b["actions"])
    sem = torch.zeros(4, 56, 56)
    sem[:, 30:40, 20:30] = 1.0
    o1 = new(b["depth"], b["goal"], b["prev_action"], b["actions"], None, sem, torch.full((4,), 0.7))
    for k in ("hazard_logit", "log_cones", "dgoal"):
        torch.testing.assert_close(o0[k], o1[k])
    assert torch.all(torch.sigmoid(o1["keepout_logit"]) < 0.05)


def _p_keepout(o: dict) -> torch.Tensor:
    never_hit = torch.full_like(o["keepout_logit"], -30.0)
    return 1.0 - survival(never_hit, o["keepout_logit"])[:, -1]


@pytest.mark.skipif(os.environ.get("AERIAL_SLOW_TESTS") != "1", reason="~3 min on CPU; set AERIAL_SLOW_TESTS=1")
def test_model_learns_to_read_the_mask() -> None:
    """After a fit, keep-out risk follows the mask: on-path objects high, off-path low, no mask low.

    From scratch the mask signal is faint through the encoder; the keep-out
    loss sits at the base rate for ~1200 steps before it locks on.
    """
    torch.manual_seed(0)
    model = ConsequenceModel(horizon=H, hidden=64, sem_channels=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    gen = torch.Generator().manual_seed(0)
    cfg = KeepoutSynthConfig(p_object=0.8, p_offpath=0.3, lateral_m=2.0, keep_radius_m=(0.4, 0.8))
    for step in range(1, 2001):
        b = synth_keepout(_batch(64, seed=step), gen, cfg)
        out = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, b["sem"], b["keep_radius"])
        loss, _ = consequence_loss(out, b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    test_gen = torch.Generator().manual_seed(99)
    scores, labels = [], []
    with torch.no_grad():
        for i in range(8):
            b = synth_keepout(_batch(64, seed=1000 + i), test_gen, cfg)
            o = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None, b["sem"], b["keep_radius"])
            p = _p_keepout(o)
            obj = b["ko_obj"] > 0
            scores.append(p[obj])
            labels.append(b["ko_hit"][obj].sum(-1))
            # Same pasted depth, mask removed: the keep-out risk must drop.
            o2 = model(b["depth"], b["goal"], b["prev_action"], b["actions"], None,
                       torch.zeros_like(b["sem"]), b["keep_radius"])
            p2 = _p_keepout(o2)
            entered = obj & (b["ko_hit"].sum(-1) > 0)
            assert float(p2[entered].mean()) < 0.2 * float(p[entered].mean())
    s, y = torch.cat(scores).numpy(), torch.cat(labels).numpy()
    pos, neg = s[y > 0], s[y == 0]
    auc = (pos[:, None] > neg[None, :]).mean()
    assert auc > 0.9, auc


def test_side_check_agrees_when_obstacle_is_where_the_hit_happens() -> None:
    B = 40
    fut = torch.zeros(B, H, 2)
    fut[:, :, 0] = 1.0
    fut[:, :, 1] = torch.where(torch.arange(B) % 2 == 0, 0.8, -0.8)[:, None]
    depth = torch.full((B, 56, 56), 0.9)
    right = (torch.arange(B) % 2 == 0)
    depth[right, :, 48:53] = 0.1  # (1, +0.8) m is ~39 deg right -> column ~50
    depth[~right, :, 3:8] = 0.1
    hit = torch.zeros(B, H)
    hit[:, 1] = 1.0
    res = keepout_side_check(depth, fut, hit)
    assert res["n"] == B and res["agree"] == 1.0
