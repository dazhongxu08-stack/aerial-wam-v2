"""Phase 2b torch-RSSM WM tests — SKIP when torch is absent (dev host is GPU-less).

These run on the H100 (torch 2.7.1+cu128) on tiny CPU tensors. Two jobs:
  1. Pin the torch primitives to the :mod:`dreamer_recipe` numpy reference
     (symlog / two-hot / categorical-KL) — the §1.5 single-source-of-truth check.
  2. Smoke the model end-to-end (build → training_loss → backward → update →
     encode/step packing) so the data plumbing and shapes are de-risked.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402  (skip whole module off-H100)

from experiments.aerial.rl import dreamer_recipe as ref  # noqa: E402
from experiments.aerial.rl.buffer import Transition  # noqa: E402
from experiments.aerial.rl.depth_geometry import cone_clearances, cone_mins_torch  # noqa: E402
from experiments.aerial.rl.dynamics_torch import (  # noqa: E402
    TorchRSSMDynamics,
    _categorical_kl,
    _cone_side_loss,
    _filter_compatible_state_dict,
    _center_median_match,
    _forward_center_depth_loss,
    _symexp,
    _symlog,
    _twohot_decode,
    _twohot_targets,
)
from experiments.aerial.rl.env.obs import Observation  # noqa: E402


# image_size must be a multiple of 16 (encoder does 4 stride-2 downsamples; the
# decoder reconstructs via image_size//16). 16 is the smallest fast tile.
def _obs(pos=(0.0, 0.0, 0.0), collided=False, size=16):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Observation(rgb=rgb, state=state, collided=collided)


def _windows(batch=2, length=3, size=16):
    ws = []
    for b in range(batch):
        ep = []
        for t in range(length):
            ep.append(Transition(
                obs=_obs(pos=(float(t), float(b), 0.0), collided=(t == length - 1)),
                action=np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                reward=float(t) - 0.5,
                done=(t == length - 1),
            ))
        ws.append(ep)
    return ws


def _tiny_model(size=16):
    # Small dims + smallest legal image (multiple of 16, >=16) for fast CPU.
    return TorchRSSMDynamics(
        image_size=size, recurrent_dim=16, stoch_dim=4, stoch_classes=4,
        hidden_dim=16, num_bins=41, device="cpu", torch_dtype=torch.float32,
    )


# -- primitives match the numpy reference (§1.5 single source of truth) ------
def test_symlog_symexp_match_reference():
    x = np.linspace(-30.0, 30.0, 17)
    xt = torch.tensor(x, dtype=torch.float64)
    np.testing.assert_allclose(_symlog(xt).numpy(), ref.symlog(x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_symexp(xt).numpy(), ref.symexp(x), rtol=1e-10, atol=1e-10)


def test_twohot_targets_match_reference():
    bins_np = ref.make_bins(41, -10.0, 10.0)
    bins_t = torch.tensor(bins_np, dtype=torch.float64)
    x = np.array([-12.0, -3.3, 0.0, 1.7, 9.9], dtype=np.float64)
    got = _twohot_targets(torch.tensor(x, dtype=torch.float64), bins_t).numpy()
    exp = ref.two_hot_encode(x, bins_np)
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)
    # decode round-trips the expectation
    np.testing.assert_allclose(
        _twohot_decode(torch.tensor(exp), bins_t).numpy(),
        ref.two_hot_decode(exp, bins_np), rtol=1e-10, atol=1e-10,
    )


def test_categorical_kl_matches_reference():
    rng = np.random.default_rng(1)
    post = rng.dirichlet(np.ones(5), size=6)
    prior = rng.dirichlet(np.ones(5), size=6)
    got = _categorical_kl(torch.tensor(post), torch.tensor(prior)).numpy()
    np.testing.assert_allclose(got, ref.categorical_kl(post, prior), rtol=1e-8, atol=1e-8)


# -- model end-to-end --------------------------------------------------------
def test_training_loss_finite_and_backprops():
    m = _tiny_model()
    from experiments.aerial.rl.wm_data import windows_to_arrays
    sample = m._arrays_to_tensors(windows_to_arrays(_windows()))
    loss, ld = m.training_loss(sample)
    assert torch.isfinite(loss)
    for k in ("loss", "loss_pred", "loss_dyn", "loss_rep", "recon_err", "post_entropy"):
        assert k in ld and np.isfinite(ld[k])
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_update_returns_updated_shape_no_skipped():
    m = _tiny_model()
    out = m.update(_windows())
    assert "skipped" not in out                    # corrector must report "updated"
    for k in ("loss", "loss_pred", "loss_dyn", "loss_rep", "recon_err", "grad_norm"):
        assert k in out and np.isfinite(out[k])


def test_free_bits_floor_holds_on_kl_terms():
    m = _tiny_model()
    from experiments.aerial.rl.wm_data import windows_to_arrays
    sample = m._arrays_to_tensors(windows_to_arrays(_windows()))
    _, ld = m.training_loss(sample)
    # loss_dyn/rep = beta * clamp_min(KL, free_nats); never below beta*free_nats.
    assert ld["loss_dyn"] >= m.beta_dyn * m.free_nats - 1e-6
    assert ld["loss_rep"] >= m.beta_rep * m.free_nats - 1e-6


def test_encode_step_latent_packing():
    m = _tiny_model()
    z = m.encode(_obs())
    assert z.shape == (m.latent_dim,)
    assert m.latent_dim == m.recurrent_dim + m.stoch_dim * m.stoch_classes
    out = m.step(z, np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32))
    assert out.z_next.shape == (m.latent_dim,)
    assert 0.0 <= out.p_coll <= 1.0
    assert isinstance(out.done, bool)


def test_reward_head_is_action_goal_and_vel_conditioned():
    """V1-②: reward_head input is [proj(feature); a; reward_aux]; train/step share shape."""
    m = _tiny_model()
    assert m.reward_in_dim == m.reward_feat_proj_dim + m.action_dim + m.reward_aux_dim
    first = next(m.reward_head.parameters())
    assert first.shape[1] == m.reward_in_dim
    z = m.encode(_obs())
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    g0 = np.zeros(4, dtype=np.float32)
    g1 = np.array([10.0, 0.0, 0.0, 10.0], dtype=np.float32)
    v = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    p0 = m.step(z, a, goal_rel=g0, body_vel=v).progress
    p1 = m.step(z, a, goal_rel=g1, body_vel=v).progress
    assert np.isfinite(p0) and np.isfinite(p1)
    assert p0 != p1


def test_from_config_reads_world_model_block():
    cfg = {
        "recurrent_dim": 16, "stoch_dim": 4, "stoch_classes": 4, "num_bins": 41,
        "bin_lo": -10.0, "bin_hi": 10.0, "free_bits": 1.0,
        "loss_scales": {"pred": 1.0, "dyn": 1.0, "rep": 0.1, "reward": 10.0},
        "lr": 1e-4, "grad_clip": 1000.0, "image_size": 16,
        "decoder": {"train_only": True}, "device": "cpu",
    }
    m = TorchRSSMDynamics.from_config(cfg)
    assert m.recurrent_dim == 16 and m.stoch_dim == 4 and m.stoch_classes == 4
    assert m.beta_rep == pytest.approx(0.1)
    assert m.beta_reward == pytest.approx(10.0)
    assert m.latent_dim == 16 + 4 * 4


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny_model()
    m.update(_windows())
    p = str(tmp_path / "wm.pt")
    m.save_checkpoint(p, step=1)
    m2 = _tiny_model()
    payload = m2.load_checkpoint(p)
    assert payload["step"] == 1
    for (n1, a), (n2, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        assert torch.allclose(a, b)


def test_load_checkpoint_skips_shape_mismatch():
    """Legacy ckpts may omit reward_feat_proj; encoder/RSSM still load."""
    m = _tiny_model()
    state = m.state_dict()
    legacy = dict(state)
    legacy["reward_head.0.weight"] = torch.randn(256, 1536)
    legacy["reward_head.0.bias"] = torch.randn(256)
    filtered, skipped = _filter_compatible_state_dict(m, legacy)
    assert any("reward_head.0.weight" in s for s in skipped)
    assert "encoder" in next(k for k in filtered if k.startswith("encoder."))
    m2 = _tiny_model()
    missing, _ = m2.load_state_dict(filtered, strict=False)
    assert not any(k.startswith("encoder.") for k in missing)


def test_reward_head_finetune_load_skipped_empty(tmp_path):
    """Saved RH-finetuned ckpt must load reward_feat_proj/head without skip."""
    m = _tiny_model()
    m.apply_freeze_backbone_train_reward_head()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    out = m.update_reward_head(_windows(batch=2, length=3), optimizer=opt)
    assert np.isfinite(out["loss_reward"])
    p = str(tmp_path / "wm_rh.pt")
    m.save_checkpoint(p, optimizer=opt, step=1)
    m2 = _tiny_model()
    payload = m2.load_checkpoint(p)
    rh_skip = [s for s in payload.get("load_skipped", []) if "reward" in s]
    assert rh_skip == []


def test_set_imagination_aux_used_in_step():
    """Cached aux reaches step when kwargs omitted."""
    m = _tiny_model()
    z = m.encode(_obs())
    a = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    g = np.array([5.0, 0.0, 0.0, 5.0], dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    m.set_imagination_aux(g, v)
    p_cached = m.step(z, a).progress
    p_explicit = m.step(z, a, goal_rel=g, body_vel=v).progress
    assert np.isfinite(p_cached) and p_cached == pytest.approx(p_explicit)


def test_torch_encode_differs_from_stub_proprio4():
    """Deploy encode must not be stub proprio4 passthrough when kind=torch."""
    from experiments.aerial.rl.dynamics import StubLatentDynamics

    obs = _obs(pos=(1.0, 2.0, 3.0, 0.0))
    stub = StubLatentDynamics(goal=None, latent_dim=8)
    stub_z = stub.encode(obs)
    torch_m = _tiny_model()
    torch_z = torch_m.encode(obs)
    assert stub_z.shape == (8,)
    assert torch_z.shape == (torch_m.latent_dim,)
    assert torch_m.latent_dim != 8
    assert not np.allclose(stub_z, torch_z[:8])


def test_collision_targets_soft_near_depth_and_auto_pos_weight():
    m = TorchRSSMDynamics(
        image_size=16, device="cpu", coll_near_depth_m=5.0, coll_pos_weight=0.0
    )
    collided = torch.zeros(2, 2)
    depth = torch.ones(2, 2, 16, 16) * 20.0
    depth[0, 0] = 2.0  # near
    collided[1, 1] = 1.0  # hard contact
    # Soft near only when action is forward.
    action = torch.zeros(2, 2, 4)
    action[0, 0, 0] = 1.0  # fwd at near cell
    action[0, 1, 1] = 1.0  # lateral at far cell
    target, pw = m._collision_train_targets(collided, depth, action=action)
    assert float(target[0, 0]) == pytest.approx(0.5)
    assert float(target[0, 1]) == pytest.approx(0.0)
    assert float(target[1, 1]) == pytest.approx(1.0)
    assert float(pw) >= 1.0
    # Same near depth but lateral action → no soft boost.
    action2 = torch.zeros(2, 2, 4)
    action2[0, 0, 1] = 1.0
    target2, _ = m._collision_train_targets(collided, depth, action=action2)
    assert float(target2[0, 0]) == pytest.approx(0.0)


def test_early_forward_target_stays_in_unit_interval_and_skips_yaw():
    """A forward action into a centre obstacle inside 18 m is a soft collision.

    A pure yaw at the same depth is not. Both sides being near does not cancel
    the centre. The target never exceeds 1.
    """
    m = TorchRSSMDynamics(
        image_size=16, device="cpu", coll_early_fwd_m=18.0, coll_pos_weight=1.0,
    )
    depth = torch.ones(1, 4, 16, 16) * 20.0
    for i, dist in enumerate((16.0, 15.0, 14.0, 13.0)):
        depth[:, i, 4:12, 4:12] = dist
    collided = torch.zeros(1, 4)
    press = torch.zeros(1, 4, 4)
    press[..., 0] = 0.7
    yaw = torch.zeros(1, 4, 4)
    yaw[..., 3] = 0.25
    t_press, _ = m._collision_train_targets(collided, depth, action=press)
    t_yaw, _ = m._collision_train_targets(collided, depth, action=yaw)
    assert float(t_press.min()) > 0.0
    assert float(t_press.max()) <= 0.55
    assert float(t_yaw.max()) == pytest.approx(0.0)
    blocked = depth.clone()
    blocked[:, :, 4:12, :4] = 5.0
    blocked[:, :, 4:12, 12:] = 5.0
    t_blocked, _ = m._collision_train_targets(collided, blocked, action=press)
    assert float(t_blocked.min()) > 0.0
    assert float(t_blocked.max()) <= 0.55


def test_center_pair_labels_one_occupied_frame_without_a_press_run():
    """Centre p10 writes the forward target. Yaw stays 0. aux_w does not scale it."""
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        coll_early_fwd_m=18.0,
        coll_near_depth_m=8.0,
        coll_fwd_depth_aux_weight=5.0,
        coll_pos_weight=1.5,
    )
    assert float(m._center_proximity_target(torch.tensor(8.0))) == pytest.approx(0.55)
    assert float(m._center_proximity_target(torch.tensor(4.0))) == pytest.approx(0.55)
    assert float(m._center_proximity_target(torch.tensor(2.0))) == pytest.approx(0.75)
    assert float(m._center_proximity_target(torch.tensor(12.0))) == pytest.approx((18.0 - 12.0) / 18.0)
    assert float(m._center_proximity_target(torch.tensor(40.0))) == pytest.approx(0.0)

    depth = torch.ones(1, 1, 16, 16) * 40.0
    depth[:, :, 4:12, 4:12] = 8.0
    yaw = torch.zeros(1, 1, 4)
    yaw[..., 3] = 0.25
    press = torch.zeros(1, 1, 4)
    press[..., 0] = 0.7
    collided = torch.zeros(1, 1)
    t_yaw, _ = m._collision_train_targets(collided, depth, action=yaw)
    t_press, _ = m._collision_train_targets(collided, depth, action=press)
    assert float(t_yaw) == pytest.approx(0.0)
    assert float(t_press) == pytest.approx(0.55)

    feat = torch.zeros(1, 1, m.latent_dim)
    loss, n_pos = m._center_pair_loss(feat, depth)
    assert n_pos == 1
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert m.coll_head[0].weight.grad is not None
    assert float(m.coll_head[0].weight.grad.abs().sum()) > 0.0


def test_early_yaw_hinge_reports_gap_on_the_early_band():
    """Sequence gap is max p(straight) - max p(yaw then forward). Margin is 0.05."""
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        coll_early_fwd_m=18.0,
        coll_early_yaw_margin=0.10,
    )
    with torch.no_grad():
        for layer in m.coll_head:
            if hasattr(layer, "weight"):
                layer.weight.zero_()
            if hasattr(layer, "bias"):
                layer.bias.zero_()
    depth = torch.ones(1, 4, 16, 16) * 20.0
    for i, dist in enumerate((16.0, 15.0, 14.0, 13.0)):
        depth[:, i, 4:12, 4:12] = dist
    feat = torch.zeros(1, 4, m.latent_dim)
    press = torch.zeros(1, 4, 4)
    press[..., 0] = 0.7
    loss, gap, n = m._early_yaw_hinge_loss(feat, depth, action=press)
    assert n == 4
    assert float(gap) == pytest.approx(0.0)
    assert float(loss.detach()) == pytest.approx(0.05)
    far = torch.ones(1, 4, 16, 16) * 30.0
    loss_far, _, n_far = m._early_yaw_hinge_loss(feat, far, action=press)
    assert n_far == 0
    assert float(loss_far.detach()) == pytest.approx(0.0)
    blocked = depth.clone()
    blocked[:, :, 4:12, :4] = 5.0
    blocked[:, :, 4:12, 12:] = 5.0
    loss_side, _, n_side = m._early_yaw_hinge_loss(feat, blocked, action=press)
    assert n_side == 0
    assert float(loss_side.detach()) == pytest.approx(0.0)


def test_deploy_cones_scale_da3_and_keep_sides_outside_the_center():
    """Road-strip forward ignores flanking facades; sides sit outside that strip."""
    from experiments.aerial.rl.depth_geometry import DEPLOY_ROAD_CENTER_FRAC, deploy_cones

    depth = np.full((100, 100), 30.0, np.float64)
    dw = max(1, int(100 * DEPLOY_ROAD_CENTER_FRAC))
    c0 = (100 - dw) // 2
    r0 = c0
    depth[r0 : r0 + dw, :c0] = 8.0
    depth[r0 : r0 + dw, c0 + dw :] = 5.0
    got = deploy_cones(depth)
    assert got["forward"] == pytest.approx(60.0)
    assert got["left"] == pytest.approx(16.0)
    assert got["right"] == pytest.approx(10.0)


def test_train_lateral_clearances_match_collision_crop():
    from experiments.aerial.rl.depth_geometry import (
        early_press_mask,
        train_lateral_clearances,
    )

    depth = np.ones((16, 16), np.float32) * 20.0
    depth[4:12, 4:12] = 12.0
    d_fwd, d_left, d_right = train_lateral_clearances(depth)
    assert d_fwd == pytest.approx(12.0)
    assert d_left == pytest.approx(20.0)
    assert d_right == pytest.approx(20.0)
    assert early_press_mask(d_fwd, d_left, d_right)
    assert not early_press_mask(30.0, 40.0, 40.0)
    m = TorchRSSMDynamics(image_size=16, device="cpu")
    batched = torch.from_numpy(depth).unsqueeze(0)
    tf, tl, tr = m._fwd_lateral_depths_torch(batched)
    assert float(tf) == pytest.approx(d_fwd)
    assert float(tl) == pytest.approx(d_left)
    assert float(tr) == pytest.approx(d_right)
    from experiments.aerial.rl.depth_geometry import frontal_obstacle_frame, stable_approach
    from experiments.aerial.rl.depth_geometry import forward_view_mass
    view = np.ones((16, 16), np.float32) * 40.0
    view[4:12, 4:12] = 12.0
    mass = forward_view_mass(view)
    assert mass["near_center"] == pytest.approx(1.0)
    assert mass["near_left"] == pytest.approx(0.0)
    assert mass["med_center"] == pytest.approx(12.0)
    speck = np.ones((16, 16), np.float32) * 40.0
    speck[8, 8] = 10.0
    assert forward_view_mass(speck)["near_center"] < 0.05
    assert frontal_obstacle_frame(12.0, 20.0, 20.0)
    assert frontal_obstacle_frame(12.0, 5.0, 30.0)
    assert not frontal_obstacle_frame(12.0, 5.0, 6.0)
    assert stable_approach([16.0, 15.2, 14.4])
    assert not stable_approach([16.0, 15.8])
    assert not stable_approach([16.0, 19.5, 14.0])


def test_coll_logits_depends_on_action():
    m = _tiny_model()
    feat = torch.zeros(1, m.latent_dim)
    a_fwd = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a_left = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    # Force 2-layer MLP to pass different action columns with distinct signs.
    with torch.no_grad():
        m.coll_head[0].weight.zero_()
        m.coll_head[0].bias.zero_()
        m.coll_head[2].weight.zero_()
        m.coll_head[2].bias.zero_()
        # Last 4 weights of layer 0 = action dims; route fwd to node 0, left to node 1.
        m.coll_head[0].weight[0, -4] = 1.0
        m.coll_head[0].weight[1, -3] = 1.0
        m.coll_head[2].weight[0, 0] = 1.0
        m.coll_head[2].weight[0, 1] = -1.0
    lf = float(m._coll_logits(feat, a_fwd).item())
    ll = float(m._coll_logits(feat, a_left).item())
    assert lf > ll


def test_coll_rank_hinge_loss_and_depth_aux():
    """P0 2026-08-28: hinge separates forward vs lateral arm; depth-aux is continuous."""
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        coll_near_depth_m=5.0,
        coll_rank_hinge_weight=2.0,
        coll_rank_hinge_margin=0.20,
        coll_fwd_depth_aux_weight=0.8,
    )
    assert m.coll_rank_hinge_weight == pytest.approx(2.0)
    assert m.coll_rank_hinge_margin == pytest.approx(0.20)
    assert m.coll_fwd_depth_aux_weight == pytest.approx(0.8)

    # 1. Depth aux continuous proximity test
    collided = torch.zeros(1, 1)
    depth = torch.ones(1, 1, 16, 16) * 2.5  # half of 5.0m
    action = torch.zeros(1, 1, 4)
    action[0, 0, 0] = 1.0  # forward action
    target, _ = m._collision_train_targets(collided, depth, action=action)
    # proximity = (5.0 - 2.5) / 5.0 = 0.5  (aux_w gates continuous mode; NOT a multiplier)
    assert float(target[0, 0]) == pytest.approx(0.5)

    # 2. Hinge loss when coll_target > 0
    feat = torch.zeros(1, 1, m.latent_dim)
    # Initialize coll_head to output 0 for all actions
    with torch.no_grad():
        for layer in m.coll_head:
            if hasattr(layer, "weight"):
                layer.weight.zero_()
            if hasattr(layer, "bias"):
                layer.bias.zero_()
    loss_hinge, gap = m._coll_rank_hinge_loss(feat, target)
    # When logits are 0, p_fwd = 0.5, p_lat = 0.5 -> gap = 0.0 -> loss = relu(0.20 - 0.0) = 0.20
    assert float(gap) == pytest.approx(0.0)
    assert float(loss_hinge.detach()) == pytest.approx(0.20)

    # 3. Hinge loss is 0 when coll_target is 0 (open space)
    zero_target = torch.zeros_like(target)
    loss_hinge_zero, _ = m._coll_rank_hinge_loss(feat, zero_target)
    assert float(loss_hinge_zero.detach()) == pytest.approx(0.0)

    # 4. Geometric condition test with depth map:
    # 4a. Forward blocked (2m), lateral open (10m) -> hinge active
    depth_geom = torch.ones(1, 1, 16, 16) * 10.0
    depth_geom[0, 0, 4:12, 4:12] = 2.0  # forward obstacle
    loss_geom_active, _ = m._coll_rank_hinge_loss(feat, target, depth=depth_geom)
    assert float(loss_geom_active.detach()) == pytest.approx(0.20)

    # 4b. Open space forward (8m > 4m cutoff) -> masked out (0.0)
    depth_open = torch.ones(1, 1, 16, 16) * 8.0
    loss_geom_open, _ = m._coll_rank_hinge_loss(feat, target, depth=depth_open)
    assert float(loss_geom_open.detach()) == pytest.approx(0.0)

    # 4c. Lateral tighter than forward (e.g. narrow corridor) -> masked out (0.0)
    depth_corridor = torch.ones(1, 1, 16, 16) * 1.0  # sides 1.0m
    depth_corridor[0, 0, 4:12, 4:12] = 2.0  # forward 2.0m
    loss_geom_corridor, _ = m._coll_rank_hinge_loss(feat, target, depth=depth_corridor)
    assert float(loss_geom_corridor.detach()) == pytest.approx(0.0)


def test_collision_soft_targets_never_exceed_one():
    """aux_w must not inflate BCE targets above 1 (R4 saturation root cause)."""
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        coll_near_depth_m=5.0,
        coll_fwd_depth_aux_weight=5.0,  # historically multiplied into target
    )
    collided = torch.zeros(1, 1)
    depth = torch.ones(1, 1, 16, 16) * 0.5  # very near
    action = torch.zeros(1, 1, 4)
    action[0, 0, 0] = 1.0
    target, _ = m._collision_train_targets(collided, depth, action=action)
    assert float(target[0, 0]) <= 1.0 + 1e-6
    assert float(target[0, 0]) == pytest.approx(0.9)  # (5-0.5)/5



def test_from_config_reads_coll_geom_keys():
    cfg = {
        "recurrent_dim": 16, "stoch_dim": 4, "stoch_classes": 4, "num_bins": 41,
        "bin_lo": -10.0, "bin_hi": 10.0, "free_bits": 1.0,
        "loss_scales": {"pred": 1.0, "dyn": 1.0, "rep": 0.1, "reward": 10.0},
        "lr": 1e-4, "grad_clip": 1000.0, "image_size": 16,
        "decoder": {"train_only": True}, "device": "cpu",
        "coll_rank_hinge_weight": 1.5,
        "coll_rank_hinge_margin": 0.25,
        "coll_hinge_fwd_max_m": 4.5,
        "coll_fwd_depth_aux_weight": 0.6,
    }
    m = TorchRSSMDynamics.from_config(cfg)
    assert m.coll_rank_hinge_weight == pytest.approx(1.5)
    assert m.coll_rank_hinge_margin == pytest.approx(0.25)
    assert m.coll_hinge_fwd_max_m == pytest.approx(4.5)
    assert m.coll_fwd_depth_aux_weight == pytest.approx(0.6)


def test_d_fwd_hat_gated_until_depth_decoder_trained():
    """Random-init depth_decoder must not emit d_fwd_hat (OA reward noise)."""
    import tempfile
    from pathlib import Path

    m = TorchRSSMDynamics(image_size=16, device="cpu", recurrent_dim=16, stoch_dim=4, stoch_classes=4)
    assert m.depth_decoder_trained is False
    z = np.zeros(m.latent_dim, dtype=np.float64)
    a = np.zeros(4, dtype=np.float64)
    out = m.step(z, a)
    assert out.d_fwd_hat is None

    m.depth_decoder_trained = True
    out2 = m.step(z, a)
    # Flag on → decoder runs; median may be finite or None if no positive patch.
    assert out2.d_fwd_hat is None or float(out2.d_fwd_hat) > 0.0

    # Checkpoint without depth_decoder.* leaves the flag False.
    bare = {k: v for k, v in m.state_dict().items() if not k.startswith("depth_decoder.")}
    path = Path(tempfile.mkdtemp()) / "wm.pt"
    torch.save({"model": bare, "step": 0}, path)
    m2 = TorchRSSMDynamics(image_size=16, device="cpu", recurrent_dim=16, stoch_dim=4, stoch_classes=4)
    payload = m2.load_checkpoint(str(path))
    assert payload["depth_decoder_trained"] is False
    assert m2.depth_decoder_trained is False
    assert m2.step(z, a).d_fwd_hat is None


def test_cone_mins_match_numpy_and_swapped_side_is_costly():
    d = np.full((8, 8), 20.0, dtype=np.float64)
    d[2:6, :4] = 4.0
    d[2:6, 4:] = 12.0
    cones = cone_clearances(d, center_frac=0.5)
    fwd, left, right = cone_mins_torch(torch.from_numpy(d), center_frac=0.5)
    assert abs(float(fwd) - cones["forward"]) < 1e-5
    assert abs(float(left) - cones["left"]) < 1e-5
    assert abs(float(right) - cones["right"]) < 1e-5

    gt = torch.from_numpy(d).unsqueeze(0)
    swapped = gt.flip(-1)
    def _sym(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    bad = float(_cone_side_loss(_sym(swapped), gt))
    good = float(_cone_side_loss(_sym(gt), gt))
    assert bad > good


def test_forward_center_loss_penalizes_a_far_read_of_a_near_obstacle():
    """A flat ~26 m decode must cost more than the true center depth."""
    gt = torch.full((1, 16, 16), 40.0)
    gt[:, 4:12, 4:12] = 11.0
    far = _symlog(torch.full((1, 16, 16), 26.0))
    matched = _symlog(gt)
    assert float(_forward_center_depth_loss(far, gt)) > float(
        _forward_center_depth_loss(matched, gt)
    ) + 0.2
    far = far.detach().requires_grad_(True)
    _forward_center_depth_loss(far, gt).backward()
    assert float(far.grad[:, 4:12, 4:12].abs().mean()) > 0.0


def test_forward_center_loss_penalizes_reading_a_far_center_as_near():
    """A far center painted at 8 m costs more than the true depth, and 4 m is not free for an 8 m wall."""
    far_gt = torch.full((1, 16, 16), 40.0)
    too_close = _symlog(torch.full((1, 16, 16), 8.0))
    matched = _symlog(far_gt)
    assert float(_forward_center_depth_loss(too_close, far_gt)) > float(
        _forward_center_depth_loss(matched, far_gt)
    ) + 0.2
    wall = torch.full((1, 16, 16), 8.0)
    short = _symlog(torch.full((1, 16, 16), 4.0))
    assert float(_forward_center_depth_loss(short, wall)) > float(
        _forward_center_depth_loss(_symlog(wall), wall)
    ) + 0.05


def test_center_loss_charges_reading_a_wall_too_close():
    """4 m for a 7 m wall is not the cheap direction."""
    gt = torch.full((1, 16, 16), 7.0)
    under = float(_forward_center_depth_loss(_symlog(torch.full((1, 16, 16), 4.0)), gt))
    over = float(_forward_center_depth_loss(_symlog(torch.full((1, 16, 16), 12.0)), gt))
    assert under > 0.5 * over


def test_far_center_read_closer_than_truth_costs_metres():
    """A 70 m center read as 50 m costs more than the true distance."""
    gt = torch.full((1, 16, 16), 70.0)
    short = float(_forward_center_depth_loss(_symlog(torch.full((1, 16, 16), 50.0)), gt))
    matched = float(_forward_center_depth_loss(_symlog(gt), gt))
    assert short > matched + 2.0


def test_center_loss_grows_with_metre_error():
    """A 9 m miss on a 22 m center costs more than a 2 m miss."""
    gt = torch.full((1, 16, 16), 22.0)
    close = float(_forward_center_depth_loss(_symlog(torch.full((1, 16, 16), 20.0)), gt))
    far = float(_forward_center_depth_loss(_symlog(torch.full((1, 16, 16), 13.0)), gt))
    assert far > close + 1.0


def test_center_median_matches_each_frame():
    """A 3 m frame and a 9 m frame must not cancel into one mid reading."""
    gt = torch.stack([torch.full((16, 16), 3.0), torch.full((16, 16), 9.0)])
    level = float(torch.expm1(0.5 * (_symlog(gt[0, 0, 0]) + _symlog(gt[1, 0, 0]))))
    pred = torch.full_like(gt, level)
    valid = torch.ones_like(gt, dtype=torch.bool)
    assert float(_center_median_match(pred, gt, valid)) > 0.05


def test_collision_head_only_leaves_the_backbone_fixed():
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        recurrent_dim=16,
        stoch_dim=4,
        stoch_classes=4,
        coll_near_depth_m=8.0,
        coll_pos_weight=1.0,
        coll_fwd_depth_aux_weight=1.0,
        center_depth_weight=4.0,
    )
    m.freeze_backbone_train_collision_head(lr=1e-4, early_fwd_m=18.0)
    assert m.encoder.conv[0].weight.requires_grad is False
    assert m.coll_head[0].weight.requires_grad is True
    assert m.coll_loss_scale == 1.0
    assert m.center_depth_weight == 0.0
    depth = torch.full((1, 2, 16, 16), 30.0)
    depth[:, :, 4:12, 4:12] = 6.0
    sample = {
        "rgb": torch.randint(0, 255, (1, 2, 16, 16, 3), dtype=torch.uint8),
        "proprio": torch.zeros(1, 2, 4),
        "action": torch.tensor([[[0.7, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0]]]),
        "reward": torch.zeros(1, 2),
        "done": torch.zeros(1, 2, dtype=torch.bool),
        "collided": torch.zeros(1, 2),
        "depth": depth,
    }
    loss, _stats = m.training_loss(sample)
    m.zero_grad(set_to_none=True)
    loss.backward()
    assert m.coll_head[0].weight.grad is not None
    assert float(m.coll_head[0].weight.grad.abs().sum()) > 0.0
    assert m.encoder.conv[0].weight.grad is None or float(m.encoder.conv[0].weight.grad.abs().sum()) == 0.0


def test_center_depth_weight_enters_the_loss_and_collision_can_be_held():
    m = TorchRSSMDynamics(
        image_size=16,
        device="cpu",
        recurrent_dim=16,
        stoch_dim=4,
        stoch_classes=4,
        center_depth_weight=4.0,
        coll_loss_scale=0.0,
        coll_pos_weight=1.5,
        coll_near_depth_m=8.0,
    )
    sample = {
        "rgb": torch.randint(0, 255, (1, 2, 16, 16, 3), dtype=torch.uint8),
        "proprio": torch.zeros(1, 2, 4),
        "action": torch.zeros(1, 2, 4),
        "reward": torch.zeros(1, 2),
        "done": torch.zeros(1, 2, dtype=torch.bool),
        "collided": torch.zeros(1, 2),
        "depth": torch.full((1, 2, 16, 16), 11.0),
    }
    loss, stats = m.training_loss(sample)
    assert np.isfinite(stats["loss_center_depth"])
    assert stats["loss_center_depth"] > 0.0
    m.zero_grad(set_to_none=True)
    loss.backward()
    coll_grad = m.coll_head[0].weight.grad
    assert coll_grad is None or float(coll_grad.abs().sum()) == 0.0
    assert m.depth_decoder.fc.weight.grad is not None
    assert float(m.depth_decoder.fc.weight.grad.abs().sum()) > 0.0

