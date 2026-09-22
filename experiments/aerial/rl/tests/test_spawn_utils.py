"""spawn_utils: z-lift and retry plan."""
import numpy as np

from experiments.aerial.rl.spawn_utils import lift_episode_z, nudge_episode_z, spawn_retry_plan


def _ep(z0: float) -> dict:
    return {"pos": [[0.0, 0.0, z0], [10.0, 0.0, z0 + 1.0]], "yaw": [0.0, 0.0]}


def test_lift_episode_z_raises_uniformly():
    out = lift_episode_z(_ep(12.0), 24.0)
    pts = np.asarray(out["pos"], dtype=np.float64)
    assert pts[0, 2] == 24.0
    assert pts[1, 2] == 25.0


def test_lift_episode_z_noop_when_already_high():
    out = lift_episode_z(_ep(30.0), 24.0)
    assert float(np.asarray(out["pos"])[0, 2]) == 30.0


def test_nudge_episode_z_adds_delta():
    out = nudge_episode_z(_ep(24.0), 5.0)
    pts = np.asarray(out["pos"], dtype=np.float64)
    assert pts[0, 2] == 29.0
    assert pts[1, 2] == 30.0


def test_spawn_retry_plan_stacks_on_lifted_base():
    tries = spawn_retry_plan(
        _ep(12.0),
        min_spawn_z=24.0,
        spawn_z_retry_m=5.0,
        spawn_z_max_retries=2,
        spawn_xy_nudge_m=12.0,
    )
    z0 = [float(np.asarray(t["pos"])[0, 2]) for t in tries]
    # First 3 are z-only; remaining are XY offsets on the highest z.
    assert z0[:3] == [24.0, 29.0, 34.0]
    assert len(tries) == 3 + 20
    xy0 = np.asarray(tries[3]["pos"], dtype=np.float64)[0, :2]
    assert abs(float(xy0[0]) - 12.0) < 1e-6 or abs(float(xy0[1]) - 12.0) < 1e-6


def test_nudge_episode_xy():
    from experiments.aerial.rl.spawn_utils import nudge_episode_xy

    out = nudge_episode_xy(_ep(24.0), 3.0, -1.0)
    pts = np.asarray(out["pos"], dtype=np.float64)
    assert pts[0, 0] == 3.0 and pts[0, 1] == -1.0
    assert pts[1, 0] == 13.0 and pts[1, 1] == -1.0
