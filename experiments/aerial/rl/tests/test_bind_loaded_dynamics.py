"""bind_loaded_dynamics keeps collector/planner on the loaded WM instance."""
from __future__ import annotations

from types import SimpleNamespace

from experiments.aerial.rl.train_rl import bind_loaded_dynamics


def test_bind_loaded_dynamics_shares_one_object():
    loaded = object()
    stale = object()
    planner = SimpleNamespace(dynamics=stale)
    collector = SimpleNamespace(dynamics=stale, planner=planner)
    loop = SimpleNamespace(dynamics=stale, collector=collector)

    bind_loaded_dynamics(loop, loaded)

    assert loop.dynamics is loaded
    assert collector.dynamics is loaded
    assert planner.dynamics is loaded
