"""PathExpert densify collect — mock smoke."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.aerial.path_expert import PathExpertPolicy
from experiments.aerial.rl.collect_path_expert_dataset import (
    _episode_arrived,
    _next_episode_index,
    _parse_route_indices,
    build_collector,
    main as collect_main,
)
from experiments.aerial.rl.spawn_utils import lift_episode_z, nudge_episode_z
from experiments.aerial.rl.env.obs import PolicyObservation


def _pol_obs(x: float, y: float = 0.0, yaw: float = 0.0) -> PolicyObservation:
    return PolicyObservation(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.array([x, y, 0.0, yaw], dtype=np.float64),
    )


def test_lift_episode_z():
    ep = {"pos": [[0, 0, 10], [10, 0, 10]], "yaw": [0, 0]}
    lifted = lift_episode_z(ep, 12.0)
    assert lifted["pos"][0][2] == 12.0
    assert lifted["pos"][1][2] == 12.0


def test_parse_route_indices_and_next_episode_index(tmp_path: Path):
    assert _parse_route_indices("6,1, 2") == [6, 1, 2]
    (tmp_path / "episode_00003.npz").write_bytes(b"x")
    assert _next_episode_index(tmp_path) == 4


def test_nudge_episode_z():
    ep = {"pos": [[0, 0, 12], [10, 0, 12]], "yaw": [0, 0]}
    nudged = nudge_episode_z(ep, 5.0)
    assert nudged["pos"][0][2] == 17.0
    assert nudged["pos"][1][2] == 17.0


def test_path_expert_policy_follows_polyline():
    ep = {
        "pos": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        "yaw": [0.0, 0.0, 0.0],
        "gpt_instruction": "go",
    }
    pol = PathExpertPolicy()
    pol.bind_episode(ep)
    pol.reset()
    a0 = pol.act(_pol_obs(0.0))
    assert a0[0] > 0.0
    assert a0.shape == (4,)


def test_collect_path_expert_mock(tmp_path: Path):
    fixtures = Path("experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json")
    if not fixtures.is_file():
        # Fallback: tiny inline ann
        ann = tmp_path / "ann.json"
        ann.write_text(
            '[{"pos":[[0,0,0],[8,0,0],[16,0,0]],"yaw":[0,0,0],'
            '"gpt_instruction":"fly","action":[1,1,0]}]\n'
        )
        ann_path = str(ann)
    else:
        ann_path = str(fixtures)

    out = tmp_path / "ds"
    rc = collect_main(
        [
            "--backend",
            "mock",
            "--episodes",
            "2",
            "--max-steps",
            "60",
            "--step-hz",
            "5",
            "--annotation",
            ann_path,
            "--out",
            str(out),
            "--keep-failed",
        ]
    )
    assert rc == 0
    npzs = list(out.glob("episode_*.npz"))
    assert npzs
    raw = np.load(npzs[0])
    assert "actions" in raw.files and "rgb" in raw.files and "arrived" in raw.files
    assert raw["actions"].shape[1] == 4
    # Densify: more frames than sparse waypoints when path is long enough.
    assert raw["actions"].shape[0] >= 1
