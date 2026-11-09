"""The near-wall set is the route-1 face, not route 0's open street or its arrival."""
import math

import numpy as np

from experiments.aerial.rl.near_wall_turn import (
    WallBitPassage,
    near_wall_east,
    west_turn_action,
)


def test_near_wall_is_the_route1_face_only():
    assert near_wall_east(-946.9, 46.0, 1.6)
    assert near_wall_east(-933.0, 63.7, 17.3)
    assert not near_wall_east(-927.7, 71.0, 22.5)
    assert not near_wall_east(-958.2, 19.5, 16.4)
    assert not near_wall_east(-950.2, 70.3, 77.2)
    assert not near_wall_east(-962.0, 50.0, 8.0)


def test_reserved_bits_start_the_passage_and_route0_codes_do_not():
    passage = WallBitPassage(np.array([1, 4]))
    quiet = np.zeros(8)
    quiet[0] = 1.0
    assert passage.apply(quiet, (-930.0, 70.0, 24.0), 0.0, (-952.0, -14.0)) is None
    hit = quiet.copy()
    hit[1] = 1.0
    out = passage.apply(hit, (-930.0, 70.0, 24.0), 0.0, (-952.0, -14.0))
    assert out is not None
    action, phase = out
    assert phase == "west"
    assert action.shape == (4,)
    # Stays on the passage even if the next code looks like route 0.
    assert passage.apply(quiet, (-940.0, 70.0, 24.0), math.pi, (-952.0, -14.0)) is not None


def test_west_turn_yaws_toward_west_from_the_southwest_nose():
    action = west_turn_action((-933.0, 64.0, 24.0), math.radians(-121.0))
    assert action.shape == (4,)
    assert abs(action[3]) > 0.2
    assert action[0] == 0.0
