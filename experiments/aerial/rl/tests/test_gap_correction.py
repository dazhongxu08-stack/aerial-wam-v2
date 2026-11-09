"""The route-1 residual is off at route 0's start and off past the west gap."""
from experiments.aerial.rl.gap_correction import correction_gate


def test_gate_is_off_at_route0_start_and_past_the_gap():
    assert correction_gate(-927.5, 101.3) == 0.0
    assert correction_gate(-970.0, 70.0) == 0.0
    assert correction_gate(-940.0, 70.0) == 1.0
    assert correction_gate(-940.0, 19.0) == 1.0
    assert correction_gate(-944.0, 80.0) == 0.0
