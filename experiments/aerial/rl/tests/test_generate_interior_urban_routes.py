import json
import math
from pathlib import Path

from experiments.aerial.scripts.generate_interior_urban_routes import generate_variants


def _spawn_tangent_yaw_rad(route: dict) -> float:
    p0, p1 = route["pos"][0], route["pos"][1]
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def test_generate_twenty_r09_interior_routes():
    root = Path(__file__).resolve().parents[4]
    routes = json.loads((root / "artifacts/seen_airsim16_m1a20.json").read_text())
    from experiments.aerial.phase3_unified.region_geometry import load_regions

    regions = load_regions()
    selected, _ = generate_variants(routes[8], regions=regions, n_target=20)
    assert len(selected) == 20
    for r in selected:
        assert r["source_route_label"] == "R09"
        assert r["pos"][0][2] >= 12.0
        assert r["complexity"]["spawn_inland_m"] >= 100.0
        # Horizontal flyable path: no in-place climb prefix; yaw matches first leg.
        p0, p1 = r["pos"][0], r["pos"][1]
        horiz = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        assert horiz >= 1.0
        tang = _spawn_tangent_yaw_rad(r)
        assert abs(r["yaw"][0] - tang) < 0.02
