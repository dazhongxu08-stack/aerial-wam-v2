import json
from pathlib import Path

from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions
from experiments.aerial.scripts.select_outdoor_complex_routes import (
    score_spawn,
    select_complex_routes,
)


def _route(spawn_z: float, xy: tuple[float, float], cat: str = "low_long") -> dict:
    x, y = xy
    return {
        "pos": [[x, y, spawn_z], [x + 10.0, y, spawn_z]],
        "yaw": [0.0, 0.0],
        "image_path": f"env_airsim_16/astar_data/{cat}/x",
        "gpt_instruction": "test",
    }


def test_r04_spawn_is_water_rejected():
    regions = load_regions()
    meta = classify_spawn_xy(-1112.809, -346.613, regions)
    assert meta["in_water"]
    assert not meta["spawn_ok"]


def test_r10_spawn_rejected_as_waterfront():
    regions = load_regions()
    meta = classify_spawn_xy(164.0, -130.0, regions)
    assert not meta["spawn_ok"]
    assert meta["in_water"] or not meta["urban_region_ids"]


def test_score_spawn_rejects_water_even_with_low_z():
    regions = load_regions()
    row = score_spawn(3, _route(15.0, (-1113.0, -347.0)), regions=regions, max_spawn_z=22.0)
    assert row is None


def test_select_prefers_urban_low_spawn():
    regions = load_regions()
    routes = [
        _route(45.0, (-1113.0, -347.0), "medium_short"),
        _route(7.5, (164.0, -130.0), "low_average"),
        _route(16.0, (-969.0, -359.0), "low_long"),
        _route(15.0, (-680.0, -100.0), "low_long"),
        _route(21.0, (36.0, 79.0), "low_average"),
        _route(19.0, (-1251.0, -533.0), "low_long"),
        _route(20.0, (92.0, 111.0), "low_long"),
    ]
    selected, _ = select_complex_routes(
        routes,
        regions=regions,
        n_select=3,
        max_spawn_z=22.0,
        min_spawn_inland_m=0.0,
        min_path_inland_m=0.0,
    )
    labels = {r["complexity"]["route_label"] for r in selected}
    assert "R01" not in labels  # water R04 analogue
    assert any("R" in x for x in labels)


def test_r17_spawn_is_water_after_smoke():
    regions = load_regions()
    meta = classify_spawn_xy(-1290.0, 100.0, regions)
    assert meta["in_water"]
    assert not meta["spawn_ok"]


def test_r14_waterfront_excluded_when_interior_routes_exist():
    regions = load_regions()
    routes = json.loads(
        (Path(__file__).resolve().parents[4] / "artifacts/seen_airsim16_m1a20.json").read_text()
    )
    selected, _ = select_complex_routes(routes, regions=regions, n_select=4, max_spawn_z=22.0)
    labels = {r["complexity"]["route_label"] for r in selected}
    assert "R09" in labels
    assert "R17" not in labels
    assert "R14" not in labels
