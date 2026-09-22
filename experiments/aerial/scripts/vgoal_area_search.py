"""Area-search helpers for Phase-2 vgoal M3 (AreaSearchPlanner wiring)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def search_area_bounds(
    spawn_pos: np.ndarray,
    *,
    half_m: float,
    route_info: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) for a search box centered on spawn."""
    sa = (route_info or {}).get("search_area") or {}
    cx, cy = float(spawn_pos[0]), float(spawn_pos[1])
    half = float(sa.get("half_m", half_m))
    min_x = float(sa.get("min_x", cx - half))
    max_x = float(sa.get("max_x", cx + half))
    min_y = float(sa.get("min_y", cy - half))
    max_y = float(sa.get("max_y", cy + half))
    return min_x, max_x, min_y, max_y


def corridor_waypoints(
    min_x: float,
    max_x: float,
    y_center: float,
    altitude_z: float,
    spacing_m: float,
    *,
    start_x: Optional[float] = None,
) -> list[np.ndarray]:
    """Eastbound cruise along Humen corridor at fixed y (probe best: yaw≈45° toward bridge)."""
    x0 = float(start_x if start_x is not None else min_x)
    x1 = float(max_x)
    if x0 > x1:
        x0, x1 = x1, x0
    xs = np.arange(x0, x1 + 1e-3, float(spacing_m))
    if len(xs) < 2:
        xs = np.array([x0, x1], dtype=np.float64)
    return [np.array([x, float(y_center), float(altitude_z)], dtype=np.float64) for x in xs]


def make_area_search_planner(
    spawn_pos: np.ndarray,
    *,
    pattern: str,
    altitude_z: float,
    half_m: float = 40.0,
    sweep_spacing_m: float = 15.0,
    waypoint_reach_radius_m: float = 3.5,
    spiral_max_radius_m: float = 25.0,
    route_info: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Build ``AreaSearchPlanner`` for lawnmower/spiral/corridor; ``None`` for legacy scan."""
    kind = str(pattern).lower()
    if kind in ("scan", "yaw", "legacy", ""):
        return None

    from vgoal.search_planner import AreaSearchPlanner, SearchAreaConfig

    sa = (route_info or {}).get("search_area") or {}
    min_x, max_x, min_y, max_y = search_area_bounds(
        spawn_pos, half_m=half_m, route_info=route_info
    )
    cfg = SearchAreaConfig(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        altitude_z=float(sa.get("altitude_z", altitude_z)),
        sweep_spacing_m=float(sa.get("sweep_spacing_m", sweep_spacing_m)),
        waypoint_reach_radius_m=float(
            sa.get("waypoint_reach_radius_m", waypoint_reach_radius_m)
        ),
        loop=bool(sa.get("loop", True)),
    )
    planner = AreaSearchPlanner(cfg)
    if kind in ("spiral", "expanding_spiral"):
        cx, cy = float(spawn_pos[0]), float(spawn_pos[1])
        planner.generate_spiral_waypoints(
            center=[cx, cy, float(cfg.altitude_z)],
            max_radius=float(sa.get("spiral_max_radius_m", spiral_max_radius_m)),
        )
    elif kind == "corridor":
        y_c = float(sa.get("corridor_y", sa.get("y_center", float(spawn_pos[1]))))
        loop = bool(sa.get("loop", False))
        cfg.loop = loop
        planner.config.loop = loop
        wps = corridor_waypoints(
            min_x,
            max_x,
            y_c,
            float(cfg.altitude_z),
            float(sa.get("sweep_spacing_m", sweep_spacing_m)),
            start_x=float(sa.get("corridor_start_x", float(spawn_pos[0]))),
        )
        planner.waypoints = wps
        planner.current_wp_idx = 0
        planner.is_completed = False
    return planner
