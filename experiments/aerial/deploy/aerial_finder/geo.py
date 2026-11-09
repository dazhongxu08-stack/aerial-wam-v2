"""Local world frame <-> WGS84.

World frame everywhere in this package: x north, y east, z up (meters), origin
at the vehicle home point. This is AirSim NED with z negated, the frame the
consequence model was trained in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class GeoRef:
    lat: float
    lon: float
    alt: float

    def _radii(self) -> Tuple[float, float]:
        s = math.sin(math.radians(self.lat))
        w = 1.0 - WGS84_E2 * s * s
        return WGS84_A * (1.0 - WGS84_E2) / w ** 1.5, WGS84_A / math.sqrt(w)

    def to_gps(self, north: float, east: float, up: float) -> Tuple[float, float, float]:
        """Tangent-plane offset -> (lat, lon, alt); sub-cm error within a few km."""
        rn, re = self._radii()
        lat = self.lat + math.degrees(north / rn)
        lon = self.lon + math.degrees(east / (re * math.cos(math.radians(self.lat))))
        return lat, lon, self.alt + up

    def to_local(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        rn, re = self._radii()
        north = math.radians(lat - self.lat) * rn
        east = math.radians(lon - self.lon) * re * math.cos(math.radians(self.lat))
        return north, east, alt - self.alt


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
