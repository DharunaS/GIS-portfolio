"""Small, dependency-free geodesic helpers."""

from __future__ import annotations

import math

EARTH_RADIUS_KM: float = 6371.0088  # IUGG mean Earth radius


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points.

    The haversine formula is accurate to ~0.5 % versus an ellipsoidal model,
    which is well below typical geocoding error and keeps evaluation fast.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        Distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bbox_diagonal_km(south: float, north: float, west: float, east: float) -> float:
    """Length of a bounding-box diagonal, used as a proxy for feature size.

    Args:
        south: Minimum latitude.
        north: Maximum latitude.
        west: Minimum longitude.
        east: Maximum longitude.

    Returns:
        Diagonal length in kilometres.
    """
    return haversine_km(south, west, north, east)
