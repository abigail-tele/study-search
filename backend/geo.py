"""ZIP-code distance helpers.

Why: patients want trials near them. We look up a ZIP's center point from the
public-domain Census file and measure straight-line (great-circle) miles to
each trial site. Everything here is a pure function: the user's ZIP is never
stored or logged (rule 8).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Mapping

from backend import config

LatLon = tuple[float, float]


@lru_cache(maxsize=4)
def load_centroids(path: str | Path = config.ZIP_CENTROIDS_PATH) -> dict[str, LatLon]:
    """Read zip,lat,lon CSV once and cache it in memory."""
    out: dict[str, LatLon] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["zip"]] = (float(row["lat"]), float(row["lon"]))
    return out


def lookup(zip_code: str, path: str | Path = config.ZIP_CENTROIDS_PATH) -> LatLon | None:
    """Return (lat, lon) for a 5-digit ZIP (ZIP+4 accepted), else None."""
    z = (zip_code or "").strip()[:5]
    if len(z) != 5 or not z.isdigit():
        return None
    return load_centroids(path).get(z)


def haversine_miles(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * config.EARTH_RADIUS_MILES * asin(sqrt(h))


def nearest_site(locations: Iterable[Mapping], zip_code: str,
                 path: str | Path = config.ZIP_CENTROIDS_PATH) -> dict | None:
    """Closest site with coordinates -> {city, state, distance_mi}, or None.

    None when the ZIP is unknown or no site has coordinates.
    """
    origin = lookup(zip_code, path)
    if origin is None:
        return None
    best = None
    for loc in locations:
        lat, lon = loc["lat"], loc["lon"]
        if lat is None or lon is None:
            continue
        d = haversine_miles(origin, (lat, lon))
        if best is None or d < best["distance_mi"]:
            best = {"city": loc["city"], "state": loc["state"],
                    "distance_mi": round(d, 1)}
    return best
