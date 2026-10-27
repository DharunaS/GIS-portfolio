"""Shared fixtures: fake NER/geocoder backends and the offline demo data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoparser.geocoder import Geocoder  # noqa: E402
from geoparser.models import BoundingBox, Candidate, Entity  # noqa: E402
from geoparser.ner import NERBackend  # noqa: E402

EXAMPLES = ROOT / "examples"


class FakeNER(NERBackend):
    """Tags every occurrence of the given names as LOC."""

    name = "fake"

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def extract(self, text: str, lang: str) -> list[Entity]:
        out = []
        for n in self.names:
            start = text.find(n)
            while start >= 0:
                out.append(Entity(n, start, start + len(n), "LOC", 0.99, "fake"))
                start = text.find(n, start + len(n))
        return sorted(out, key=lambda e: e.start)


class FakeGeocoder(Geocoder):
    """Returns canned candidates and records every query."""

    def __init__(self, table: dict[str, list[Candidate]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def search(self, query: str, lang: str | None = None) -> list[Candidate]:
        self.calls.append(query)
        return list(self.table.get(query, []))


def cand(
    name: str,
    lat: float,
    lon: float,
    place_type: str = "city",
    cc: str = "de",
    importance: float = 0.5,
    population: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> Candidate:
    """Concise candidate constructor for tests (bbox = south, north, west, east)."""
    return Candidate(
        name=name,
        lat=lat,
        lon=lon,
        bbox=BoundingBox(*bbox) if bbox else None,
        place_type=place_type,
        country_code=cc,
        importance=importance,
        population=population,
    )


@pytest.fixture
def demo_cache() -> Path:
    return EXAMPLES / "geocode_cache.json"


@pytest.fixture
def demo_gold() -> Path:
    return EXAMPLES / "gold_annotations.jsonl"
