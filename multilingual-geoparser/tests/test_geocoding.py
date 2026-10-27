"""Geocoder parsing, caching and error handling; spatial disambiguation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeGeocoder, cand

from geoparser.disambiguation import DisambiguationConfig, Disambiguator
from geoparser.exceptions import ConfigurationError, GeocodingError
from geoparser.geo_utils import haversine_km
from geoparser.geocoder import CachedGeocoder, NominatimGeocoder, candidate_from_nominatim
from geoparser.models import BoundingBox, Entity

RAW = {
    "display_name": "Köln, North Rhine-Westphalia, Germany",
    "lat": "50.9384",
    "lon": "6.9599",
    "boundingbox": ["50.83", "51.08", "6.77", "7.16"],
    "class": "boundary",
    "type": "administrative",
    "addresstype": "city",
    "importance": 0.76,
    "osm_type": "relation",
    "osm_id": 1,
    "address": {"country_code": "de"},
    "extratags": {"population": "1,083,498"},
}


class TestGeoUtils:
    def test_haversine_known_distance(self) -> None:
        # Berlin → Munich ≈ 504 km
        assert haversine_km(52.5200, 13.4050, 48.1351, 11.5820) == pytest.approx(504, rel=0.01)

    def test_haversine_zero(self) -> None:
        assert haversine_km(10, 10, 10, 10) == 0.0


class TestModels:
    def test_bbox_from_nominatim_and_geojson_order(self) -> None:
        bb = BoundingBox.from_nominatim(["1", "2", "3", "4"])
        assert bb.as_geojson_bbox() == [3.0, 1.0, 4.0, 2.0]

    @pytest.mark.parametrize("vals", [[2, 1, 0, 0], [0, 1, 0, 200]])
    def test_bbox_validation(self, vals: list[float]) -> None:
        with pytest.raises(ValueError):
            BoundingBox.from_nominatim(vals)


class TestNominatimParsing:
    def test_candidate_fields(self) -> None:
        c = candidate_from_nominatim(RAW)
        assert (c.lat, c.lon, c.place_type, c.country_code) == (50.9384, 6.9599, "city", "de")
        assert c.population == 1083498
        assert c.osm_id == "relation/1"
        assert c.bbox is not None and c.bbox.contains(50.94, 6.96)

    def test_bad_bbox_is_ignored(self) -> None:
        c = candidate_from_nominatim({**RAW, "boundingbox": ["x"], "extratags": {"population": "many"}})
        assert c.bbox is None and c.population is None

    def test_missing_coordinates_raise(self) -> None:
        with pytest.raises(KeyError):
            candidate_from_nominatim({"display_name": "nowhere"})

    def test_placeholder_user_agent_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            NominatimGeocoder(user_agent="geopy")

    def test_public_rate_limit_enforced(self) -> None:
        with pytest.raises(ConfigurationError):
            NominatimGeocoder(user_agent="test-suite/1.0 (ci@example.org)", min_delay_seconds=0.1)

    def test_transient_errors_return_empty(self) -> None:
        from geopy.exc import GeocoderTimedOut

        g = NominatimGeocoder(user_agent="test-suite/1.0 (ci@example.org)")

        def boom(*a: object, **k: object) -> None:
            raise GeocoderTimedOut("slow")

        g._geocode = boom  # type: ignore[assignment]
        assert g.search("Köln") == []

    def test_policy_errors_raise(self) -> None:
        from geopy.exc import GeocoderInsufficientPrivileges

        g = NominatimGeocoder(user_agent="test-suite/1.0 (ci@example.org)")

        def denied(*a: object, **k: object) -> None:
            raise GeocoderInsufficientPrivileges("403")

        g._geocode = denied  # type: ignore[assignment]
        with pytest.raises(GeocodingError):
            g.search("Köln")


class TestCache:
    def test_roundtrip_and_single_upstream_call(self, tmp_path: Path) -> None:
        inner = FakeGeocoder({"Köln": [candidate_from_nominatim(RAW)]})
        path = tmp_path / "c.json"
        with CachedGeocoder(path, inner=inner) as cg:
            first = cg.search("Köln")
            second = cg.search("  köln ")
        assert inner.calls == ["Köln"]
        assert first[0].lat == second[0].lat
        assert cg.hits == 1 and cg.misses == 1

        reloaded = CachedGeocoder(path, offline=True)
        assert reloaded.search("Köln")[0].bbox == first[0].bbox

    def test_offline_miss_is_empty(self, tmp_path: Path) -> None:
        assert CachedGeocoder(tmp_path / "none.json", offline=True).search("Atlantis") == []

    def test_offline_requires_no_inner_but_online_does(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            CachedGeocoder(tmp_path / "c.json", inner=None, offline=False)

    def test_corrupt_cache(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(GeocodingError):
            CachedGeocoder(p, offline=True)

    def test_demo_cache_loads(self, demo_cache: Path) -> None:
        cg = CachedGeocoder(demo_cache, offline=True)
        assert len(cg.search("Frankfurt")) == 2
        assert json.loads(demo_cache.read_text(encoding="utf-8"))


# --- Disambiguation -------------------------------------------------------

PARIS_FR = cand("Paris, France", 48.8535, 2.3484, cc="fr", importance=0.88, population=2_133_111)
PARIS_TX = cand("Paris, Texas, United States", 33.6609, -95.5555, "town", "us", 0.52, 24_476)
TEXAS = cand(
    "Texas, United States", 31.26, -98.55, "state", "us", 0.80, 29_000_000, bbox=(25.84, 36.50, -106.65, -93.51)
)
DALLAS = cand("Dallas, Texas", 32.7763, -96.7969, "city", "us", 0.72, 1_304_379)

FFM = cand("Frankfurt am Main", 50.1106, 8.6821, importance=0.80, population=773_068)
FFO = cand("Frankfurt (Oder)", 52.3471, 14.5506, importance=0.62, population=57_015)
BRANDENBURG = cand("Brandenburg", 52.41, 12.53, "state", importance=0.74, bbox=(51.36, 53.56, 11.27, 14.77))


def _ents(*names: str) -> list[Entity]:
    return [Entity(n, i * 10, i * 10 + len(n)) for i, n in enumerate(names)]


class TestDisambiguator:
    def test_paris_texas_via_admin_context(self) -> None:
        res = Disambiguator().resolve(_ents("Paris", "Texas", "Dallas"), [[PARIS_FR, PARIS_TX], [TEXAS], [DALLAS]])
        assert res[0].candidate is PARIS_TX
        assert res[0].score_breakdown["admin_context"] == 1.0

    def test_frankfurt_oder_via_context(self) -> None:
        res = Disambiguator().resolve(_ents("Frankfurt", "Brandenburg"), [[FFM, FFO], [BRANDENBURG]])
        assert res[0].candidate is FFO

    def test_without_context_prominence_wins(self) -> None:
        cfg = DisambiguationConfig(
            weights={
                "importance": 1,
                "population": 1,
                "type_prior": 1,
                "name_match": 1,
                "admin_context": 0,
                "coherence": 0,
            }
        )
        res = Disambiguator(cfg).resolve(_ents("Frankfurt", "Brandenburg"), [[FFM, FFO], [BRANDENBURG]])
        assert res[0].candidate is FFM

    def test_country_hint(self) -> None:
        cfg = DisambiguationConfig(country_hints={"us"})
        res = Disambiguator(cfg).resolve(_ents("Paris"), [[PARIS_FR, PARIS_TX]])
        assert res[0].score_breakdown["admin_context"] in (0.0, 1.0)  # feature present
        # A hint alone (no other mentions) adds 0.15 — not enough to beat Paris FR's prominence.
        assert res[0].candidate is PARIS_FR

    def test_no_candidates_unresolved(self) -> None:
        res = Disambiguator().resolve(_ents("Atlantis"), [[]])
        assert not res[0].is_resolved and res[0].n_candidates == 0

    def test_single_candidate_full_confidence(self) -> None:
        assert Disambiguator().resolve(_ents("Dallas"), [[DALLAS]])[0].confidence == 1.0

    def test_min_confidence_threshold(self) -> None:
        cfg = DisambiguationConfig(min_confidence=0.99)
        res = Disambiguator(cfg).resolve(_ents("Paris"), [[PARIS_FR, PARIS_TX]])
        assert res[0].candidate is None and res[0].confidence < 0.99

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Disambiguator().resolve(_ents("a", "b"), [[]])

    def test_negative_weights_rejected(self) -> None:
        with pytest.raises(ValueError):
            Disambiguator(DisambiguationConfig(weights={"importance": -1}))
