"""Metrics, exports and end-to-end pipeline/CLI runs on the offline fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeGeocoder, FakeNER, cand

from geoparser.cli import main
from geoparser.evaluation import (
    MAX_ERROR_KM,
    GoldDocument,
    GoldToponym,
    _auc,
    evaluate,
    format_report,
    load_gold,
)
from geoparser.exceptions import EvaluationError
from geoparser.export import build_map, to_geodataframe, to_geojson, write_geojson
from geoparser.models import Document, Entity, ResolvedLocation
from geoparser.pipeline import GeoParser

TEXT = "Flooding hit Köln and Bonn."
KOELN = cand("Köln", 50.9384, 6.9599, bbox=(50.83, 51.08, 6.77, 7.16))
BONN_WRONG = cand("Bonn far away", 40.0, 6.9)


def _doc(locs: list[ResolvedLocation]) -> Document:
    return Document(TEXT, "d1", "de", [loc.entity for loc in locs], locs)


def _gold() -> GoldDocument:
    return GoldDocument(
        "d1", TEXT, "de", [GoldToponym(13, 17, "Köln", 50.9375, 6.9603), GoldToponym(22, 26, "Bonn", 50.7374, 7.0982)]
    )


class TestMetrics:
    def test_perfect_recognition_with_errors(self) -> None:
        doc = _doc(
            [
                ResolvedLocation(Entity("Köln", 13, 17), KOELN, 1.0, 1),
                ResolvedLocation(Entity("Bonn", 22, 26), BONN_WRONG, 1.0, 1),
            ]
        )
        m = evaluate([doc], [_gold()]).overall
        assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
        assert m.acc_at_161km == 0.5
        assert m.median_error_km is not None and m.mean_error_km is not None
        assert m.mean_error_km == pytest.approx(597, abs=5)  # (≈0 + ≈1194 km) / 2

    def test_strict_vs_lenient(self) -> None:
        doc = _doc([ResolvedLocation(Entity("Köln and", 13, 21), KOELN, 1.0, 1)])
        strict = evaluate([doc], [_gold()], "strict").overall
        lenient = evaluate([doc], [_gold()], "lenient").overall
        assert strict.tp == 0 and lenient.tp == 1
        assert lenient.precision == 1.0 and lenient.recall == 0.5

    def test_unresolved_true_positive_counts_as_max_error(self) -> None:
        doc = _doc([ResolvedLocation(Entity("Köln", 13, 17), None, 0.0, 0)])
        m = evaluate([doc], [_gold()]).overall
        assert m.tp == 1 and m.n_resolved_tp == 0
        assert m.mean_error_km is None and m.acc_at_161km == 0.0 and m.auc == 1.0

    def test_missing_prediction_document(self) -> None:
        m = evaluate([], [_gold()]).overall
        assert (m.recall, m.precision, m.f1) == (0.0, 0.0, 0.0)

    def test_auc_bounds(self) -> None:
        assert _auc([0.0, 0.0]) == 0.0
        assert _auc([MAX_ERROR_KM] * 3) == pytest.approx(1.0)
        assert 0 < _auc([1, 10, 100, 1000]) < 1

    def test_report_markdown(self) -> None:
        md = format_report(evaluate([], [_gold()]))
        assert "| **all** |" in md and "| de |" in md

    def test_invalid_mode(self) -> None:
        with pytest.raises(EvaluationError):
            evaluate([], [], mode="fuzzy")  # type: ignore[arg-type]


class TestGoldLoading:
    def test_demo_gold_is_consistent(self, demo_gold: Path) -> None:
        gold = load_gold(demo_gold)
        assert {g.language for g in gold} == {"en", "de", "ta", "hi"}

    def test_offset_mismatch_detected(self, tmp_path: Path) -> None:
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps({"id": "x", "text": "Köln", "toponyms": [{"start": 0, "end": 3, "text": "Köln"}]}),
            encoding="utf-8",
        )
        with pytest.raises(EvaluationError):
            load_gold(p)


class TestExport:
    def _docs(self) -> list[Document]:
        return [
            _doc(
                [
                    ResolvedLocation(Entity("Köln", 13, 17), KOELN, 0.9, 1),
                    ResolvedLocation(Entity("Bonn", 22, 26), None, 0.0, 0),
                ]
            )
        ]

    def test_geojson_structure(self) -> None:
        fc = to_geojson(self._docs())
        feat = fc["features"][0]
        assert feat["geometry"]["coordinates"] == [6.9599, 50.9384]  # lon, lat
        assert feat["bbox"] == [6.77, 50.83, 7.16, 51.08]
        assert feat["properties"]["mention"] == "Köln"
        assert fc["metadata"]["n_unresolved"] == 1

    def test_write_geojson_utf8(self, tmp_path: Path) -> None:
        path = write_geojson(self._docs(), tmp_path / "o" / "x.geojson")
        assert "Köln" in path.read_text(encoding="utf-8")

    def test_geodataframe(self) -> None:
        gdf = to_geodataframe(self._docs())
        assert len(gdf) == 1 and gdf.crs.to_epsg() == 4326
        assert gdf.geometry.iloc[0].x == pytest.approx(6.9599)

    def test_folium_map(self, tmp_path: Path) -> None:
        out = tmp_path / "m.html"
        build_map(self._docs()).save(str(out))
        html = out.read_text(encoding="utf-8")
        assert "leaflet" in html.lower() and "Köln" in html

    def test_empty_map(self) -> None:
        assert build_map([]) is not None


class TestPipeline:
    def test_parse_with_fakes(self) -> None:
        geo = FakeGeocoder({"Köln": [KOELN]})
        parser = GeoParser(FakeNER(["Köln", "Atlantis"]), geo)
        doc = parser.parse("Von Köln nach Atlantis und zurück nach Köln.", lang="de")
        assert [loc.is_resolved for loc in doc.locations] == [True, False, True]
        assert geo.calls.count("Köln") == 1  # memoised within a document

    def test_tamil_suffix_fallback_query(self) -> None:
        chennai = cand("Chennai", 13.08, 80.27, cc="in")
        geo = FakeGeocoder({"சென்னை": [chennai]})
        doc = GeoParser(FakeNER(["சென்னையில்"]), geo).parse("சென்னையில் மழை.", lang="ta")
        assert doc.locations[0].candidate is chennai
        assert geo.calls == ["சென்னையில்", "சென்னை"]

    def test_empty_text(self) -> None:
        doc = GeoParser(FakeNER([]), FakeGeocoder({})).parse("   ", lang="en")
        assert doc.entities == [] and doc.locations == []


class TestCLI:
    def test_offline_evaluate_end_to_end(
        self, tmp_path: Path, demo_gold: Path, demo_cache: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "cache.json"
        cache.write_bytes(demo_cache.read_bytes())
        report = tmp_path / "report.json"
        rc = main(
            [
                "evaluate",
                "--gold",
                str(demo_gold),
                "--ner-backend",
                "gazetteer",
                "--offline",
                "--cache",
                str(cache),
                "--report",
                str(report),
                "--map",
                str(tmp_path / "m.html"),
            ]
        )
        assert rc == 0
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["overall"]["f1"] > 0.9
        assert data["overall"]["acc_at_161km"] == 1.0
        assert set(data["per_language"]) == {"en", "de", "ta", "hi"}
        assert "| **all** |" in capsys.readouterr().out

    def test_offline_parse_text(self, tmp_path: Path, demo_cache: Path) -> None:
        out = tmp_path / "o.geojson"
        rc = main(
            [
                "parse",
                "--text",
                "Das Hochwasser erreichte Frankfurt und Słubice; Brandenburg handelte.",
                "--ner-backend",
                "gazetteer",
                "--offline",
                "--cache",
                str(demo_cache),
                "--geojson",
                str(out),
            ]
        )
        assert rc == 0
        names = [f["properties"]["resolved_name"] for f in json.loads(out.read_text("utf-8"))["features"]]
        assert names[0].startswith("Frankfurt (Oder)")

    def test_online_without_user_agent_fails_cleanly(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEOPARSER_USER_AGENT", raising=False)
        assert main(["parse", "--text", "Köln", "--cache", str(tmp_path / "c.json")]) == 1
