"""Export resolved locations as GeoJSON, GeoPandas and interactive Folium maps."""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from geoparser.models import Document, ResolvedLocation

logger = logging.getLogger(__name__)

LANGUAGE_COLORS: dict[str, str] = {
    "en": "blue",
    "de": "red",
    "ta": "green",
    "hi": "orange",
}


def _feature(doc: Document, loc: ResolvedLocation, context_chars: int = 60) -> dict[str, Any]:
    c = loc.candidate
    assert c is not None
    s = max(0, loc.entity.start - context_chars)
    e = min(len(doc.text), loc.entity.end + context_chars)
    props: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "language": doc.language,
        "mention": loc.entity.text,
        "start": loc.entity.start,
        "end": loc.entity.end,
        "ner_label": loc.entity.label,
        "ner_score": round(loc.entity.score, 4),
        "ner_source": loc.entity.source,
        "resolved_name": c.name,
        "place_type": c.place_type,
        "place_class": c.place_class,
        "country_code": c.country_code,
        "osm_id": c.osm_id,
        "importance": c.importance,
        "population": c.population,
        "confidence": loc.confidence,
        "n_candidates": loc.n_candidates,
        "context": doc.text[s:e],
        "score_breakdown": loc.score_breakdown,
    }
    feature: dict[str, Any] = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(c.lon, 7), round(c.lat, 7)]},
        "properties": props,
    }
    if c.bbox is not None:
        feature["bbox"] = c.bbox.as_geojson_bbox()
    return feature


def to_geojson(documents: Iterable[Document]) -> dict[str, Any]:
    """Build an RFC 7946 FeatureCollection of resolved mentions.

    Unresolved mentions are excluded from features but counted in the
    collection's ``metadata`` so downstream users can see coverage.

    Args:
        documents: Processed documents.

    Returns:
        A JSON-serialisable FeatureCollection dict.
    """
    features: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for doc in documents:
        for loc in doc.locations:
            if loc.is_resolved:
                features.append(_feature(doc, loc))
            else:
                unresolved.append({"doc_id": doc.doc_id, "mention": loc.entity.text, "start": loc.entity.start})
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if features:
        lons = [f["geometry"]["coordinates"][0] for f in features]
        lats = [f["geometry"]["coordinates"][1] for f in features]
        fc["bbox"] = [min(lons), min(lats), max(lons), max(lats)]
    fc["metadata"] = {"n_resolved": len(features), "n_unresolved": len(unresolved), "unresolved": unresolved}
    return fc


def write_geojson(documents: Iterable[Document], path: str | Path) -> Path:
    """Serialise :func:`to_geojson` output to ``path`` (UTF-8, non-ASCII kept).

    Args:
        documents: Processed documents.
        path: Destination file.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_geojson(documents), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote GeoJSON to %s", path)
    return path


def to_geodataframe(documents: Iterable[Document]) -> Any:
    """Return resolved mentions as a ``geopandas.GeoDataFrame`` (EPSG:4326).

    Args:
        documents: Processed documents.

    Returns:
        A GeoDataFrame with one row per resolved mention.

    Raises:
        ImportError: If GeoPandas is not installed.
    """
    import geopandas as gpd  # local import keeps the core light

    fc = to_geojson(documents)
    if not fc["features"]:
        return gpd.GeoDataFrame(columns=["mention", "geometry"], geometry="geometry", crs="EPSG:4326")
    flat = []
    for f in fc["features"]:
        f = json.loads(json.dumps(f))
        f["properties"]["score_breakdown"] = json.dumps(f["properties"]["score_breakdown"])
        flat.append(f)
    return gpd.GeoDataFrame.from_features(flat, crs="EPSG:4326")


def build_map(
    documents: Iterable[Document],
    draw_bboxes: bool = True,
    max_bbox_km: float | None = 3000.0,
    tiles: str = "OpenStreetMap",
) -> Any:
    """Build an interactive Folium map of resolved mentions.

    Markers are coloured by document language and grouped into one toggleable
    layer per language; bounding boxes are drawn in a separate layer. Very
    large boxes (countries) are skipped by default because they hide detail.

    Args:
        documents: Processed documents.
        draw_bboxes: Whether to draw feature extents.
        max_bbox_km: Skip boxes whose diagonal exceeds this; ``None`` = keep all.
        tiles: Folium tile set name.

    Returns:
        A ``folium.Map``.
    """
    import folium
    from folium.plugins import MarkerCluster

    from geoparser.geo_utils import bbox_diagonal_km

    docs = list(documents)
    fc = to_geojson(docs)
    features = fc["features"]
    if features:
        w, s, e, n = fc["bbox"]
        fmap = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=3, tiles=tiles, control_scale=True)
        fmap.fit_bounds([[s, w], [n, e]], padding=(30, 30))
    else:
        fmap = folium.Map(location=[20.0, 30.0], zoom_start=2, tiles=tiles)

    layers: dict[str, Any] = {}
    bbox_layer = folium.FeatureGroup(name="Bounding boxes", show=draw_bboxes)
    for feat in features:
        p = feat["properties"]
        lang = p.get("language") or "?"
        if lang not in layers:
            layers[lang] = MarkerCluster(name=f"Language: {lang}").add_to(fmap)
        lon, lat = feat["geometry"]["coordinates"]
        popup_html = (
            f"<b>{html.escape(p['mention'])}</b> → {html.escape(p['resolved_name'])}<br>"
            f"<i>{html.escape(str(p['place_type']))}</i> · {html.escape(str(p['country_code'] or ''))}"
            f" · conf {p['confidence']:.2f} · {p['n_candidates']} candidate(s)<br>"
            f"<small>{lat:.4f}, {lon:.4f} · {html.escape(p['doc_id'])} [{html.escape(lang)}]</small><br>"
            f"<small>…{html.escape(p['context'])}…</small>"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=360),
            tooltip=f"{p['mention']} ({lang})",
            icon=folium.Icon(color=LANGUAGE_COLORS.get(lang, "gray"), icon="map-marker", prefix="fa"),
        ).add_to(layers[lang])

        if draw_bboxes and "bbox" in feat:
            bw, bs, be, bn = feat["bbox"]
            if max_bbox_km is None or bbox_diagonal_km(bs, bn, bw, be) <= max_bbox_km:
                folium.Rectangle(
                    bounds=[[bs, bw], [bn, be]],
                    color=LANGUAGE_COLORS.get(lang, "gray"),
                    weight=1,
                    fill=True,
                    fill_opacity=0.08,
                    tooltip=f"Extent: {p['resolved_name'].split(',')[0]}",
                ).add_to(bbox_layer)

    bbox_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def write_map(documents: Iterable[Document], path: str | Path, **kwargs: Any) -> Path:
    """Build a map with :func:`build_map` and save it as standalone HTML.

    Args:
        documents: Processed documents.
        path: Destination ``.html`` file.
        **kwargs: Forwarded to :func:`build_map`.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_map(documents, **kwargs).save(str(path))
    logger.info("Wrote map to %s", path)
    return path
