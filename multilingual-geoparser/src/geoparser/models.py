"""Core data structures shared across the geo-parsing pipeline.

All coordinates are WGS84 (EPSG:4326). Internally we store ``lat``/``lon``
explicitly; GeoJSON export converts to the RFC 7946 ``[lon, lat]`` order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned geographic bounding box.

    Attributes:
        south: Minimum latitude.
        north: Maximum latitude.
        west: Minimum longitude.
        east: Maximum longitude.
    """

    south: float
    north: float
    west: float
    east: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.south <= self.north <= 90.0):
            raise ValueError(f"Invalid latitude range: {self.south}..{self.north}")
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError(f"Invalid longitude range: {self.west}..{self.east}")

    @classmethod
    def from_nominatim(cls, raw_bbox: list[Any]) -> BoundingBox:
        """Build a box from Nominatim's ``[south, north, west, east]`` strings.

        Args:
            raw_bbox: Four numeric values (often strings) in Nominatim order.

        Returns:
            A validated ``BoundingBox``.

        Raises:
            ValueError: If the list does not contain four parseable numbers.
        """
        if len(raw_bbox) != 4:
            raise ValueError(f"Expected 4 bbox values, got {len(raw_bbox)}")
        s, n, w, e = (float(v) for v in raw_bbox)
        return cls(south=s, north=n, west=w, east=e)

    def as_geojson_bbox(self) -> list[float]:
        """Return the RFC 7946 bbox ``[west, south, east, north]``."""
        return [self.west, self.south, self.east, self.north]

    def contains(self, lat: float, lon: float) -> bool:
        """Check whether a point lies inside the box.

        TODO: breaks for boxes crossing the 180th meridian (Fiji, Chukotka).
        """
        return self.south <= lat <= self.north and self.west <= lon <= self.east


@dataclass
class Entity:
    """A location mention recognised in text.

    Attributes:
        text: Surface form exactly as it appears in the document.
        start: Character offset (inclusive) in the source document.
        end: Character offset (exclusive) in the source document.
        label: Normalised entity label (``LOC``, ``GPE`` or ``FAC``).
        score: Recogniser confidence in ``[0, 1]``.
        source: Name of the backend that produced the entity.
    """

    text: str
    start: int
    end: int
    label: str = "LOC"
    score: float = 1.0
    source: str = "unknown"

    def overlaps(self, other: Entity) -> bool:
        """Return ``True`` if the two character spans intersect."""
        return self.start < other.end and other.start < self.end


@dataclass
class Candidate:
    """One gazetteer/geocoder candidate for a toponym.

    Attributes:
        name: Display name returned by the geocoder.
        lat: Latitude of the representative point.
        lon: Longitude of the representative point.
        bbox: Optional extent of the feature.
        place_type: Fine-grained type, e.g. ``city``, ``state``, ``river``.
        place_class: OSM class, e.g. ``place``, ``boundary``, ``natural``.
        country_code: ISO 3166-1 alpha-2 code (lower case) if known.
        importance: Nominatim importance in ``[0, 1]`` (Wikipedia-derived).
        population: Population if known (from OSM ``extratags``).
        osm_id: OSM identifier formatted as ``<type>/<id>``.
        raw: Unmodified provider payload for provenance.
    """

    name: str
    lat: float
    lon: float
    bbox: BoundingBox | None = None
    place_type: str = "unknown"
    place_class: str = "unknown"
    country_code: str | None = None
    importance: float = 0.0
    population: int | None = None
    osm_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ResolvedLocation:
    """A recognised entity linked to its chosen geographic referent.

    Attributes:
        entity: The originating text mention.
        candidate: The selected candidate, or ``None`` if unresolved.
        confidence: Disambiguation confidence in ``[0, 1]``.
        n_candidates: Number of candidates considered.
        score_breakdown: Per-feature scores for the selected candidate.
    """

    entity: Entity
    candidate: Candidate | None
    confidence: float = 0.0
    n_candidates: int = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Whether a candidate was selected."""
        return self.candidate is not None


@dataclass
class Document:
    """A single input text plus its pipeline outputs.

    Attributes:
        text: Raw document text.
        doc_id: Stable identifier (used in exports and evaluation).
        language: ISO 639-1 language code (detected or provided).
        entities: Recognised location entities.
        locations: Resolved locations (one per entity).
    """

    text: str
    doc_id: str = "doc-0"
    language: str | None = None
    entities: list[Entity] = field(default_factory=list)
    locations: list[ResolvedLocation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain Python types (drops raw provider payloads)."""
        data = asdict(self)
        for loc in data["locations"]:
            if loc["candidate"] is not None:
                loc["candidate"].pop("raw", None)
        return data
