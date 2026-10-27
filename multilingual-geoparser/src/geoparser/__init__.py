"""Multilingual geoparsing: NER, geocoding, spatial disambiguation and evaluation."""

__version__ = "1.0.0"

from geoparser.config import PipelineConfig
from geoparser.disambiguation import DisambiguationConfig, Disambiguator
from geoparser.models import BoundingBox, Candidate, Document, Entity, ResolvedLocation
from geoparser.pipeline import GeoParser

__all__ = [
    "BoundingBox",
    "Candidate",
    "DisambiguationConfig",
    "Disambiguator",
    "Document",
    "Entity",
    "GeoParser",
    "PipelineConfig",
    "ResolvedLocation",
    "__version__",
]
