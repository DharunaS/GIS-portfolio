"""Exception hierarchy for the geo-parsing pipeline.

Catching :class:`GeoParserError` handles every failure raised deliberately by
this package, while letting genuine programming errors propagate.
"""


class GeoParserError(Exception):
    """Base class for all package-specific errors."""


class ConfigurationError(GeoParserError):
    """Raised when configuration values are missing or inconsistent."""


class ModelLoadError(GeoParserError):
    """Raised when an NER backend or model cannot be loaded."""


class GeocodingError(GeoParserError):
    """Raised when the geocoding service fails irrecoverably."""


class EvaluationError(GeoParserError):
    """Raised when gold data is malformed or incompatible with predictions."""
