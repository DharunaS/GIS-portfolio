"""Typed pipeline configuration, loadable from JSON and overridable from the CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from geoparser.disambiguation import DisambiguationConfig
from geoparser.exceptions import ConfigurationError
from geoparser.ner import DEFAULT_HF_MODELS


@dataclass
class PipelineConfig:
    """All settings needed to build a :class:`~geoparser.pipeline.GeoParser`.

    Attributes:
        ner_backend: ``transformers`` | ``spacy`` | ``stanza`` | ``gazetteer``.
        hf_models: Language → Hugging Face model id (transformers backend).
        ner_min_score: Minimum NER confidence.
        device: Torch device for transformers (``-1`` = CPU).
        gazetteer_path: Text file with one name per line (gazetteer backend);
            if unset, names are taken from the geocoder cache.
        user_agent: Nominatim User-Agent (falls back to ``GEOPARSER_USER_AGENT``).
        nominatim_domain: Nominatim host.
        candidate_limit: Candidates requested per toponym.
        cache_path: JSON geocoder cache.
        offline: Use only the cache (no network).
        disambiguation: Resolver hyper-parameters.
    """

    ner_backend: str = "transformers"
    hf_models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HF_MODELS))
    ner_min_score: float = 0.5
    device: int | str = -1
    gazetteer_path: str | None = None
    user_agent: str | None = None
    nominatim_domain: str = "nominatim.openstreetmap.org"
    candidate_limit: int = 5
    cache_path: str = ".cache/geocode_cache.json"
    offline: bool = False
    disambiguation: DisambiguationConfig = field(default_factory=DisambiguationConfig)

    def resolved_user_agent(self) -> str:
        """Return the configured or environment-provided user agent.

        Raises:
            ConfigurationError: If neither is set and the run is online.
        """
        ua = self.user_agent or os.environ.get("GEOPARSER_USER_AGENT")
        if not ua:
            raise ConfigurationError(
                "A Nominatim user agent is required for online geocoding. "
                "Pass --user-agent 'name/version (email)' or set GEOPARSER_USER_AGENT, "
                "or use --offline with a populated cache."
            )
        return ua

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a plain dict, rejecting unknown keys.

        Args:
            data: Mapping as found in a JSON config file.

        Returns:
            The config.

        Raises:
            ConfigurationError: On unknown keys or invalid nested values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"Unknown config keys: {sorted(unknown)}")
        data = dict(data)
        if "disambiguation" in data:
            d = dict(data["disambiguation"])
            d_known = {f.name for f in fields(DisambiguationConfig)}
            bad = set(d) - d_known
            if bad:
                raise ConfigurationError(f"Unknown disambiguation keys: {sorted(bad)}")
            if "country_hints" in d:
                d["country_hints"] = {c.lower() for c in d["country_hints"]}
            if "weights" in d:
                d["weights"] = {**DisambiguationConfig().weights, **d["weights"]}
            data["disambiguation"] = DisambiguationConfig(**d)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The config.

        Raises:
            ConfigurationError: If the file is unreadable or invalid.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot load config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config {path} must be a JSON object")
        return cls.from_dict(data)
