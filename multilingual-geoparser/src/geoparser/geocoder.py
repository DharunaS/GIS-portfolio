"""Nominatim candidate lookup plus a JSON cache.

The public Nominatim API allows 1 request/s and wants an identifying
User-Agent, so both are enforced here rather than trusted to the caller.

The cache matters more than it looks: Nominatim's data changes daily, so
without a frozen cache the same experiment gives different numbers a month
later. offline=True reads only the cache, which is how the tests run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from geoparser.exceptions import ConfigurationError, GeocodingError
from geoparser.models import BoundingBox, Candidate

logger = logging.getLogger(__name__)

_PLACEHOLDER_AGENTS = {"", "geopy", "my-app", "geoparser"}


class Geocoder(ABC):
    """Interface: map a query string to ranked candidate places."""

    @abstractmethod
    def search(self, query: str, lang: str | None = None) -> list[Candidate]:
        """Return candidate referents for ``query``.

        Args:
            query: Toponym surface form (any script).
            lang: Document language, used as a hint by some providers.

        Returns:
            Candidates, possibly empty. Implementations must not raise for
            "not found"; they raise :class:`GeocodingError` only for
            unrecoverable failures (auth, policy violations).
        """


def candidate_from_nominatim(raw: dict[str, Any]) -> Candidate:
    """Convert a raw Nominatim JSON record into a :class:`Candidate`.

    Args:
        raw: One element of Nominatim's ``/search`` JSON response.

    Returns:
        The parsed candidate.

    Raises:
        ValueError: If mandatory coordinates are missing.
    """
    bbox: BoundingBox | None = None
    if raw.get("boundingbox"):
        try:
            bbox = BoundingBox.from_nominatim(raw["boundingbox"])
        except ValueError as exc:
            logger.debug("Ignoring invalid bbox %s: %s", raw.get("boundingbox"), exc)

    population: int | None = None
    pop_raw = (raw.get("extratags") or {}).get("population")
    if pop_raw:
        try:
            population = int(str(pop_raw).replace(",", "").replace(" ", "").split(";")[0])
        except ValueError:
            population = None

    osm_type, osm_id = raw.get("osm_type"), raw.get("osm_id")
    return Candidate(
        name=str(raw.get("display_name") or raw.get("name") or ""),
        lat=float(raw["lat"]),
        lon=float(raw["lon"]),
        bbox=bbox,
        place_type=str(raw.get("addresstype") or raw.get("type") or "unknown"),
        place_class=str(raw.get("class") or raw.get("category") or "unknown"),
        country_code=((raw.get("address") or {}).get("country_code") or None),
        importance=float(raw.get("importance") or 0.0),
        population=population,
        osm_id=f"{osm_type}/{osm_id}" if osm_type and osm_id else None,
        raw=raw,
    )


class NominatimGeocoder(Geocoder):
    """Nominatim (OpenStreetMap) candidate generator.

    Args:
        user_agent: Contact-identifying UA string, e.g.
            ``"multilingual-geoparser/1.0 (you@example.org)"``.
        domain: Nominatim host; point it at a self-hosted instance for bulk use.
        limit: Maximum candidates per query.
        timeout: HTTP timeout in seconds.
        min_delay_seconds: Delay between calls (policy: ≥ 1 s on the public API).
        max_retries: Retries on transient errors.
        result_language: ``Accept-Language`` for display names.

    Raises:
        ConfigurationError: If ``user_agent`` is a placeholder.
    """

    def __init__(
        self,
        user_agent: str,
        domain: str = "nominatim.openstreetmap.org",
        limit: int = 5,
        timeout: float = 10.0,
        min_delay_seconds: float = 1.0,
        max_retries: int = 2,
        result_language: str = "en",
    ) -> None:
        if user_agent.strip().lower() in _PLACEHOLDER_AGENTS:
            raise ConfigurationError(
                "Nominatim requires an identifying user agent, e.g. "
                "'multilingual-geoparser/1.0 (you@example.org)'. Set --user-agent "
                "or GEOPARSER_USER_AGENT."
            )
        if domain == "nominatim.openstreetmap.org" and min_delay_seconds < 1.0:
            raise ConfigurationError("The public Nominatim API allows at most 1 request/second")
        try:
            from geopy.extra.rate_limiter import RateLimiter
            from geopy.geocoders import Nominatim
        except ImportError as exc:  # pragma: no cover - geopy is a core dependency
            raise ConfigurationError("geopy is required for NominatimGeocoder") from exc

        self.limit = limit
        self.result_language = result_language
        self._client = Nominatim(user_agent=user_agent, domain=domain, timeout=timeout)
        self._geocode = RateLimiter(
            self._client.geocode,
            min_delay_seconds=min_delay_seconds,
            max_retries=max_retries,
            error_wait_seconds=max(2.0, min_delay_seconds * 2),
            swallow_exceptions=False,
        )

    def search(self, query: str, lang: str | None = None) -> list[Candidate]:
        """See :meth:`Geocoder.search`."""
        from geopy.exc import (
            GeocoderAuthenticationFailure,
            GeocoderInsufficientPrivileges,
            GeopyError,
        )

        if not query.strip():
            return []
        try:
            results = self._geocode(
                query,
                exactly_one=False,
                limit=self.limit,
                addressdetails=True,
                extratags=True,
                namedetails=True,
                language=self.result_language,
            )
        except (GeocoderAuthenticationFailure, GeocoderInsufficientPrivileges) as exc:
            raise GeocodingError(f"Nominatim rejected the request (policy/auth): {exc}") from exc
        except GeopyError as exc:
            logger.warning("Geocoding failed for %r after retries: %s", query, exc)
            return []

        candidates: list[Candidate] = []
        for loc in results or []:
            try:
                candidates.append(candidate_from_nominatim(loc.raw))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping malformed Nominatim record: %s", exc)
        return candidates


def candidate_to_dict(c: Candidate) -> dict[str, Any]:
    """Serialise a candidate (including ``raw``) for the cache."""
    return asdict(c)


def candidate_from_dict(d: dict[str, Any]) -> Candidate:
    """Inverse of :func:`candidate_to_dict`."""
    d = dict(d)
    if d.get("bbox"):
        d["bbox"] = BoundingBox(**d["bbox"])
    return Candidate(**d)


class CachedGeocoder(Geocoder):
    """JSON-file cache in front of another geocoder.

    The cache key is the *query string* only; results do not depend on the
    document language, and sharing entries across languages saves requests.

    Args:
        cache_path: JSON file to read/write. Created on first save.
        inner: Wrapped geocoder; may be ``None`` when ``offline`` is set.
        offline: Never call ``inner``; cache misses return no candidates.
        autosave_every: Persist after this many new entries.
    """

    def __init__(
        self,
        cache_path: str | os.PathLike[str],
        inner: Geocoder | None = None,
        offline: bool = False,
        autosave_every: int = 20,
    ) -> None:
        if inner is None and not offline:
            raise ConfigurationError("CachedGeocoder needs an inner geocoder unless offline=True")
        self.cache_path = Path(cache_path)
        self.inner = inner
        self.offline = offline
        self.autosave_every = autosave_every
        self._lock = threading.Lock()
        self._dirty = 0
        self.hits = 0
        self.misses = 0
        self._cache: dict[str, list[dict[str, Any]]] = self._read()

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise GeocodingError(f"Cannot read geocoder cache {self.cache_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise GeocodingError(f"Cache {self.cache_path} must contain a JSON object")
        return data

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.split()).casefold()

    def search(self, query: str, lang: str | None = None) -> list[Candidate]:
        """Serve from cache; on a miss query ``inner`` (unless offline) and store."""
        key = self._key(query)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return [candidate_from_dict(c) for c in cached]

        self.misses += 1
        if self.offline or self.inner is None:
            logger.debug("Offline cache miss for %r", query)
            return []

        results = self.inner.search(query, lang)
        with self._lock:
            self._cache[key] = [candidate_to_dict(c) for c in results]
            self._dirty += 1
            should_save = self._dirty >= self.autosave_every
        if should_save:
            self.save()
        return results

    def entries(self) -> dict[str, list[dict[str, Any]]]:
        """Return a shallow copy of the cache contents (query → candidates)."""
        with self._lock:
            return dict(self._cache)

    def save(self) -> None:
        """Atomically write the cache to disk (no-op when nothing changed)."""
        with self._lock:
            if not self._dirty:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.cache_path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._cache, fh, ensure_ascii=False, indent=1)
                os.replace(tmp, self.cache_path)
            except OSError as exc:
                Path(tmp).unlink(missing_ok=True)
                raise GeocodingError(f"Cannot write cache {self.cache_path}: {exc}") from exc
            self._dirty = 0

    def __enter__(self) -> CachedGeocoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.save()
