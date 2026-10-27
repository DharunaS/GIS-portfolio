"""End-to-end orchestration: language ID → NER → candidates → resolution."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from geoparser.config import PipelineConfig
from geoparser.disambiguation import Disambiguator
from geoparser.exceptions import ConfigurationError, GeocodingError
from geoparser.geocoder import CachedGeocoder, Geocoder, NominatimGeocoder
from geoparser.language import SUPPORTED_LANGUAGES, detect_language
from geoparser.models import Candidate, Document, Entity
from geoparser.ner import GazetteerNER, NERBackend, build_ner_backend
from geoparser.normalize import query_variants

logger = logging.getLogger(__name__)


class GeoParser:
    """Multilingual geoparser.

    Args:
        ner: Location recogniser.
        geocoder: Candidate generator.
        disambiguator: Resolver; defaults to :class:`Disambiguator`.

    Example:
        >>> parser = GeoParser.from_config(PipelineConfig(offline=True))  # doctest: +SKIP
        >>> doc = parser.parse("Das DLR hat einen Standort in Oberpfaffenhofen.")  # doctest: +SKIP
    """

    def __init__(
        self,
        ner: NERBackend,
        geocoder: Geocoder,
        disambiguator: Disambiguator | None = None,
    ) -> None:
        self.ner = ner
        self.geocoder = geocoder
        self.disambiguator = disambiguator or Disambiguator()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> GeoParser:
        """Build all components from a :class:`PipelineConfig`.

        Args:
            config: Pipeline settings.

        Returns:
            A ready-to-use parser (models load lazily).

        Raises:
            ConfigurationError: If components cannot be configured.
        """
        inner: Geocoder | None = None
        if not config.offline:
            inner = NominatimGeocoder(
                user_agent=config.resolved_user_agent(),
                domain=config.nominatim_domain,
                limit=config.candidate_limit,
            )
        geocoder = CachedGeocoder(config.cache_path, inner=inner, offline=config.offline)

        if config.ner_backend == "gazetteer":
            names = _load_gazetteer_names(config.gazetteer_path, geocoder)
            ner: NERBackend = GazetteerNER(names)
        elif config.ner_backend == "transformers":
            ner = build_ner_backend(
                "transformers", models=config.hf_models, device=config.device, min_score=config.ner_min_score
            )
        else:
            ner = build_ner_backend(config.ner_backend)
        return cls(ner, geocoder, Disambiguator(config.disambiguation))

    def _candidates(self, entity: Entity, lang: str, memo: dict[str, list[Candidate]]) -> list[Candidate]:
        for variant in query_variants(entity.text, lang):
            if variant not in memo:
                memo[variant] = self.geocoder.search(variant, lang)
            if memo[variant]:
                return memo[variant]
        return []

    def parse(self, text: str, doc_id: str = "doc-0", lang: str | None = None) -> Document:
        """Geoparse a single document.

        Args:
            text: Document text.
            doc_id: Identifier carried into outputs.
            lang: ISO 639-1 code; auto-detected when ``None``.

        Returns:
            The processed :class:`Document`.

        Raises:
            ModelLoadError: If the NER model cannot be loaded.
            GeocodingError: On unrecoverable geocoder failures.
        """
        if lang is None:
            lang = detect_language(text)
        elif lang not in SUPPORTED_LANGUAGES:
            logger.warning("Language %r is not officially supported; continuing", lang)
        doc = Document(text=text, doc_id=doc_id, language=lang)
        if not text.strip():
            return doc

        doc.entities = self.ner.extract(text, lang)
        memo: dict[str, list[Candidate]] = {}
        candidates = [self._candidates(e, lang, memo) for e in doc.entities]
        doc.locations = self.disambiguator.resolve(doc.entities, candidates, lang)
        logger.info(
            "%s [%s]: %d entities, %d resolved",
            doc_id,
            lang,
            len(doc.entities),
            sum(loc.is_resolved for loc in doc.locations),
        )
        return doc

    def parse_many(self, records: Iterable[tuple[str, str, str | None]]) -> Iterator[Document]:
        """Geoparse ``(doc_id, text, lang)`` records lazily.

        A failure in one document is logged and yields an empty document, so a
        single bad record does not abort a batch run. Configuration and model
        errors still propagate.

        Args:
            records: Iterable of ``(doc_id, text, lang_or_None)``.

        Yields:
            Processed documents in input order.
        """
        for doc_id, text, lang in records:
            try:
                yield self.parse(text, doc_id=doc_id, lang=lang)
            except GeocodingError:
                raise
            except (ValueError, KeyError, RuntimeError) as exc:
                logger.error("Failed to process %s: %s", doc_id, exc)
                yield Document(text=text, doc_id=doc_id, language=lang)

    def close(self) -> None:
        """Flush caches to disk."""
        if isinstance(self.geocoder, CachedGeocoder):
            self.geocoder.save()


def _load_gazetteer_names(path: str | None, geocoder: CachedGeocoder) -> list[str]:
    if path:
        try:
            return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError as exc:
            raise ConfigurationError(f"Cannot read gazetteer {path}: {exc}") from exc
    # Fall back to OSM names (supported languages only) of the cached candidates.
    keys = {"name", "short_name"} | {f"{p}:{lang}" for p in ("name", "alt_name") for lang in SUPPORTED_LANGUAGES}
    names: set[str] = set()
    for entries in geocoder.entries().values():
        for e in entries:
            nd = (e.get("raw") or {}).get("namedetails") or {}
            names.update(str(v) for k, v in nd.items() if k in keys and len(str(v)) >= 3)
    if not names:
        raise ConfigurationError("Gazetteer backend needs --gazetteer or a populated geocoder cache")
    return sorted(names)


def read_input_records(path: str | Path) -> list[tuple[str, str, str | None]]:
    """Read documents from ``.txt`` (one doc per non-empty line) or ``.jsonl``.

    JSONL records need ``text`` and may carry ``id`` and ``lang``.

    Args:
        path: Input file.

    Returns:
        ``(doc_id, text, lang)`` tuples.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"Cannot read input {path}: {exc}") from exc

    records: list[tuple[str, str, str | None]] = []
    if path.suffix == ".jsonl":
        for i, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                records.append((str(rec.get("id", f"{path.stem}-{i}")), rec["text"], rec.get("lang")))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ConfigurationError(f"{path}:{i}: invalid JSONL record ({exc})") from exc
    else:
        for i, line in enumerate((ln for ln in content.splitlines() if ln.strip()), 1):
            records.append((f"{path.stem}-{i}", line.strip(), None))
    return records
