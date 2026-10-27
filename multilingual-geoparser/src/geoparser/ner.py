"""Location NER backends.

All backends return Entity spans with document-level offsets and a label in
{LOC, GPE, FAC}, so everything downstream can ignore which one ran.

- TransformersNER: Hugging Face token classification, routed per language.
  WikiNEuRal for en/de, IndicNER for hi/ta (WikiNEuRal has no Indic data).
- SpacyNER / StanzaNER: mostly for comparison. Stanza has no hi/ta NER.
- GazetteerNER: plain dictionary matching. Offline and deterministic, so the
  tests and CI use it, and it doubles as a baseline.

transformers/spacy/stanza are imported inside methods so the package and its
tests work without them installed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from geoparser.exceptions import ConfigurationError, ModelLoadError
from geoparser.models import Entity
from geoparser.normalize import clean_surface

logger = logging.getLogger(__name__)

#: Raw label → normalised location label. Anything not listed is dropped.
LOCATION_LABEL_MAP: dict[str, str] = {
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "GPE",
    "FAC": "FAC",
    "FACILITY": "FAC",
    # Some Indic datasets use these spellings.
    "PLACE": "LOC",
}

#: Default Hugging Face model per language.
DEFAULT_HF_MODELS: dict[str, str] = {
    "en": "Babelscape/wikineural-multilingual-ner",
    "de": "Babelscape/wikineural-multilingual-ner",
    "hi": "ai4bharat/IndicNER",
    "ta": "ai4bharat/IndicNER",
}

DEFAULT_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "hi": "xx_ent_wiki_sm",
    "ta": "xx_ent_wiki_sm",
}

# Sentence boundaries including the Devanagari danda (।, ॥).
_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")


def normalise_label(raw_label: str) -> str | None:
    """Map a backend-specific label to ``LOC``/``GPE``/``FAC``.

    Strips BIO/BILOU prefixes (``B-``, ``I-``, ``S-`` …) before lookup.

    Args:
        raw_label: Label emitted by a model, e.g. ``"B-LOC"`` or ``"GPE"``.

    Returns:
        The normalised label, or ``None`` if it is not a location type.
    """
    label = raw_label.upper()
    if len(label) > 2 and label[1] in "-_" and label[0] in "BIESLU":
        label = label[2:]
    return LOCATION_LABEL_MAP.get(label)


def chunk_text(text: str, max_chars: int = 1500) -> Iterator[tuple[int, str]]:
    """Split ``text`` into sentence-aligned chunks no longer than ``max_chars``.

    Transformer encoders have a hard token limit (512 for XLM-R). Chunking on
    sentence boundaries avoids truncating entities mid-span.

    Args:
        text: The full document.
        max_chars: Soft character budget per chunk; a single longer sentence
            is hard-split at whitespace.

    Yields:
        ``(offset, chunk)`` pairs where ``offset`` is the chunk's start index.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    # Sentence spans (start, end) in document coordinates.
    sentences: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENT_END_RE.finditer(text):
        sentences.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        sentences.append((cursor, len(text)))

    # Hard-split sentences that exceed the budget on their own.
    pieces: list[tuple[int, int]] = []
    for s, e in sentences:
        while e - s > max_chars:
            cut = text.rfind(" ", s + 1, s + max_chars)
            cut = cut if cut > s else s + max_chars
            pieces.append((s, cut))
            s = cut + 1 if text[cut : cut + 1] == " " else cut
        if e > s:
            pieces.append((s, e))

    chunk_start: int | None = None
    chunk_end = 0
    for s, e in pieces:
        if chunk_start is None:
            chunk_start, chunk_end = s, e
        elif e - chunk_start <= max_chars:
            chunk_end = e
        else:
            yield chunk_start, text[chunk_start:chunk_end]
            chunk_start, chunk_end = s, e
    if chunk_start is not None:
        yield chunk_start, text[chunk_start:chunk_end]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` so it begins and ends on a non-punctuation char."""

    def is_core(ch: str) -> bool:
        return ch.isalnum() or unicodedata.category(ch)[0] == "M"

    while start < end and not is_core(text[start]):
        start += 1
    while end > start and not is_core(text[end - 1]):
        end -= 1
    return start, end


def postprocess_entities(text: str, entities: Iterable[Entity], min_score: float = 0.0) -> list[Entity]:
    """Clean, merge and de-duplicate raw entity spans.

    Steps:
        1. Trim punctuation at span edges and re-sync ``text`` with offsets.
        2. Merge same-label spans that touch or are joined by a hyphen. Sub-word
           tokenisers split Indic words (``"दिल्" + "ली"``). Spans separated by a
           space are NOT merged, otherwise "Berlin Hamburg" becomes one place.
        3. Drop spans below ``min_score`` and resolve overlaps (keep highest).

    Args:
        text: The document the offsets refer to.
        entities: Raw entities from a backend.
        min_score: Minimum confidence to keep an entity.

    Returns:
        Sorted, non-overlapping entities.
    """
    trimmed: list[Entity] = []
    for ent in sorted(entities, key=lambda e: (e.start, -e.end)):
        s, e = _trim_span(text, ent.start, ent.end)
        if e <= s:
            continue
        trimmed.append(Entity(text[s:e], s, e, ent.label, ent.score, ent.source))

    merged: list[Entity] = []
    for ent in trimmed:
        if merged:
            prev = merged[-1]
            gap = text[prev.end : ent.start]
            # Merge only fragments of one word ("दिल्" + "ली") or hyphen-joined
            # parts. Whitespace-separated spans stay separate, so lists like
            # "Berlin Hamburg" are not collapsed into a single toponym.
            if prev.label == ent.label and ent.start >= prev.end and gap in ("", "-"):
                merged[-1] = Entity(
                    text[prev.start : ent.end],
                    prev.start,
                    ent.end,
                    prev.label,
                    min(prev.score, ent.score),
                    prev.source,
                )
                continue
        merged.append(ent)

    kept: list[Entity] = []
    for ent in sorted((e for e in merged if e.score >= min_score), key=lambda e: -e.score):
        if not any(ent.overlaps(k) for k in kept):
            kept.append(ent)
    return sorted(kept, key=lambda e: e.start)


class NERBackend(ABC):
    """Interface for all location recognisers."""

    name: str = "base"

    @abstractmethod
    def extract(self, text: str, lang: str) -> list[Entity]:
        """Recognise location entities.

        Args:
            text: Document text.
            lang: ISO 639-1 language code.

        Returns:
            Location entities with document-level offsets.
        """


class TransformersNER(NERBackend):
    """Hugging Face token-classification NER with per-language model routing.

    Args:
        models: Mapping ``lang → model id``. Missing languages fall back to
            ``models["default"]`` if present.
        device: ``-1`` for CPU, ``0..n`` for a CUDA device, or ``"mps"``.
        min_score: Confidence threshold applied after aggregation.
        max_chars: Chunk size for long documents.
        batch_size: Chunks per forward pass.
    """

    name = "transformers"

    def __init__(
        self,
        models: dict[str, str] | None = None,
        device: int | str = -1,
        min_score: float = 0.5,
        max_chars: int = 1500,
        batch_size: int = 8,
    ) -> None:
        self.models = dict(models or DEFAULT_HF_MODELS)
        self.device = device
        self.min_score = min_score
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._pipelines: dict[str, Any] = {}

    def _model_for(self, lang: str) -> str:
        model_id = self.models.get(lang) or self.models.get("default")
        if not model_id:
            raise ConfigurationError(
                f"No Transformers NER model configured for language {lang!r}; configured: {sorted(self.models)}"
            )
        return model_id

    def _load(self, model_id: str) -> Any:
        if model_id in self._pipelines:
            return self._pipelines[model_id]
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ModelLoadError("transformers is not installed. Run `pip install -e .[ml]`.") from exc
        logger.info("Loading Hugging Face model %s", model_id)
        try:
            nlp = pipeline(
                "token-classification",
                model=model_id,
                tokenizer=model_id,
                aggregation_strategy="simple",
                device=self.device,
            )
        except Exception as exc:  # network, auth, corrupt cache, OOM …
            raise ModelLoadError(f"Could not load model {model_id!r}: {exc}") from exc
        self._pipelines[model_id] = nlp
        return nlp

    def extract(self, text: str, lang: str) -> list[Entity]:
        """See :meth:`NERBackend.extract`."""
        if not text.strip():
            return []
        model_id = self._model_for(lang)
        nlp = self._load(model_id)
        chunks = list(chunk_text(text, self.max_chars))
        outputs = nlp([c for _, c in chunks], batch_size=self.batch_size)
        if chunks and outputs and isinstance(outputs[0], dict):  # single input quirk
            outputs = [outputs]

        raw: list[Entity] = []
        for (offset, _), preds in zip(chunks, outputs, strict=True):
            for p in preds:
                label = normalise_label(str(p.get("entity_group") or p.get("entity", "")))
                if label is None:
                    continue
                s, e = offset + int(p["start"]), offset + int(p["end"])
                raw.append(Entity(text[s:e], s, e, label, float(p["score"]), f"hf:{model_id}"))
        return postprocess_entities(text, raw, self.min_score)


class SpacyNER(NERBackend):
    """spaCy NER backend.

    Args:
        models: Mapping ``lang → spaCy package``.
    """

    name = "spacy"

    def __init__(self, models: dict[str, str] | None = None) -> None:
        self.models = dict(models or DEFAULT_SPACY_MODELS)
        self._nlp: dict[str, Any] = {}

    def _load(self, lang: str) -> Any:
        model = self.models.get(lang) or self.models.get("default")
        if not model:
            raise ConfigurationError(f"No spaCy model configured for language {lang!r}")
        if model not in self._nlp:
            try:
                import spacy

                self._nlp[model] = spacy.load(model, disable=["lemmatizer", "parser"])
            except ImportError as exc:
                raise ModelLoadError("spaCy is not installed. Run `pip install -e .[nlp]`.") from exc
            except OSError as exc:
                raise ModelLoadError(
                    f"spaCy model {model!r} not found. Run `python -m spacy download {model}`."
                ) from exc
        return self._nlp[model]

    def extract(self, text: str, lang: str) -> list[Entity]:
        """See :meth:`NERBackend.extract`."""
        if not text.strip():
            return []
        doc = self._load(lang)(text)
        raw = [
            Entity(ent.text, ent.start_char, ent.end_char, label, 1.0, "spacy")
            for ent in doc.ents
            if (label := normalise_label(ent.label_)) is not None
        ]
        return postprocess_entities(text, raw)


class StanzaNER(NERBackend):
    """Stanza NER backend (official NER models exist for ``en`` and ``de``).

    Args:
        use_gpu: Whether Stanza may use CUDA.
    """

    name = "stanza"

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self._nlp: dict[str, Any] = {}

    def _load(self, lang: str) -> Any:
        if lang not in self._nlp:
            try:
                import stanza
            except ImportError as exc:
                raise ModelLoadError("stanza is not installed. Run `pip install -e .[nlp]`.") from exc
            try:
                self._nlp[lang] = stanza.Pipeline(lang, processors="tokenize,ner", use_gpu=self.use_gpu, verbose=False)
            except Exception as exc:
                raise ModelLoadError(
                    f"Stanza has no usable NER pipeline for {lang!r} ({exc}). "
                    "Use the transformers backend for Hindi/Tamil."
                ) from exc
        return self._nlp[lang]

    def extract(self, text: str, lang: str) -> list[Entity]:
        """See :meth:`NERBackend.extract`."""
        if not text.strip():
            return []
        doc = self._load(lang)(text)
        raw = [
            Entity(ent.text, ent.start_char, ent.end_char, label, 1.0, "stanza")
            for ent in doc.ents
            if (label := normalise_label(ent.type)) is not None
        ]
        return postprocess_entities(text, raw)


class GazetteerNER(NERBackend):
    """Longest-match dictionary recogniser (offline, deterministic baseline).

    Word boundaries are Unicode-aware. For agglutinative languages (Tamil) a
    name may be followed by case suffixes, so only the left boundary is
    enforced and the span is extended to the end of the word; downstream
    :func:`geoparser.normalize.query_variants` strips the suffix again.

    Args:
        names: Iterable of place names (any script).
        suffixing_languages: Languages for which right-boundary checks are
            relaxed.
    """

    name = "gazetteer"

    def __init__(self, names: Iterable[str], suffixing_languages: Iterable[str] = ("ta",)) -> None:
        cleaned = {clean_surface(n) for n in names}
        cleaned.discard("")
        if not cleaned:
            raise ConfigurationError("GazetteerNER needs at least one name")
        self.suffixing_languages = frozenset(suffixing_languages)
        # Tamil stems ending in a virama (்) lose it when a vowel suffix
        # attaches (பெங்களூர் + இல் → பெங்களூரில்), so also match the bare stem.
        cleaned |= {n[:-1] for n in cleaned if n.endswith("்") and len(n) > 3}
        alternation = "|".join(re.escape(n) for n in sorted(cleaned, key=len, reverse=True))
        self._pattern = re.compile(f"(?:{alternation})")

    @staticmethod
    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or unicodedata.category(ch)[0] == "M"

    def extract(self, text: str, lang: str) -> list[Entity]:
        """See :meth:`NERBackend.extract`."""
        out: list[Entity] = []
        for m in self._pattern.finditer(text):
            s, e = m.start(), m.end()
            if s > 0 and self._is_word_char(text[s - 1]):
                continue
            if e < len(text) and self._is_word_char(text[e]):
                if lang not in self.suffixing_languages:
                    continue
                while e < len(text) and self._is_word_char(text[e]):
                    e += 1
            out.append(Entity(text[s:e], s, e, "LOC", 1.0, "gazetteer"))
        return postprocess_entities(text, out)


def build_ner_backend(name: str, **kwargs: Any) -> NERBackend:
    """Instantiate an NER backend by name.

    Args:
        name: One of ``transformers``, ``spacy``, ``stanza``, ``gazetteer``.
        **kwargs: Passed to the backend constructor.

    Returns:
        An initialised backend (models load lazily on first use).

    Raises:
        ConfigurationError: If ``name`` is unknown.
    """
    registry: dict[str, type[NERBackend]] = {
        "transformers": TransformersNER,
        "spacy": SpacyNER,
        "stanza": StanzaNER,
        "gazetteer": GazetteerNER,
    }
    try:
        cls = registry[name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown NER backend {name!r}; choose from {sorted(registry)}") from exc
    return cls(**kwargs)
