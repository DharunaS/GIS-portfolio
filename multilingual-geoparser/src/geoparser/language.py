"""Decide which language a document is in, to pick the NER model.

Tamil and Hindi are easy: count characters in the Tamil (U+0B80-U+0BFF) and
Devanagari (U+0900-U+097F) blocks. For Latin script use langdetect, falling
back to counting German vs English stop words. That is all the routing needs.

Known gap: Marathi and Nepali are also Devanagari and get routed as Hindi.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "de", "ta", "hi"})

_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "ta": (0x0B80, 0x0BFF),
    "hi": (0x0900, 0x097F),  # Devanagari; Marathi/Nepali also use it, routed as Hindi.
}

_DE_MARKERS = {
    "der",
    "die",
    "das",
    "und",
    "ist",
    "nicht",
    "mit",
    "von",
    "zu",
    "den",
    "im",
    "auf",
    "für",
    "ein",
    "eine",
    "nach",
    "bei",
    "wurde",
    "sich",
    "auch",
}
_EN_MARKERS = {
    "the",
    "and",
    "is",
    "of",
    "to",
    "in",
    "with",
    "for",
    "on",
    "was",
    "from",
    "by",
    "at",
    "an",
    "a",
    "near",
    "were",
    "this",
    "that",
    "it",
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_share(text: str, lo: int, hi: int) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def detect_language(text: str, default: str = "en") -> str:
    """Detect the language of ``text`` among the supported languages.

    Args:
        text: Input document.
        default: Code returned when the text is empty or ambiguous.

    Returns:
        An ISO 639-1 code from :data:`SUPPORTED_LANGUAGES`. Unsupported
        languages detected by ``langdetect`` are mapped to ``default``.
    """
    if not text or not text.strip():
        return default

    for lang, (lo, hi) in _SCRIPT_RANGES.items():
        if _script_share(text, lo, hi) >= 0.3:
            return lang

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic results
        lang = detect(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        logger.debug("langdetect returned unsupported language %r; using heuristic", lang)
    except ImportError:
        logger.debug("langdetect not installed; using stop-word heuristic")
    except Exception as exc:  # langdetect raises LangDetectException on noise
        logger.debug("langdetect failed (%s); using stop-word heuristic", exc)

    words = [w.lower() for w in _WORD_RE.findall(text)]
    de_hits = sum(w in _DE_MARKERS for w in words) + 2 * sum(c in "äöüß" for c in text.lower())
    en_hits = sum(w in _EN_MARKERS for w in words)
    if de_hits > en_hits:
        return "de"
    if en_hits > 0:
        return "en"
    return default
