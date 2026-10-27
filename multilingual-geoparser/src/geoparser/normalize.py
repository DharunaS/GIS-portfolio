"""Turn entity surface forms into geocoder queries.

Tamil glues case endings onto names: "சென்னையில்" is "in Chennai". German
does it less ("Münchens"). Nominatim finds neither form, so for each entity
we try the surface form first and only fall back to a suffix-stripped
variant if that returns nothing. Stripping first would break names that
really do end in something suffix-shaped.

The Tamil rules are a short hand-written list, not a real lemmatiser.
TODO: compare against Stanza's Tamil lemmatiser.
"""

from __future__ import annotations

import re
import unicodedata

# (written suffix, replacement), longest first. Tamil vowel signs merge with
# the preceding consonant, so "-இல்" after a consonant is written "ில்" and the
# stem must regain its virama "்" (e.g. பெங்களூரில் → பெங்களூர்).
_TAMIL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("யிலிருந்து", ""),
    ("விலிருந்து", ""),
    ("ிலிருந்து", "்"),
    ("யிலுள்ள", ""),
    ("விலுள்ள", ""),
    ("ிலுள்ள", "்"),
    ("யின்", ""),
    ("வின்", ""),
    ("யில்", ""),
    ("வில்", ""),
    ("ில்", "்"),
    ("ுக்கு", "்"),
    ("க்கு", ""),
    ("யை", ""),
    ("வை", ""),
)

# Hindi postpositions are usually separate tokens; strip if NER absorbed them.
_HINDI_POSTPOSITIONS: tuple[str, ...] = (" में", " से", " का", " की", " के", " को", " तक", " पर")

_WS_RE = re.compile(r"\s+")


def clean_surface(text: str) -> str:
    """NFC-normalise, collapse whitespace and strip edge punctuation.

    Combining marks (Tamil/Devanagari vowel signs, virama) are preserved;
    only leading/trailing quote and bracket characters are removed.

    Args:
        text: Raw entity surface string.

    Returns:
        The cleaned string (may be empty).
    """
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text).strip()
    # \W matches combining marks, so only strip punctuation that is not a mark.
    while text and not text[0].isalnum() and unicodedata.category(text[0])[0] != "M":
        text = text[1:]
    while text and not text[-1].isalnum() and unicodedata.category(text[-1])[0] != "M":
        text = text[:-1]
    return text


def query_variants(text: str, lang: str | None) -> list[str]:
    """Generate ordered geocoding query variants for a toponym.

    Args:
        text: Entity surface form.
        lang: ISO 639-1 language code of the source document.

    Returns:
        De-duplicated list of query strings, most faithful first.
    """
    base = clean_surface(text)
    variants: list[str] = [base] if base else []

    if lang == "ta":
        for suffix, repl in _TAMIL_SUFFIXES:
            if base.endswith(suffix) and len(base) - len(suffix) >= 2:
                variants.append(base[: -len(suffix)] + repl)
                break
    elif lang == "hi":
        for post in _HINDI_POSTPOSITIONS:
            if base.endswith(post):
                variants.append(base[: -len(post)].strip())
                break
    elif lang == "de":
        if len(base) > 4 and base.endswith("s") and base[0].isupper() and " " not in base:
            variants.append(base[:-1])

    return [v for v in dict.fromkeys(variants) if v]


def name_key(text: str) -> str:
    """Case- and accent-insensitive comparison key for Latin script names.

    Non-Latin scripts are only NFC-normalised and case-folded, because
    stripping combining marks there would destroy vowel signs.

    Args:
        text: Name to normalise.

    Returns:
        Normalised comparison key.
    """
    text = clean_surface(text).casefold()
    decomposed = unicodedata.normalize("NFD", text)
    if all(ord(c) < 0x0250 or unicodedata.category(c) == "Mn" for c in decomposed):
        decomposed = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        return unicodedata.normalize("NFC", decomposed)
    return text
