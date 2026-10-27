"""Language identification, normalisation and NER post-processing."""

from __future__ import annotations

import pytest

from geoparser.exceptions import ConfigurationError
from geoparser.language import detect_language
from geoparser.models import Entity
from geoparser.ner import (
    GazetteerNER,
    TransformersNER,
    build_ner_backend,
    chunk_text,
    normalise_label,
    postprocess_entities,
)
from geoparser.normalize import clean_surface, name_key, query_variants


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("சென்னையில் கனமழை பெய்தது.", "ta"),
        ("मुंबई में भारी बारिश हुई।", "hi"),
        ("Das Hochwasser an der Oder erreichte Frankfurt.", "de"),
        ("The storm passed over Paris, Texas.", "en"),
        ("", "en"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


class TestNormalize:
    def test_tamil_locative_after_glide(self) -> None:
        assert query_variants("சென்னையில்", "ta") == ["சென்னையில்", "சென்னை"]

    def test_tamil_locative_restores_virama(self) -> None:
        assert query_variants("பெங்களூரில்", "ta")[-1] == "பெங்களூர்"

    def test_tamil_ablative(self) -> None:
        assert query_variants("ஸ்ரீஹரிகோட்டாவிலிருந்து", "ta")[-1] == "ஸ்ரீஹரிகோட்டா"

    def test_tamil_clean_stem_unchanged(self) -> None:
        assert query_variants("மதுரை", "ta") == ["மதுரை"]

    def test_german_genitive(self) -> None:
        assert query_variants("Münchens", "de") == ["Münchens", "München"]

    def test_hindi_absorbed_postposition(self) -> None:
        assert query_variants("मुंबई में", "hi") == ["मुंबई में", "मुंबई"]

    def test_clean_surface_keeps_vowel_signs(self) -> None:
        assert clean_surface(' "मदुरै", ') == "मदुरै"
        assert clean_surface("(சென்னை)") == "சென்னை"

    def test_name_key_strips_latin_accents_only(self) -> None:
        assert name_key("Köln") == name_key("koln")
        assert name_key("மதுரை") == "மதுரை"


class TestLabels:
    @pytest.mark.parametrize("raw", ["LOC", "B-LOC", "I-LOC", "GPE", "S-GPE", "FAC", "location"])
    def test_location_labels_kept(self, raw: str) -> None:
        assert normalise_label(raw) in {"LOC", "GPE", "FAC"}

    @pytest.mark.parametrize("raw", ["PER", "B-ORG", "MISC", "O"])
    def test_other_labels_dropped(self, raw: str) -> None:
        assert normalise_label(raw) is None


class TestChunking:
    def test_offsets_are_exact(self) -> None:
        text = "Berlin is big. दिल्ली बड़ा है। " + "word " * 200
        for offset, chunk in chunk_text(text, max_chars=64):
            assert text[offset : offset + len(chunk)] == chunk
            assert len(chunk) <= 64

    def test_short_text_single_chunk(self) -> None:
        assert list(chunk_text("Köln.", 100)) == [(0, "Köln.")]

    def test_invalid_budget(self) -> None:
        with pytest.raises(ValueError):
            list(chunk_text("x", 0))


class TestPostprocess:
    def test_merges_subword_fragments(self) -> None:
        text = "दिल्ली में"
        raw = [Entity("दिल्", 0, 4, "LOC", 0.9), Entity("ली", 4, 6, "LOC", 0.8)]
        out = postprocess_entities(text, raw)
        assert [(e.text, e.start, e.end) for e in out] == [("दिल्ली", 0, 6)]
        assert out[0].score == pytest.approx(0.8)

    def test_keeps_space_separated_spans_apart(self) -> None:
        text = "Berlin Hamburg"
        raw = [Entity("Berlin", 0, 6), Entity("Hamburg", 7, 14)]
        assert len(postprocess_entities(text, raw)) == 2

    def test_trims_punctuation_and_filters_score(self) -> None:
        text = 'in "Köln", and Bonn'
        raw = [Entity('"Köln",', 3, 10, "LOC", 0.9), Entity("Bonn", 15, 19, "LOC", 0.2)]
        out = postprocess_entities(text, raw, min_score=0.5)
        assert [(e.text, e.start) for e in out] == [("Köln", 4)]

    def test_overlaps_keep_highest_score(self) -> None:
        text = "New York City"
        raw = [Entity("New York", 0, 8, "GPE", 0.7), Entity("New York City", 0, 13, "GPE", 0.95)]
        assert [e.text for e in postprocess_entities(text, raw)] == ["New York City"]


class TestGazetteer:
    def test_word_boundaries_latin(self) -> None:
        ner = GazetteerNER(["Paris", "Texas"])
        ents = ner.extract("Parisian food in Paris, Texas.", "en")
        assert [(e.text, e.start) for e in ents] == [("Paris", 17), ("Texas", 24)]

    def test_tamil_suffix_extension_and_virama_stem(self) -> None:
        ner = GazetteerNER(["சென்னை", "பெங்களூர்"])
        ents = ner.extract("சென்னையில் இருந்து பெங்களூரில் வரை", "ta")
        assert [e.text for e in ents] == ["சென்னையில்", "பெங்களூரில்"]

    def test_longest_match_wins(self) -> None:
        ner = GazetteerNER(["Frankfurt", "Frankfurt am Main"])
        assert [e.text for e in ner.extract("in Frankfurt am Main", "de")] == ["Frankfurt am Main"]

    def test_empty_gazetteer_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            GazetteerNER(["", "  "])


class TestTransformersNER:
    def test_maps_chunk_offsets_to_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = "We met in Köln. Later we flew to Chennai."

        def fake_pipeline(chunks: list[str], batch_size: int = 8) -> list[list[dict]]:
            out = []
            for chunk in chunks:
                preds = []
                for name in ("Köln", "Chennai"):
                    i = chunk.find(name)
                    if i >= 0:
                        preds.append({"entity_group": "LOC", "start": i, "end": i + len(name), "score": 0.98})
                preds.append({"entity_group": "PER", "start": 0, "end": 2, "score": 0.99})
                out.append(preds)
            return out

        ner = TransformersNER(max_chars=20)
        monkeypatch.setattr(ner, "_load", lambda model_id: fake_pipeline)
        ents = ner.extract(text, "en")
        assert [(e.text, text[e.start : e.end]) for e in ents] == [("Köln", "Köln"), ("Chennai", "Chennai")]
        assert all(e.label == "LOC" for e in ents)

    def test_unconfigured_language(self) -> None:
        with pytest.raises(ConfigurationError):
            TransformersNER(models={"en": "x"}).extract("text", "ta")


def test_unknown_backend() -> None:
    with pytest.raises(ConfigurationError):
        build_ner_backend("nope")
