#!/usr/bin/env python3
"""Build the offline demo fixtures: gold annotations and a geocoder cache.

Why a fixture? Nominatim's public API is rate-limited (1 req/s) and changes
over time, and Hugging Face models are large downloads. To keep CI and the
README demo deterministic, this script writes:

* ``examples/gold_annotations.jsonl`` — 9 documents (en/de/ta/hi) with
  character-offset toponym spans and reference coordinates. Offsets are
  computed here, never typed by hand. Linear features (rivers) are annotated
  as spans with ``lat/lon = null`` because a single point error is ill-defined
  for a 900 km river.
* ``examples/geocode_cache.json`` — Nominatim-shaped candidate lists for the
  queries the pipeline issues, *including deliberate distractors* (Frankfurt
  am Main vs. Frankfurt (Oder), Paris FR vs. Paris TX, Hyderabad IN vs. PK,
  Berlin DE vs. Berlin NH, Brandenburg state vs. city).

Candidate coordinates, extents, populations and importances are rounded,
hand-curated approximations of OpenStreetMap values for demonstration only;
OSM ids are intentionally omitted. Run the pipeline online (without
``--offline``) to populate a cache with real Nominatim responses.

Usage:
    python examples/build_demo_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def place(
    display: str,
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    addresstype: str,
    cc: str,
    importance: float,
    population: int | None = None,
    cls: str = "place",
    names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a Nominatim-shaped ``/search`` record."""
    rec: dict[str, Any] = {
        "display_name": display,
        "lat": f"{lat:.5f}",
        "lon": f"{lon:.5f}",
        "boundingbox": [f"{v:.4f}" for v in bbox],  # south, north, west, east
        "class": cls,
        "type": addresstype,
        "addresstype": addresstype,
        "importance": importance,
        "address": {"country_code": cc},
        "extratags": {"population": str(population)} if population else {},
        "namedetails": {"name": display.split(",")[0], **(names or {})},
    }
    return rec


P = {
    "oberpfaffenhofen": place(
        "Oberpfaffenhofen, Weßling, Bavaria, Germany",
        48.0742,
        11.2667,
        (48.065, 48.092, 11.245, 11.290),
        "village",
        "de",
        0.38,
        2000,
        names={"name:en": "Oberpfaffenhofen"},
    ),
    "munich": place(
        "München, Bavaria, Germany",
        48.1371,
        11.5754,
        (48.0616, 48.2482, 11.3608, 11.7229),
        "city",
        "de",
        0.80,
        1512491,
        names={"name:en": "Munich", "name:de": "München", "name:hi": "म्यूनिख", "name:ta": "மியூனிக்"},
    ),
    "ammersee": place(
        "Ammersee, Bavaria, Germany",
        48.0000,
        11.1167,
        (47.90, 48.08, 11.07, 11.20),
        "water",
        "de",
        0.50,
        cls="natural",
        names={"name:en": "Ammersee"},
    ),
    "bavaria": place(
        "Bayern, Germany",
        48.9468,
        11.4039,
        (47.27, 50.56, 8.98, 13.84),
        "state",
        "de",
        0.78,
        13369393,
        cls="boundary",
        names={"name:en": "Bavaria", "name:de": "Bayern", "name:hi": "बवेरिया", "name:ta": "பவேரியா"},
    ),
    "chennai": place(
        "Chennai, Tamil Nadu, India",
        13.0827,
        80.2707,
        (12.95, 13.23, 80.17, 80.33),
        "city",
        "in",
        0.72,
        7088000,
        names={"name:en": "Chennai", "name:ta": "சென்னை", "name:hi": "चेन्नई"},
    ),
    "madurai": place(
        "Madurai, Tamil Nadu, India",
        9.9252,
        78.1198,
        (9.87, 9.98, 78.07, 78.17),
        "city",
        "in",
        0.62,
        1017865,
        names={"name:en": "Madurai", "name:ta": "மதுரை", "name:hi": "मदुरै"},
    ),
    "coimbatore": place(
        "Coimbatore, Tamil Nadu, India",
        11.0168,
        76.9558,
        (10.90, 11.13, 76.86, 77.05),
        "city",
        "in",
        0.63,
        1050721,
        names={"name:en": "Coimbatore", "name:ta": "கோயம்புத்தூர்", "name:hi": "कोयंबटूर"},
    ),
    "hyderabad_in": place(
        "Hyderabad, Telangana, India",
        17.3850,
        78.4867,
        (17.20, 17.60, 78.20, 78.70),
        "city",
        "in",
        0.72,
        6809970,
        names={"name:en": "Hyderabad", "name:hi": "हैदराबाद"},
    ),
    "hyderabad_pk": place(
        "Hyderabad, Sindh, Pakistan",
        25.3960,
        68.3578,
        (25.30, 25.50, 68.25, 68.45),
        "city",
        "pk",
        0.62,
        1732693,
        names={"name:en": "Hyderabad"},
    ),
    "tamil_nadu": place(
        "Tamil Nadu, India",
        10.9094,
        78.3665,
        (8.07, 13.56, 76.23, 80.35),
        "state",
        "in",
        0.74,
        72147030,
        cls="boundary",
        names={"name:en": "Tamil Nadu", "name:ta": "தமிழ்நாடு", "name:hi": "तमिलनाडु"},
    ),
    "paris_fr": place(
        "Paris, Île-de-France, France",
        48.8535,
        2.3484,
        (48.8156, 48.9022, 2.2241, 2.4699),
        "city",
        "fr",
        0.88,
        2133111,
        names={"name:en": "Paris", "name:hi": "पेरिस"},
    ),
    "paris_tx": place(
        "Paris, Lamar County, Texas, United States",
        33.6609,
        -95.5555,
        (33.60, 33.73, -95.63, -95.45),
        "town",
        "us",
        0.52,
        24476,
    ),
    "paris_tn": place(
        "Paris, Henry County, Tennessee, United States",
        36.3020,
        -88.3267,
        (36.26, 36.34, -88.37, -88.28),
        "town",
        "us",
        0.45,
        10156,
    ),
    "texas": place(
        "Texas, United States",
        31.2639,
        -98.5456,
        (25.84, 36.50, -106.65, -93.51),
        "state",
        "us",
        0.80,
        29145505,
        cls="boundary",
        names={"name:en": "Texas"},
    ),
    "dallas": place(
        "Dallas, Texas, United States", 32.7763, -96.7969, (32.62, 33.02, -96.99, -96.46), "city", "us", 0.72, 1304379
    ),
    "koeln": place(
        "Köln, North Rhine-Westphalia, Germany",
        50.9384,
        6.9599,
        (50.83, 51.08, 6.77, 7.16),
        "city",
        "de",
        0.76,
        1083498,
        names={"name:en": "Cologne", "name:de": "Köln"},
    ),
    "berlin": place(
        "Berlin, Germany",
        52.5170,
        13.3889,
        (52.34, 52.68, 13.09, 13.76),
        "city",
        "de",
        0.85,
        3755251,
        names={"name:en": "Berlin", "name:hi": "बर्लिन", "name:ta": "பெர்லின்"},
    ),
    "berlin_nh": place(
        "Berlin, Coös County, New Hampshire, United States",
        44.4687,
        -71.1851,
        (44.42, 44.53, -71.27, -71.12),
        "city",
        "us",
        0.40,
        9425,
    ),
    "braunschweig": place(
        "Braunschweig, Lower Saxony, Germany",
        52.2646,
        10.5236,
        (52.19, 52.34, 10.40, 10.65),
        "city",
        "de",
        0.63,
        251364,
        names={"name:en": "Brunswick"},
    ),
    "oder": place(
        "Oder, Germany",
        52.6000,
        14.6000,
        (50.00, 53.90, 14.00, 18.00),
        "river",
        "de",
        0.58,
        cls="waterway",
        names={"name:de": "Oder", "name:pl": "Odra"},
    ),
    "frankfurt_main": place(
        "Frankfurt am Main, Hesse, Germany",
        50.1106,
        8.6821,
        (50.015, 50.227, 8.472, 8.800),
        "city",
        "de",
        0.80,
        773068,
        names={"short_name": "Frankfurt"},
    ),
    "frankfurt_oder": place(
        "Frankfurt (Oder), Brandenburg, Germany",
        52.3471,
        14.5506,
        (52.25, 52.40, 14.40, 14.66),
        "city",
        "de",
        0.62,
        57015,
        names={"short_name": "Frankfurt"},
    ),
    "slubice": place(
        "Słubice, Lubusz Voivodeship, Poland", 52.3502, 14.5606, (52.33, 52.37, 14.53, 14.62), "town", "pl", 0.45, 16000
    ),
    "brandenburg_state": place(
        "Brandenburg, Germany",
        52.4125,
        12.5316,
        (51.36, 53.56, 11.27, 14.77),
        "state",
        "de",
        0.74,
        2573135,
        cls="boundary",
    ),
    "brandenburg_city": place(
        "Brandenburg an der Havel, Brandenburg, Germany",
        52.4084,
        12.5625,
        (52.35, 52.50, 12.38, 12.70),
        "city",
        "de",
        0.55,
        72000,
    ),
    "sriharikota": place(
        "Sriharikota, Tirupati, Andhra Pradesh, India",
        13.7200,
        80.2300,
        (13.60, 13.83, 80.17, 80.25),
        "island",
        "in",
        0.50,
        names={"name:en": "Sriharikota", "name:ta": "ஸ்ரீஹரிகோட்டா", "name:hi": "श्रीहरिकोटा"},
    ),
    "bengaluru": place(
        "Bengaluru, Karnataka, India",
        12.9716,
        77.5946,
        (12.83, 13.14, 77.46, 77.78),
        "city",
        "in",
        0.75,
        8443675,
        names={
            "name:en": "Bengaluru",
            "name:ta": "பெங்களூரு",
            "name:hi": "बेंगलुरु",
            "alt_name:ta": "பெங்களூர்",
            "old_name:en": "Bangalore",
        },
    ),
    "mumbai": place(
        "Mumbai, Maharashtra, India",
        19.0760,
        72.8777,
        (18.89, 19.27, 72.77, 72.99),
        "city",
        "in",
        0.80,
        12442373,
        names={"name:en": "Mumbai", "name:hi": "मुंबई", "name:ta": "மும்பை"},
    ),
    "pune": place(
        "Pune, Maharashtra, India",
        18.5204,
        73.8567,
        (18.41, 18.63, 73.74, 73.98),
        "city",
        "in",
        0.70,
        3124458,
        names={"name:en": "Pune", "name:hi": "पुणे"},
    ),
    "maharashtra": place(
        "Maharashtra, India",
        19.7515,
        75.7139,
        (15.60, 22.03, 72.64, 80.89),
        "state",
        "in",
        0.76,
        112374333,
        cls="boundary",
        names={"name:en": "Maharashtra", "name:hi": "महाराष्ट्र"},
    ),
    "varanasi": place(
        "Varanasi, Uttar Pradesh, India",
        25.3176,
        82.9739,
        (25.23, 25.38, 82.90, 83.05),
        "city",
        "in",
        0.66,
        1198491,
        names={"name:en": "Varanasi", "name:hi": "वाराणसी"},
    ),
    "ganga": place(
        "Ganga, India",
        25.3000,
        83.0200,
        (21.50, 31.00, 77.50, 89.00),
        "river",
        "in",
        0.70,
        cls="waterway",
        names={"name:en": "Ganges", "name:hi": "गंगा"},
    ),
}

#: query (as issued by the pipeline) → candidate keys, best-ranked first as Nominatim would.
CACHE: dict[str, list[str]] = {
    "oberpfaffenhofen": ["oberpfaffenhofen"],
    "munich": ["munich"],
    "ammersee": ["ammersee"],
    "bavaria": ["bavaria"],
    "chennai": ["chennai"],
    "madurai": ["madurai"],
    "hyderabad": ["hyderabad_in", "hyderabad_pk"],
    "tamil nadu": ["tamil_nadu"],
    "paris": ["paris_fr", "paris_tx", "paris_tn"],
    "texas": ["texas"],
    "dallas": ["dallas"],
    "köln": ["koeln"],
    "berlin": ["berlin", "berlin_nh"],
    "braunschweig": ["braunschweig"],
    "oder": ["oder"],
    "frankfurt": ["frankfurt_main", "frankfurt_oder"],
    "słubice": ["slubice"],
    "brandenburg": ["brandenburg_state", "brandenburg_city"],
    "சென்னை": ["chennai"],
    "மதுரை": ["madurai"],
    "கோயம்புத்தூர்": ["coimbatore"],
    "ஸ்ரீஹரிகோட்டா": ["sriharikota"],
    "பெங்களூர்": ["bengaluru"],
    "मुंबई": ["mumbai"],
    "पुणे": ["pune"],
    "महाराष्ट्र": ["maharashtra"],
    "बेंगलुरु": ["bengaluru"],
    "श्रीहरिकोटा": ["sriharikota"],
    "वाराणसी": ["varanasi"],
    "गंगा": ["ganga"],
}

#: (doc id, lang, text, [(surface, lat, lon)]) — reference coordinates are
#: city-centre / centroid values independent of the candidate points above.
DOCS: list[tuple[str, str, str, list[tuple[str, float | None, float | None]]]] = [
    (
        "en-1",
        "en",
        "The DLR Earth Observation Center in Oberpfaffenhofen, near Munich, processes Sentinel-2 "
        "scenes of the Ammersee and the rest of Bavaria.",
        [
            ("Oberpfaffenhofen", 48.0830, 11.2770),
            ("Munich", 48.1351, 11.5820),
            ("Ammersee", 47.9970, 11.1200),
            ("Bavaria", 48.7904, 11.4979),
        ],
    ),
    (
        "en-2",
        "en",
        "After the cyclone, relief teams travelled from Chennai to Madurai and Kanchipuram, while analysts "
        "in Hyderabad mapped flood extents across Tamil Nadu.",
        [
            ("Chennai", 13.0827, 80.2707),
            ("Madurai", 9.9252, 78.1198),
            ("Kanchipuram", 12.8342, 79.7036),
            ("Hyderabad", 17.3850, 78.4867),
            ("Tamil Nadu", 11.1271, 78.6569),
        ],
    ),
    (
        "en-3",
        "en",
        "The storm passed over Paris, Texas, about 160 km north-east of Dallas.",
        [("Paris", 33.6609, -95.5555), ("Texas", 31.9686, -99.9018), ("Dallas", 32.7767, -96.7970)],
    ),
    (
        "de-1",
        "de",
        "Das Deutsche Zentrum für Luft- und Raumfahrt hat seinen Hauptsitz in Köln und betreibt "
        "Standorte in Berlin, Braunschweig und Oberpfaffenhofen.",
        [
            ("Köln", 50.9375, 6.9603),
            ("Berlin", 52.5200, 13.4050),
            ("Braunschweig", 52.2689, 10.5268),
            ("Oberpfaffenhofen", 48.0830, 11.2770),
        ],
    ),
    (
        "de-2",
        "de",
        "Das Hochwasser an der Oder erreichte Frankfurt und Słubice; Brandenburg rief den Katastrophenfall aus.",
        [
            ("Oder", None, None),
            ("Frankfurt", 52.3471, 14.5506),
            ("Słubice", 52.3500, 14.5600),
            ("Brandenburg", 52.4125, 12.5316),
        ],
    ),
    (
        "ta-1",
        "ta",
        "சென்னையில் பெய்த கனமழையால் மதுரை மற்றும் கோயம்புத்தூர் பகுதிகளில் வெள்ளம் ஏற்பட்டது.",
        [("சென்னையில்", 13.0827, 80.2707), ("மதுரை", 9.9252, 78.1198), ("கோயம்புத்தூர்", 11.0168, 76.9558)],
    ),
    (
        "ta-2",
        "ta",
        "இஸ்ரோ ஸ்ரீஹரிகோட்டாவிலிருந்து செயற்கைக்கோளை ஏவியது; பெங்களூரில் உள்ள கட்டுப்பாட்டு மையம் அதைக் கண்காணித்தது.",
        [("ஸ்ரீஹரிகோட்டாவிலிருந்து", 13.7331, 80.2350), ("பெங்களூரில்", 12.9716, 77.5946)],
    ),
    (
        "hi-1",
        "hi",
        "भारी बारिश के कारण मुंबई और पुणे में बाढ़ आ गई, जबकि महाराष्ट्र सरकार ने राहत कार्य शुरू किया।",
        [("मुंबई", 19.0760, 72.8777), ("पुणे", 18.5204, 73.8567), ("महाराष्ट्र", 19.6633, 75.3003)],
    ),
    (
        "hi-2",
        "hi",
        "इसरो का मुख्यालय बेंगलुरु में है और उपग्रह श्रीहरिकोटा से प्रक्षेपित किए जाते हैं; वाराणसी गंगा के तट पर स्थित है।",
        [
            ("बेंगलुरु", 12.9716, 77.5946),
            ("श्रीहरिकोटा", 13.7331, 80.2350),
            ("वाराणसी", 25.3176, 82.9739),
            ("गंगा", None, None),
        ],
    ),
]


def build_gold() -> list[dict[str, Any]]:
    """Compute character offsets for each annotated surface form (first unused occurrence)."""
    out = []
    for doc_id, lang, text, tops in DOCS:
        cursor = 0
        spans = []
        for surface, lat, lon in tops:
            start = text.find(surface, cursor)
            if start < 0:
                raise ValueError(f"{doc_id}: {surface!r} not found after offset {cursor}")
            spans.append({"start": start, "end": start + len(surface), "text": surface, "lat": lat, "lon": lon})
            cursor = start + len(surface)
        out.append({"id": doc_id, "lang": lang, "text": text, "toponyms": spans})
    return out


def main() -> None:
    """Write both fixture files next to this script."""
    gold_path = HERE / "gold_annotations.jsonl"
    with gold_path.open("w", encoding="utf-8") as fh:
        for rec in build_gold():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cache = {q: [_as_candidate(P[k]) for k in keys] for q, keys in CACHE.items()}
    cache_path = HERE / "geocode_cache.json"
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {gold_path.name} ({len(DOCS)} docs) and {cache_path.name} ({len(cache)} queries)")


def _as_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw record into the cache's serialised Candidate form."""
    import sys

    sys.path.insert(0, str(HERE.parent / "src"))
    from geoparser.geocoder import candidate_from_nominatim, candidate_to_dict

    return candidate_to_dict(candidate_from_nominatim(raw))


if __name__ == "__main__":
    main()
