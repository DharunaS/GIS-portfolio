#!/usr/bin/env python3
r"""Entry point for the Multilingual Geo-Parsing Pipeline.

Examples:
    python main.py parse --text "Das DLR forscht in Oberpfaffenhofen bei München." \\
        --user-agent "my-geoparser/1.0 (me@example.org)" --geojson out/de.geojson --map out/de.html

    python main.py evaluate --gold examples/gold_annotations.jsonl \\
        --ner-backend gazetteer --offline --cache examples/geocode_cache.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a fresh clone without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geoparser.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
