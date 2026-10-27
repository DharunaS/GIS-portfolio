"""Command-line interface (``python main.py`` or the ``geoparser`` script)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from geoparser import __version__
from geoparser.config import PipelineConfig
from geoparser.evaluation import evaluate, format_report, load_gold
from geoparser.exceptions import GeoParserError
from geoparser.export import to_geojson, write_geojson, write_map
from geoparser.pipeline import GeoParser, read_input_records

logger = logging.getLogger("geoparser")


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pipeline")
    g.add_argument("--config", type=Path, help="JSON config file (CLI flags override it)")
    g.add_argument(
        "--ner-backend",
        choices=["transformers", "spacy", "stanza", "gazetteer"],
        help="NER backend (default: transformers)",
    )
    g.add_argument("--gazetteer", help="Names file for the gazetteer backend (one per line)")
    g.add_argument("--device", help="Torch device for transformers: -1 (CPU), 0, 1 … or mps")
    g.add_argument("--ner-min-score", type=float, help="Minimum NER confidence (default 0.5)")
    g.add_argument("--user-agent", help="Nominatim User-Agent, e.g. 'my-geoparser/1.0 (me@uni.de)'")
    g.add_argument("--nominatim-domain", help="Nominatim host (self-hosted instance for bulk runs)")
    g.add_argument("--cache", help="Geocoder JSON cache path")
    g.add_argument("--offline", action="store_true", help="Use only the geocoder cache, no network")
    g.add_argument(
        "--country-hint",
        action="append",
        default=[],
        metavar="CC",
        help="ISO alpha-2 country prior for disambiguation (repeatable)",
    )
    g.add_argument("--min-confidence", type=float, help="Leave mentions below this unresolved")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="geoparser",
        description="Multilingual geoparsing: NER → geocoding → spatial disambiguation → GeoJSON/maps.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Geoparse text or a file of documents")
    src = p_parse.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Inline document text")
    src.add_argument("--input", type=Path, help=".txt (one doc per line) or .jsonl ({id,text,lang})")
    p_parse.add_argument("--lang", choices=["auto", "en", "de", "ta", "hi"], default="auto")
    p_parse.add_argument("--geojson", type=Path, help="Write a GeoJSON FeatureCollection here")
    p_parse.add_argument("--map", type=Path, help="Write an interactive Folium HTML map here")
    p_parse.add_argument("--no-bbox", action="store_true", help="Do not draw bounding boxes on the map")
    _add_common(p_parse)

    p_eval = sub.add_parser("evaluate", help="Score against gold annotations (JSONL)")
    p_eval.add_argument("--gold", type=Path, required=True, help="Gold JSONL file")
    p_eval.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    p_eval.add_argument("--report", type=Path, help="Write the full JSON report here")
    p_eval.add_argument("--geojson", type=Path, help="Also export predictions as GeoJSON")
    p_eval.add_argument("--map", type=Path, help="Also export predictions as a map")
    _add_common(p_eval)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Merge a JSON config (if any) with CLI overrides.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The effective pipeline configuration.
    """
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.ner_backend:
        cfg.ner_backend = args.ner_backend
    if args.gazetteer:
        cfg.gazetteer_path = args.gazetteer
    if args.device is not None:
        cfg.device = int(args.device) if args.device.lstrip("-").isdigit() else args.device
    if args.ner_min_score is not None:
        cfg.ner_min_score = args.ner_min_score
    if args.user_agent:
        cfg.user_agent = args.user_agent
    if args.nominatim_domain:
        cfg.nominatim_domain = args.nominatim_domain
    if args.cache:
        cfg.cache_path = args.cache
    if args.offline:
        cfg.offline = True
    if args.country_hint:
        cfg.disambiguation.country_hints |= {c.lower() for c in args.country_hint}
    if args.min_confidence is not None:
        cfg.disambiguation.min_confidence = args.min_confidence
    return cfg


def _run_parse(args: argparse.Namespace, parser: GeoParser) -> int:
    lang: str | None = None if args.lang == "auto" else args.lang
    if args.text is not None:
        records = [("cli-0", args.text, lang)]
    else:
        records = [(i, t, lang or doc_lang) for i, t, doc_lang in read_input_records(args.input)]
    docs = list(parser.parse_many(records))

    if args.geojson:
        write_geojson(docs, args.geojson)
    if args.map:
        write_map(docs, args.map, draw_bboxes=not args.no_bbox)
    if not args.geojson:
        json.dump(to_geojson(docs), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    for d in docs:
        for loc in d.locations:
            c = loc.candidate
            where = f"{c.lat:.4f}, {c.lon:.4f}  {c.name[:60]}" if c else "UNRESOLVED"
            print(
                f"[{d.doc_id}|{d.language}] {loc.entity.text!r:28} → {where}  (conf {loc.confidence:.2f})",
                file=sys.stderr,
            )
    return 0


def _run_evaluate(args: argparse.Namespace, parser: GeoParser) -> int:
    gold = load_gold(args.gold)
    docs = list(parser.parse_many((g.doc_id, g.text, g.language) for g in gold))
    report = evaluate(docs, gold, mode=args.mode)
    print(format_report(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.geojson:
        write_geojson(docs, args.geojson)
    if args.map:
        write_map(docs, args.map)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` success, ``1`` pipeline error, ``130`` interrupted.
    """
    args = build_arg_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        parser = GeoParser.from_config(config_from_args(args))
        try:
            return _run_parse(args, parser) if args.command == "parse" else _run_evaluate(args, parser)
        finally:
            parser.close()
    except GeoParserError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
