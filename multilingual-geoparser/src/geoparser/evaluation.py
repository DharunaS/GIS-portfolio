"""Score recognition and resolution separately (Gritta et al., 2018).

Recognition: P/R/F1 over spans, strict (exact offsets) or lenient (overlap),
one-to-one matching.

Resolution, on matched spans only:
- mean / median error in km (resolved matches only)
- acc@161: share within 161 km (100 mi); unresolved counts as a miss
- AUC of log errors, scaled to [0, 1], lower is better; unresolved gets the
  max error of 20,039 km (half the Earth's circumference)

Mean error alone is misleading: one Paris-France-instead-of-Texas mistake
(~7,800 km) swamps everything else, so all four are reported.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from geoparser.exceptions import EvaluationError
from geoparser.geo_utils import haversine_km
from geoparser.models import Document, ResolvedLocation

logger = logging.getLogger(__name__)

MatchMode = Literal["strict", "lenient"]
MAX_ERROR_KM: float = 20039.0
ACC_THRESHOLD_KM: float = 161.0


@dataclass(frozen=True)
class GoldToponym:
    """A gold-standard toponym annotation.

    Attributes:
        start: Character offset (inclusive).
        end: Character offset (exclusive).
        text: Surface form (validated against the document text).
        lat: Gold latitude, or ``None`` if only the span is annotated.
        lon: Gold longitude, or ``None``.
    """

    start: int
    end: int
    text: str
    lat: float | None = None
    lon: float | None = None


@dataclass
class GoldDocument:
    """A gold-standard document."""

    doc_id: str
    text: str
    language: str | None
    toponyms: list[GoldToponym]


@dataclass
class Metrics:
    """Aggregated scores for one slice of the evaluation set."""

    n_gold: int = 0
    n_pred: int = 0
    tp: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_resolved_tp: int = 0
    mean_error_km: float | None = None
    median_error_km: float | None = None
    acc_at_161km: float | None = None
    auc: float | None = None


@dataclass
class EvaluationReport:
    """Overall and per-language metrics plus mention-level diagnostics."""

    mode: MatchMode
    overall: Metrics
    per_language: dict[str, Metrics] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def load_gold(path: str | Path) -> list[GoldDocument]:
    """Load gold annotations from JSON Lines.

    Each line: ``{"id": str, "text": str, "lang": str, "toponyms":
    [{"start": int, "end": int, "text": str, "lat": float, "lon": float}]}``.

    Args:
        path: ``.jsonl`` file.

    Returns:
        Parsed and validated gold documents.

    Raises:
        EvaluationError: On malformed JSON, missing keys or offsets that do
            not match the annotated surface text.
    """
    path = Path(path)
    docs: list[GoldDocument] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"Cannot read gold file {path}: {exc}") from exc

    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            text = rec["text"]
            toponyms = []
            for t in rec.get("toponyms", []):
                gt = GoldToponym(int(t["start"]), int(t["end"]), t["text"], t.get("lat"), t.get("lon"))
                if text[gt.start : gt.end] != gt.text:
                    raise EvaluationError(
                        f"{path}:{lineno}: span [{gt.start}:{gt.end}] is "
                        f"{text[gt.start : gt.end]!r}, annotation says {gt.text!r}"
                    )
                toponyms.append(gt)
            docs.append(GoldDocument(str(rec.get("id", f"doc-{lineno}")), text, rec.get("lang"), toponyms))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"{path}:{lineno}: invalid gold record ({exc})") from exc
    return docs


def _match(
    gold: Sequence[GoldToponym], pred: Sequence[ResolvedLocation], mode: MatchMode
) -> list[tuple[GoldToponym, ResolvedLocation]]:
    """Greedy one-to-one matching, preferring larger overlaps."""
    pairs: list[tuple[int, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            ps, pe = p.entity.start, p.entity.end
            if mode == "strict":
                if (ps, pe) == (g.start, g.end):
                    pairs.append((g.end - g.start, gi, pi))
            else:
                overlap = min(pe, g.end) - max(ps, g.start)
                if overlap > 0:
                    pairs.append((overlap, gi, pi))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[GoldToponym, ResolvedLocation]] = []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gold[gi], pred[pi]))
    return matched


def _auc(errors_km: Sequence[float]) -> float:
    """Normalised area under the log-error curve (Gritta et al., 2018).

    With errors sorted ascending, the curve is ``log(1 + e_i)`` over the unit
    interval. Normalising by ``log(1 + MAX_ERROR_KM)`` maps it to ``[0, 1]``.
    """
    if not errors_km:
        return 0.0
    logs = sorted(math.log1p(min(e, MAX_ERROR_KM)) for e in errors_km)
    if len(logs) == 1:
        return logs[0] / math.log1p(MAX_ERROR_KM)
    # Trapezoidal rule over x in [0, 1].
    dx = 1.0 / (len(logs) - 1)
    area = sum((logs[i] + logs[i + 1]) / 2 * dx for i in range(len(logs) - 1))
    return area / math.log1p(MAX_ERROR_KM)


class _Accumulator:
    def __init__(self) -> None:
        self.n_gold = 0
        self.n_pred = 0
        self.tp = 0
        self.errors: list[float] = []  # resolved TPs with gold coords
        self.auc_errors: list[float] = []  # all TPs with gold coords (unresolved = max)

    def finalise(self) -> Metrics:
        p = self.tp / self.n_pred if self.n_pred else 0.0
        r = self.tp / self.n_gold if self.n_gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        m = Metrics(self.n_gold, self.n_pred, self.tp, round(p, 4), round(r, 4), round(f1, 4), len(self.errors))
        if self.errors:
            m.mean_error_km = round(statistics.fmean(self.errors), 2)
            m.median_error_km = round(statistics.median(self.errors), 2)
        if self.auc_errors:
            m.acc_at_161km = round(sum(e <= ACC_THRESHOLD_KM for e in self.auc_errors) / len(self.auc_errors), 4)
            m.auc = round(_auc(self.auc_errors), 4)
        return m


def evaluate(
    predictions: Iterable[Document],
    gold: Iterable[GoldDocument],
    mode: MatchMode = "strict",
) -> EvaluationReport:
    """Score pipeline output against gold annotations.

    Documents are aligned by ``doc_id``. Gold documents without a prediction
    count as all-false-negative; predictions without gold are ignored with a
    warning.

    Args:
        predictions: Processed documents.
        gold: Gold documents.
        mode: ``"strict"`` (exact spans) or ``"lenient"`` (overlap).

    Returns:
        An :class:`EvaluationReport`.

    Raises:
        EvaluationError: If ``mode`` is invalid.
    """
    if mode not in ("strict", "lenient"):
        raise EvaluationError(f"Unknown match mode {mode!r}")
    pred_by_id = {d.doc_id: d for d in predictions}
    gold_list = list(gold)
    extra = set(pred_by_id) - {g.doc_id for g in gold_list}
    if extra:
        logger.warning("Ignoring %d predicted documents without gold: %s", len(extra), sorted(extra)[:5])

    overall = _Accumulator()
    by_lang: dict[str, _Accumulator] = defaultdict(_Accumulator)
    diagnostics: list[dict[str, Any]] = []

    for g in gold_list:
        pred_doc = pred_by_id.get(g.doc_id)
        preds = pred_doc.locations if pred_doc else []
        lang = g.language or (pred_doc.language if pred_doc else None) or "unknown"
        matches = _match(g.toponyms, preds, mode)
        matched_pred_ids = {id(p) for _, p in matches}
        matched_gold_ids = {id(gt) for gt, _ in matches}

        for acc in (overall, by_lang[lang]):
            acc.n_gold += len(g.toponyms)
            acc.n_pred += len(preds)
            acc.tp += len(matches)

        for gt, p in matches:
            if gt.lat is None or gt.lon is None:
                continue
            if p.candidate is None:
                err = MAX_ERROR_KM
                resolved = False
            else:
                err = haversine_km(gt.lat, gt.lon, p.candidate.lat, p.candidate.lon)
                resolved = True
            for acc in (overall, by_lang[lang]):
                acc.auc_errors.append(err)
                if resolved:
                    acc.errors.append(err)
            if not resolved or err > ACC_THRESHOLD_KM:
                diagnostics.append(
                    {
                        "doc_id": g.doc_id,
                        "kind": "resolution",
                        "mention": gt.text,
                        "error_km": round(err, 1),
                        "predicted": p.candidate.name if p.candidate else None,
                    }
                )

        diagnostics += [
            {"doc_id": g.doc_id, "kind": "false_negative", "mention": gt.text}
            for gt in g.toponyms
            if id(gt) not in matched_gold_ids
        ]
        diagnostics += [
            {"doc_id": g.doc_id, "kind": "false_positive", "mention": p.entity.text}
            for p in preds
            if id(p) not in matched_pred_ids
        ]

    return EvaluationReport(
        mode=mode,
        overall=overall.finalise(),
        per_language={k: v.finalise() for k, v in sorted(by_lang.items())},
        errors=diagnostics,
    )


def format_report(report: EvaluationReport) -> str:
    """Render a report as a Markdown table (suitable for the README/CI logs).

    Args:
        report: Output of :func:`evaluate`.

    Returns:
        Markdown string.
    """

    def fmt(v: float | None, digits: int = 3) -> str:
        return "–" if v is None else f"{v:.{digits}f}"

    rows = [("**all**", report.overall)] + list(report.per_language.items())
    lines = [
        f"Match mode: `{report.mode}`",
        "",
        "| Lang | Gold | Pred | TP | P | R | F1 | Mean err (km) | Median err (km) | Acc@161 | AUC ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m.n_gold} | {m.n_pred} | {m.tp} | {fmt(m.precision)} | {fmt(m.recall)} | "
            f"{fmt(m.f1)} | {fmt(m.mean_error_km, 1)} | {fmt(m.median_error_km, 1)} | "
            f"{fmt(m.acc_at_161km)} | {fmt(m.auc)} |"
        )
    return "\n".join(lines)
