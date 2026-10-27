"""Pick one referent per toponym.

Each candidate gets a weighted sum of six features in [0, 1]. Four only look
at the candidate itself (importance, population, place type, name match).
Two look at the rest of the document:

- admin_context: the candidate sits inside a state/region/country that is
  also mentioned in the text (or passed as a country hint). This is what
  gets "Paris, Texas" and "Frankfurt ... Brandenburg" right.
- coherence: exp(-d / scale), d = median distance to what the other mentions
  currently resolve to. Places in one text tend to be near each other.

Because the document features depend on the other picks, resolution loops:
pick by local score, recompute document features, re-pick, until nothing
changes (max 5 rounds). Confidence is the softmax of the winning score.

The weights are hand-set. TODO: learn them from a labelled corpus.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from geoparser.geo_utils import haversine_km
from geoparser.models import Candidate, Entity, ResolvedLocation
from geoparser.normalize import name_key, query_variants

logger = logging.getLogger(__name__)

#: How plausible each place type is as the referent of a bare name. Picked by
#: hand: a bare "Bremen" is far more likely the city than a street called Bremen.
TYPE_PRIORS: dict[str, float] = {
    "country": 1.0,
    "state": 0.9,
    "region": 0.75,
    "province": 0.85,
    "county": 0.6,
    "state_district": 0.6,
    "city": 0.95,
    "town": 0.75,
    "municipality": 0.7,
    "village": 0.45,
    "suburb": 0.4,
    "hamlet": 0.25,
    "continent": 0.8,
    "island": 0.6,
    "river": 0.55,
    "mountain": 0.5,
    "peak": 0.5,
    "lake": 0.5,
    "water": 0.5,
    "aerodrome": 0.6,
    "building": 0.3,
    "road": 0.15,
    "unknown": 0.3,
}

_FAC_TYPES = {"aerodrome", "building", "station", "university", "stadium", "attraction", "museum"}
_ADMIN_TYPES = {"country", "state", "province", "region", "county", "city", "town", "municipality"}


@dataclass
class DisambiguationConfig:
    """Weights and hyper-parameters of the resolver.

    Attributes:
        weights: Linear weights per feature (normalised to sum to 1).
        coherence_scale_km: Distance at which coherence decays to ``1/e``.
        softmax_temperature: Lower → sharper confidence.
        max_iterations: Rounds of document-level refinement.
        min_confidence: Mentions below this are left unresolved.
        country_hints: ISO alpha-2 codes known a priori (e.g. ``{"de"}``).
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "importance": 0.30,
            "population": 0.10,
            "type_prior": 0.15,
            "name_match": 0.15,
            "admin_context": 0.15,
            "coherence": 0.15,
        }
    )
    coherence_scale_km: float = 400.0
    softmax_temperature: float = 0.1
    max_iterations: int = 5
    min_confidence: float = 0.0
    country_hints: set[str] = field(default_factory=set)

    def normalised_weights(self) -> dict[str, float]:
        """Return weights scaled to sum to one (negative weights rejected)."""
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Disambiguation weights must be non-negative")
        total = sum(self.weights.values()) or 1.0
        return {k: v / total for k, v in self.weights.items()}


def _name_similarity(entity: Entity, cand: Candidate, lang: str | None) -> float:
    names: set[str] = {cand.name.split(",")[0]}
    namedetails = cand.raw.get("namedetails") if cand.raw else None
    if isinstance(namedetails, dict):
        names.update(str(v) for v in namedetails.values())
    keys = {name_key(n) for n in names if n}
    best = 0.0
    for variant in query_variants(entity.text, lang):
        q = name_key(variant)
        for k in keys:
            if q == k:
                return 1.0
            best = max(best, SequenceMatcher(None, q, k).ratio())
    return best


def _type_prior(entity: Entity, cand: Candidate) -> float:
    prior = TYPE_PRIORS.get(cand.place_type, TYPE_PRIORS["unknown"])
    if entity.label == "FAC":
        is_facility = cand.place_type in _FAC_TYPES or cand.place_class in {"aeroway", "building", "amenity"}
        prior = 1.0 if is_facility else prior * 0.6
    elif entity.label == "GPE":
        prior = prior if cand.place_type in _ADMIN_TYPES else prior * 0.6
    return prior


def _population_score(cand: Candidate) -> float:
    if not cand.population or cand.population <= 0:
        return 0.0
    return min(1.0, math.log10(cand.population + 1) / 7.5)  # 30 M → 1.0


_CONTAINER_TYPES = {"state", "province", "region", "county", "state_district"}


def _is_country(cand: Candidate) -> bool:
    return cand.place_type == "country"


def _inside_admin(cand: Candidate, admin: Candidate) -> bool:
    """Return whether ``cand`` lies inside a (different, larger) admin unit's bbox."""
    if admin.place_type not in _CONTAINER_TYPES or admin.bbox is None:
        return False
    if cand.place_type == admin.place_type:
        return False
    return admin.bbox.contains(cand.lat, cand.lon)


class Disambiguator:
    """Spatially-aware toponym resolver.

    Args:
        config: Weights and hyper-parameters; defaults if omitted.
    """

    def __init__(self, config: DisambiguationConfig | None = None) -> None:
        self.config = config or DisambiguationConfig()
        self._w = self.config.normalised_weights()

    def _local(self, entity: Entity, cand: Candidate, lang: str | None) -> dict[str, float]:
        return {
            "importance": max(0.0, min(1.0, cand.importance)),
            "population": _population_score(cand),
            "type_prior": _type_prior(entity, cand),
            "name_match": _name_similarity(entity, cand, lang),
        }

    def _softmax_top(self, scores: list[float]) -> float:
        t = max(self.config.softmax_temperature, 1e-6)
        m = max(scores)
        exps = [math.exp((s - m) / t) for s in scores]
        return exps[scores.index(m)] / sum(exps)

    def resolve(
        self,
        entities: list[Entity],
        candidates: list[list[Candidate]],
        lang: str | None = None,
    ) -> list[ResolvedLocation]:
        """Resolve every entity to (at most) one candidate.

        Args:
            entities: Recognised mentions, in document order.
            candidates: ``candidates[i]`` are the options for ``entities[i]``.
            lang: Document language (affects name-variant matching).

        Returns:
            One :class:`ResolvedLocation` per entity, same order.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(entities) != len(candidates):
            raise ValueError("entities and candidates must have equal length")
        if not entities:
            return []

        local = [[self._local(e, c, lang) for c in cands] for e, cands in zip(entities, candidates, strict=True)]
        w = self._w

        def score(i: int, j: int, doc_feats: dict[str, float]) -> float:
            feats = {**local[i][j], **doc_feats}
            return sum(w.get(k, 0.0) * v for k, v in feats.items())

        # Initial picks from local evidence only.
        picks: list[int | None] = []
        for i, cands in enumerate(candidates):
            if not cands:
                picks.append(None)
                continue
            vals = [score(i, j, {}) for j in range(len(cands))]
            picks.append(vals.index(max(vals)))

        def doc_features(i: int, cand: Candidate) -> dict[str, float]:
            countries = set(self.config.country_hints)
            others: list[Candidate] = []
            for k, pk in enumerate(picks):
                if k == i or pk is None:
                    continue
                other = candidates[k][pk]
                # Repeated mentions of the same name are not independent evidence.
                if entities[k].text == entities[i].text:
                    continue
                others.append(other)
                if _is_country(other) and other.country_code:
                    countries.add(other.country_code.lower())
            in_country = bool(cand.country_code and cand.country_code.lower() in countries)
            admin_context = 1.0 if (in_country or any(_inside_admin(cand, o) for o in others)) else 0.0
            if others:
                d = statistics.median(haversine_km(cand.lat, cand.lon, o.lat, o.lon) for o in others)
                coherence = math.exp(-d / self.config.coherence_scale_km)
            else:
                coherence = 0.0
            return {"admin_context": admin_context, "coherence": coherence}

        final_scores: list[list[float]] = [[] for _ in entities]
        final_feats: list[dict[str, float]] = [{} for _ in entities]
        for iteration in range(max(1, self.config.max_iterations)):
            changed = False
            for i, cands in enumerate(candidates):
                if not cands:
                    continue
                feats = [doc_features(i, c) for c in cands]
                vals = [score(i, j, feats[j]) for j in range(len(cands))]
                best = vals.index(max(vals))
                final_scores[i] = vals
                final_feats[i] = {**local[i][best], **feats[best]}
                if best != picks[i]:
                    picks[i] = best
                    changed = True
            if not changed:
                logger.debug("Disambiguation converged after %d iteration(s)", iteration + 1)
                break

        results: list[ResolvedLocation] = []
        for i, entity in enumerate(entities):
            cands = candidates[i]
            pick = picks[i]
            if pick is None:
                results.append(ResolvedLocation(entity, None, 0.0, 0))
                continue
            conf = self._softmax_top(final_scores[i])
            breakdown = {k: round(v, 4) for k, v in final_feats[i].items()}
            breakdown["total"] = round(final_scores[i][pick], 4)
            chosen: Candidate | None = cands[pick] if conf >= self.config.min_confidence else None
            results.append(ResolvedLocation(entity, chosen, round(conf, 4), len(cands), breakdown))
        return results
