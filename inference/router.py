"""Query router — dispatches ``(lat, lon, week)`` across registered models.

Default merge policy: ``best_coverage``. The highest-priority model whose
``bounds`` include the query point owns the prediction; everything its
vocab doesn't cover is filled in by the next-priority model, and so on.

Callers can swap in other policies later (rank fusion etc.) once we have
calibration data. See ``DISTRIBUTION_MODELS_PLAN.md`` section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import DistributionModel
from .registry import ModelRegistry


@dataclass
class RoutedPrediction:
    """One species' combined prediction with provenance.

    ``probability`` and ``source`` are always set. ``per_model`` holds every
    model that produced an opinion about this species, so downstream code
    can decide how to surface disagreement.
    """

    species_code: str
    probability: float
    source: str
    per_model: Dict[str, float] = field(default_factory=dict)


class Router:
    """Dispatch + merge across the registry."""

    def __init__(self, registry: ModelRegistry, *, policy: str = 'best_coverage') -> None:
        if policy != 'best_coverage':
            raise ValueError(f'unknown policy {policy!r} '
                             f'(only "best_coverage" implemented)')
        self._registry = registry
        self._policy = policy

    def predict(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float] = None,
    ) -> Dict[str, RoutedPrediction]:
        """Return ``{species_code: RoutedPrediction}`` for this location."""
        models = self._registry.applicable(lat, lon)
        models.sort(key=lambda m: -m.priority)  # highest first

        # Run every applicable model — each has its own cache so repeat calls
        # at the same station are cheap.
        raw: List[tuple] = [(m, m.predict(lat, lon, week, altitude)) for m in models]

        merged: Dict[str, RoutedPrediction] = {}
        for model, probs in raw:
            for code, p in probs.items():
                if code in merged:
                    merged[code].per_model[model.model_id] = p
                    continue
                merged[code] = RoutedPrediction(
                    species_code=code,
                    probability=p,
                    source=model.model_id,
                    per_model={model.model_id: p},
                )
        return merged

    def predict_top_k(
        self,
        lat: float,
        lon: float,
        week: int,
        k: int,
        altitude: Optional[float] = None,
    ) -> List[RoutedPrediction]:
        """Convenience: return top-k species by merged probability."""
        merged = self.predict(lat, lon, week, altitude)
        return sorted(merged.values(), key=lambda r: -r.probability)[:k]
