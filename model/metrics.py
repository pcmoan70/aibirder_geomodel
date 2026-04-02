"""Validation metric helpers for BirdNET Geomodel."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


# Component weights: (metric_key, weight)
# Values are transformed to [0, 1] where higher = better.
_GEOSCORE_COMPONENTS: List[Tuple[str, float]] = [
    ('map',                0.20),
    ('f1_10',              0.20),
    ('list_ratio_10',      0.15),
    ('watchlist_mean_ap',  0.10),
    ('holdout_map',        0.10),
    ('map_density_ratio',  0.20),
    ('pred_density_corr',  0.05),
]


def compute_geoscore(metrics: Dict[str, float]) -> float:
    """Compute the GeoScore composite quality metric.

    Missing components are skipped and weights are renormalized.

    Component transforms (all map to [0, 1], higher = better):
    - map, f1_10, watchlist_mean_ap, holdout_map: used directly (already [0, 1])
    - list_ratio_10: exp(-|log(r)|) — symmetric penalty around ratio=1,
      equivalent to min(r, 1/r); never collapses to zero for finite ratios
    - map_density_ratio: min(r, 1/r) — peaks at 1.0 for perfect
      sparse/dense balance; penalizes both sparse-dominant and dense-dominant
      bias symmetrically
    - pred_density_corr: 1 - |r| — lower absolute correlation is better
    """
    scored: List[Tuple[float, float]] = []

    for key, weight in _GEOSCORE_COMPONENTS:
        raw = metrics.get(key, float('nan'))
        if raw is None or math.isnan(raw):
            continue

        if key == 'list_ratio_10':
            # Symmetric penalty for over/under prediction around ratio=1.
            # exp(-|log(r)|) = min(r, 1/r): smooth, never zero for finite r.
            val = math.exp(-abs(math.log(max(raw, 1e-8))))
        elif key == 'map_density_ratio':
            # Perfect balance (ratio=1.0) scores 1.0; deviations in either
            # direction are penalized equally.  Prevents gaming via
            # aggressive propagation that makes sparse > dense.
            val = min(raw, 1.0 / max(raw, 1e-8))
        elif key == 'pred_density_corr':
            # Lower absolute correlation is better.
            val = max(0.0, 1.0 - abs(raw))
        else:
            val = float(raw)

        scored.append((val, weight))

    if not scored:
        return 0.0

    total_weight = sum(w for _, w in scored)
    return sum(v * w for v, w in scored) / total_weight
