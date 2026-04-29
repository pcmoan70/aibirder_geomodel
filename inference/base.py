"""Abstract ``DistributionModel`` interface + caching mixin."""

from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple


Bounds = Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)


@dataclass(frozen=True)
class _CacheKey:
    lat: float
    lon: float
    week: int
    altitude: Optional[int]


class DistributionModel(abc.ABC):
    """Uniform interface for "what species occur at (lat, lon, week)?".

    Implementations translate their native vocab to eBird/BirdNET
    ``species_code`` (6-letter) so the router can merge outputs across
    backends.

    Attributes
    ----------
    model_id : str
        Short unique id used in logs and the registry ("nordic",
        "birdnet_global_12k").
    bounds : (lon_min, lat_min, lon_max, lat_max)
        Region where this model is valid. Use a global bbox for planetary
        models. The router will only call us inside these bounds.
    priority : int
        Higher wins when multiple models cover the same point. Regional
        specialists should outrank planetary fallbacks.
    vocab : frozenset[str]
        The set of ``species_code`` values this model can return.
    """

    def __init__(
        self,
        model_id: str,
        bounds: Bounds,
        priority: int,
        vocab: FrozenSet[str],
    ) -> None:
        self.model_id = model_id
        self.bounds = tuple(bounds)  # type: ignore[assignment]
        self.priority = int(priority)
        self.vocab = frozenset(vocab)

    # -- public API --------------------------------------------------------

    def supports(self, lat: float, lon: float) -> bool:
        """True if (lat, lon) is within this model's declared bounds."""
        lon_min, lat_min, lon_max, lat_max = self.bounds
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    def predict(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return ``{species_code: probability}`` at this location and week.

        Subclasses implement ``_predict_uncached``; this wrapper applies
        the cache (see :class:`CachedModelMixin` for the cache state).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_uncached(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float],
    ) -> Dict[str, float]:
        """Run the underlying model. Implementations MUST override this."""

    def __repr__(self) -> str:  # pragma: no cover — convenience
        return (
            f'<{type(self).__name__} id={self.model_id!r} '
            f'bounds={self.bounds} priority={self.priority} '
            f'vocab_size={len(self.vocab)}>'
        )


class CachedModelMixin(DistributionModel, abc.ABC):
    """Mixin that adds an LRU cache to :meth:`DistributionModel.predict`.

    Inference of these models is per-station or per-hour, not per
    detection, so re-running the forward pass for every call is wasteful.
    The cache quantizes ``lat``, ``lon``, and ``altitude`` to limit the
    number of distinct keys (typical station = one cache entry per week).

    Parameters
    ----------
    cache_size : int, default 4096
        Maximum distinct (lat, lon, week[, altitude]) entries to keep.
    coord_precision : int, default 3
        Decimal places used to round lat/lon before caching. 3 → ~111 m
        at the equator, fine for station-scale queries.
    altitude_precision : int, default 50
        Metres resolution for altitude quantization. GPS altitude noise is
        ~±20–30 m so snapping to 50-m bins has no real information loss.
    """

    def __init__(
        self,
        model_id: str,
        bounds: Bounds,
        priority: int,
        vocab: FrozenSet[str],
        *,
        cache_size: int = 4096,
        coord_precision: int = 3,
        altitude_precision: int = 50,
    ) -> None:
        super().__init__(model_id, bounds, priority, vocab)
        self._cache: 'OrderedDict[_CacheKey, Dict[str, float]]' = OrderedDict()
        self._cache_size = int(cache_size)
        self._coord_precision = int(coord_precision)
        self._altitude_precision = int(altitude_precision)
        self._cache_hits = 0
        self._cache_misses = 0

    def _quantize_key(
        self, lat: float, lon: float, week: int, altitude: Optional[float]
    ) -> _CacheKey:
        qlat = round(float(lat), self._coord_precision)
        qlon = round(float(lon), self._coord_precision)
        qalt: Optional[int] = None
        if altitude is not None:
            step = max(self._altitude_precision, 1)
            qalt = int(round(float(altitude) / step) * step)
        return _CacheKey(qlat, qlon, int(week), qalt)

    def predict(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float] = None,
    ) -> Dict[str, float]:
        key = self._quantize_key(lat, lon, week, altitude)
        cached = self._cache.get(key)
        if cached is not None:
            # LRU: move to end
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return cached

        probs = self._predict_uncached(lat, lon, week, altitude)
        self._cache[key] = probs
        self._cache_misses += 1
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return probs

    def cache_info(self) -> Dict[str, int]:
        """Stats for diagnostics and tests."""
        return {
            'size': len(self._cache),
            'capacity': self._cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
