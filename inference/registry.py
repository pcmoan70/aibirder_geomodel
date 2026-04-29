"""Model registry — loads distribution models from YAML manifests.

A manifest describes one :class:`DistributionModel` instance:

.. code-block:: yaml

    # models/nordic/manifest.yaml
    id: nordic
    backend: pytorch
    priority: 100
    bounds: [4.0, 53.0, 32.0, 71.5]
    weights: checkpoint_best.pt
    labels: labels.txt
    taxonomy: ../../taxonomy.csv

    # models/global_12k/manifest.yaml
    id: birdnet_global_12k
    backend: tflite
    priority: 10
    bounds: [-180, -90, 180, 90]
    weights: BirdNET+_Geomodel_V3.0.2_Global_12K_FP16.tflite
    labels: BirdNET+_Geomodel_V3.0.2_Global_12K_Labels.txt
    taxonomy: ../../taxonomy.csv

Paths are resolved relative to the manifest file. Registry walks a root
directory, finds every ``manifest.yaml``, and instantiates the matching
backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import DistributionModel
from .torch_adapter import TorchDistribution
from .onnx_adapter import OnnxDistribution
from .tflite_adapter import TfliteDistribution


_BACKENDS = {
    'pytorch': TorchDistribution,
    'torch':   TorchDistribution,
    'onnx':    OnnxDistribution,
    'tflite':  TfliteDistribution,
}


@dataclass
class Manifest:
    """Parsed representation of one ``manifest.yaml``."""

    path: Path
    id: str
    backend: str
    bounds: tuple
    priority: int
    weights: Path
    labels: Path
    taxonomy: Path
    cache_size: int = 4096
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_yaml(cls, path: Path) -> 'Manifest':
        with path.open(encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        here = path.parent

        def _rel(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (here / pp).resolve()

        required = ('id', 'backend', 'bounds', 'weights', 'labels', 'taxonomy')
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f'{path}: manifest missing keys: {missing}')

        return cls(
            path=path,
            id=str(raw['id']),
            backend=str(raw['backend']).lower(),
            bounds=tuple(float(x) for x in raw['bounds']),
            priority=int(raw.get('priority', 0)),
            weights=_rel(raw['weights']),
            labels=_rel(raw['labels']),
            taxonomy=_rel(raw['taxonomy']),
            cache_size=int(raw.get('cache_size', 4096)),
            extra=raw,
        )


class ModelRegistry:
    """Container for all loaded distribution models.

    Instantiation is lazy — manifests are discovered at construction but the
    models themselves load only when first accessed via :meth:`get`,
    :meth:`all`, or :meth:`applicable`.
    """

    def __init__(self, manifests: Optional[List[Manifest]] = None) -> None:
        self._manifests: Dict[str, Manifest] = {}
        self._models: Dict[str, DistributionModel] = {}
        for m in manifests or []:
            self.register_manifest(m)

    # -- discovery --------------------------------------------------------

    @classmethod
    def from_tree(cls, root: str = 'models') -> 'ModelRegistry':
        """Scan *root* for ``manifest.yaml`` files and build a registry."""
        root_path = Path(root)
        if not root_path.is_dir():
            return cls([])
        manifests: List[Manifest] = []
        for yml in sorted(root_path.rglob('manifest.yaml')):
            try:
                manifests.append(Manifest.from_yaml(yml))
            except Exception as exc:
                print(f'[ModelRegistry] skipping {yml}: {exc}')
        return cls(manifests)

    def register_manifest(self, m: Manifest) -> None:
        if m.id in self._manifests:
            raise ValueError(f'duplicate model id {m.id!r} '
                             f'(already registered from {self._manifests[m.id].path})')
        self._manifests[m.id] = m

    def register_model(self, model: DistributionModel) -> None:
        """Directly register an already-instantiated model (no manifest)."""
        if model.model_id in self._models:
            raise ValueError(f'duplicate model id {model.model_id!r}')
        self._models[model.model_id] = model

    # -- access -----------------------------------------------------------

    def ids(self) -> List[str]:
        return sorted(set(self._manifests.keys()) | set(self._models.keys()))

    def get(self, model_id: str) -> DistributionModel:
        if model_id in self._models:
            return self._models[model_id]
        if model_id not in self._manifests:
            raise KeyError(f'no model {model_id!r} in registry')
        m = self._manifests[model_id]
        backend_cls = _BACKENDS.get(m.backend)
        if backend_cls is None:
            raise ValueError(f'{m.path}: unknown backend {m.backend!r}. '
                             f'Known: {sorted(_BACKENDS)}')
        if backend_cls is TorchDistribution:
            model = TorchDistribution.from_checkpoint(
                checkpoint=str(m.weights),
                taxonomy=str(m.taxonomy),
                model_id=m.id,
                bounds=m.bounds,
                priority=m.priority,
                cache_size=m.cache_size,
                labels=str(m.labels) if m.labels.name != 'labels.txt' else None,
            )
        elif backend_cls is OnnxDistribution:
            model = OnnxDistribution.from_files(
                onnx=str(m.weights),
                labels=str(m.labels),
                taxonomy=str(m.taxonomy),
                model_id=m.id,
                bounds=m.bounds,
                priority=m.priority,
                cache_size=m.cache_size,
            )
        elif backend_cls is TfliteDistribution:
            model = OnnxDistribution.from_tflite(
                tflite=str(m.weights),
                labels=str(m.labels),
                taxonomy=str(m.taxonomy),
                model_id=m.id,
                bounds=m.bounds,
                priority=m.priority,
                cache_size=m.cache_size,
            )
        else:  # pragma: no cover — guarded above
            raise ValueError(f'unsupported backend {m.backend!r}')
        self._models[model_id] = model
        return model

    def all(self) -> List[DistributionModel]:
        return [self.get(mid) for mid in self.ids()]

    def applicable(self, lat: float, lon: float) -> List[DistributionModel]:
        """Return models whose declared bounds include ``(lat, lon)``."""
        return [m for m in self.all() if m.supports(lat, lon)]

    def __repr__(self) -> str:  # pragma: no cover
        return f'<ModelRegistry ids={self.ids()}>'
