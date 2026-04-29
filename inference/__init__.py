"""Distribution-model adapter layer.

The classes in this package provide a uniform interface for asking
"what species occur at (lat, lon, week)?" from multiple backends (our
PyTorch model, BirdNET TFLite geomodel, future ones). See
``DISTRIBUTION_MODELS_PLAN.md`` for the full design.

Public API:

    from inference import DistributionModel, TorchDistribution

    m = TorchDistribution.from_checkpoint(
        checkpoint='checkpoints/nordic/run1/checkpoint_best.pt',
        taxonomy='taxonomy.csv',
        model_id='nordic',
        bounds=(4.0, 53.0, 32.0, 71.5),
        priority=100,
    )
    probs = m.predict(59.9, 10.75, week=17)   # {'gretit1': 0.92, ...}
"""

from .base import DistributionModel, CachedModelMixin
from .torch_adapter import TorchDistribution
from .onnx_adapter import OnnxDistribution
from .tflite_adapter import TfliteDistribution
from .registry import Manifest, ModelRegistry
from .router import Router, RoutedPrediction

__all__ = ['DistributionModel', 'CachedModelMixin',
           'TorchDistribution', 'OnnxDistribution', 'TfliteDistribution',
           'Manifest', 'ModelRegistry',
           'Router', 'RoutedPrediction']
