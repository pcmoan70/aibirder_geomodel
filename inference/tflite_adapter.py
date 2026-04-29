"""Backward-compat shim: ``TfliteDistribution`` is an alias for
:class:`OnnxDistribution` that accepts a ``.tflite`` via ``from_files``.

Retained so pre-existing manifests with ``backend: tflite`` keep working.
New manifests should use ``backend: onnx`` and point at a ``.onnx`` file
directly (see :mod:`.onnx_adapter`).
"""

from __future__ import annotations

from typing import Any

from .onnx_adapter import OnnxDistribution


class TfliteDistribution(OnnxDistribution):
    """Kept for backward compatibility. Delegates to :class:`OnnxDistribution`."""

    @classmethod
    def from_files(cls, tflite: str, **kwargs: Any) -> OnnxDistribution:  # type: ignore[override]
        """Load a ``.tflite`` via :meth:`OnnxDistribution.from_tflite`."""
        return OnnxDistribution.from_tflite(tflite=tflite, **kwargs)
