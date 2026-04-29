"""ONNX backend for :class:`DistributionModel`.

Wraps an ONNX model file (typically BirdNET's geomodel) via
:mod:`onnxruntime`. Expects a single input tensor with final axis width 3
(``[latitude_degrees, longitude_degrees, week_number]``) and a single
output tensor with final axis = vocabulary size. Output values are
treated as sigmoid probabilities.

If you only have a TFLite file, use :meth:`OnnxDistribution.from_tflite`
— it runs :mod:`tf2onnx` once and caches the result next to the source.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import Bounds, CachedModelMixin


def _load_session(onnx_path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(
        str(onnx_path), providers=['CPUExecutionProvider'],
    )


def _ensure_onnx_from_tflite(tflite_path: Path) -> Path:
    """Return an ONNX sibling of *tflite_path*, converting on demand."""
    onnx_path = tflite_path.with_suffix('.onnx')
    if onnx_path.exists() and onnx_path.stat().st_mtime >= tflite_path.stat().st_mtime:
        return onnx_path
    print(
        f'[OnnxDistribution] converting {tflite_path.name} → {onnx_path.name} '
        f'via tf2onnx (one-time)...', file=sys.stderr,
    )
    try:
        subprocess.run(
            [sys.executable, '-m', 'tf2onnx.convert',
             '--tflite', str(tflite_path), '--output', str(onnx_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'tf2onnx conversion failed for {tflite_path}: {exc.stderr}'
        ) from exc
    return onnx_path


def _load_geomodel_labels(labels_path: Path) -> List[tuple]:
    """Return ``[(taxonKey, sci_name, com_name), ...]`` in row order."""
    rows: List[tuple] = []
    with labels_path.open(encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts or not parts[0]:
                continue
            tk = parts[0].strip()
            sci = (parts[1] if len(parts) > 1 else tk).strip()
            com = (parts[2] if len(parts) > 2 else sci).strip()
            rows.append((tk, sci, com))
    return rows


def _build_sci_to_code(taxonomy_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with taxonomy_path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sci = (row.get('sci_name') or '').strip()
            code = (row.get('species_code') or '').strip()
            if sci and code:
                mapping[sci] = code
    return mapping


class OnnxDistribution(CachedModelMixin):
    """DistributionModel backed by an ONNX model file.

    Use :meth:`from_files` for ONNX input, :meth:`from_tflite` to load a
    TFLite (auto-converts via tf2onnx on first use).
    """

    def __init__(
        self,
        *,
        model_id: str,
        bounds: Bounds,
        priority: int,
        session,
        input_name: str,
        output_name: str,
        idx_to_code: Dict[int, str],
        cache_size: int = 4096,
    ) -> None:
        super().__init__(
            model_id=model_id,
            bounds=bounds,
            priority=priority,
            vocab=frozenset(idx_to_code.values()),
            cache_size=cache_size,
        )
        self._sess = session
        self._input_name = input_name
        self._output_name = output_name
        self._idx_to_code = dict(idx_to_code)
        order = sorted(idx_to_code.keys())
        self._output_indices = np.array(order, dtype=np.int64)
        self._codes = [idx_to_code[i] for i in order]

    # -- factories ---------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        onnx: str,
        labels: str,
        taxonomy: str,
        *,
        model_id: str = 'birdnet_global',
        bounds: Bounds = (-180.0, -90.0, 180.0, 90.0),
        priority: int = 10,
        cache_size: int = 4096,
    ) -> 'OnnxDistribution':
        return cls._build(
            onnx_path=Path(onnx), labels_path=Path(labels), taxonomy_path=Path(taxonomy),
            model_id=model_id, bounds=bounds, priority=priority, cache_size=cache_size,
        )

    @classmethod
    def from_tflite(
        cls,
        tflite: str,
        labels: str,
        taxonomy: str,
        *,
        model_id: str = 'birdnet_global',
        bounds: Bounds = (-180.0, -90.0, 180.0, 90.0),
        priority: int = 10,
        cache_size: int = 4096,
    ) -> 'OnnxDistribution':
        onnx_path = _ensure_onnx_from_tflite(Path(tflite))
        return cls._build(
            onnx_path=onnx_path, labels_path=Path(labels), taxonomy_path=Path(taxonomy),
            model_id=model_id, bounds=bounds, priority=priority, cache_size=cache_size,
        )

    @classmethod
    def _build(
        cls,
        *,
        onnx_path: Path,
        labels_path: Path,
        taxonomy_path: Path,
        model_id: str,
        bounds: Bounds,
        priority: int,
        cache_size: int,
    ) -> 'OnnxDistribution':
        for p, kind in [(onnx_path, 'onnx'), (labels_path, 'labels'),
                        (taxonomy_path, 'taxonomy')]:
            if not p.is_file():
                raise FileNotFoundError(f'{kind} not found: {p}')

        sess = _load_session(onnx_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        n_out = int(sess.get_outputs()[0].shape[-1])

        sci_to_code = _build_sci_to_code(taxonomy_path)
        rows = _load_geomodel_labels(labels_path)
        if len(rows) != n_out:
            print(
                f'[OnnxDistribution:{model_id}] label rows ({len(rows)}) '
                f"!= model output width ({n_out}); assuming row-order "
                f"matches output index.", file=sys.stderr,
            )

        idx_to_code: Dict[int, str] = {}
        n_unmapped = 0
        for i, (_tk, sci, _com) in enumerate(rows):
            if i >= n_out:
                break
            code = sci_to_code.get(sci)
            if code:
                idx_to_code[i] = code
            else:
                n_unmapped += 1
        if n_unmapped:
            print(
                f'[OnnxDistribution:{model_id}] {n_unmapped} of '
                f'{len(rows)} labels had no species_code in {taxonomy_path.name} '
                f'and are omitted from predictions.', file=sys.stderr,
            )

        return cls(
            model_id=model_id, bounds=bounds, priority=priority,
            session=sess, input_name=input_name, output_name=output_name,
            idx_to_code=idx_to_code, cache_size=cache_size,
        )

    # -- inference ---------------------------------------------------------

    def _predict_uncached(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float],   # ignored — model has no altitude input
    ) -> Dict[str, float]:
        inp = np.asarray([[float(lat), float(lon), float(week)]], dtype=np.float32)
        out = self._sess.run([self._output_name], {self._input_name: inp})[0][0]
        selected = out[self._output_indices]
        return {code: float(p) for code, p in zip(self._codes, selected)}
