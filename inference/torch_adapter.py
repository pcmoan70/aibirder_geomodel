"""PyTorch backend for :class:`DistributionModel`.

Wraps a checkpoint trained by ``train.py``. The checkpoint indexes species
by GBIF ``taxonKey``; this adapter translates to eBird ``species_code``
via the taxonomy CSV so the output is comparable with BirdNET's TFLite
geomodel.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import torch

from .base import Bounds, CachedModelMixin

# Add repo root to sys.path so ``from model.model import create_model`` works
# regardless of where this package is imported from.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from model.model import create_model  # noqa: E402


def _build_taxon_to_code(taxonomy_path: Path) -> Dict[str, str]:
    """Return ``taxonKey -> species_code`` from a BirdNET-style taxonomy CSV.

    The repo's ``taxonomy.csv`` keys species by ``species_code`` and carries
    ``sci_name``. Our model's labels key by ``taxonKey`` via ``sci_name``.
    We therefore join on ``sci_name`` to obtain the taxonKey → species_code
    mapping.

    Also accepts ``combined_taxonomy.csv`` (produced by ``utils/combine.py``)
    where ``taxonKey`` and ``species_code`` are both present — that case
    short-circuits the sci_name lookup. Caller will then need the
    secondary ``taxonomy.csv`` to get the *actual* 6-letter eBird code, but
    many callers just want *any* stable id.
    """
    mapping: Dict[str, str] = {}
    with taxonomy_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = [f.lower() for f in (reader.fieldnames or [])]
        has_taxonkey = 'taxonkey' in fieldnames
        for row in reader:
            code = (row.get('species_code') or '').strip()
            if not code:
                continue
            if has_taxonkey:
                tk = str(row.get('taxonKey') or '').strip()
                if tk:
                    mapping[tk] = code
    return mapping


def _build_sci_to_code(taxonomy_path: Path) -> Dict[str, str]:
    """Return ``sci_name -> species_code``, used as a fallback join key."""
    mapping: Dict[str, str] = {}
    with taxonomy_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sci = (row.get('sci_name') or '').strip()
            code = (row.get('species_code') or '').strip()
            if sci and code:
                mapping[sci] = code
    return mapping


def _load_labels_file(labels_path: Path) -> List[tuple]:
    """Return list of (taxonKey_str, sci_name, com_name) in model-index order."""
    rows: List[tuple] = []
    with labels_path.open(encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            tk = parts[0] if parts else ''
            sci = parts[1] if len(parts) > 1 else tk
            com = parts[2] if len(parts) > 2 else sci
            rows.append((tk, sci, com))
    return rows


class TorchDistribution(CachedModelMixin):
    """DistributionModel backed by a ``train.py`` PyTorch checkpoint.

    The constructor expects an already-loaded model; use
    :meth:`from_checkpoint` for the common "load from disk" case.
    """

    def __init__(
        self,
        *,
        model_id: str,
        bounds: Bounds,
        priority: int,
        model: torch.nn.Module,
        idx_to_code: Dict[int, str],   # model output index → species_code
        device: torch.device,
        has_altitude_input: bool = False,
        cache_size: int = 4096,
    ) -> None:
        super().__init__(
            model_id=model_id,
            bounds=bounds,
            priority=priority,
            vocab=frozenset(idx_to_code.values()),
            cache_size=cache_size,
        )
        self._model = model
        self._idx_to_code = dict(idx_to_code)
        # Prebuilt torch tensor of the indices we actually return.
        self._indices = torch.tensor(
            sorted(idx_to_code.keys()), dtype=torch.long, device=device,
        )
        self._codes = [idx_to_code[i] for i in self._indices.tolist()]
        self._device = device
        self._has_altitude = bool(has_altitude_input)

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        taxonomy: str,
        *,
        model_id: str,
        bounds: Bounds,
        priority: int = 100,
        device: Optional[str] = None,
        cache_size: int = 4096,
        labels: Optional[str] = None,
    ) -> 'TorchDistribution':
        """Load a checkpoint and return a ready-to-call wrapper.

        Parameters
        ----------
        checkpoint : str
            Path to ``checkpoint_best.pt`` / ``checkpoint_latest.pt``.
        taxonomy : str
            BirdNET-style taxonomy CSV providing ``species_code`` per
            ``sci_name`` (repo root's ``taxonomy.csv``).
        model_id, bounds, priority
            See :class:`DistributionModel`.
        device : str, optional
            'cuda', 'cpu', or None = auto.
        labels : str, optional
            Override the labels file that lives next to the checkpoint.
        """
        ckpt_path = Path(checkpoint)
        tax_path = Path(taxonomy)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
        if not tax_path.is_file():
            raise FileNotFoundError(f'taxonomy not found: {tax_path}')

        dev = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        mcfg = ckpt['model_config']
        vocab = ckpt['species_vocab']
        idx_to_taxonkey = vocab['idx_to_species']  # int → numpy.int64 taxonKey

        model = create_model(
            n_species=mcfg['n_species'],
            n_env_features=mcfg['n_env_features'],
            model_scale=mcfg.get('model_scale', 1.0),
            coord_harmonics=mcfg.get('coord_harmonics', 8),
            week_harmonics=mcfg.get('week_harmonics', 4),
            habitat_head=mcfg.get('habitat_head', False),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(dev).eval()

        # Build taxonKey → species_code by joining on sci_name via the
        # checkpoint's labels file. If labels lack names, fall back to the
        # taxonomy CSV's own taxonKey column where available.
        labels_path = Path(labels) if labels else (ckpt_path.parent / 'labels.txt')
        if not labels_path.exists():
            labels_path = ckpt_path.parent / f'{ckpt_path.stem}_labels.txt'
        idx_to_code: Dict[int, str] = {}
        sci_to_code = _build_sci_to_code(tax_path)
        tax_taxon_to_code = _build_taxon_to_code(tax_path)
        n_missing = 0
        if labels_path.exists():
            rows = _load_labels_file(labels_path)
            for idx, (tk, sci, _com) in enumerate(rows):
                code = None
                if tk in tax_taxon_to_code:
                    code = tax_taxon_to_code[tk]
                elif sci and sci in sci_to_code:
                    code = sci_to_code[sci]
                if code is not None:
                    idx_to_code[idx] = code
                else:
                    n_missing += 1
        else:
            # No labels file — try the taxonomy CSV's taxonKey column only.
            for idx_key, tk_raw in idx_to_taxonkey.items():
                tk = str(int(tk_raw))
                code = tax_taxon_to_code.get(tk)
                if code:
                    idx_to_code[int(idx_key)] = code
                else:
                    n_missing += 1
        if n_missing:
            print(
                f'[TorchDistribution:{model_id}] {n_missing} species '
                f'could not be mapped to a species_code and will be silently '
                f'omitted from predictions.',
                file=sys.stderr,
            )

        return cls(
            model_id=model_id,
            bounds=bounds,
            priority=priority,
            model=model,
            idx_to_code=idx_to_code,
            device=dev,
            cache_size=cache_size,
        )

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def _predict_uncached(
        self,
        lat: float,
        lon: float,
        week: int,
        altitude: Optional[float],
    ) -> Dict[str, float]:
        lat_t = torch.tensor([float(lat)], dtype=torch.float32, device=self._device)
        lon_t = torch.tensor([float(lon)], dtype=torch.float32, device=self._device)
        week_t = torch.tensor([float(week)], dtype=torch.float32, device=self._device)

        # model API: model(lat, lon, week, return_env=False)
        # altitude is currently unsupported by the trained checkpoint; the
        # parameter here is kept for future compatibility.
        out = self._model(lat_t, lon_t, week_t, return_env=False)
        logits = out['species_logits'][0]            # (n_species,)
        logits = logits.clamp(-30, 30)
        probs = torch.sigmoid(logits)
        selected = probs.index_select(0, self._indices).cpu().numpy()
        return {code: float(p) for code, p in zip(self._codes, selected)}
