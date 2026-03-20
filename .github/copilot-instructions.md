# BirdNET Geomodel Project

## Coding Guidelines

1. **Fix root causes, not symptoms.** Never add hacky fallbacks, arbitrary clamps, or
   band-aid workarounds. Diagnose *why* something fails (e.g. unbounded FiLM gamma
   causing FP16 overflow) and fix the architecture or logic that produces the bad
   state. If a fix involves a magic constant or a `try/except` that silences an
   error, it's wrong.

2. **Document consistently.** Every non-trivial change must be documented in *all*
   relevant places — code docstrings, README, and MkDocs (`docs/`). Keep them in
   sync. If a feature is added or changed, update the matching docs page. Use
   Google-style docstrings. Add inline comments only where the *why* isn't obvious.

3. **Use American English.** All prose — code comments, docstrings, documentation,
   and report text — must use American English spelling (e.g. "modeling" not
   "modelling", "optimization" not "optimisation", "maximize" not "maximise").

## Overview

Spatiotemporal species occurrence prediction using H3 geospatial cells and weekly
temporal data. Predicts which species occur at a given (lat, lon, week) — no
environmental inputs at inference.

## Architecture

Multi-task model: raw (lat, lon, week) → multi-harmonic circular encoding →
FiLM-conditioned residual encoder → two heads:
- **Species head**: multi-label classification (BCE/ASL/focal/AN loss)
- **Env head**: regression on environmental features (auxiliary, training only)
- **Habitat head** (optional, `--habitat_head`): env_pred → species logits,
  gate-combined with direct species head

FiLM conditioning: week encoding → per-block (γ, β); γ bounded via tanh in (0, 2),
zero-init output layers. Encoder uses pre-norm residual blocks
(LayerNorm eps=1e-4 for FP16 safety). Model size controlled by continuous
`model_scale` factor (0.5 → ~1.8M, 1.0 → ~7.2M, 2.0 → ~36M params).

**Training**: AMP on CUDA, gradient clipping (max_norm=1.0), cosine LR with
3-epoch warmup, early stopping on GeoScore. Data cached to
`checkpoints/.data_cache/`. Optuna autotune via `--autotune`.

**Inference** (`predict.py`): (lat, lon, week) → species probabilities.
**Export** (`convert.py`): PyTorch → ONNX/TFLite/SavedModel with FP16/INT8 quantisation.

## Data

Parquet files with H3 cells × 48 weekly species lists + environmental features.
Species identified by eBird codes (birds) or iNaturalist IDs (non-birds).
Key pipeline classes in `utils/data.py`: `H3DataLoader`, `H3DataPreprocessor`,
`BirdSpeciesDataset`.

## Project Structure

```
train.py / predict.py / convert.py    — Training, inference, export
model/model.py                        — Network architecture
model/loss.py                         — Loss functions (BCE, ASL, focal, AN, masked MSE)
model/metrics.py                      — GeoScore and validation metrics
model/autotune.py                     — Optuna hyperparameter search
utils/data.py                         — Data loading, preprocessing, Dataset
utils/geoutils.py                     — Earth Engine feature extraction
utils/gbifutils.py                    — GBIF occurrence retrieval
utils/combine.py                      — Merge EE + GBIF into parquet
utils/regions.py                      — Holdout region definitions
scripts/plot_*.py                     — Visualization scripts
docs/                                 — MkDocs documentation site
```

## Key Design Decisions

- Env features are **training targets only** — never model inputs
- Location-based train/val split prevents spatial data leakage
- Sparse species encoding (packed index arrays) avoids memory bloat with forked workers
- FiLM gamma uses `1 + tanh(raw)` (bounded, smooth) — not raw addition — to prevent
  compound FP16 overflow through deep encoder stacks
- GeoScore (weighted composite of mAP, F1, list-ratio, watchlist AP, holdout mAP,
  density ratio) is the primary optimisation target
