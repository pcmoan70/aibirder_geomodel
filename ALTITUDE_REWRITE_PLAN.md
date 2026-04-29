# Altitude-as-input + Two-phase Architecture — Implementation Plan

This document proposes a staged rewrite of the current `BirdNETGeoModel`:

1. Treat altitude as a *first-class input* (not a predicted env feature).
2. Split training into **habitat → species** (phase A) and
   **(lat, lon, week) → env** (phase B), then joint fine-tune (phase C).

Motivation lives in conversation history:

- Phone GPS provides altitude directly, so the model doesn't have to derive it
  from `(lat, lon)` — a regression that is lossy at mountainous sub-cell
  scale.
- Elevation currently lives in the *output* of the env head; predicted
  elevation is what feeds the habitat head, which is strictly worse than the
  true local value.
- A clean split makes the habitat model *location-agnostic* and reusable
  across regions where an env vector is available.

The plan is deliberately staged so each step is independently useful.

---

## Current architecture (for reference)

```
f_unified : (lat, lon, week)
              └── SpatioTemporalEncoder → hidden
                    ├── SpeciesPredictionHead       → species_logits (direct)
                    ├── EnvironmentalPredictionHead → env_pred  (regularizer)
                    └── HabitatSpeciesHead(env_pred.detach()) → habitat_logits
                         gated combination → final species_logits
```

env features = 22 (`elevation_m`, water/temp/precip/canopy/…).

---

## Step 1 — Altitude as a model input  (minimum viable change)

**Goal:** keep the current unified architecture but feed altitude as an input
alongside `(lat, lon, week)`. Drop `elevation_m` from the env outputs.

### Files to change

| file                                 | change                                                                                                                                                                                                                   |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `utils/data.py:60`                   | exclude `elevation_m` from `env_columns`; expose a new `altitude` column to the loader                                                                                                                                   |
| `utils/data.py:145`                  | `flatten_to_samples` returns `(lats, lons, weeks, altitudes, species_lists, env_features_df)`                                                                                                                            |
| `utils/data.py:1235`                 | `n_env_features` drops by 1 (22 → 21)                                                                                                                                                                                    |
| `model/model.py:106–224` `SpatioTemporalEncoder` | take a 4th input `altitude`; encode with a small MLP (2–4 hidden dims) rather than harmonics; concatenate to `[lat_features, lon_features]` before `input_proj`; bump `input_proj` in-dim by the altitude embed size |
| `model/model.py:597` `create_model`  | new arg `altitude_embed_dim=16`                                                                                                                                                                                         |
| `train.py:322, 441`                  | pass `altitude` through to the model; apply jitter (σ ≈ 20 m) to match phone GPS noise                                                                                                                                  |
| `predict.py:104–107`                 | require `--altitude` at inference, fall back to DEM lookup when absent                                                                                                                                                  |
| `scripts/plot_range_maps.py:299` `predict_grid` | fetch altitude from the cell's DEM value at grid build time                                                                                                                                                    |
| dataset cache key `train.py:57`      | add `altitude_feature` to the hash                                                                                                                                                                                       |

### Pros

- Tiny change surface; reversible — roll back by restoring one column.
- Tests the altitude hypothesis before committing to architectural split.
- Keeps all the training machinery (cosine LR, AMP, scheduler state).
- Immediate inference win: phone altitude is a real per-observation signal,
  not a cell-level approximation.

### Cons

- Still a single joint model; doesn't get the clean two-phase benefits.
- `n_env_features` change invalidates all existing checkpoints — full retrain
  required.
- Altitude distribution mismatch (DEM vs phone GPS) remains; only mitigated
  by jitter.

**Decision point:** if step 1 lifts mAP/top-k recall meaningfully, consider
stopping here. If gains are marginal, proceed to step 2.

---

## Step 2 — Habitat classifier as a standalone module

**Goal:** introduce `f_habitat(env_vec, altitude, week) → species_logits` that
takes **true** env features as input. Train it independently of geography.

### Files to change

| file                                | change                                                                                                                                |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `model/model.py` (new class `HabitatClassifier`) | MLP stack with FiLM on `week` features; no geography inputs                                                               |
| `train.py` (new `--phase habitat`)  | iterates over (cell, week) samples, uses EE-sampled env features **as input**, not as target                                          |
| `utils/data.py:140`                 | add a mode that yields `(env_vec_true, altitude_true, week, species_labels)` tuples                                                   |

**Loss:** same species loss family (BCE/AN/…). No env-loss term — the habitat
model doesn't predict env.

**Training augmentation:**

- Altitude: gaussian σ ≈ 20 m (phone GPS realism).
- Env features: small noise on temperature/precip (~5 % σ) to prevent
  overfitting to the exact EE bins.

### Pros

- Pure separation: the habitat model is *location-agnostic* and transferable
  to any bbox where env features are available.
- Phase A converges fast (few params, clean signal — env features are
  low-dimensional).
- Becomes a frozen artifact you can pair with different geography
  predictors.
- Publishable as its own result: "what species live in this habitat?".

### Cons

- Biogeography loss: identical habitats in Norway vs Scotland collapse to
  the same prediction. Mitigation: pass a coarse lat/lon encoding (4
  harmonics only) as auxiliary input — partial fix.
- Requires a new training loop and dataset schema.
- Two artifacts to keep in sync (habitat model + env predictor).

---

## Step 3 — Environment predictor + joint fine-tune

**Goal:** keep the existing env head but as a standalone network; then jointly
fine-tune the full pipeline on species loss.

### Files to change

| file                                        | change                                                                                                                   |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `model/model.py` (new class `EnvPredictor`) | essentially the current `SpatioTemporalEncoder + EnvironmentalPredictionHead` combined                                    |
| `train.py` (new `--phase env`)              | trains `(lat, lon, week) → env_vec` with masked MSE                                                                      |
| `model/model.py` new `Pipeline` wrapper     | `species = habitat_classifier(env_predictor(lat, lon, week), altitude, week)`                                            |
| `train.py` (new `--phase joint`)            | loads pretrained habitat + env, low-LR fine-tune end-to-end, species-loss-only                                           |

### Pros

- Fine-tune recovers the joint-learning benefit of the current architecture.
- You can **swap** which env predictor to use — e.g. a simple DEM+WorldClim
  lookup vs a learned predictor.
- Inference flexibility: at test time, if the user has env features from a
  local sensor (NDVI from phone camera?), skip `env_predictor` entirely and
  feed them directly.

### Cons

- Three training runs instead of one.
- Orchestration complexity (checkpoint paths for each phase, frozen vs
  trainable flags).
- End-to-end fine-tune can overfit small labeled regions — use a low LR
  (1e-5) and few epochs.

---

## Recommended sequencing

```
  Week 0   step 1 implementation + retrain → compare mAP vs current
  Week 1   if step 1 wins → start step 2 habitat classifier
  Week 2   step 3 env predictor + joint fine-tune
  Week 3   evaluation pass: per-species AP, density-stratified mAP, watchlist
```

## Common pitfalls across all three steps

1. **Train/test altitude datum mismatch.** Norway's DEM is orthometric
   (NN2000); phone GPS is WGS84 ellipsoid, ~40 m offset. Decide one datum and
   convert at inference.
2. **Feature scaling.** `altitude` in metres ranges 0–2500; env features are
   largely normalized. Add a `StandardScaler` step (already exists) and
   ensure `altitude` is included in it.
3. **Cache invalidation.** Changing env-feature count or adding altitude is a
   breaking schema change — bump the cache key so old `preprocessed_*.pkl`
   files are ignored.
4. **Checkpoint compatibility.** State dicts won't match across architectural
   changes — fresh training required at each step.

## When to abandon the rewrite

- If step 1's joint retrain doesn't beat run4's `val_top10_recall=0.284`,
  the altitude signal isn't the bottleneck and the split is unlikely to
  help. Look elsewhere (week encoding, label propagation, species loss
  choice).
- If step 2's habitat classifier underperforms the unified model at mAP by
  >5 %, biogeography leakage is dominant; add the coarse lat/lon auxiliary
  input or keep the unified architecture.
