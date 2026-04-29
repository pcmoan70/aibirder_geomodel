# Training & Validation Measures

This document explains every loss term and validation metric printed during
`train.py`. Cross-references point at the source of truth so you can audit the
math without re-reading the whole training loop.

---

## 1. Loss components

The model has two output heads:

- `species_logits`  →  per-species presence/absence logits (binary multi-label)
- `env_pred`        →  predicted environmental features (continuous)

The total optimization target is a weighted sum of the per-head losses,
combined in `model/loss.py:259` (`MultiTaskLoss`):

```
total = species_weight · species_loss
      + env_weight     · env_loss
      [+ habitat_weight · habitat_loss]    # only if --habitat_head is on
```

Defaults: `species_weight = 1.0`, `env_weight = 0.1`, `habitat_weight = 0.0`.

### 1.1 `species_loss`

Multi-label classification loss on the species head. The functional form is
selected with `--species_loss {bce,asl,focal,an}`:

| Choice | Loss                        | When to use                                                    |
|--------|-----------------------------|----------------------------------------------------------------|
| `bce`  | Binary cross-entropy        | Default. Treats every absent species in a checklist as a true negative. |
| `an`   | "Assume-Negative"           | Counts unobserved species as soft negatives with up-weighted positives — robust to incomplete checklists (the realistic case). |
| `focal`| Focal loss (α, γ)           | Down-weights easy negatives — useful with very long-tailed labels. |
| `asl`  | Asymmetric loss             | Different focusing for positives vs negatives + a small probability margin on negatives. |

All variants reduce to a single scalar per batch and are averaged over batches
for the printed value.

### 1.2 `env_loss`

Mean-squared error between `env_pred` and the per-cell environmental features
(elevation, climate, land-cover fractions, …). Computed by `masked_mse` so
NaN/missing channels are excluded from the average. Lower is better; magnitude
depends on the feature normalization, so compare relative trend rather than
absolute value.

### 1.3 `total_loss`

The exact scalar that gets `.backward()` called on it. Because `env_weight` is
small by default, `total ≈ species_loss + 0.1·env_loss`. If you change
`--species_weight` or `--env_weight`, keep this in mind — `total` is no longer
directly comparable across runs with different weights.

### 1.4 `habitat_loss` (optional)

Auxiliary loss applied directly to the *habitat-species head* logits before
gating, when `--habitat_head` is used. Same loss family as `species_loss`.
Gives the habitat head a full-strength gradient even while the gate keeps its
contribution small in the early epochs (`model/model.py:535`).

---

## 2. Validation metrics

Computed in `Trainer.validate` (`train.py:379`). All quality metrics use a
clamped sigmoid of the logits as the predicted probability.

### 2.1 `mAP` — mean Average Precision

Per-sample average precision averaged over samples that have at least one
positive species, then averaged across the validation set. For one sample:

```
AP = (1 / n_positives) · Σ_k  Precision@k · 1{rank-k species is a true positive}
```

where species are sorted by predicted probability descending.

This measures **ranking quality**: do the species the user actually saw get
high scores relative to the ones they didn't? Threshold-free, so it isolates
the model's ranking from any decision rule.

### 2.2 Top-k Recall (`top10_recall`, `top30_recall`)

Of all true positives across the validation batch, what fraction land in the
model's top-k predictions for that sample?

```
top-k recall = (Σ_samples #{true positives in top-k}) / (Σ_samples #{true positives})
```

`top10_recall` is the most product-relevant number — it answers "if the app
shows the user 10 species suggestions, how often does the actual species
appear?". `top30_recall` is the loose version (longer suggestion list).

### 2.3 F1 by threshold (`f1_5`, `f1_10`, `f1_25`)

At each threshold τ ∈ {0.05, 0.10, 0.25}, predict species whose probability
exceeds τ, then compute micro-averaged precision/recall/F1 from the
batch-summed TP/FP/FN counters:

```
precision_τ = TP_τ / (TP_τ + FP_τ)
recall_τ    = TP_τ / (TP_τ + FN_τ)
f1_τ        = 2 · precision_τ · recall_τ / (precision_τ + recall_τ)
```

`f1_10` (threshold = 10 %) is the headline number — it's also a GeoScore
component. Going from τ=5 % to τ=25 % traces the precision/recall trade-off:
lower threshold = longer lists, higher recall, lower precision; higher
threshold = the opposite.

### 2.4 Species List-length ratio (`list_ratio_τ`, `mean_list_len_τ`)

For a sample with `n_true` positives, the model emits `n_pred(τ)` species
above threshold τ. Two related diagnostics:

- **`list_ratio_τ`**: per-sample ratio averaged across samples with positives.
  ```
  list_ratio_τ = mean over samples_with_positives of (n_pred(τ) / n_true)
  ```
  - = 1 → predicted list length matches the observed list length on average.
  - > 1 → over-predicting (long lists).
  - < 1 → under-predicting (short lists).
- **`mean_list_len_τ`**: average number of predicted species per *all* samples
  (including those with no observed positives). Calibration aid — if the user
  expects ~5 species but the model returns 50, that's a usability problem
  even when ranking quality (mAP) is fine.

GeoScore uses `list_ratio_10` with a symmetric penalty so 0.5 and 2.0 score
identically (`model/metrics.py:43`).

### 2.5 Density-stratified mAP (`map_sparse`, `map_dense`, `map_density_ratio`)

Validation samples are split into quartiles by observation density (a per-cell
count from the data pipeline). Two sub-mAPs are reported:

- `map_sparse`: mAP on the bottom 25 % density quartile (under-surveyed cells).
- `map_dense`:  mAP on the top 25 % density quartile (well-surveyed cells).
- `map_density_ratio = map_sparse / map_dense`.

Ratio close to 1 means the model generalizes evenly across the survey-effort
spectrum; ratio ≪ 1 means it over-fits popular cells. GeoScore penalizes
*both* directions of imbalance (`min(r, 1/r)`), so artificially boosting
sparse mAP via heavy label propagation does not game the score.

### 2.6 Prediction-density correlation (`pred_density_corr`)

Pearson correlation between observation density and the number of species
predicted at τ = 0.10, computed across all validation samples. A high
correlation (positive or negative) means the model's prediction count tracks
survey effort rather than ecology — i.e., it's predicting "popular cells get
longer lists" instead of "richer habitats get longer lists". GeoScore prefers
`|r| → 0`.

### 2.7 Watchlist per-species AP (`ap_<species_code>`, `watchlist_mean_ap`)

For a curated list of conservation-relevant species (`WATCHLIST_SPECIES` in
`train.py`), AP is computed *per species* across the entire validation set
(rank validation samples by that species' predicted probability, score against
the binary presence label). `watchlist_mean_ap` is the unweighted mean across
species that have at least one positive sample. This catches degradation on
endemic / restricted-range species that would be invisible in the global mAP.

---

## 3. GeoScore — composite early-stopping metric

`compute_geoscore` (`model/metrics.py:22`) collapses the metrics above into a
single number used for "best model" selection and early stopping. Components
and weights:

| Component             | Weight | Transform                                  |
|-----------------------|-------:|--------------------------------------------|
| `map`                 |   0.20 | identity                                   |
| `f1_10`               |   0.20 | identity                                   |
| `list_ratio_10`       |   0.15 | `exp(-|log r|)` = `min(r, 1/r)`            |
| `watchlist_mean_ap`   |   0.10 | identity                                   |
| `holdout_map`         |   0.10 | identity                                   |
| `map_density_ratio`   |   0.20 | `min(r, 1/r)`                              |
| `pred_density_corr`   |   0.05 | `1 - |r|`                                  |

All transforms map into `[0, 1]` with higher = better. Missing components are
dropped and the remaining weights renormalized, so a run without the holdout
loader still produces a valid GeoScore (just with the other components
reweighted).

The training loop logs `GeoScore` each validation epoch and saves a new "best"
checkpoint whenever it improves. Early stopping (`--patience`) triggers after
`patience` epochs without improvement.

---

## 4. Quick reading guide

A healthy run typically shows:

- `train_loss`, `val_loss`: monotonically decreasing for the first many epochs;
  divergence (val going up while train still drops) = over-fit.
- `species_loss` ≫ `env_loss` × `env_weight`: expected; species head is the
  primary target.
- `mAP`, `f1_10`, `top10_recall`: increasing — the headline quality signals.
- `list_ratio_10` → 1.0: the model emits lists of the right length.
- `map_density_ratio` → 1.0: equal performance on under- and well-surveyed
  cells.
- `pred_density_corr` → 0: prediction count not driven by survey effort.
- `GeoScore`: monotonically increasing; this is what early stopping watches.

If `mAP` is rising but `GeoScore` is flat, look at `list_ratio_10` and
`map_density_ratio` — the model is probably winning on ranking while losing
on calibration or spatial bias.
