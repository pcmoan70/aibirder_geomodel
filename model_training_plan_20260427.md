# Model Training Plan

**File:** `model_training_plan_20260427.md`
**Date:** 2026-04-27
**Status:** Training and experiment plan

---

## 1. Objective

Train a Norway-optimized Nordic bird occurrence model in stages:

1. Shared Nordic model.
2. Shared Nordic model with altitude and learned week.
3. Shared Nordic model with common environment and smoothing.
4. Norway residual branch.
5. Norway residual branch with mineral and biotic features.
6. Joint fine-tuning.
7. Final selection and export.

Each stage must be validated before moving to the next.

---

## 2. Data splits

Use multiple validation schemes.

### 2.1 Random observation split

Purpose:

```text
fast sanity check
overfitting detection
general metric tracking
```

Not sufficient for final selection.

### 2.2 Spatial block split

Hold out Norwegian regions:

```text
Østlandet
Vestlandet
Trøndelag
Nordland
Troms
Finnmark
mountain regions
coastal regions
wetland-rich regions
forest-rich regions
```

Purpose:

```text
test spatial generalization
test interpolation into unseen areas
```

### 2.3 Cross-border split

Hold out border zones:

```text
Norway-Sweden border
Norway-Finland border
```

Purpose:

```text
test whether Swedish/Finnish data help Norwegian predictions
test border continuity
```

### 2.4 Altitude-band split

Hold out altitude bands:

```text
0–100 m
100–300 m
300–700 m
700–1200 m
>1200 m
```

Purpose:

```text
test vertical generalization
detect altitude overfit
```

### 2.5 Ecosystem split

Hold out ecosystem classes:

```text
old forest
rich fen
bog/mire
coastal heath
alpine snowbed
semi-natural grassland
calcareous forest
wet meadow/marshland
```

Purpose:

```text
test Norway high-resolution feature generalization
```

### 2.6 Temporal split

Hold out years or year ranges.

Purpose:

```text
test temporal robustness
prevent future leakage
evaluate biotic indicator time-safety
```

---

## 3. Training stage 0 — Baseline

### Model

```text
lat/lon/week only
shared species head
no altitude
no Norway residual
```

### Purpose

Establish baseline metrics and maps.

### Outputs

```text
baseline_metrics.json
baseline_maps/
baseline_week_curves/
baseline_altitude_curves_if_altitude_probed/
```

### Decision

Proceed when baseline training is reproducible.

---

## 4. Training stage 1 — Altitude + learned week

### Model

```text
lat/lon
altitude
hybrid week encoder
shared species head
```

### Loss

```text
species loss
week smoothness loss
week curvature loss
```

### Experiments

```text
A0: baseline
A1: altitude only
A2: learned week only
A3: altitude + learned week
A4: altitude + learned week + altitude dropout
```

### Target improvements

```text
alpine species
coastal/lowland separation
migration phenology
autumn/spring asymmetry
```

### Diagnostics

```text
probability vs altitude
probability vs week
spring vs autumn maps
```

---

## 5. Training stage 2 — Shared Nordic environment and smoothing

### Model

```text
shared trunk
country embedding
common environment encoder
environment head
optional habitat head
shared smoothing
border smoothing
```

### Loss

```text
L_species
+ lambda_env L_common_env
+ lambda_smooth L_shared_smooth
+ lambda_border L_border
+ lambda_week L_week
```

### Experiments

```text
S0: no common env
S1: common env input only
S2: common env + environment head
S3: common env + environment head + habitat head
S4: S3 + shared smoothing
S5: S4 + border smoothing
```

### Target improvements

```text
better broad Nordic generalization
less spatial speckle
better cross-border continuity
better rare species behavior
```

---

## 6. Training stage 3 — Norway residual v1

### Model

```text
shared trunk frozen
Norway feature encoder
Norway species bias residual
Norway low-rank habitat residual
species-level gate
Norway residual smoothing
```

### Loss

```text
L_species_NO
+ lambda_NO_smooth L_smooth_NO
+ lambda_residual mean(z_NO_residual^2)
+ lambda_gate mean(g_NO^2)
```

### Batch composition

Primarily Norway:

```text
Norway observations: 80–100%
small Nordic calibration batch: 0–20%
```

### Experiments

```text
R0: shared model only
R1: species bias residual only
R2: habitat residual from topography/ecosystem
R3: R2 + geology/minerals
R4: R3 + biotic indicators
R5: R4 + residual smoothing
```

### Acceptance

Residual must improve Norway validation without producing noisy maps.

---

## 7. Training stage 4 — Norway residual v2

### Model

Add:

```text
location-dependent gate
sample-count gate prior
Norway auxiliary ecosystem/mineral/biotic heads
```

### Loss

```text
L_species_NO
+ L_smooth_NO
+ L_residual_magnitude
+ L_gate_prior
+ L_NO_aux
```

### Gate prior

For species `s`:

```text
g_prior_s = sigmoid(a + b * log(1 + n_NO_s))
```

This lets common well-sampled Norwegian species use stronger residuals than rare unsupported species.

### Acceptance

```text
gates make ecological sense
large gates occur for species with Norway data support
sparse species remain mostly shared-model driven
```

---

## 8. Training stage 5 — Joint fine-tuning

### Model

Full model:

```text
shared Nordic trunk
Norway residual branch
all heads
```

### Learning rates

```text
shared trunk:   1e-5 to 3e-5
shared heads:   3e-5 to 1e-4
Norway branch:  1e-4
gates:          1e-4
```

### Batch mix

Start:

```text
Norway: 60%
Sweden: 15%
Finland: 15%
Denmark: 10%
```

Alternative:

```text
Norway: 50%
Sweden/Finland/Denmark: 50%
```

### Loss

Full loss:

```text
L =
    L_species
  + lambda_env L_common_env
  + lambda_NO_env L_norway_aux
  + lambda_smooth L_smooth_shared
  + lambda_NO_smooth L_smooth_NO
  + lambda_border L_border
  + lambda_week L_week
  + lambda_week2 L_week_curvature
  + lambda_residual L_residual_magnitude
  + lambda_gate L_gate_regularization
```

### Acceptance

```text
Norway improves
non-Norway does not degrade materially
border discontinuity remains low
residual remains bounded
maps are ecologically plausible
```

---

## 9. Optimizer and training details

Suggested optimizer:

```text
AdamW
```

Initial hyperparameters:

```text
lr_shared_stage1: 1e-4
lr_residual_stage3: 1e-4
lr_joint_shared: 1e-5 to 3e-5
lr_joint_residual: 1e-4
weight_decay: 1e-4
batch_size: as large as GPU allows
mixed_precision: true
gradient_clip_norm: 1.0
```

Loss for species:

```text
BCEWithLogitsLoss
or Asymmetric Loss for imbalanced labels
```

Use per-species weights carefully. Overweighting rare species can create artifacts.

---

## 10. Sampling strategy

Training batches should include:

```text
positive bird observation samples
background/pseudo-absence samples
grid-based regularization samples
smoothing neighbor samples
border samples
```

### Norway residual batches

Include extra Norway samples from:

```text
old forest
wetland
rich fen
alpine
coastal
calcareous areas
lowland agriculture
urban/edge
```

This ensures the residual sees all major Norwegian habitats.

---

## 11. Hyperparameter tuning

Tune in this order:

1. altitude dropout,
2. learned week dimension and smoothness,
3. shared smoothing lambda,
4. Norway residual rank,
5. residual L2 penalty,
6. Norway smoothing lambda,
7. gate prior strength,
8. biotic indicator feature set.

Do not tune everything at once.

### Suggested search ranges

```text
week learned_dim:        [32, 64, 128]
week_smooth_lambda:      [1e-5, 1e-4, 1e-3]
shared_smooth_lambda:    [1e-5, 1e-4, 3e-4, 1e-3]
NO_residual_rank:        [32, 64, 128]
NO_residual_l2:          [3e-4, 1e-3, 3e-3, 1e-2]
NO_smooth_lambda:        [1e-4, 3e-4, 1e-3]
gate_initial:            [0.1, 0.2, 0.3]
```

---

## 12. Model selection

Do not select solely by global mAP.

Selection criteria:

```text
Norway validation score
held-out Norwegian region score
rare/species-group score
spatial roughness
residual roughness
border discontinuity
calibration
migration map plausibility
altitude response plausibility
map inspection for selected species
```

Recommended model card should include:

```text
global metrics
Norway metrics
species-group metrics
failure cases
known limitations
data source limitations
```

---

## 13. Required diagnostic plots

For each candidate model:

```text
overall validation curves
loss component curves
calibration plots
spatial roughness maps
Norway residual maps
shared vs residual vs final maps
week curves for selected species
altitude curves for selected species
spring/autumn migration maps
border discontinuity maps
feature importance / ablation summaries
```

Species groups for inspection:

```text
alpine birds
coastal birds
forest specialists
wetland/marshland birds
old-growth-associated birds
calcareous habitat-associated birds
migrants
rare range-edge species
common residents
```

---

## 14. Final training run

When the configuration is selected:

1. Freeze data version.
2. Freeze feature version.
3. Freeze splits.
4. Run final training with at least 3 random seeds.
5. Compare seed variability.
6. Select either best seed or ensemble if deployment allows.
7. Export model.
8. Generate final model report.

Final outputs:

```text
checkpoint.pt
model_config.yaml
feature_stats.json
species_taxonomy.csv
validation_report.md
model_card.md
exported_model.onnx
```

---


## References and source notes

The design is intentionally independent of one exact upstream code commit, but it assumes the current public BirdNET GeoModel shape:
a multi-task neural network using raw latitude/longitude/week inputs, internal circular encoding, multi-label species classification,
an auxiliary environmental regression head, and an optional habitat head that maps predicted environmental features to species logits.

Public sources checked on 2026-04-27:

- BirdNET GeoModel repository: https://github.com/birdnet-team/geomodel
- BirdNET GeoModel README/model summary: https://github.com/birdnet-team/geomodel
- Artsdatabanken / Norwegian Species Observation Service IPT: https://ipt.artsdatabanken.no/resource?r=speciesobservationsservice2
- GBIF dataset page for Norwegian Species Observation Service: https://www.gbif.org/dataset/b124e1e0-4755-430f-9eab-894f25a9b59c
- Artsdatabanken observation access notes: https://artsdatabanken.no/Pages/180954/Observations
- NGU geological datasets: https://www.ngu.no/en/geologiske-kart/datasett
- NGU downloads: https://geo.ngu.no/download/order?lang=en
- NGU bedrock ecology WMS notes: https://www.ngu.no/en/taxonomy/term/36
- NIBIO National Land Resource Map / AR5: https://www.nibio.no/en/subjects/soil/national-land-resource-map
- NIBIO AR5 WMS: https://www.nibio.no/tjenester/wms-tjenester/wms-tjenester-ar5

