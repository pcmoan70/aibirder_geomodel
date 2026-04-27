# Implementation Plan: Norway + Nordic Residual GeoModel

**File:** `implementation_plan_20260427.md`
**Date:** 2026-04-27
**Status:** Engineering plan

---

## 1. Goal

Implement a Norway-specialized Nordic bird occurrence model with:

- shared Nordic trunk,
- altitude input,
- non-harmonic learned week dependence,
- common environment head,
- optional habitat head,
- Norway-only residual branch,
- Norway geology/mineral features,
- Norway non-bird biotic indicator features,
- shared and residual smoothing,
- robust training and test infrastructure.

The implementation should be incremental. Each stage must produce a runnable model and diagnostic outputs.

---

## 2. Repository layout

Recommended structure:

```text
geomodel_norway/
    configs/
        base_shared_nordic.yaml
        norway_residual_v1.yaml
        norway_residual_v2.yaml

    data/
        raw/
        interim/
        processed/
        features/
        splits/

    model/
        coordinate_encoder.py
        altitude_encoder.py
        week_encoder.py
        country_encoder.py
        shared_trunk.py
        species_heads.py
        environment_heads.py
        norway_feature_encoders.py
        norway_residual.py
        smoothing.py
        losses.py
        full_model.py

    data_pipeline/
        download_occurrences.py
        clean_occurrences.py
        build_grid.py
        extract_common_env.py
        extract_norway_features.py
        build_biotic_indicators.py
        combine_dataset.py
        validate_dataset.py

    training/
        train.py
        train_shared.py
        train_norway_residual.py
        train_joint.py
        callbacks.py
        metrics.py
        samplers.py

    evaluation/
        evaluate.py
        plot_maps.py
        plot_week_curves.py
        plot_altitude_curves.py
        residual_diagnostics.py
        calibration.py
        spatial_roughness.py
        border_diagnostics.py

    export/
        export_onnx.py
        export_tflite.py
        test_export.py

    tests/
        unit/
        integration/
        regression/
```

This structure keeps data, model, training, evaluation, and export code separate.

---

## 3. Implementation stages

## Stage 1 — Baseline wrapper

### Objective

Reproduce a current-style shared Nordic model before adding Norway residuals.

### Tasks

1. Create data schema loader for:
   - lat,
   - lon,
   - week,
   - species label vector.
2. Implement or wrap the existing coordinate/week encoder.
3. Implement shared species head.
4. Train on Nordic observations.
5. Produce baseline diagnostics.

### Deliverables

```text
configs/base_shared_nordic.yaml
model/full_model.py with residual disabled
training/train_shared.py
reports/baseline_shared_nordic.md
```

### Acceptance criteria

- Model trains end-to-end.
- Prediction works for one lat/lon/week.
- Validation metrics are reproducible with fixed seed.
- Baseline maps can be produced.

---

## Stage 2 — Altitude input

### Objective

Add altitude as a model input.

### Files

```text
model/altitude_encoder.py
data_pipeline/extract_common_env.py
training/metrics.py
```

### Encoder

```python
class AltitudeEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, altitude_norm, missing_flag, source_flag):
        x = torch.stack([altitude_norm, missing_flag, source_flag], dim=-1)
        return self.net(x)
```

### Data fields

```text
altitude_m
altitude_norm
altitude_missing_flag
altitude_source_flag
```

### Tasks

1. Add DEM altitude extraction.
2. Add optional user/GPS altitude at inference.
3. Add altitude dropout during training.
4. Add altitude-response plots.

### Acceptance criteria

- Model works with and without altitude input.
- Missing altitude does not crash inference.
- Altitude ablation report exists.

---

## Stage 3 — Hybrid week encoder

### Objective

Allow non-harmonic seasonal structure.

### Files

```text
model/week_encoder.py
model/losses.py
evaluation/plot_week_curves.py
```

### Encoder

```python
class HybridWeekEncoder(nn.Module):
    def __init__(self, n_weeks=48, fourier_harmonics=4, learned_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_weeks, learned_dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * fourier_harmonics + learned_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
```

### Regularization

```text
L_week = sum ||E[w+1] - E[w]||^2
L_week_curvature = sum ||E[w+1] - 2E[w] + E[w-1]||^2
```

Use lower smoothness penalty during migration windows.

### Acceptance criteria

- Week curves are smooth but can be asymmetric.
- Spring/autumn maps can differ.
- Learned week embeddings do not oscillate strongly week-to-week.

---

## Stage 4 — Common Nordic environment and country embeddings

### Objective

Add shared common environment features and country-specific flexibility.

### Files

```text
model/country_encoder.py
model/environment_heads.py
data_pipeline/extract_common_env.py
```

### Tasks

1. Add `country_id`.
2. Add small country embedding.
3. Add common environmental feature encoder.
4. Add environment prediction head.
5. Add masked MSE loss for environment targets.

### Acceptance criteria

- Non-Norway countries train with common features.
- Shared model improves or remains stable.
- Country embedding does not dominate predictions.

---

## Stage 5 — Shared smoothing

### Objective

Implement smoothing in shared logits.

### Files

```text
model/smoothing.py
training/samplers.py
evaluation/spatial_roughness.py
```

### Tasks

1. Build H3/grid neighbor sampler.
2. Compute neighbor pairs in training batch.
3. Compute ecological weights from:
   - distance,
   - altitude difference,
   - common environment distance.
4. Add shared smoothing loss.
5. Log smoothing loss separately.

### Acceptance criteria

- Roughness decreases without large validation loss.
- Maps lose isolated speckles.
- Coast/inland and mountain/lowland boundaries remain plausible.

---

## Stage 6 — Norway feature extraction

### Objective

Build Norway-only high-resolution features.

### Files

```text
data_pipeline/extract_norway_features.py
data_pipeline/validate_dataset.py
```

### Feature groups

```text
topography
hydrology
ecosystem/land cover
geology/minerals
biotic indicators
effort controls
```

### Tasks

1. Implement feature extraction per grid cell or occurrence point.
2. Aggregate features at:
   - 50 m,
   - 250 m,
   - 1000 m,
   - 5000 m.
3. Add Norway masks.
4. Normalize/transform features.
5. Write feature metadata.

### Acceptance criteria

- Norway rows have Norway features.
- Non-Norway rows have Norway masks equal to zero.
- Feature distributions are logged.
- Missingness is explicit.

---

## Stage 7 — Norway residual v1

### Objective

Implement the first Norway residual branch.

### Files

```text
model/norway_feature_encoders.py
model/norway_residual.py
model/full_model.py
```

### Components

```text
NorwayFeatureEncoder
Norway species bias residual
Norway low-rank habitat residual
species-level Norway gate
residual magnitude penalty
Norway residual smoothing
```

### Formula

```text
z_final = z_shared + g_NO * z_NO_residual
```

### v1 exclusions

Disable local spatial residual.

### Acceptance criteria

- Residual branch can be trained with shared trunk frozen.
- Residual maps are smooth and interpretable.
- Residual is small for unsupported species.
- Norway validation improves on at least target species groups.

---

## Stage 8 — Norway residual v2

### Objective

Add better gates and auxiliary tasks.

### Components

```text
location-dependent gate
species sample-count gate prior
Norway ecosystem auxiliary head
Norway mineral/geochemistry auxiliary head
Norway biotic indicator auxiliary head
```

### Acceptance criteria

- Gates are lower for sparse/uncertain cases.
- Large residuals are associated with meaningful feature groups.
- Residual improves validation, not only training score.

---

## Stage 9 — Joint fine-tuning

### Objective

Align shared Nordic trunk and Norway residual.

### Tasks

1. Unfreeze upper shared layers.
2. Use low learning rate for shared trunk.
3. Use mixed batches:
   - 50–60% Norway,
   - 40–50% Sweden/Finland/Denmark.
4. Monitor cross-border discontinuity.
5. Monitor shared model degradation outside Norway.

### Acceptance criteria

- Norway improves.
- Sweden/Finland/Denmark do not degrade materially.
- Cross-border shared predictions remain smooth.
- Residual remains bounded.

---

## Stage 10 — Export and inference

### Objective

Create deployable prediction model.

### Files

```text
export/export_onnx.py
export/test_export.py
predict.py
```

### Tasks

1. Export shared-only path.
2. Export Norway-residual path.
3. Add inference feature loader.
4. Add diagnostic mode.
5. Test with missing Norway features.
6. Benchmark model size and latency.

### Acceptance criteria

- Inference works inside and outside Norway.
- Missing Norway features return shared model prediction.
- Exported model agrees with PyTorch within tolerance.
- Diagnostic output includes shared, residual, gate, final.

---

## 4. Python class sketch

```python
class NorwayNordicGeoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coord_encoder = CoordinateEncoder(config)
        self.alt_encoder = AltitudeEncoder(config)
        self.week_encoder = HybridWeekEncoder(config)
        self.country_encoder = CountryEncoder(config)
        self.common_env_encoder = CommonEnvEncoder(config)

        self.shared_trunk = SharedTrunk(config)
        self.shared_species_head = LowRankSpeciesHead(config)
        self.env_head = EnvironmentHead(config)
        self.habitat_head = HabitatHead(config)

        self.norway_residual = NorwayResidualBranch(config)

    def forward(self, batch):
        h_coord = self.coord_encoder(batch["lat"], batch["lon"])
        h_alt = self.alt_encoder(
            batch["altitude_norm"],
            batch["altitude_missing"],
            batch["altitude_source"],
        )
        h_week = self.week_encoder(batch["week"])
        h_country = self.country_encoder(batch["country_id"])
        h_env = self.common_env_encoder(batch["common_env"], batch["common_env_mask"])

        h_shared = self.shared_trunk(
            h_coord=h_coord,
            h_alt=h_alt,
            h_week=h_week,
            h_country=h_country,
            h_env=h_env,
        )

        z_direct = self.shared_species_head(h_shared)
        env_pred = self.env_head(h_shared)
        z_habitat = self.habitat_head(env_pred.detach())
        z_shared = combine_shared_logits(z_direct, z_habitat)

        z_final = z_shared

        if self.norway_residual.enabled:
            z_no, gate_no = self.norway_residual(batch, h_shared, h_week)
            is_no = batch["country_id"] == COUNTRY_NO
            z_final = torch.where(
                is_no[:, None],
                z_shared + gate_no * z_no,
                z_shared,
            )

        return {
            "logits": z_final,
            "z_shared": z_shared,
            "env_pred": env_pred,
            "z_no_residual": z_no if self.norway_residual.enabled else None,
            "gate_no": gate_no if self.norway_residual.enabled else None,
        }
```

---

## 5. Configuration-driven design

All major decisions should be config-controlled:

```yaml
altitude:
  enabled: true

week_encoder:
  type: fourier_plus_learned

smoothing:
  shared:
    enabled: true
  norway_residual:
    enabled: true

norway_residual:
  enabled: true
  local_spatial_residual: false

biotic_features:
  enabled: true
  use_embeddings: false
```

Avoid hard-coding model variants.

---

## 6. Logging

Each run must log:

```text
data version
feature version
taxonomy version
config hash
git commit
random seed
training stage
loss components
validation metrics
map diagnostics
model size
inference latency
```

Use TensorBoard, Weights & Biases, MLflow, or simple JSON logs.

---

## 7. Minimum viable implementation

The first useful model should include:

```text
shared Nordic trunk
altitude input
hybrid week encoder
country embedding
common environment head
shared smoothing
Norway feature encoder
Norway low-rank residual
species-level Norway gate
Norway residual smoothing
```

Delay:

```text
local spatial residual
learned biotic assemblage embedding
advanced gates
complex uncertainty modeling
```

until v1 works.

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

