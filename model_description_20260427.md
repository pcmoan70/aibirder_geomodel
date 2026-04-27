# Norway + Nordic Bird Occurrence GeoModel

**File:** `model_description_20260427.md`
**Date:** 2026-04-27
**Status:** Model design specification
**Target:** A Norway-optimized bird occurrence prior using all available Nordic observations and Norway-specific high-resolution ecological data.

---

## 1. Purpose

This document describes a proposed extension of the BirdNET GeoModel idea for a high-accuracy Norway model.

The model should:

1. Use observations from Norway, Sweden, Finland, Denmark, and optionally nearby regions.
2. Learn broad Nordic bird ecology, migration, altitude response, habitat response, and species correlations.
3. Use Norway-only high-resolution features to sharpen predictions inside Norway.
4. Preserve smoothness across geography, altitude, habitat, geology, and biotic indicators.
5. Allow non-harmonic seasonal patterns, including asymmetric spring and autumn migration routes.
6. Remain simple enough to train, debug, export, and run in a practical inference pipeline.

The central design is:

```text
final Norway logit =
    shared Nordic logit
  + Norway gate * Norway residual logit
```

Outside Norway:

```text
final logit = shared Nordic logit
```

The Norway branch is therefore a residual correction, not a full replacement model.

---

## 2. Summary of the upstream baseline

The current public BirdNET GeoModel is a compact spatiotemporal occurrence model. Its public README describes a multi-task neural network that learns spatial-temporal patterns from raw `(lat, lon, week)`, performs multi-label species classification, uses environmental feature regression as an auxiliary regularizer, and optionally uses a habitat head where predicted environmental features are mapped to species logits and combined with the direct species head through a learned gate.

This proposal keeps those principles and extends them with:

- altitude as a first-class input,
- learned/non-harmonic week dependence,
- country embeddings,
- common Nordic environmental features,
- Norway-only high-resolution features,
- Norway residual correction,
- Norway residual smoothing,
- biotic indicator features from non-bird taxa,
- mineral/geology features,
- rigorous leakage controls.

---

## 3. High-level architecture

```text
Inputs:
    lat, lon, altitude, week
    country_id
    common Nordic environmental features
    Norway-only high-resolution features
    non-bird biotic indicator features
    feature masks

Shared Nordic model:
    coordinate encoder
    altitude encoder
    hybrid week encoder
    country embedding
    common environment encoder
        ↓
    shared ecological embedding
        ↓
    shared direct species head
    shared environment head
    shared habitat head

Norway residual model:
    Norway topography/hydrology/ecosystem encoder
    Norway geology/mineral encoder
    Norway non-bird biotic indicator encoder
    Norway residual habitat head
    Norway species bias residual
    optional Norway local spatial residual
    Norway residual gate

Output:
    if outside Norway:
        logits = shared_logits

    if inside Norway:
        logits = shared_logits + gate_NO * residual_logits_NO
```

The architecture deliberately separates:

```text
shared model = general Nordic ecology and migration
Norway residual = local high-resolution correction
```

This separation is the most important design decision.

---

## 4. Why not a separate Norway-only model?

A Norway-only model would have good local detail but would waste useful structure from Sweden, Finland, Denmark, and possibly surrounding regions.

The shared Nordic model can learn:

- migration timing and routes,
- broad boreal/alpine/coastal ecology,
- species correlations,
- rare-species habitat preferences,
- cross-border continuity,
- altitude response from a larger ecological sample.

The Norway residual can then learn:

- high-resolution Norwegian terrain effects,
- geology and calcium effects,
- local ecosystem and vegetation effects,
- old-growth/marshland/fen indicators,
- fine habitat corrections,
- Norway-specific observation/reporting effects.

This gives a better bias-variance tradeoff than either a pure Nordic model or a pure Norway model.

---

## 5. Inputs and feature groups

All input features are grouped by availability and role.

### 5.1 Universal features

Available for all countries:

```text
lat
lon
week
altitude_m
altitude_missing_flag
altitude_source_flag
country_id
```

`country_id` is a categorical value:

```text
NO, SE, FI, DK
```

Additional countries can be added if data are included.

### 5.2 Altitude input

Altitude should be a first-class input, not only an environmental target.

Suggested normalized altitude:

```text
altitude_norm = clip((altitude_m - 500) / 1500, -2, 4)
```

Altitude source flag:

```text
0 = DEM/grid altitude
1 = GPS/user altitude
2 = unknown/imputed
```

The model should learn different reliability for DEM altitude and GPS altitude.

### 5.3 Week input

The model should not force all seasonal structure to be harmonic.

Use:

```text
week_features =
    Fourier/circular week features
  + learned week embedding
```

This allows:

- sharp migration peaks,
- asymmetric spring and autumn routes,
- autumn abundance much higher than spring,
- different phenology by species and region.

Suggested initial migration windows:

```text
spring: weeks 10–22
autumn: weeks 30–43
```

These are only regularization windows, not hard-coded phenology.

### 5.4 Common Nordic environmental features

Features available for all countries:

```text
coarse_elevation
temperature_summaries
precipitation_summaries
land_cover
water_fraction
urban_fraction
forest_fraction
distance_to_coast
distance_to_freshwater
canopy_height_if_consistent
snow_or_climate_proxy
```

These go into the shared Nordic trunk and the shared environmental auxiliary head.

### 5.5 Norway-only high-resolution features

Available only inside Norway:

```text
topography:
    elevation_10m
    slope
    aspect_sin
    aspect_cos
    terrain_ruggedness
    topographic_position_index
    topographic_wetness_index
    solar_exposure
    cold_air_pooling_proxy

hydrology/coast:
    distance_to_coast
    distance_to_freshwater
    distance_to_river
    distance_to_lake
    distance_to_wetland
    wetland_fraction_250m
    wetland_fraction_1000m
    marine_influence_proxy

land cover / ecosystem:
    forest_fraction_50m
    forest_fraction_250m
    forest_fraction_1000m
    conifer_forest_fraction
    deciduous_forest_fraction
    old_forest_fraction
    open_mountain_fraction
    bog_mire_fraction
    agricultural_fraction
    urban_fraction
    NiN_nature_type_embedding
    AR5_land_resource_embedding

climate/snow:
    mean_temperature
    precipitation
    snow_persistence_proxy
    growing_degree_days
    frost_frequency_proxy
```

### 5.6 Norway geology, calcium, and mineral features

Norway-specific geology/mineral block:

```text
bedrock_lithology_embedding
carbonate_bedrock_fraction_250m
carbonate_bedrock_fraction_1000m
base_rich_bedrock_fraction_1000m
acidic_bedrock_fraction_1000m
mafic_bedrock_fraction_1000m

calcium_ca_mean_250m
calcium_ca_mean_1000m
calcium_ca_p90_1000m
magnesium_mg_mean_1000m
phosphorus_p_mean_1000m
potassium_k_mean_1000m
sodium_na_mean_1000m
sulfur_s_mean_1000m
iron_fe_mean_1000m
manganese_mn_mean_1000m
zinc_zn_mean_1000m
copper_cu_mean_1000m

base_richness_index
calcareousness_index
nutrient_richness_index
acidic_soil_proxy
soil_ph_proxy
cation_richness_index
trace_metal_stress_index
```

Recommended first-version subset:

```text
calcium_ca
magnesium_mg
phosphorus_p
potassium_k
carbonate_bedrock_fraction
base_richness_index
calcareousness_index
acidic_bedrock_fraction
nutrient_richness_index
soil_ph_proxy
```

### 5.7 Norway non-bird biotic indicator features

Use flora, fungi, lichens, mosses, insects, molluscs, and other non-bird observations as derived ecological indicators.

Do not feed raw non-bird observations directly. Convert them into smoothed ecological indicator features.

Recommended v1 indicators:

```text
old_growth_forest_indicator
deadwood_forest_indicator
rich_deciduous_forest_indicator
calcareous_forest_indicator
boreal_conifer_forest_indicator
rich_fen_indicator
poor_bog_indicator
marshland_reedbed_indicator
wet_meadow_indicator
eutrophic_freshwater_indicator
oligotrophic_freshwater_indicator
coastal_heath_indicator
saltmarsh_shore_meadow_indicator
sand_dune_indicator
alpine_heath_indicator
alpine_snowbed_indicator
calcareous_alpine_indicator
dry_warm_slope_indicator
semi_natural_grassland_indicator
urban_ruderal_indicator
```

Effort-control features:

```text
nonbird_observation_count_250m
nonbird_observation_count_1000m
nonbird_observer_count_1000m
nonbird_species_count_1000m
nonbird_observation_days_1000m
nonbird_taxon_group_count_1000m
```

Optional learned features:

```text
biotic_embedding_01 ... biotic_embedding_32
```

These should be learned from non-bird taxa only.

---

## 6. Data table schema

The combined model table should have one row per training sample or grid cell.

```text
sample_id
taxon_id
species_label_vector
lat
lon
week
year
country_id
altitude_m
altitude_missing_flag
altitude_source_flag

common_env_*
common_env_mask_*

norway_topography_*
norway_hydrology_*
norway_ecosystem_*
norway_geology_*
norway_minerals_*
norway_biotic_indicator_*
norway_env_mask_*

observation_weight
source_id
spatial_cell_id
region_holdout_id
temporal_holdout_id
quality_flags
```

For non-Norway rows:

```text
norway_* = 0
norway_env_mask_* = 0
```

For Norway rows:

```text
norway_* = real values
norway_env_mask_* = 1
```

Never impute fake Norway-only features outside Norway. Missingness must be explicit.

---

## 7. Shared Nordic model

### 7.1 Encoders

The shared model encodes:

```text
lat/lon
altitude
week
country_id
common environment
```

Coordinate encoder:

```text
lat/lon -> Fourier/circular spatial features -> MLP -> spatial embedding
```

Altitude encoder:

```text
altitude_norm, missing_flag, source_flag -> MLP -> altitude embedding
```

Week encoder:

```text
Fourier week encoding + learned week embedding -> week embedding
```

Country embedding:

```text
country_id -> small embedding, e.g. dim 8
```

Common environmental encoder:

```text
common_env, common_env_mask -> MLP -> common environment embedding
```

Concatenate:

```text
h0 = concat(spatial, altitude, week, country, common_env)
```

Then pass through residual blocks with FiLM modulation from the week embedding.

### 7.2 Shared direct species head

Use a low-rank/factorized species head:

```text
h_shared -> latent factors -> species logits
```

Conceptual form:

```text
z_shared_direct_s = dot(A h_shared, E_s) + b_s
```

This encourages species to share ecological factors.

### 7.3 Shared environment head

The shared environmental head predicts common environmental features:

```text
h_shared -> predicted_common_env
```

This is an auxiliary training-only regularizer.

### 7.4 Shared habitat head

Optional but recommended:

```text
predicted_common_env -> habitat_species_logits
```

Then:

```text
z_shared =
    z_shared_direct
  + gate_shared_habitat * z_shared_habitat
```

This keeps an explicit environment-to-species pathway.

---

## 8. Norway residual model

The Norway residual branch should be active only when Norway features are available.

### 8.1 Branch inputs

```text
h_shared
week_embedding
altitude_embedding
norway_topography
norway_hydrology
norway_ecosystem
norway_geology
norway_minerals
norway_biotic_indicators
norway_feature_masks
```

### 8.2 Norway feature encoders

Use several small encoders rather than one giant unstructured MLP:

```text
TopographyEncoder
HydrologyEncoder
EcosystemEncoder
GeologyMineralEncoder
BioticIndicatorEncoder
```

Then:

```text
h_NO = concat(
    h_topography,
    h_hydrology,
    h_ecosystem,
    h_geology_mineral,
    h_biotic,
    h_shared_summary
)
```

Final projection:

```text
h_NO -> MLP -> h_NO_residual
```

### 8.3 Residual components

The Norway residual logit is:

```text
z_NO_residual =
    z_NO_species_bias
  + z_NO_habitat_residual
  + z_NO_local_spatial_residual
```

The local spatial residual should be disabled in v1 and only added if necessary.

### 8.4 Norway species bias residual

A per-species scalar:

```text
z_NO_species_bias_s = b_NO_s
```

This captures Norway-specific range offsets or systematic reporting differences.

Regularize strongly:

```text
L_bias = lambda_bias * mean_s(b_NO_s^2)
```

### 8.5 Norway habitat residual

Main high-resolution correction:

```text
h_NO_residual -> low-rank factors -> species residual logits
```

Formula:

```text
u_NO = Linear(h_NO_residual)                 # K-dimensional
z_NO_habitat = u_NO @ SpeciesResidualMatrix.T
```

where:

```text
SpeciesResidualMatrix shape = [n_species, K]
```

Start with:

```text
K = 64
```

This encourages shared Norway-specific ecological corrections rather than per-species memorization.

### 8.6 Optional Norway local spatial residual

Disable in v1.

If enabled later:

```text
lat, lon, altitude -> LocalNorwayEncoder -> low-rank species residual
```

Regularize more strongly than the habitat residual. This component is most likely to memorize observer hotspots.

### 8.7 Norway residual gate

Final Norway formula:

```text
z_final = z_shared + g_NO * z_NO_residual
```

First version:

```text
g_NO_s = sigmoid(a_s)
```

Initialize:

```text
g_NO_s ≈ 0.2
```

Improved version:

```text
g_NO_s(x) =
    sigmoid(
        a_s
      + GateMLP(h_shared, h_NO, feature_quality, observation_density, species_NO_count)
    )
```

The gate should be small when:

- features are missing,
- local observation effort is very high but ecological evidence is weak,
- species has few Norwegian observations,
- location is near border and Nordic shared model is likely reliable,
- residual would be large but unsupported.

---

## 9. Smoothing and correlation structure

The model uses smoothing in logit space.

### 9.1 Shared Nordic smoothing

For nearby/similar samples:

```text
L_smooth_shared =
    w_ij * ||z_shared_i - z_shared_j||^2
```

where:

```text
w_ij =
    exp(-d_geo^2 / sigma_geo^2)
  * exp(-d_alt^2 / sigma_alt^2)
  * exp(-d_common_env^2 / sigma_env^2)
```

### 9.2 Norway residual smoothing

Inside Norway:

```text
L_smooth_NO =
    w_NO_ij * ||z_NO_residual_i - z_NO_residual_j||^2
```

where:

```text
w_NO_ij =
    w_geo
  * w_alt
  * w_ecosystem
  * w_geochem
  * w_biotic
```

Use biotic and geochemical similarity to avoid smoothing across real ecological boundaries.

### 9.3 Cross-border smoothing

Near Norway-Sweden and Norway-Finland borders:

```text
L_border =
    w_border_ij * ||z_shared_NO_side - z_shared_neighbor_side||^2
```

Apply only to shared logits, not Norway residual logits.

### 9.4 Temporal smoothing

For learned week embeddings:

```text
L_week =
    sum_w alpha_w ||E[w+1] - E[w]||^2
```

and optionally:

```text
L_week_curvature =
    sum_w ||E[w+1] - 2E[w] + E[w-1]||^2
```

Use smaller `alpha_w` during spring and autumn migration windows.

---

## 10. Observation weighting

Observation weight:

```text
weight =
    species_weight
  * country_weight
  * spatial_thinning_weight
  * source_quality_weight
  * target_region_weight
```

Suggested country weights for Norway-targeted training:

```text
Norway:              1.00
Sweden near Norway:  0.80
Finland near Norway: 0.70
Denmark:             0.40–0.60
```

The exact weights should be tuned using Norway-only validation and held-out regions.

---

## 11. Loss function

Total loss:

```text
L =
    L_species
  + lambda_env * L_common_env
  + lambda_NO_env * L_norway_aux
  + lambda_smooth * L_smooth_shared
  + lambda_NO_smooth * L_smooth_NO
  + lambda_border * L_border
  + lambda_week * L_week
  + lambda_week2 * L_week_curvature
  + lambda_residual * L_residual_magnitude
  + lambda_gate * L_gate_regularization
```

Initial values:

```text
lambda_env        = 0.10
lambda_NO_env     = 0.05
lambda_smooth     = 1e-4
lambda_NO_smooth  = 3e-4
lambda_border     = 1e-4
lambda_week       = 1e-4
lambda_week2      = 5e-5
lambda_residual   = 1e-3
lambda_gate       = 1e-4
```

The residual magnitude penalty is essential:

```text
L_residual_magnitude = mean(z_NO_residual^2)
```

It keeps the Norway branch from becoming a full replacement model.

---

## 12. Recommended initial model configuration

```yaml
model:
  name: norway_nordic_residual_geomodel
  region_mode: nordic_with_norway_residual

  inputs:
    lat: true
    lon: true
    week: true
    altitude: true
    country_id: true
    common_env: true
    norway_env: true
    norway_geology_minerals: true
    norway_biotic_indicators: true
    feature_masks: true

  coordinate_encoder:
    type: fourier
    dim: 128

  altitude_encoder:
    enabled: true
    dim: 32
    normalize:
      center_m: 500
      scale_m: 1500
      clip_min: -2.0
      clip_max: 4.0
    use_missing_flag: true
    use_source_flag: true
    dropout_prob: 0.15

  week_encoder:
    type: fourier_plus_learned
    n_weeks: 48
    fourier_harmonics: 4
    learned_dim: 64
    smooth_lambda: 1.0e-4
    curvature_lambda: 5.0e-5
    migration_windows:
      spring: [10, 22]
      autumn: [30, 43]
    migration_smooth_weight: 0.25

  country_embedding:
    enabled: true
    countries: [NO, SE, FI, DK]
    dim: 8

  shared_trunk:
    hidden_dim: 512
    n_blocks: 6
    residual_blocks: true
    film_from_week: true
    dropout: 0.10

  shared_species_head:
    type: low_rank
    rank: 128

  environment_head:
    enabled: true
    target: common_env
    loss_weight: 0.10

  habitat_head:
    enabled: true
    gate: learned

  norway_residual:
    enabled: true
    active_country: NO

    encoders:
      topography_dim: 64
      hydrology_dim: 32
      ecosystem_dim: 64
      geology_mineral_dim: 64
      biotic_indicator_dim: 64

    fusion:
      hidden_dims: [256, 128, 64]
      dropout: 0.10
      layer_norm: true

    components:
      species_bias: true
      habitat_residual: true
      local_spatial_residual: false

    habitat_residual:
      type: low_rank
      rank: 64

    gate:
      type: species_and_location
      initial_value: 0.20
      use_sample_count_prior: true

    regularization:
      residual_l2: 1.0e-3
      gate_l2: 1.0e-4
```

---

## 13. Inference behavior

The prediction API should accept:

```bash
python predict.py --lat 61.5 --lon 8.2 --week 23
```

and:

```bash
python predict.py --lat 61.5 --lon 8.2 --week 23 --altitude 1350
```

Inside Norway:

```text
load common env
load Norway high-resolution env
load geology/mineral features
load biotic indicator features
compute shared logits
compute Norway residual logits
compute Norway gate
return final probabilities
```

Outside Norway:

```text
load common env
skip Norway branch
return shared probabilities
```

If Norway high-resolution features are unavailable:

```text
set Norway feature mask to 0
gate_NO should go close to zero
use shared Nordic prediction
```

Diagnostic mode should return:

```text
species
p_shared
p_final
delta_logit_NO
gate_NO
largest contributing Norway feature groups
```

---

## 14. Expected benefits

Expected improvements:

- better alpine/lowland separation,
- better coastal/inland separation,
- more realistic Norway maps,
- better old-growth forest bird predictions,
- better rich-fen/marshland/wetland species behavior,
- better calcium/calcareous habitat specialization,
- better migration asymmetry,
- better cross-border consistency,
- fewer isolated false hotspots through smoothing,
- better rare-species behavior via shared Nordic learning.

---

## 15. Main risks

| Risk | Cause | Mitigation |
|---|---|---|
| Norway branch memorizes observer hotspots | Too much residual capacity | residual L2, gate regularization, no local residual in v1 |
| Biotic indicators leak observation effort | Citizen-science bias | effort controls, cross-fitting, no bird-derived features |
| Over-smoothing | Too large smoothing penalty | tune roughness-vs-accuracy, use ecological weights |
| Altitude overfit | DEM/GPS inconsistency | source flags, altitude dropout |
| Week embedding overfit | too flexible temporal representation | week smoothness and curvature penalties |
| Cross-border discontinuity | Norway residual too strong near border | border smoothing on shared logits and gate dampening |
| Data source inconsistency | mixed resolution and taxonomy | strict data cleaning, schema validation, source-specific QA |

---

## 16. Versioning

Recommended naming:

```text
data/nordic_norway_vYYYYMMDD.parquet
configs/norway_nordic_residual_vYYYYMMDD.yaml
checkpoints/norway_nordic_residual_vYYYYMMDD.pt
reports/model_report_vYYYYMMDD.md
```

All model runs should log:

```text
git commit
data version
feature version
species taxonomy version
training config hash
random seed
validation split version
```


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

