# Project document: Detailed Nordic + Norway residual GeoModel

**File:** `norway_detailed_model_project_20260427.md`  
**Date:** 2026-04-27  
**Target codebase:** `birdnet-team/geomodel`  
**Goal:** modify the existing BirdNET GeoModel into a Nordic shared ecological model with a Norway-specialized residual branch.

---

## 1. Executive summary

The current BirdNET GeoModel predicts species occurrence priors from geographic and temporal inputs, with an auxiliary environmental head. The proposed extension keeps the existing model philosophy but adds a more detailed hierarchy:

```text
Nordic shared model
    learns broad geography, altitude, seasonality, habitat and inter-country structure

Norway residual model
    learns Norway-specific corrections from high-resolution Norwegian data

Interaction / biotic features
    allow flora, insects, geology, minerals and prey/guild indicators to refine the Norway model
```

The model should not become a separate Norway-only model. It should be a **shared Nordic prior plus a regularized Norway correction**:

```text
final_logit_NO(species) = shared_logit(species) + norway_gate(species, x) * norway_residual_logit(species)
```

Outside Norway:

```text
final_logit = shared_logit
```

This keeps the statistical strength of Sweden, Finland, Denmark and Norway while allowing Norwegian high-resolution terrain, ecosystem, geology, mineral and biotic data to sharpen predictions inside Norway.

---

## 2. Design principles

### 2.1 Keep the current model usable

The modified model should still support the old style inference:

```bash
python predict.py --lat 61.5 --lon 8.2 --week 23
```

and add richer inference when available:

```bash
python predict.py --lat 61.5 --lon 8.2 --week 23 --altitude 1350 --country NO
```

If Norway-only features are missing, the model must fall back to the shared Nordic prediction.

### 2.2 Use all data, but specialize Norway

The shared model uses all Nordic observations:

```text
Norway + Sweden + Finland + Denmark
```

The Norway residual uses only Norway-specific high-resolution features and Norwegian residual training targets, but it is regularized to remain a correction to the Nordic model.

### 2.3 Avoid leakage

Do not feed raw bird observations as features for other birds. If prey/guild features are used, they must be generated as **out-of-fold predicted suitability surfaces**, not direct local observation counts.

For non-bird flora, insects, fungi, lichens, mosses and molluscs, use derived ecological indicators rather than raw point observations.

### 2.4 Smooth the right objects

Use separate smoothing terms:

```text
shared smoothing:       smooth shared Nordic logits across similar Nordic habitats
Norway residual smoothing: smooth the residual correction across similar Norwegian habitats
border smoothing:       smooth shared logits across Norway-Sweden/Finland borders
```

Do not smooth the final prediction blindly.

---

## 3. High-level architecture

```text
Inputs:
    lat, lon, week
    altitude and altitude metadata
    country_id
    common Nordic environmental features
    Norway-only high-resolution features
    non-bird biotic indicator features
    prey/guild suitability features
    feature masks

Shared Nordic encoder:
    coordinate encoder
    altitude encoder
    hybrid week encoder
    common environment encoder
    country embedding
    residual blocks with temporal FiLM

Shared heads:
    direct species head
    common environment auxiliary head
    optional habitat species head

Norway residual branch:
    Norway topography/geology/ecosystem/mineral encoder
    Norway biotic indicator encoder
    Norway prey/guild feature encoder
    low-rank residual species head
    Norway gate
    optional local spatial residual

Output:
    if country != NO:
        logits = shared_logits
    if country == NO:
        logits = shared_logits + norway_gate * norway_residual_logits
```

---

## 4. Inputs and data schema

Create a versioned training table, for example:

```text
combined_nordic_v2.parquet
```

Recommended columns:

```text
sample_id
source_id
observation_year
lat
lon
week
country_id
h3_cell
region_holdout_id
altitude_m
altitude_missing
altitude_source
common_env_*
common_env_mask_*
norway_topography_*
norway_hydrology_*
norway_ecosystem_*
norway_geology_*
norway_mineral_*
norway_biotic_*
norway_prey_guild_*
norway_feature_mask_*
species_label_vector
observation_weight
```

### 4.1 Country id

Use a small country embedding:

```text
NO, SE, FI, DK
```

This allows the shared model to represent broad domain shifts without creating separate country models.

### 4.2 Altitude input

Altitude becomes a true input feature, not only an environmental target.

Fields:

```text
altitude_m
altitude_missing
altitude_source
```

Suggested source codes:

```text
0 = DEM/grid altitude
1 = GPS/user altitude
2 = unknown/imputed
```

Suggested normalization:

```text
altitude_norm = clip((altitude_m - 500) / 1500, -2.0, 4.0)
```

Use altitude dropout during training so the model remains robust when altitude is missing or noisy.

---

## 5. Common Nordic environmental features

Common features should be available in all Nordic countries.

Examples:

```text
coarse elevation
coarse slope / ruggedness if available
mean annual temperature
seasonal temperature summaries
precipitation summaries
snow proxy
land cover class
forest fraction
water fraction
urban fraction
distance to coast
distance to freshwater
canopy height if consistently available
```

These feed the shared Nordic trunk and the common environmental auxiliary head.

---

## 6. Norway-only high-resolution feature blocks

Norway features are only available inside Norway. For non-Norway samples, set feature values to zero and masks to zero.

### 6.1 Topography

```text
elevation_10m
slope_mean_50m
slope_mean_250m
slope_std_250m
aspect_sin
aspect_cos
terrain_ruggedness_250m
terrain_ruggedness_1000m
topographic_position_index
topographic_wetness_index
solar_exposure
cold_air_pooling_proxy
valley_bottom_proxy
ridge_proxy
cliff_ruggedness_1000m
```

### 6.2 Hydrology and coast

```text
distance_to_coast
distance_to_freshwater
distance_to_river
distance_to_lake
distance_to_wetland
wetland_fraction_250m
wetland_fraction_1000m
marine_influence_proxy
fjord_distance
shoreline_complexity_1000m
```

### 6.3 Land cover and ecosystem

```text
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
```

### 6.4 Climate and snow

```text
mean_temperature
summer_temperature
winter_temperature
precipitation
snow_persistence_proxy
growing_degree_days
frost_frequency_proxy
```

### 6.5 Geology and minerals

This is important for Norway because bedrock and soil chemistry strongly influence vegetation and invertebrate communities.

Recommended v1 features:

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
base_richness_index
calcareousness_index
nutrient_richness_index
acidic_soil_proxy
soil_ph_proxy
cation_richness_index
```

Optional later features:

```text
sodium_na_mean_1000m
sulfur_s_mean_1000m
iron_fe_mean_1000m
manganese_mn_mean_1000m
zinc_zn_mean_1000m
copper_cu_mean_1000m
trace_metal_stress_index
```

Use robust transforms such as quantile normalization or clipped z-scores. Do not use raw heavy-tailed mineral concentrations directly.

### 6.6 Non-bird biotic indicator features

Use flora, mosses, lichens, fungi, insects, molluscs and other non-bird observations as ecological indicators.

Do not use raw point observations directly. Build smoothed, effort-corrected indicator surfaces.

Recommended v1 indicators:

```text
old_growth_forest_indicator
deadwood_forest_indicator
rich_deciduous_forest_indicator
calcareous_forest_indicator
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

Effort controls:

```text
nonbird_observation_count_250m
nonbird_observation_count_1000m
nonbird_observer_count_1000m
nonbird_species_count_1000m
nonbird_observation_days_1000m
```

### 6.7 Prey and guild suitability features

These are useful for predators and ecological interactions.

Use out-of-fold predicted suitability, not raw observations.

Examples:

```text
ptarmigan_suitability_5km
ptarmigan_suitability_10km
forest_grouse_suitability_5km
hare_suitability_5km
hare_suitability_10km
lemming_small_mammal_proxy_5km
waterbird_suitability_5km
seabird_suitability_10km
wetland_insect_suitability_1000m
old_forest_insect_suitability_1000m
open_land_insect_suitability_1000m
```

These should mainly affect selected predator, scavenger and insectivore species through a small interaction residual.

---

## 7. Feature aggregation strategy

For habitat, geology, minerals and biotic indicators, use multi-scale summaries:

```text
50 m
250 m
1000 m
5000 m
10000 m for large predators / prey fields
```

Examples:

```text
forest_fraction_250m
forest_fraction_1000m
calcium_ca_mean_1000m
calcium_ca_p90_1000m
old_growth_indicator_1000m
ptarmigan_suitability_10km
```

The `p90` or high-quantile features are useful because small patches of rare habitat can matter even when the local mean is low.

---

## 8. Model modules

### 8.1 Coordinate encoder

Continue using circular/Fourier encodings for latitude and longitude, projected into a spatial embedding.

```text
coord_embedding = CoordinateEncoder(lat, lon)
```

### 8.2 Altitude encoder

```python
class AltitudeEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, altitude_norm, missing_flag, source_flag):
        x = torch.stack([altitude_norm, missing_flag, source_flag], dim=-1)
        return self.net(x)
```

### 8.3 Hybrid week encoder

Use both harmonic and learned week structure.

```text
week_embedding = FourierWeek(week) + LearnedWeekEmbedding(week)
```

This allows asymmetric spring/autumn migration routes and different migration intensities.

Use FiLM modulation from week embedding:

```text
scale, shift = WeekFiLM(week_embedding)
h = scale * h + shift
```

### 8.4 Common environment encoder

Encode common Nordic environmental features:

```text
common_env_embedding = CommonEnvEncoder(common_env, common_env_mask)
```

### 8.5 Shared Nordic trunk

Concatenate:

```text
coord_embedding
altitude_embedding
week_embedding
country_embedding
common_env_embedding
```

Then pass through residual blocks.

```text
h_shared = SharedTrunk(concat(...))
```

### 8.6 Shared species head

Use a low-rank species head:

```text
z_direct_shared = LowRankSpeciesHead(h_shared)
```

This implicitly captures inter-species correlations through shared latent factors.

### 8.7 Shared environment head

```text
env_pred_common = CommonEnvHead(h_shared)
```

This regularizes the shared latent space to remain ecological.

### 8.8 Shared habitat head

Optional but recommended:

```text
z_habitat_shared = HabitatSpeciesHead(env_pred_common.detach())
z_shared = z_direct_shared + gate_habitat * z_habitat_shared
```

Detaching the predicted environment before the habitat head keeps the environment head honest as an environment regressor.

---

## 9. Norway residual branch

The Norway branch learns corrections, not full predictions.

### 9.1 Norway feature encoder

Input:

```text
norway_topography_*
norway_hydrology_*
norway_ecosystem_*
norway_geology_*
norway_mineral_*
norway_biotic_*
norway_prey_guild_*
feature masks
```

Suggested MLP:

```text
Linear(input_dim, 256)
LayerNorm
GELU
Dropout(0.10)
Linear(256, 128)
LayerNorm
GELU
Linear(128, 64)
```

```text
h_NO = NorwayFeatureEncoder(norway_features, masks)
```

### 9.2 Residual decomposition

```text
z_NO_residual =
    z_NO_country_species_bias
  + z_NO_habitat_residual
  + z_NO_interaction_residual
  + z_NO_local_spatial_residual_optional
```

### 9.3 Country/species bias residual

```text
z_NO_country_species_bias_s = b_NO_s
```

This captures broad Norway-specific abundance/range differences. Regularize strongly.

### 9.4 Habitat residual

Low-rank form:

```text
u_NO = Linear(h_NO)                       # dim K, e.g. 64
z_NO_habitat = u_NO @ SpeciesResidualMatrix.T
```

This lets related species share Norway-specific corrections.

### 9.5 Interaction residual

Small residual for prey/guild/species interaction features.

Example for gyrfalcon:

```text
interaction inputs:
    ptarmigan_suitability_5km
    ptarmigan_suitability_10km
    open_mountain_fraction_10km
    cliff_ruggedness_5km
    week_embedding
```

Use a small low-rank residual gated to selected species. Do not let it affect all species strongly.

### 9.6 Optional local spatial residual

Only add after the habitat residual works.

```text
r_local = LocalNorwayEncoder(lat, lon, altitude)
z_NO_local = r_local @ SpeciesLocalResidualMatrix.T
```

Regularize strongly and smooth. This can easily memorize observer hotspots, so it should be disabled in v1.

### 9.7 Norway gate

First version:

```text
g_NO_s = sigmoid(a_s)
```

Initialize near 0.2.

Better version:

```text
g_NO_s(x) = sigmoid(a_s + v_s · GateMLP(h_shared, h_NO, feature_quality, sample_count_prior))
```

Inputs:

```text
feature availability
local observation density
species Norwegian sample count
distance to border
altitude
week
```

Final Norway formula:

```text
z_final = z_shared + g_NO * z_NO_residual
```

---

## 10. Inter-species correlation handling

The model has four correlation mechanisms:

1. **Shared trunk**: all species use the same ecological latent space.
2. **Low-rank species heads**: species share latent factors.
3. **Habitat/environment heads**: species are coupled through environmental structure.
4. **Interaction residuals**: selected predator/prey/guild relationships use predicted prey/guild suitability.

Recommended relation graph types:

```text
habitat_similarity
prey_of
taxonomic_similarity
migration_similarity
biotic_indicator_similarity
```

Optional embedding regularizer:

```text
L_graph = Σ_edges w_ij ||E_i - E_j||²
```

For predator-prey edges, do not necessarily force embeddings to be identical. Use relation-specific interaction features instead.

---

## 11. Loss functions

Total loss:

```text
L =
    L_species
  + λ_env L_common_env
  + λ_NO_env L_norway_aux
  + λ_smooth L_shared_smooth
  + λ_NO_smooth L_norway_residual_smooth
  + λ_border L_border_shared
  + λ_week L_week_smooth
  + λ_week2 L_week_curvature
  + λ_residual L_residual_magnitude
  + λ_gate L_gate_regularization
  + λ_graph L_species_graph_optional
```

Suggested initial weights:

```text
λ_env           = 0.10
λ_NO_env        = 0.05
λ_smooth        = 1e-4
λ_NO_smooth     = 3e-4
λ_border        = 1e-4
λ_week          = 1e-4
λ_week2         = 5e-5
λ_residual      = 1e-3
λ_gate          = 1e-4
λ_graph         = 1e-5 initially, optional
```

### 11.1 Species loss

Use BCE, ASL or assume-negative loss depending on current baseline.

Keep BCE as the first implementation for comparability.

### 11.2 Environment auxiliary loss

Use masked MSE or Huber loss:

```text
L_common_env = masked_mse(env_pred_common, env_target_common, common_env_mask)
```

### 11.3 Norway auxiliary loss

Predict key Norway ecological indices:

```text
base_richness_index
calcareousness_index
nutrient_richness_index
old_growth_indicator
rich_fen_indicator
open_mountain_indicator
wetland_indicator
```

This makes the Norway embedding ecological rather than just spatial.

### 11.4 Residual magnitude penalty

```text
L_residual_magnitude = mean(z_NO_residual²)
```

This is essential. It prevents the Norway residual from becoming a full Norway-only model.

### 11.5 Gate regularization

```text
L_gate = mean(g_NO²)
```

or:

```text
L_gate = mean((g_NO - g_prior_species)²)
```

where `g_prior_species` is based on the number of Norwegian records for the species.

---

## 12. Smoothing losses

### 12.1 Shared Nordic smoothing

For neighboring or sampled pairs `i, j`:

```text
L_shared_smooth = w_ij ||z_shared_i - z_shared_j||²
```

with:

```text
w_ij = exp(-d_geo² / σ_geo²)
     * exp(-d_alt² / σ_alt²)
     * exp(-d_common_env² / σ_env²)
```

Start values:

```text
σ_geo = 50 km
σ_alt = 300 m
σ_env = 1.5 normalized units
```

### 12.2 Norway residual smoothing

Smooth the residual, not only the final prediction.

```text
L_NO_smooth = w_NO_ij ||z_NO_residual_i - z_NO_residual_j||²
```

with:

```text
w_NO_ij = w_geo * w_alt * w_ecosystem * w_geochem * w_biotic
```

Start values:

```text
σ_geo_NO = 10 km
σ_alt_NO = 150 m
σ_env_NO = 1.0 normalized units
```

### 12.3 Border smoothing

Near Norway-Sweden/Finland borders:

```text
L_border = w_border ||z_shared_NO_side - z_shared_SE_FI_side||²
```

Apply this to shared logits only, not to Norway residuals.

### 12.4 Week smoothness

Learned week embeddings:

```text
L_week = Σ_w α_w ||E[w+1] - E[w]||²
L_week_curvature = Σ_w ||E[w+1] - 2E[w] + E[w-1]||²
```

Use lower penalty during migration windows:

```text
spring: weeks 10-22
autumn: weeks 30-43
migration α = 0.25 * normal α
```

---

## 13. Data cleaning plan

### 13.1 Occurrence data

For all observation sources:

```text
remove records without coordinates
remove records with impossible coordinates
remove records with coordinate uncertainty above threshold
remove records with invalid dates
map species names to stable taxon ids
remove records with unresolved or ambiguous taxa
remove duplicates by species/date/location/source
```

### 13.2 Spatial thinning and bias control

For dense species or hotspots:

```text
cap observations per species per H3 cell/week/year
compute observation density features
compute source-quality weights
compute spatial thinning weights
```

### 13.3 Non-bird biotic observations

Exclude birds from non-bird features.

Allowed groups:

```text
vascular plants
mosses
lichens
fungi
beetles
butterflies
dragonflies
molluscs
other useful invertebrates
```

Create indicator scores with:

```text
indicator_species_weight
distance_kernel
time_decay
observation_quality_weight
effort correction
```

### 13.4 Prey/guild surfaces

Use out-of-fold predictions:

```text
split data into spatial folds
train prey/guild suitability models on folds != k
predict prey/guild surfaces for fold k
use these predictions as features for target model
```

This prevents leakage.

---

## 14. Training stages

### Stage 0: baseline reproduction

Train or load current-style GeoModel.

Deliverables:

```text
baseline metrics
Norway-only validation
spatial roughness maps
spring/autumn maps
altitude response plots
```

### Stage 1: shared Nordic model

Train on all Nordic data using:

```text
lat, lon, altitude, week, country_id, common_env
```

No Norway residual yet.

Losses:

```text
species loss
common env loss
shared smoothing
border smoothing
week smoothness
```

### Stage 2: Norway residual v1

Freeze most of shared trunk.

Train:

```text
NorwayFeatureEncoder
Norway habitat residual
Norway species bias
Norway gate
Norway auxiliary ecological heads
```

Use Norwegian samples for Norway residual loss.

### Stage 3: joint fine-tuning

Unfreeze upper shared layers.

Suggested batch mix:

```text
60% Norway
15% Sweden
15% Finland
10% Denmark
```

Learning rates:

```text
shared trunk:  1e-5 to 3e-5
Norway branch: 1e-4
heads/gates:   1e-4
```

### Stage 4: interaction residual

Add prey/guild interaction features after the basic Norway residual works.

Train with small residual penalties and inspect predator species carefully.

### Stage 5: model selection and pruning

For each species, check:

```text
shared AP
Norway corrected AP
residual magnitude
gate strength
map quality
```

Shrink or disable residuals for species where they hurt validation or create artifacts.

---

## 15. Validation and tests

### 15.1 Splits

Use several validation modes:

```text
random observation split
spatial block split
Norwegian region holdout
altitude band holdout
ecosystem holdout
cross-border holdout
migration season holdout
```

### 15.2 Metrics

```text
global mAP
Norway mAP
rare species mAP
top-k recall
calibration error
Brier score
GeoScore
spatial roughness
residual roughness
border discontinuity score
altitude response smoothness
spring/autumn route asymmetry
```

### 15.3 Norway residual diagnostics

For selected species, produce:

```text
shared map
Norway residual map
final map
gate map
week curve shared vs final
altitude response shared vs final
```

Good residuals should be ecologically meaningful and smooth within similar habitats.

Bad residuals:

```text
observer hotspot memorization
isolated speckles
large unsupported corrections
border discontinuities
unrealistic altitude cliffs
```

---

## 16. Code changes in existing repository

Recommended new/modified files:

```text
model/model.py
    add altitude encoder, hybrid week encoder, NorwayResidualBranch

model/loss.py
    add smoothing losses, week smoothness, residual penalties, gate penalties

utils/data.py
    extend dataset loader for grouped features and masks

utils/features_common.py
    common Nordic feature extraction helpers

utils/features_norway.py
    Norway high-resolution feature extraction helpers

utils/biotic_features.py
    non-bird indicator construction

utils/prey_features.py
    out-of-fold prey/guild suitability generation

train.py
    add staged training modes and new CLI/config options

predict.py
    add altitude, country and Norway feature handling

convert.py
    ensure export works with optional Norway branch

configs/nordic_norway_residual.yaml
    full model and training config
```

---

## 17. CLI/config additions

Example options:

```bash
--country_features
--altitude_input
--week_encoder fourier_plus_learned
--common_env_features
--norway_residual
--norway_feature_path
--norway_residual_rank 64
--norway_gate species_and_location
--smooth_shared
--smooth_norway_residual
--smooth_border
--interaction_features
--training_stage shared_nordic|norway_residual|joint|interaction
```

Prefer a YAML config for real runs.

---

## 18. Recommended initial config

```yaml
model:
  region_mode: nordic_with_norway_residual

  inputs:
    lat: true
    lon: true
    week: true
    altitude: true
    country_id: true
    common_env: true
    norway_env: true
    biotic_indicators: true
    prey_guild_features: false
    feature_masks: true

  coordinate_encoder:
    type: fourier
    dim: 128

  altitude_encoder:
    enabled: true
    dim: 32
    dropout_prob: 0.15
    normalize:
      center_m: 500
      scale_m: 1500
      clip_min: -2.0
      clip_max: 4.0

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
    dim: 8

  shared_trunk:
    hidden_dim: 512
    n_blocks: 6
    dropout: 0.10
    film_from_week: true

  shared_species_head:
    type: low_rank
    rank: 128

  environment_head:
    enabled: true
    loss_weight: 0.10

  habitat_head:
    enabled: true
    gate: learned

  norway_residual:
    enabled: true
    active_country: NO
    feature_encoder:
      hidden_dims: [256, 128, 64]
      dropout: 0.10
      layer_norm: true
    components:
      country_species_bias: true
      habitat_residual: true
      interaction_residual: false
      local_spatial_residual: false
    habitat_residual:
      type: low_rank
      rank: 64
    gate:
      type: species
      initial_value: 0.20
      use_sample_count_prior: true
    regularization:
      residual_l2: 1.0e-3
      gate_l2: 1.0e-4

  smoothing:
    shared:
      enabled: true
      lambda: 1.0e-4
      sigma_geo_km: 50
      sigma_alt_m: 300
      sigma_env: 1.5
    norway_residual:
      enabled: true
      lambda: 3.0e-4
      sigma_geo_km: 10
      sigma_alt_m: 150
      sigma_env: 1.0
    border:
      enabled: true
      lambda: 1.0e-4
      width_km: 50
```

---

## 19. Inference behavior

### 19.1 Outside Norway

```text
load common env
compute shared logits
return shared probabilities
```

### 19.2 Inside Norway with full features

```text
load common env
load Norway features
compute shared logits
compute Norway residual
compute gate
return final probabilities
```

### 19.3 Inside Norway without high-resolution features

```text
compute shared logits
set Norway gate close to zero
return shared probabilities with warning/diagnostic flag
```

### 19.4 Diagnostic output

For debugging:

```text
species
p_shared
residual_logit
gate
p_final
```

---

## 20. First implementation milestone

The first useful version should include:

```text
altitude input
hybrid week encoder
country embedding
shared Nordic model
Norway feature encoder
Norway low-rank habitat residual
Norway species-level gate
Norway residual smoothing
common/shared smoothing
border smoothing
non-bird biotic indicator features
mineral/geology features
```

Do not include in v1:

```text
local spatial residual
complex graph neural network
raw bird observation interaction features
large unregularized Norway-specific head
```

---

## 21. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Norway branch memorizes observer hotspots | residual L2, gate penalty, residual smoothing, spatial holdouts |
| Biotic features leak observation effort | cross-fitting, effort controls, no bird-derived features |
| Altitude overfits | altitude dropout, source flags, altitude-band holdouts |
| Learned week embeddings become noisy | week smoothness and curvature penalties |
| Smoothing washes out real boundaries | condition smoothing on altitude, ecosystem, minerals and biotic indicators |
| Border discontinuity | shared-logit border smoothing |
| Predator/prey features leak bird observations | use out-of-fold predicted prey suitability, not raw local records |
| Too many Norway features | start with curated groups and low-rank residuals |

---

## 22. Summary

The proposed model is a conservative but powerful extension of the existing BirdNET GeoModel:

```text
shared Nordic ecological prior
+ explicit altitude input
+ non-harmonic learned week structure
+ common environment and habitat heads
+ country embedding
+ Norway-only high-resolution residual branch
+ geology/mineral features
+ non-bird biotic indicator features
+ optional prey/guild interaction features
+ smoothing at shared, residual and border levels
```

This gives a model that uses all Nordic observations, learns broad ecological correlations, and sharpens predictions in Norway using data that are only available there.
