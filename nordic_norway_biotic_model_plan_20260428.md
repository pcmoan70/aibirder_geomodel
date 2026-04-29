# Nordic + Norway Biotic GeoModel Plan

**File:** `nordic_norway_biotic_model_plan_20260428.md`  
**Date:** 2026-04-28  
**Target repository:** `aibirder_geomodel`  
**Model family:** BirdNET-style spatiotemporal prior with Nordic shared ecology, plant-derived habitat features, bird and mammal range heads, and optional Norway residual refinement.

---

## 1. Goal

Build a Norway-focused biological range model that predicts bird and mammal occurrence priors from:

- location
- week or season
- altitude
- country/domain
- coarse environmental layers
- plant-derived habitat indicators across Norway and surrounding countries
- optional Norway-only high-resolution terrain, geology, ecosystem and plant refinements

The model should use all available Nordic and surrounding-country data, but it should still allow Norway-specific corrections where Norway has better ecological feature coverage.

The central principle is:

```text
animal ranges = broad geography + seasonality + climate + terrain + habitat + survey process
```

Plants should be used as a habitat translator:

```text
plants -> habitat conditions -> bird and mammal suitability
```

not as raw biological co-occurrence shortcuts.

---

## 2. Recommended First Production Model

Use a shared ecological trunk with separate animal heads:

```text
Inputs:
    lat, lon, week
    altitude_m, altitude_missing, altitude_source
    country_id
    common environmental features
    plant-derived habitat indicators
    effort and source-quality features
    optional Norway-only high-resolution features

Shared Nordic ecological trunk:
    coordinate encoder
    week encoder
    altitude encoder
    country embedding
    common environment encoder
    plant habitat encoder
    feature mask encoder

Output heads:
    bird species head
    mammal species head
    optional environment reconstruction head
    optional habitat reconstruction head

Norway residual branch:
    active only for Norway
    uses high-resolution Norway features
    predicts regularized correction logits
```

Final logits:

```text
z_bird_shared = f_bird(h_shared)
z_mammal_shared = f_mammal(h_shared)
```

For samples outside Norway:

```text
z_bird_final = z_bird_shared
z_mammal_final = z_mammal_shared
```

For samples inside Norway:

```text
z_bird_final =
    z_bird_shared
  + g_bird_NO(x) * r_bird_NO(x)

z_mammal_final =
    z_mammal_shared
  + g_mammal_NO(x) * r_mammal_NO(x)
```

where:

```text
g_NO(x) in [0, 1]
```

is a learned residual gate, initialized small.

Convert logits to probabilities with:

```text
p(species present | x) = sigmoid(z_species_final)
```

---

## 3. Species Scope

### 3.1 Birds

Birds are seasonal and often strongly week-dependent.

Recommended target:

```text
bird_species_label_vector[0:n_birds]
```

For birds, keep weekly labels:

```text
week = 1..48
```

Use the current BirdNET 48-week convention unless there is a strong reason to change it.

### 3.2 Mammals

Mammals are usually less migratory, but observations may still be seasonal because of detectability, snow tracking, hunting seasons, roadkill reporting, camera traps, and tourist effort.

Recommended target:

```text
mammal_species_label_vector[0:n_mammals]
```

Use week as an input for mammals too, but treat mammal validation carefully. A weekly mammal probability may reflect both true seasonal activity and observation process.

### 3.3 Why separate heads?

Bird and mammal records have different observation processes. A single output head can work, but separate heads are cleaner:

```text
h_shared -> bird_head
h_shared -> mammal_head
```

This allows:

- separate loss weights
- separate calibration
- separate minimum observation thresholds
- separate diagnostics
- different handling of seasonal response

---

## 4. Data Products

The project should produce several versioned data products instead of one ambiguous table.

### 4.1 Raw occurrence archives

Store raw downloaded archives without modification.

Suggested layout:

```text
data/raw/gbif/
data/raw/artsobservasjoner/
data/raw/finbif/
data/raw/artportalen/
data/raw/naturbasen/
data/raw/nibio/
data/raw/ngu/
```

Each archive should have a sidecar metadata file:

```yaml
source_name:
download_date:
query:
country:
taxon_scope:
year_start:
year_end:
license:
record_count:
download_url_or_key:
processing_script:
```

### 4.2 Clean occurrence table

Create one normalized occurrence table:

```text
occurrences_clean_v1.parquet
```

Recommended columns:

```text
occurrence_id
source_id
source_record_id
taxon_key
stable_taxon_id
scientific_name
common_name
kingdom
class_name
taxon_group
country_id
observation_date
observation_year
month
day
week
lat
lon
coordinate_uncertainty_m
coordinate_precision
basis_of_record
dataset_key
observer_id_hash
event_id_hash
license
quality_flags
```

Recommended `taxon_group` values:

```text
bird
mammal
vascular_plant
moss
lichen
fungus
insect
mollusc
other_nonbird
```

### 4.3 Grid table

Create an H3 grid for the full model domain:

```text
nordic_grid_v1.parquet
```

Recommended domain:

```text
Norway
Sweden
Finland
Denmark
Estonia
Latvia
Lithuania
northern Germany / Poland buffer if useful
North Sea / Baltic coastal buffer if marine birds are included
```

Recommended columns:

```text
h3_index
h3_resolution
target_km
geometry
lat_centroid
lon_centroid
country_id
distance_to_country_border_km
distance_to_norway_border_km
```

Resolution guidance:

```text
coarse shared model: 5-25 km grid
Norway residual feature extraction: 250 m to 1 km source features aggregated to model grid
high-resolution inference later: sample features at finer points, then feed model
```

### 4.4 Animal label table

Aggregate birds and mammals by H3 cell and week:

```text
animal_labels_h3_week_v1.parquet
```

Recommended columns:

```text
h3_index
week
bird_species_ids
mammal_species_ids
bird_obs_count
mammal_obs_count
bird_observer_count
mammal_observer_count
bird_observation_days
mammal_observation_days
source_mix
```

For a sample `i`, define:

```text
y_bird_i,s = 1 if bird species s is observed in cell/week i, else 0
y_mammal_i,s = 1 if mammal species s is observed in cell/week i, else 0
```

These are presence-only / pseudo-negative labels, not true absence labels.

### 4.5 Common environmental feature table

Create:

```text
common_env_h3_v1.parquet
```

Recommended common features available across all countries:

```text
elevation_m
slope_mean
terrain_ruggedness
temperature_mean_annual
temperature_winter
temperature_summer
precipitation_annual
precipitation_summer
snow_persistence_proxy
water_fraction
wetland_fraction
forest_fraction
conifer_fraction
deciduous_fraction
open_land_fraction
agricultural_fraction
urban_fraction
distance_to_coast_km
distance_to_freshwater_km
landcover_class
canopy_height_m
```

Each feature should have an optional mask:

```text
common_env_mask_<feature>
```

Mask value:

```text
1 = observed or reliable
0 = missing, imputed, outside coverage, or low confidence
```

### 4.6 Plant habitat feature table

Create:

```text
plant_habitat_h3_v1.parquet
```

This is the key product for using plant data well.

Recommended plant habitat features:

```text
plant_obs_count_1km
plant_obs_count_5km
plant_species_count_1km
plant_species_count_5km
plant_observation_days_5km
plant_observer_count_5km
plant_data_age_median
plant_distance_to_nearest_record_km

calcareous_indicator_1km
calcareous_indicator_5km
acidic_bog_indicator_1km
poor_bog_indicator_1km
rich_fen_indicator_1km
rich_fen_indicator_5km
wet_meadow_indicator_1km
reedbed_marsh_indicator_1km
eutrophic_lake_margin_indicator_1km
oligotrophic_freshwater_indicator_1km

alpine_heath_indicator_1km
alpine_snowbed_indicator_1km
calcareous_alpine_indicator_1km
coastal_heath_indicator_1km
saltmarsh_indicator_1km
sand_dune_indicator_1km

old_forest_indicator_1km
old_forest_indicator_5km
rich_deciduous_forest_indicator_1km
deadwood_forest_indicator_1km

semi_natural_grassland_indicator_1km
dry_warm_slope_indicator_1km
ruderal_urban_indicator_1km
```

Use the same feature names across countries. If a country lacks a source, set the value to zero and the mask to zero:

```text
plant_mask_<feature> = 0
```

### 4.7 Norway high-resolution feature table

Create:

```text
norway_highres_h3_v1.parquet
```

Recommended Norway-only features:

```text
elevation_10m_mean
elevation_10m_p10
elevation_10m_p90
slope_50m_mean
slope_250m_mean
slope_250m_std
aspect_sin
aspect_cos
ruggedness_250m
ruggedness_1000m
topographic_position_index
topographic_wetness_index
solar_exposure
cold_air_pooling_proxy
valley_bottom_proxy
ridge_proxy
cliff_ruggedness_1000m

ar5_forest_fraction
ar5_agriculture_fraction
ar5_mire_fraction
ar5_open_land_fraction
ar5_urban_fraction
nin_nature_type_class
nin_nature_type_confidence

bedrock_lithology_class
carbonate_bedrock_fraction
base_rich_bedrock_fraction
acidic_bedrock_fraction
mafic_bedrock_fraction
soil_ph_proxy
calcium_index
magnesium_index
phosphorus_index
base_richness_index
nutrient_richness_index
```

These feed only the Norway residual branch.

---

## 5. Data Sources

This section lists good source categories. Exact licenses and access requirements must be checked before production training.

### 5.1 Animal occurrence data

Recommended sources:

```text
GBIF occurrence downloads
Artsobservasjoner / Artsdatabanken
eBird if licensed/available
FinBIF for Finland
Artportalen for Sweden if available
DOFbasen / Danish biodiversity sources if available
museum or monitoring datasets if licensed
camera trap datasets for mammals if available
roadkill / hunting / monitoring datasets only with careful source flags
```

Download filters:

```text
HAS_COORDINATE = true
HAS_GEOSPATIAL_ISSUE = false
OCCURRENCE_STATUS = PRESENT
YEAR >= chosen start year
COUNTRY in model domain
TAXON_KEY in Aves or Mammalia
```

Recommended year range:

```text
birds: 2000-present, optionally with recent weighting
mammals: 2000-present, but inspect source changes
plants: wider historical window allowed, with data-age feature
```

### 5.2 Plant occurrence data

Recommended sources:

```text
GBIF Plantae
Artsobservasjoner plant records
FinBIF vascular plants, mosses, fungi and lichens where relevant
Artportalen plant/moss/lichen records
national red-list or habitat indicator datasets if licensed
vegetation plot datasets if accessible
```

Plant taxonomic groups:

```text
vascular plants
mosses
lichens
fungi, especially habitat-indicating fungi
```

Do not use plant records directly as animal co-occurrence features. Convert them into habitat indicator surfaces.

### 5.3 Coarse environmental data

Recommended global/Nordic-compatible sources:

```text
Copernicus DEM or equivalent elevation
SRTM/GMTED fallback where needed
WorldClim or CHELSA climate normals
ERA5-Land climate/snow summaries
Copernicus Global Land Cover
CORINE Land Cover
JRC Global Surface Water
Global Forest Canopy Height
HydroSHEDS or national hydrography
OpenStreetMap-derived distance to coast/freshwater if quality is acceptable
```

The existing repository already has Earth Engine extraction utilities that can be extended.

### 5.4 Norway high-resolution sources

Recommended Norway sources:

```text
Kartverket DEM / hoydedata for terrain
NIBIO AR5 for land resource classes
NIBIO SR16 or forest resources if available
Miljodirektoratet / Naturbase for protected and nature-type layers
Artsdatabanken NiN-derived nature type layers where available
NGU bedrock geology
NGU surficial deposits
NGU geochemistry/mineral products where licensed and relevant
NVE hydrology/snow proxies where useful
```

### 5.5 Surrounding-country high-value data

If available, add equivalent national features for Sweden, Finland and Denmark. However, do not block v1 on perfect parity. A robust mask design is more important:

```text
feature_value = 0
feature_mask = 0
```

for unavailable features.

---

## 6. Plant Feature Construction

### 6.1 Indicator species groups

Create curated indicator groups from plant ecology knowledge.

Example:

```text
G_rich_fen = {plant species associated with base-rich fens}
G_poor_bog = {Sphagnum and bog-associated species}
G_calcareous_alpine = {calciphilous alpine species}
G_old_forest = {old forest vascular plants, lichens, mosses and fungi}
```

For each H3 cell `c`, plant group `g`, and scale radius `R`, define:

```text
I_g,R(c) =
    sum_{j in plant records of group g} K(d(c, j); R) * q_j * a_j
    ---------------------------------------------------------------
    epsilon + sum_{j in all plant records} K(d(c, j); R) * q_j
```

where:

```text
K(d; R) = exp(-d^2 / (2 R^2))
q_j = record quality weight
a_j = time-decay weight
epsilon = small stabilizer
```

Suggested time decay:

```text
a_j = exp(-max(0, year_now - year_j - grace_years) / tau_years)
```

Start values:

```text
grace_years = 10
tau_years = 25
```

For stable habitats, avoid aggressive decay. Old plant observations can still be useful if the habitat is persistent.

### 6.2 Effort correction

Raw indicator scores must be corrected for plant recording effort.

Define effort:

```text
E_R(c) =
    log1p(
        sum_{j in all plant records} K(d(c, j); R) * q_j
    )
```

Use both:

```text
I_g,R(c)
E_R(c)
```

as model inputs. Do not divide away all effort; the model needs to know when plant features are weakly supported.

### 6.3 Empirical Bayes shrinkage

For sparse plant data, shrink group indicators toward regional means.

```text
I_shrunk_g,R(c) =
    (n_eff(c) * I_g,R(c) + alpha * mu_g,region)
    --------------------------------------------
    n_eff(c) + alpha
```

where:

```text
n_eff(c) = effective plant effort around c
mu_g,region = mean indicator score for region/country/ecozone
alpha = shrinkage strength
```

Start values:

```text
alpha = 5 to 20
```

### 6.4 Cross-fitted plant features

To avoid leakage through shared observer hotspots:

```text
Split grid cells into spatial folds F_1..F_K.

For each fold k:
    build plant surfaces using plant records outside F_k
    predict/smooth plant indicators for cells inside F_k
    write cross-fitted plant features for F_k
```

Training uses:

```text
plant_habitat_crossfit_v1.parquet
```

Inference can use:

```text
plant_habitat_fullfit_v1.parquet
```

but validation metrics must be computed with cross-fitted features.

### 6.5 Plant community embedding option

After curated indicators work, add a low-rank plant community encoder.

Build a plant group vector:

```text
v_plant(c) in R^G
```

where `G` is number of plant indicator groups.

Then:

```text
h_plant = MLP([v_plant, effort_features, masks])
```

Optional self-supervised target:

```text
predict held-out plant indicator groups from environment
```

Do not start with thousands of plant species as direct animal-model inputs.

---

## 7. Input Schema For Training

Create:

```text
combined_nordic_biotic_v1.parquet
```

One row per:

```text
h3_index, week
```

Recommended columns:

```text
sample_id
h3_index
week
lat
lon
country_id
region_holdout_id
ecozone_id
observation_year_bucket

altitude_m
altitude_missing
altitude_source

common_env_*
common_env_mask_*

plant_*
plant_mask_*

animal_effort_bird_*
animal_effort_mammal_*

norway_highres_*
norway_highres_mask_*

bird_species_ids
mammal_species_ids
bird_label_weight
mammal_label_weight
```

Recommended country encoding:

```text
NO = 0
SE = 1
FI = 2
DK = 3
EE = 4
LV = 5
LT = 6
DE = 7
PL = 8
other = 9
```

---

## 8. Model Architecture

### 8.1 Coordinate encoder

Use the current Fourier/circular coordinate encoding:

```text
phi_lat(lat) = [sin(k lat_rad), cos(k lat_rad)] for k = 1..K
phi_lon(lon) = [sin(k lon_rad), cos(k lon_rad)] for k = 1..K
```

Then:

```text
h_coord = MLP([phi_lat, phi_lon])
```

### 8.2 Week encoder

Use Fourier week features plus learned week embedding.

```text
theta_w = 2 pi (week - 1) / 48
phi_week(w) = [sin(k theta_w), cos(k theta_w)] for k = 1..K_w
e_week = Embedding(week)
h_week = MLP([phi_week, e_week])
```

For week `0` yearly samples, either:

```text
use special learned yearly embedding
```

or avoid yearly samples in the biotic model v1.

### 8.3 Altitude encoder

Normalize:

```text
altitude_norm = clip((altitude_m - 500) / 1500, -2.0, 4.0)
```

Input:

```text
x_alt = [altitude_norm, altitude_missing, altitude_source_onehot]
h_alt = MLP(x_alt)
```

Apply altitude dropout during training:

```text
with probability p_alt_dropout:
    altitude_missing = 1
    altitude_norm = 0
    altitude_source = unknown
```

Start:

```text
p_alt_dropout = 0.10 to 0.20
```

### 8.4 Country/domain encoder

```text
h_country = Embedding(country_id)
```

Start:

```text
country_embedding_dim = 8
```

### 8.5 Common environment encoder

```text
x_common = concat(common_env_values, common_env_masks)
h_common = MLP_common(x_common)
```

Use robust normalization:

```text
x_cont = clip((x - median) / IQR, -5, 5)
```

for heavy-tailed variables.

### 8.6 Plant habitat encoder

```text
x_plant = concat(plant_indicator_values, plant_effort_values, plant_masks)
h_plant = MLP_plant(x_plant)
```

The plant encoder should be modest at first:

```text
Linear(input_dim, 128)
LayerNorm
GELU
Dropout(0.10)
Linear(128, 64)
LayerNorm
GELU
```

### 8.7 Shared trunk

Concatenate:

```text
h0 = concat(
    h_coord,
    h_week,
    h_alt,
    h_country,
    h_common,
    h_plant
)
```

Project:

```text
h = Linear(h0, D)
```

Then use residual blocks:

```text
h_{l+1} = h_l + Block_l(LayerNorm(h_l))
```

Apply week FiLM:

```text
[gamma_l, beta_l] = MLP_l(h_week)
h_{l+1} = gamma_l * h_{l+1} + beta_l
```

Bound gamma for stability:

```text
gamma_l = 1 + tanh(raw_gamma_l)
```

### 8.8 Bird and mammal heads

Use low-rank heads:

```text
u_bird = MLP_bird(h_shared) in R^K_bird
z_bird = u_bird W_bird^T + b_bird
```

```text
u_mammal = MLP_mammal(h_shared) in R^K_mammal
z_mammal = u_mammal W_mammal^T + b_mammal
```

where:

```text
W_bird in R^(n_birds x K_bird)
W_mammal in R^(n_mammals x K_mammal)
```

Start ranks:

```text
K_bird = 128
K_mammal = 64
```

### 8.9 Auxiliary heads

Add auxiliary reconstruction heads:

```text
env_pred = EnvHead(h_shared)
plant_habitat_pred = PlantHabitatHead(h_shared)
```

These encourage ecological structure in the shared representation.

Use masked losses:

```text
L_env = masked_huber(env_pred, env_target, env_mask)
L_plant_aux = masked_huber(plant_pred, plant_target, plant_mask)
```

### 8.10 Norway residual branch

For Norway:

```text
h_NO = NorwayEncoder([norway_highres_values, norway_highres_masks])
```

Residual:

```text
r_bird_NO = u_bird_NO W_bird_NO^T + b_bird_NO
r_mammal_NO = u_mammal_NO W_mammal_NO^T + b_mammal_NO
```

Gate:

```text
g_bird_NO = sigmoid(a_bird + GateBird([h_shared, h_NO, feature_quality]))
g_mammal_NO = sigmoid(a_mammal + GateMammal([h_shared, h_NO, feature_quality]))
```

Initialize:

```text
a = logit(0.15) = log(0.15 / 0.85) ~= -1.735
```

If Norway features are missing:

```text
g_NO -> 0
```

---

## 9. Loss Functions

### 9.1 Species losses

For birds:

```text
L_bird = SpeciesLoss(z_bird_final, y_bird)
```

For mammals:

```text
L_mammal = SpeciesLoss(z_mammal_final, y_mammal)
```

Recommended first loss:

```text
Assume-negative loss or ASL
```

Presence-only data should not be treated as perfect absence. If using BCE, use it as a baseline only.

### 9.2 Assume-negative loss

For sample `i`, positives `P_i`, sampled negatives `N_i`:

```text
L_AN(i) =
    1 / S * (
        lambda_pos * sum_{s in P_i} BCE(z_i,s, 1)
      + scale_neg * sum_{s in N_i} BCE(z_i,s, 0)
    )
```

where:

```text
S = number of species in the head
scale_neg = true_negative_count / sampled_negative_count
```

### 9.3 Asymmetric loss

For target `y` and probability `p = sigmoid(z)`:

```text
p_m = max(p - m, 0)
L_ASL =
    - y (1 - p)^gamma_pos log(p)
    - (1 - y) p_m^gamma_neg log(1 - p_m)
```

Start:

```text
gamma_pos = 0
gamma_neg = 2
m = 0.05
```

### 9.4 Total loss

```text
L_total =
    lambda_bird L_bird
  + lambda_mammal L_mammal
  + lambda_env L_env
  + lambda_plant_aux L_plant_aux
  + lambda_res L_residual
  + lambda_gate L_gate
  + lambda_smooth L_shared_smooth
  + lambda_NO_smooth L_NO_residual_smooth
  + lambda_border L_border
  + lambda_week L_week
  + lambda_week2 L_week_curvature
```

Start weights:

```text
lambda_bird = 1.0
lambda_mammal = 1.0
lambda_env = 0.05 to 0.10
lambda_plant_aux = 0.02 to 0.05
lambda_res = 1.0e-3
lambda_gate = 1.0e-4
lambda_smooth = 1.0e-4
lambda_NO_smooth = 3.0e-4
lambda_border = 1.0e-4
lambda_week = 1.0e-4
lambda_week2 = 5.0e-5
```

### 9.5 Residual penalty

```text
L_residual =
    mean_NO(r_bird_NO^2)
  + mean_NO(r_mammal_NO^2)
```

### 9.6 Gate penalty

```text
L_gate =
    mean_NO(g_bird_NO^2)
  + mean_NO(g_mammal_NO^2)
```

Optional species-count prior:

```text
g_prior_s = clip(log1p(n_NO_s) / log1p(n_ref), 0, 1) * g_max
L_gate_prior = mean_s (g_s - g_prior_s)^2
```

### 9.7 Shared smoothing

For neighbor pairs `(i, j)`:

```text
L_shared_smooth =
    sum_{(i,j)} w_ij || z_shared_i - z_shared_j ||^2
    -----------------------------------------------
    sum_{(i,j)} w_ij
```

with:

```text
w_ij =
    exp(-d_geo(i,j)^2 / sigma_geo^2)
  * exp(-d_alt(i,j)^2 / sigma_alt^2)
  * exp(-d_env(i,j)^2 / sigma_env^2)
```

Start:

```text
sigma_geo = 50 km
sigma_alt = 300 m
sigma_env = 1.5 normalized units
```

### 9.8 Norway residual smoothing

```text
L_NO_residual_smooth =
    sum_{(i,j) in NO} w_NO_ij || r_NO_i - r_NO_j ||^2
    -------------------------------------------------
    sum_{(i,j) in NO} w_NO_ij
```

with:

```text
w_NO_ij =
    exp(-d_geo^2 / sigma_geo_NO^2)
  * exp(-d_alt^2 / sigma_alt_NO^2)
  * exp(-d_plant_habitat^2 / sigma_plant_NO^2)
  * exp(-d_geology^2 / sigma_geology_NO^2)
```

Start:

```text
sigma_geo_NO = 10 km
sigma_alt_NO = 150 m
sigma_plant_NO = 1.0 normalized units
sigma_geology_NO = 1.0 normalized units
```

### 9.9 Border smoothing

Apply only to shared logits near country borders:

```text
L_border =
    mean || z_shared_NO_side - z_shared_neighbor_side ||^2
```

Do not smooth Norway residuals across borders.

### 9.10 Week embedding smoothness

```text
L_week = sum_w alpha_w || E_w+1 - E_w ||^2
```

Curvature:

```text
L_week2 = sum_w || E_w+1 - 2 E_w + E_w-1 ||^2
```

Use lower smoothness during migration:

```text
alpha_w = 0.25 * alpha_base for weeks 10-22 and 30-43
alpha_w = alpha_base otherwise
```

---

## 10. Training Process

### Stage 0: Data inventory and reproducibility

Deliverables:

```text
source inventory
license table
record counts by country/taxon/source/year
coordinate uncertainty histograms
taxon resolution report
H3 coverage maps
```

Checklist:

- [ ] List all raw occurrence sources.
- [ ] Record license and citation requirements.
- [ ] Record exact download queries and dates.
- [ ] Count bird, mammal and plant records by country.
- [ ] Count unique species by taxon group and country.
- [ ] Inspect coordinate uncertainty distributions.
- [ ] Decide maximum coordinate uncertainty thresholds.
- [ ] Create stable taxon IDs.
- [ ] Save raw archives as immutable inputs.

### Stage 1: Clean occurrence processing

Filters:

```text
valid coordinates
valid date
known taxon
species-level ID where required
coordinate uncertainty <= threshold
not obvious centroid/country capital coordinates
not fossil/captive/cultivated unless intentionally included
present records only
```

Suggested thresholds:

```text
animal max coordinate uncertainty: 1000 m to 5000 m
plant max coordinate uncertainty: 1000 m to 5000 m for fine features, larger only for coarse features
```

Checklist:

- [ ] Normalize GBIF/FinBIF/Artsobservasjoner schemas.
- [ ] Remove records without usable coordinates.
- [ ] Remove records with invalid dates.
- [ ] Remove records above uncertainty thresholds.
- [ ] Remove captive/cultivated records unless explicitly useful.
- [ ] Deduplicate by taxon/date/location/source/event.
- [ ] Assign `week` using the 48-week convention.
- [ ] Assign H3 cell.
- [ ] Assign country and region holdout ID.

### Stage 2: Build environmental grid

Checklist:

- [ ] Build H3 grid over full Nordic/surrounding-country domain.
- [ ] Extract common environmental features.
- [ ] Add masks for every feature.
- [ ] Normalize continuous features using robust statistics.
- [ ] Validate maps for each environmental feature.
- [ ] Check missingness by country.
- [ ] Check border artifacts.

### Stage 3: Build plant habitat surfaces

Checklist:

- [ ] Define plant taxon groups.
- [ ] Define curated plant indicator species lists.
- [ ] Compute plant effort features.
- [ ] Compute plant indicator surfaces at 1 km and 5 km.
- [ ] Apply time decay.
- [ ] Apply empirical Bayes shrinkage.
- [ ] Generate cross-fitted plant features.
- [ ] Generate full-fit plant features for inference.
- [ ] Validate indicator maps against known habitat maps.
- [ ] Inspect plant effort maps.
- [ ] Confirm sparse areas are masked or shrunk.

### Stage 4: Build animal labels

Checklist:

- [ ] Aggregate bird records by H3 cell and week.
- [ ] Aggregate mammal records by H3 cell and week.
- [ ] Build bird species vocabulary.
- [ ] Build mammal species vocabulary.
- [ ] Apply minimum observation thresholds.
- [ ] Compute label frequency statistics.
- [ ] Compute animal effort features.
- [ ] Create train/validation/test splits.

Suggested thresholds:

```text
bird min observations: 10 to 50 depending on scope
mammal min observations: 10 to 30, inspect rare species manually
```

### Stage 5: Baseline current-style model

Train a baseline using only:

```text
lat, lon, week
```

Optional:

```text
coarse env as auxiliary target only
```

Checklist:

- [ ] Train baseline bird head.
- [ ] Train baseline mammal head or separate mammal baseline.
- [ ] Compute Norway-only metrics.
- [ ] Compute spatial holdout metrics.
- [ ] Plot selected species range maps.
- [ ] Plot week curves.
- [ ] Save calibration diagnostics.

### Stage 6: Shared Nordic ecological model

Inputs:

```text
lat, lon, week
altitude
country_id
common_env
plant_habitat
feature masks
```

No Norway residual yet.

Checklist:

- [ ] Add dataset support for grouped inputs.
- [ ] Add altitude encoder.
- [ ] Add country embedding.
- [ ] Add common environment input encoder.
- [ ] Add plant habitat encoder.
- [ ] Add separate bird and mammal heads.
- [ ] Train shared Nordic model.
- [ ] Compare against baseline.
- [ ] Run plant feature ablation.
- [ ] Run altitude/country ablation.
- [ ] Inspect species that improve/worsen most.

### Stage 7: Norway residual v1

Inputs:

```text
shared trunk output
Norway high-resolution features
Norway feature masks
```

Train:

```text
Norway residual encoder
Norway residual low-rank bird head
Norway residual low-rank mammal head
Norway residual gates
Norway auxiliary heads
```

Checklist:

- [ ] Freeze most shared trunk initially.
- [ ] Train Norway residual branch.
- [ ] Apply residual magnitude penalty.
- [ ] Apply gate penalty.
- [ ] Add Norway residual smoothing.
- [ ] Inspect residual maps.
- [ ] Disable residual for species where validation worsens.
- [ ] Verify no hard discontinuity at Sweden/Finland borders.

### Stage 8: Joint fine-tuning

Batch mix:

```text
Norway: 50-60 percent
Sweden: 10-20 percent
Finland: 10-20 percent
Denmark and Baltic/buffer: remainder
```

Learning rates:

```text
shared lower trunk: 0 or 1e-5
shared upper trunk: 1e-5 to 3e-5
animal heads: 1e-4
Norway residual branch: 1e-4
gates: 1e-4
```

Checklist:

- [ ] Unfreeze upper shared layers.
- [ ] Use lower LR for shared trunk.
- [ ] Continue residual penalties.
- [ ] Monitor border discontinuity.
- [ ] Monitor Norway holdout metrics.
- [ ] Monitor non-Norway metrics for regression.

### Stage 9: Calibration and pruning

Checklist:

- [ ] Calibrate bird probabilities.
- [ ] Calibrate mammal probabilities separately.
- [ ] Compute per-species AP and Brier score.
- [ ] Compute reliability curves.
- [ ] Prune species with insufficient support.
- [ ] Shrink or disable Norway residuals that create artifacts.
- [ ] Export diagnostic metadata.

---

## 11. Validation Design

Use multiple validation modes.

### 11.1 Random split

Useful for debugging only. Not enough for range validation.

### 11.2 Spatial block split

Split by H3 parent cells or custom blocks:

```text
train: blocks not in validation
validation: held-out blocks
```

### 11.3 Norway region holdouts

Recommended Norway holdout regions:

```text
southeast lowland
southwest coast
western fjords
Trondelag
Nordland coast
Troms
Finnmark
inland valleys
alpine/high mountain
boreal forest interior
```

### 11.4 Altitude band holdout

Example bands:

```text
0-100 m
100-300 m
300-700 m
700-1100 m
1100+ m
```

### 11.5 Habitat holdout

Hold out cells dominated by:

```text
wetland
old forest
alpine
coastal heath
agriculture
urban
rich deciduous forest
```

### 11.6 Cross-border validation

Hold out bands near:

```text
Norway-Sweden
Norway-Finland
Sweden-Finland
Denmark-Sweden
```

Measure discontinuity:

```text
D_border =
    mean_{matched border pairs} || p_left - p_right ||_1
```

Use species-specific interpretation. Some discontinuity is real, but hard artificial jumps are bad.

---

## 12. Metrics

Core metrics:

```text
mAP bird
mAP mammal
Norway mAP bird
Norway mAP mammal
rare species mAP
top-k recall
Brier score
expected calibration error
GeoScore
```

Spatial diagnostics:

```text
spatial roughness
residual roughness
border discontinuity
altitude response smoothness
habitat response plausibility
plant-feature sensitivity
```

For species `s`, range-map roughness:

```text
R_s =
    mean_{neighbor cells i,j} (p_i,s - p_j,s)^2
```

Altitude response:

```text
A_s(a) = mean probability for species s over samples binned by altitude a
```

Residual magnitude:

```text
M_s_NO = mean_NO |g_s_NO(x) * r_s_NO(x)|
```

Gate strength:

```text
G_s_NO = mean_NO g_s_NO(x)
```

---

## 13. Diagnostics To Generate

For selected birds and mammals:

```text
shared probability map
final probability map
Norway residual map
Norway gate map
plant habitat indicator overlay
altitude response curve
weekly curve
validation observation overlay
calibration curve
```

Species groups to inspect:

```text
common forest birds
alpine birds
wetland birds
coastal birds
large raptors
owls
woodpeckers
deer species
hare species
large carnivores
small mustelids
bats if included
semi-aquatic mammals
```

Bad signs:

```text
isolated hotspot speckles
hard country-border jumps
high residual where plant effort is high but habitat is not special
range expansion caused by plant recorder density
unrealistic alpine/coastal leakage
high mammal probabilities around reporting hotspots only
```

---

## 14. Leakage Controls

### 14.1 Do not use raw animal observations as features

Do not feed:

```text
local bird counts
local mammal counts
raw prey records
raw predator records
```

as direct features for other animal species.

### 14.2 If prey/guild features are added later

Use out-of-fold predicted suitability:

```text
For fold k:
    train prey model on folds != k
    predict prey suitability on fold k
    use predicted prey suitability as feature
```

Formula:

```text
prey_feature_s,k(x) = sigmoid(f_prey_trained_without_fold_k(x))
```

Never:

```text
prey_feature(x) = raw prey observations near x
```

### 14.3 Plant leakage controls

Plants are allowed as habitat indicators, but still require controls:

- [ ] Cross-fit plant surfaces.
- [ ] Include plant effort features.
- [ ] Include plant data age.
- [ ] Smooth and shrink sparse plant indicators.
- [ ] Validate plant indicators independently.
- [ ] Run ablation with and without plant effort controls.

---

## 15. Suggested Configuration

```yaml
model:
  name: nordic_norway_biotic_v1
  n_weeks: 48

  inputs:
    coordinates: true
    week: true
    altitude: true
    country_id: true
    common_env: true
    plant_habitat: true
    norway_highres: true
    feature_masks: true

  coordinate_encoder:
    harmonics: 4
    dim: 128

  week_encoder:
    fourier_harmonics: 8
    learned_dim: 64
    film: true

  altitude_encoder:
    dim: 32
    dropout: 0.15
    center_m: 500
    scale_m: 1500
    clip: [-2.0, 4.0]

  country_embedding:
    dim: 8

  common_env_encoder:
    hidden_dims: [128, 64]
    dropout: 0.10

  plant_encoder:
    hidden_dims: [128, 64]
    dropout: 0.10
    use_effort_controls: true

  shared_trunk:
    hidden_dim: 512
    blocks: 6
    dropout: 0.10

  bird_head:
    type: low_rank
    rank: 128
    hidden_dim: 512

  mammal_head:
    type: low_rank
    rank: 64
    hidden_dim: 256

  auxiliary_heads:
    env_reconstruction: true
    plant_habitat_reconstruction: true

  norway_residual:
    enabled: true
    active_country: NO
    rank_bird: 64
    rank_mammal: 32
    gate_initial: 0.15
    residual_l2: 1.0e-3
    gate_l2: 1.0e-4
    local_spatial_residual: false

training:
  species_loss: asl
  asl_gamma_pos: 0.0
  asl_gamma_neg: 2.0
  asl_clip: 0.05
  batch_size: 1024
  epochs: 200
  lr_shared: 2.0e-4
  lr_residual: 1.0e-4
  weight_decay: 1.0e-4
  patience: 30
  mixed_precision: true

loss_weights:
  bird: 1.0
  mammal: 1.0
  env: 0.10
  plant_aux: 0.03
  residual_l2: 1.0e-3
  gate_l2: 1.0e-4
  shared_smooth: 1.0e-4
  norway_residual_smooth: 3.0e-4
  border: 1.0e-4
  week_smooth: 1.0e-4
  week_curvature: 5.0e-5
```

---

## 16. Repository Implementation Checklist

### Data pipeline

- [ ] Add `utils/features_common.py`.
- [ ] Add `utils/features_plants.py`.
- [ ] Add `utils/features_norway.py`.
- [ ] Add `utils/build_biotic_training_table.py`.
- [ ] Add source metadata writer.
- [ ] Extend `utils/gbifutils.py` or add a normalized occurrence converter.
- [ ] Extend `utils/combine.py` or create a new combiner for separate bird/mammal labels.
- [ ] Add grouped feature masks.
- [ ] Add cross-fitted plant feature generation.

### Model code

- [ ] Extend dataset to return grouped tensors.
- [ ] Add altitude encoder.
- [ ] Add country embedding.
- [ ] Add common environment input encoder.
- [ ] Add plant habitat encoder.
- [ ] Add bird head.
- [ ] Add mammal head.
- [ ] Add optional Norway residual branch.
- [ ] Add gate outputs to model diagnostics.
- [ ] Keep backwards-compatible lat/lon/week inference path.

### Loss code

- [ ] Add dual-head species loss.
- [ ] Add plant auxiliary reconstruction loss.
- [ ] Add residual L2 loss.
- [ ] Add gate penalty.
- [ ] Add shared smoothing loss.
- [ ] Add Norway residual smoothing loss.
- [ ] Add border smoothing loss.
- [ ] Add week embedding smoothness and curvature losses.

### Training CLI

- [ ] Add YAML config support.
- [ ] Add `--training_stage`.
- [ ] Add `--country_features`.
- [ ] Add `--altitude_input`.
- [ ] Add `--common_env_input`.
- [ ] Add `--plant_features`.
- [ ] Add `--norway_residual`.
- [ ] Add `--bird_head` and `--mammal_head` options if needed.
- [ ] Add split config for spatial holdouts.

### Inference

- [ ] Support old mode: `lat`, `lon`, `week`.
- [ ] Support richer mode: altitude, country, common env, plant features.
- [ ] Support Norway feature lookup.
- [ ] Return diagnostic output:
  - [ ] shared probability
  - [ ] residual logit
  - [ ] gate
  - [ ] final probability
  - [ ] missing feature flags

### Plotting and reports

- [ ] Bird range maps.
- [ ] Mammal range maps.
- [ ] Plant indicator maps.
- [ ] Residual maps.
- [ ] Gate maps.
- [ ] Border discontinuity plots.
- [ ] Altitude response plots.
- [ ] Plant-feature ablation report.
- [ ] Calibration report.

---

## 17. Minimum Useful Version

The first version that is worth training should include:

- [ ] Bird labels.
- [ ] Mammal labels.
- [ ] Shared H3 grid.
- [ ] Common environmental features.
- [ ] Altitude input.
- [ ] Country embedding.
- [ ] Plant effort features.
- [ ] 8-15 curated plant habitat indicators.
- [ ] Separate bird and mammal heads.
- [ ] Spatial holdout validation.
- [ ] Plant ablation experiment.

Do not include in minimum useful version:

- [ ] Raw prey features.
- [ ] Local spatial Norway residual.
- [ ] Thousands of raw plant species inputs.
- [ ] Graph neural network.
- [ ] Unregularized Norway-only species head.
- [ ] Non-cross-fitted plant surfaces for validation.

---

## 18. Recommended Ablations

Run these in order:

```text
A0: lat/lon/week only
A1: + altitude
A2: + country
A3: + common environment inputs
A4: + plant effort controls only
A5: + plant habitat indicators
A6: + Norway high-resolution features
A7: + Norway residual branch
```

For each ablation, report:

```text
global bird mAP
Norway bird mAP
global mammal mAP
Norway mammal mAP
rare species mAP
spatial holdout mAP
calibration
roughness
border discontinuity
```

Key question:

```text
Do plant indicators improve spatial holdout performance after controlling for plant effort?
```

If yes, they are probably adding habitat information. If no, they may only encode survey density or redundant environment.

---

## 19. Main Risks

| Risk | Mitigation |
|---|---|
| Plant data encodes botanist effort rather than habitat | Cross-fitting, effort features, data age, shrinkage |
| Mammal records are too biased | Separate mammal head, source flags, source-specific diagnostics |
| Norway residual memorizes hotspots | Small gate, residual L2, residual smoothing, spatial holdouts |
| Border discontinuities | Shared trunk across countries, border smoothing on shared logits |
| Too many plant indicators | Start curated and low-dimensional |
| Coarse env differs by country/source | Country embedding, masks, source metadata |
| Rare species unstable | Minimum observations, per-species diagnostics, residual pruning |
| Mammal seasonality confused with detectability | Separate mammal calibration and seasonal diagnostics |

---

## 20. Summary

The strongest model for this dataset is not a Norway-only model and not a raw species co-occurrence model. It should be a shared Nordic ecological model with plant-derived habitat indicators available across Norway and surrounding countries, plus a regularized Norway residual branch for high-resolution Norwegian feature corrections.

The best use of plant data is:

```text
plant records
  -> effort-corrected habitat indicator surfaces
  -> shared ecological trunk
  -> bird and mammal range probabilities
```

The model should first prove that plant indicators improve spatial holdout performance beyond altitude, climate, landcover and plant effort. Only after that should more complex Norway residuals or prey/guild interaction features be added.

