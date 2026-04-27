# Data Cleaning and Feature Engineering Plan

**File:** `data_cleaning_plan_20260427.md`
**Date:** 2026-04-27
**Status:** Data and feature engineering plan

---

## 1. Goal

Build a clean, versioned, reproducible dataset for a Norway-specialized Nordic bird occurrence model.

The dataset must combine:

- bird observations from Norway, Sweden, Finland, Denmark,
- common environmental features available across countries,
- Norway-only high-resolution features,
- Norway geology/mineral features,
- Norway non-bird biotic indicator features,
- observation effort controls,
- validation split metadata.

The core rule:

```text
all countries contribute to shared Nordic ecology,
Norway-only data contributes only through masked Norway residual features.
```

---

## 2. Raw data sources

### 2.1 Bird observations

Candidate sources:

```text
GBIF
Artsdatabanken / Artsobservasjoner
eBird, if licensed/available
national biodiversity portals
research/museum datasets
```

For the Norway model, the minimal Nordic countries are:

```text
Norway
Sweden
Finland
Denmark
```

### 2.2 Non-bird observations for Norway

Use only non-bird taxa for biotic features:

```text
vascular plants
mosses
lichens
fungi
beetles
butterflies
moths if reliable
dragonflies
molluscs
other useful invertebrates
```

Exclude:

```text
birds
uncertain taxonomic groups
records with poor spatial precision
records with unresolved taxonomy
records likely to be captive/cultivated when relevant
```

### 2.3 Environmental/geospatial data

Common Nordic:

```text
DEM/elevation
climate
land cover
water/coast
urban
forest/canopy if consistent
```

Norway-specific:

```text
10 m / high-resolution DEM
AR5 / land resource classes
NiN / nature types if available
forest type/productivity/age proxies
wetlands
hydrology
coastline
NGU bedrock/geology
NGU calcium/ecological bedrock classes
geochemistry/mineral datasets where available
```

---

## 3. General observation cleaning

### 3.1 Required fields

Each observation must have:

```text
taxon_id
scientific_name
event_date or year
latitude
longitude
coordinate_uncertainty_m
country
basis/source
```

Optional but useful:

```text
observer_id
dataset_id
institution_code
sampling_protocol
occurrence_status
individual_count
sex
life_stage
behavior
```

### 3.2 Coordinate cleaning

Reject or flag records with:

```text
missing coordinates
coordinates outside country boundary
coordinate_uncertainty_m > threshold
zero/centroid coordinates
obvious institution coordinates
swapped lat/lon
marine coordinates for terrestrial species unless expected
```

Suggested uncertainty thresholds:

```text
bird training observations:
    accept <= 1000 m, prefer <= 250 m

biotic indicator features:
    accept <= 1000 m for broad features
    accept <= 250 m for fine-scale indicators

geology/ecosystem features:
    grid-derived, no observation uncertainty
```

For rare species, use stricter QA if records are sensitive or spatially generalized.

### 3.3 Temporal cleaning

Fields:

```text
year
month
day
week_48
```

Convert date to 48-week BirdNET-style week index if matching upstream.

Reject or flag:

```text
missing date
implausible date
future date
large date uncertainty
```

For static habitat indicators, older records can be useful. For dynamic indicators, apply time decay.

### 3.4 Taxonomy normalization

Use one taxonomy table for all sources.

Build mapping:

```text
source_taxon_id -> canonical_taxon_id -> model_species_id
```

Handle:

```text
synonyms
subspecies
species aggregates
hybrids
uncertain identifications
taxonomic splits/lumps
Norwegian/common names
scientific names
```

Rules:

```text
species-level records preferred
subspecies rolled up unless model has subspecies label
genus-only records excluded from species labels
uncertain records excluded or downweighted
```

### 3.5 Duplicate handling

Deduplicate records that likely represent the same observation.

Candidate keys:

```text
canonical_taxon_id
rounded_lat
rounded_lon
date
observer_id
dataset_id
source_record_id
```

Use spatial rounding according to uncertainty.

Keep source IDs to allow audit.

---

## 4. Bird observation-specific cleaning

### 4.1 Label construction

The model is multi-label per location/week.

Two possible training layouts:

1. Occurrence-positive sampling:
   - each bird observation is a positive species event,
   - negatives are sampled from background/pseudo-absence.

2. Cell-week multi-label:
   - aggregate observations to H3 cell × week × year,
   - species vector has positives for observed species.

Recommended for this model:

```text
cell-week-year multi-label table
+ source-quality weights
+ background sampling
```

### 4.2 Observation effort

Citizen science data are biased.

Estimate effort proxies:

```text
bird_observation_count_cell_week
bird_species_count_cell_week
observer_count_cell_week
days_with_observations_cell_week
distance_to_road
distance_to_settlement
distance_to_trail
protected_area_indicator
```

Use effort for:

```text
sample weighting
background selection
bias diagnostics
calibration
```

Do not let effort become the main predictor of species probability.

### 4.3 Spatial thinning

Thin overly dense clusters.

Options:

```text
one observation per species per small grid per day
cap observations per species per H3 cell/week
weighted sampling inversely proportional to local density
```

Do not over-thin rare species.

---

## 5. Non-bird biotic indicator construction

### 5.1 Exclude birds

For bird model features:

```text
exclude all Aves from biotic features
```

This prevents circularity.

### 5.2 Indicator group catalogue

Build a curated file:

```text
biotic_indicator_groups.yaml
```

Example:

```yaml
old_growth_forest_indicator:
  description: Species associated with old or continuity forest.
  taxa:
    - taxon_id: ...
      indicator_strength: 1.0
    - taxon_id: ...
      indicator_strength: 0.7

rich_fen_indicator:
  description: Calcium-rich fen, base-rich wetland.
  taxa:
    - taxon_id: ...
      indicator_strength: 1.0
```

Each group should include:

```text
description
ecological interpretation
taxon list
indicator strength
taxonomic groups included
recommended time decay
recommended spatial radius
known biases
```

### 5.3 Indicator score formula

For location `x`, radius `r`, group `G`:

```text
score_G,r(x) =
    sum_i indicator_strength_i
        * quality_weight_i
        * distance_kernel(d_i, r)
        * time_decay_i
        / effort_correction(x)
```

Distance kernel:

```text
exp(-d^2 / sigma_r^2)
```

Time decay:

```text
exp(-(prediction_year - observation_year) / tau_G)
```

Suggested `tau_G`:

```text
old-growth forest / lichens / fungi: 20–50 years
geology-linked flora: 20–50 years
wet meadow / reedbed / marsh: 5–20 years
insect communities: 5–15 years
temporary wetlands: 3–10 years
```

### 5.4 Effort controls

For each radius:

```text
nonbird_observation_count
nonbird_observer_count
nonbird_species_count
nonbird_observation_days
taxon_group_count
```

Keep these separate from indicator scores.

### 5.5 Cross-fitting

To avoid leakage:

```text
for each validation fold:
    build biotic indicator surfaces without records from that fold
```

For static production maps, a full-data indicator surface can be built later, but validation must be cross-fitted.

### 5.6 Learned biotic embeddings

Later version:

```text
H3 cell × non-bird taxon matrix
    -> TF-IDF / normalized counts
    -> PCA, NMF, or autoencoder
    -> 16–32 dimensional biotic embedding
```

Rules:

```text
exclude birds
cross-fit embeddings
include effort controls
avoid very rare taxa unless grouped
```

---

## 6. Geology and mineral feature construction

### 6.1 Bedrock features

For each grid cell and buffer:

```text
carbonate_bedrock_fraction
base_rich_bedrock_fraction
acidic_bedrock_fraction
mafic_bedrock_fraction
ultramafic_bedrock_fraction
lithology_embedding
```

### 6.2 Calcium and mineral features

Raw elements:

```text
Ca, Mg, P, K, Na, S, Fe, Mn, Zn, Cu
```

Derived indices:

```text
base_richness_index = z(Ca) + z(Mg) + 0.5*z(K)
calcareousness_index = z(Ca) + carbonate_bedrock_indicator
nutrient_richness_index = z(P) + z(K) + z(Ca) + z(Mg)
acidic_soil_proxy = acidic_bedrock_fraction - base_rich_bedrock_fraction
trace_metal_stress_index = z(Cu) + z(Zn) + z(Ni) + z(Cr) + z(Co)
```

Use robust transforms:

```text
log1p for positive concentrations
winsorization
quantile normalization
robust z-score
```

### 6.3 Aggregation radii

Compute at:

```text
50 m
250 m
1000 m
5000 m
```

For sparse geochemistry points, prefer interpolated surfaces or coarse class indicators over unreliable point lookup.

---

## 7. Terrain, hydrology, ecosystem feature construction

### 7.1 Terrain

From DEM:

```text
elevation
slope
aspect_sin
aspect_cos
ruggedness
topographic_position_index
topographic_wetness_index
solar exposure
cold-air pooling proxy
```

### 7.2 Hydrology

```text
distance_to_coast
distance_to_lake
distance_to_river
distance_to_wetland
wetland_fraction
freshwater_fraction
marine_influence_proxy
```

### 7.3 Land cover/ecosystem

```text
forest_fraction
conifer_fraction
deciduous_fraction
old_forest_proxy
bog_fraction
open_mountain_fraction
agriculture_fraction
urban_fraction
AR5 classes
NiN classes
```

Aggregate fractions in buffers.

---

## 8. Missingness and masks

Each feature group must have masks.

```text
common_env_mask_*
norway_env_mask_*
biotic_indicator_mask_*
geology_mask_*
```

Rules:

```text
real missingness is not zero
zero means zero only when mask says valid
model receives mask
normalization ignores missing values
```

---

## 9. Normalization

Store fitted transforms from training data only.

Recommended:

```text
continuous positive skew: log1p + robust z-score
fractions: logit transform or keep 0–1 with clipping
distances: log1p(distance_m)
categoricals: integer IDs + embeddings
counts: log1p counts
indicator scores: quantile transform or log1p + robust z-score
```

Save:

```text
feature_stats.json
category_maps.json
normalization_version
```

---

## 10. Dataset versioning

Each dataset build should output:

```text
combined_nordic_norway_YYYYMMDD.parquet
feature_stats_YYYYMMDD.json
data_manifest_YYYYMMDD.json
schema_YYYYMMDD.json
```

Manifest contents:

```text
source datasets
download dates
record counts before/after cleaning
taxon mapping version
feature rasters and versions
normalization parameters
validation split IDs
code commit
```

---

## 11. Data quality reports

Produce reports:

```text
record_count_by_country
record_count_by_species
record_count_by_year
record_count_by_source
coordinate_uncertainty_distribution
taxon_mapping_failures
duplicate_count
missing_feature_rates
feature_distribution_plots
biotic_indicator_maps
Norway feature coverage maps
```

Do not train until the reports look sane.

---

## 12. Leakage controls

Critical rules:

```text
do not use bird observations as model input features
do not build biotic features from validation fold records
do not use future non-bird records for time-strict validation
do not use validation region records to compute local residual features
do not use model labels or validation labels to select indicator taxa
```

If using full-data indicator surfaces for final production, report validation with cross-fitted features separately.

---

## 13. Output tables

### 13.1 Observation table

```text
observations_clean.parquet
```

### 13.2 Grid feature table

```text
grid_features.parquet
```

### 13.3 Training table

```text
combined_training.parquet
```

### 13.4 Validation split table

```text
validation_splits.parquet
```

### 13.5 Feature metadata

```text
feature_metadata.yaml
```

Each feature should have:

```text
name
group
source
resolution
aggregation_radius
transform
valid_range
missing_value_rule
mask_column
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

